"""LLM-based intent parser using OpenRouter (free tier).

Falls back to the rule-based parser when ``OPENROUTER_API_KEY`` is not set
or when the LLM call fails, so the system always produces a result.
"""

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from aegis.models.state import AegisState, TrainingSpec, RunEvent
from aegis.nodes.intent import parse_intent_node  # rule-based fallback


# ---------------------------------------------------------------------------
# Structured output schema for the LLM
# ---------------------------------------------------------------------------

class ParsedIntent(BaseModel):
    """Structured output the LLM should return."""
    method: Literal["full_finetune", "lora", "qlora"] = "lora"
    model_name: str = Field(description="Full HuggingFace model ID, e.g. 'anthonym21/Eve-2-MoE-272M'")
    dataset_path: str = Field(default="./data/sample.jsonl", description="HuggingFace dataset name or local path")
    num_epochs: int = Field(default=3, ge=1, le=100)
    micro_batch_size: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=5e-5, gt=0)
    seq_len: int = Field(default=512, ge=64, le=8192)
    target_gpu: str = Field(default="a10g", description="GPU type: t4, a10g, or a100")
    estimated_params_b: float = Field(default=0.0, ge=0, description="Estimated total parameter count in billions (e.g. 0.272 for 272M, 7.0 for 7B). Set to 0 if unknown.")
    is_moe: bool = Field(default=False, description="True if the model is a Mixture-of-Experts architecture")
    streaming: bool = Field(default=False, description="True if the dataset should be streamed (for large datasets >1GB)")
    max_steps: int = Field(default=0, ge=0, description="Fixed step count. Use >0 for streaming datasets or very large datasets (e.g. 10000). 0 = use num_epochs.")
    trust_remote_code: bool = Field(default=False, description="True for custom model architectures (MoE, custom attention, etc.)")

    @field_validator("learning_rate")
    @classmethod
    def clamp_learning_rate(cls, v: float) -> float:
        """Clamp absurd learning rates. Anything above 1e-2 is almost certainly wrong."""
        if v > 1e-2:
            return 2e-4
        return v


# ---------------------------------------------------------------------------
# LLM parser
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a training configuration assistant for the Aegis-ML fine-tuning orchestrator.

Given a user's natural language description of a fine-tuning job, extract a structured training specification.

IMPORTANT RULES:
1. Use the EXACT model name the user provides (e.g. "anthonym21/Eve-2-MoE-272M" stays as-is). Do NOT substitute different models.
2. If the user provides a HuggingFace URL like https://huggingface.co/org/model, extract "org/model" as the model_name.
3. For dataset, use the name the user provides. If they say "cosmopedia", use "HuggingFaceTB/cosmopedia". If a URL like https://huggingface.co/datasets/org/name, extract "org/name".
4. learning_rate MUST be between 1e-6 and 1e-2. Typical values: 5e-5 for full finetune, 2e-4 for LoRA/QLoRA. NEVER return values like 5.0 or 1.0.
5. Estimate the model's total parameter count in billions (estimated_params_b). Look for hints in the name: "272M" = 0.272, "7B" = 7.0, "70B" = 70.0.
6. Set is_moe=true if the model name contains "MoE", "mixture", or "expert".
7. For MoE models: use qlora, micro_batch_size=1 or 2, a100 GPU, and trust_remote_code=true.
8. For models > 3B params: recommend a100 GPU. For < 1B: a10g is fine. For 1-3B: a10g with qlora.
9. For HuggingFace Hub datasets (anything that's NOT a local file path): set streaming=true and max_steps=10000 (or scale with user-specified token count: 1B tokens ~ 10000 steps, 100M ~ 1000 steps).
10. For local file datasets (paths starting with ./ or /): set streaming=false and max_steps=0.
11. Set trust_remote_code=true for custom architectures (MoE, custom attention, any non-standard model).

Default values (use when user doesn't specify):
- method: lora (qlora for large/MoE models)
- num_epochs: 3
- micro_batch_size: 4 (smaller for large models)
- learning_rate: 2e-4 for lora/qlora, 5e-5 for full_finetune
- seq_len: 512
- target_gpu: a10g (a100 for >3B or MoE)
- dataset_path: ./data/sample.jsonl
- streaming: false (true for Hub datasets)
- max_steps: 0 (>0 for streaming)
- trust_remote_code: false (true for MoE/custom models)"""


def _get_llm():
    """Create a ChatOpenAI instance pointing at OpenRouter."""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    return ChatOpenAI(
        model="arcee-ai/trinity-large-preview:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=512,
    )


def parse_intent_llm(state: AegisState, user_input: str) -> AegisState:
    """Parse user intent via LLM with structured output.

    Falls back to the rule-based parser if:
    - ``OPENROUTER_API_KEY`` is not set
    - The LLM call raises an exception
    """
    llm = _get_llm()
    if llm is None:
        return parse_intent_node(state, user_input=user_input)

    try:
        structured_llm = llm.with_structured_output(ParsedIntent)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        parsed: ParsedIntent = structured_llm.invoke(messages)

        spec = TrainingSpec(
            method=parsed.method,
            model_name=parsed.model_name,
            dataset_path=parsed.dataset_path,
            num_epochs=parsed.num_epochs,
            micro_batch_size=parsed.micro_batch_size,
            learning_rate=parsed.learning_rate,
            seq_len=parsed.seq_len,
            target_gpu=parsed.target_gpu,
            streaming=parsed.streaming,
            max_steps=parsed.max_steps,
            trust_remote_code=parsed.trust_remote_code,
        )

        event = RunEvent(
            phase="parse_intent",
            status="completed",
            message=f"LLM extracted {spec.method} spec for {spec.model_name}",
            data={
                "parsed_spec": spec.model_dump(),
                "parser": "llm",
                "estimated_params_b": parsed.estimated_params_b,
                "is_moe": parsed.is_moe,
            },
        )

        return state.model_copy(update={"spec": spec, "events": state.events + [event]})

    except Exception as e:
        # Log the LLM failure then fall back to rule-based
        event = RunEvent(
            phase="parse_intent",
            status="completed",
            message=f"LLM parse failed ({e}), falling back to rule-based parser",
            data={"parser": "rule_fallback", "error": str(e)},
        )
        fallback_state = parse_intent_node(state, user_input=user_input)
        return fallback_state.model_copy(
            update={"events": [event] + list(fallback_state.events)}
        )
