import re
from aegis.models.state import AegisState, TrainingSpec, RunEvent


def parse_intent_node(state: AegisState, user_input: str) -> AegisState:
    """Parse user intent and extract TrainingSpec. MVP: Rule-based extraction."""
    try:
        # Extract method
        method = "lora"  # default
        if "full" in user_input.lower() and "finetune" in user_input.lower():
            method = "full_finetune"
        elif "qlora" in user_input.lower() or "q-lora" in user_input.lower():
            method = "qlora"
        elif "lora" in user_input.lower():
            method = "lora"

        # Extract model name
        model_map = {
            "tinyllama": "tinyllama/tinyllama-272m",
            "llama-7b": "meta-llama/Llama-2-7b-hf",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        }
        model_name = "tinyllama/tinyllama-272m"  # default
        for key, value in model_map.items():
            if key in user_input.lower():
                model_name = value
                break

        # Extract batch size if mentioned
        batch_match = re.search(r'batch\s*(?:size)?\s*(\d+)', user_input.lower())
        batch_size = int(batch_match.group(1)) if batch_match else 4

        # Extract epochs if mentioned
        epoch_match = re.search(r'(\d+)\s*epoch', user_input.lower())
        num_epochs = int(epoch_match.group(1)) if epoch_match else 3

        spec = TrainingSpec(
            method=method,
            model_name=model_name,
            dataset_path="./data/sample.jsonl",
            num_epochs=num_epochs,
            micro_batch_size=batch_size,
        )

        event = RunEvent(
            phase="parse_intent",
            status="completed",
            message=f"Extracted {method} training spec for {model_name}",
            data={"parsed_spec": spec.model_dump()}
        )

        return state.model_copy(update={"spec": spec, "events": state.events + [event]})

    except Exception as e:
        event = RunEvent(
            phase="parse_intent",
            status="failed",
            message=f"Failed to parse intent: {str(e)}"
        )
        return state.model_copy(update={"events": state.events + [event]})
