# fitcheck

> **Know before you train** — VRAM estimation for LLM fine-tuning.

fitcheck predicts GPU memory usage from first principles. Given a model, GPU, and training method, it tells you whether your config will fit — before you spend an hour discovering it won't.

## Why fitcheck?

Fine-tuning LLMs means guessing at batch sizes and hoping you don't OOM. The feedback loop is brutal: pick a config, wait for the run to start, crash 2 minutes in, adjust, repeat.

fitcheck collapses that loop. It computes each VRAM component — model weights, optimizer states, gradients, activations, the logits buffer, eval KV-cache spikes — and produces a breakdown with confidence bounds.

## What it computes

| Component | What it is | Why it matters |
|-----------|-----------|----------------|
| **Model weights** | Base params in training dtype (bf16/NF4) | 4.2 GB for QLoRA 8B, 16 GB for full bf16 |
| **Optimizer states** | AdamW momentum + variance per trainable param | Dominates full fine-tune (~60 GB for 8B) |
| **Gradients** | One gradient per trainable param | Small for LoRA, huge for full FT |
| **Activations** | Per-layer stored tensors for backward pass | Flash-attention-aware, scales with batch × seq |
| **Logits buffer** | batch × seq × vocab × 4 bytes (float32) | The surprise OOM — 2 GB at bs=4 with 128k vocab |
| **Eval KV-cache** | Spike during evaluation steps | Can exceed training steady-state |

## Quick Start

```bash
pip install -r requirements.txt

# Run tests
pytest
```

```python
from fitcheck.hub.resolver import resolve_from_config
from fitcheck.hardware.registry import get_hardware
from fitcheck.profilers.vram.engine import VRAMEstimator
from fitcheck.models.profiles import TrainingMethod, LoRAConfig

# QLoRA Llama 8B on an RTX 3090
estimator = VRAMEstimator()
breakdown = estimator.estimate(
    model=resolve_from_config("meta-llama/Llama-3.1-8B", config),
    hardware=get_hardware("3090"),
    method=TrainingMethod.QLORA,
    batch_size=4,
    seq_len=1024,
    lora_config=LoRAConfig(rank=16),
)

print(f"Steady-state: {breakdown.steady_state_gb:.1f} GB")
print(f"Usable VRAM:  {get_hardware('3090').usable_vram_gb} GB")
# Steady-state: 16.6 GB
# Usable VRAM:  22.8 GB  ← fits with 6 GB headroom
```

## Development

```bash
pytest                                              # Run all tests (51)
pytest tests/fitcheck/profilers/test_estimator.py -v  # End-to-end estimator tests
ruff format fitcheck tests                          # Format code
```

## License

MIT
