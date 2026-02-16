"""Shared test fixtures for fitcheck tests."""

from fitcheck.models.profiles import ModelProfile


def make_llama_8b() -> ModelProfile:
    """Create a mock Llama 3.1 8B ModelProfile for testing.

    Uses real Llama 3.1 8B dimensions. The total_params count is
    approximate (computed from dimensions, not from safetensors).
    """
    return ModelProfile(
        model_id="meta-llama/Llama-3.1-8B",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        total_params=8_030_000_000,
        total_params_b=8.03,
    )
