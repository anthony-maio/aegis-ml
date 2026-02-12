import re
from aegis.models.state import TrainingSpec, CostEstimate

# GPU pricing (USD/second) - source: modal.com/pricing
GPU_PRICING = {
    "a10g": 0.000306,  # $1.10/hr
    "t4": 0.000126,    # $0.45/hr
    "a100": 0.000793,  # $2.85/hr
}

# Known model sizes (total params in billions) for quick lookup
KNOWN_MODEL_SIZES: dict[str, float] = {
    "tinyllama": 0.272,
    "llama-2-7b": 7.0,
    "llama-2-13b": 13.0,
    "meta-llama-3-8b": 8.0,
    "meta-llama-3-70b": 70.0,
    "mistral-7b": 7.0,
    "mixtral-8x7b": 46.7,  # MoE: 8 experts x 7B (but only 2 active)
    "phi-2": 2.7,
    "phi-3-mini": 3.8,
    "gemma-2b": 2.0,
    "gemma-7b": 7.0,
}


def _estimate_params_b(model_name: str) -> float:
    """Estimate model parameter count in billions from the model name.

    Tries known models first, then regex patterns like '7B', '272M', '13b'.
    Returns a conservative default (1.0B) if nothing matches.
    """
    name_lower = model_name.lower().replace("/", "-").replace("_", "-")

    # Check known models
    for key, size in KNOWN_MODEL_SIZES.items():
        if key in name_lower:
            return size

    # Try regex: "7b", "13B", "70b", etc.
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[^a-zA-Z])", model_name)
    if match:
        return float(match.group(1))

    # Try regex: "272M", "1.3M", etc.
    match = re.search(r"(\d+(?:\.\d+)?)\s*[mM](?:\b|[^a-zA-Z])", model_name)
    if match:
        return float(match.group(1)) / 1000.0

    return 1.0  # conservative default


def _is_moe(model_name: str) -> bool:
    """Detect MoE architecture from model name."""
    lower = model_name.lower()
    return any(kw in lower for kw in ["moe", "mixture", "mixtral", "expert", "switch"])


def _estimate_model_vram_gb(params_b: float, method: str, is_moe: bool) -> float:
    """Estimate base model VRAM in GB.

    Full precision: ~4 bytes/param (fp32) or ~2 bytes/param (fp16/bf16).
    LoRA: base model in fp16 + small adapter.
    QLoRA: base model in 4-bit (~0.5 bytes/param) + adapter.
    MoE: all experts are loaded, only a subset is active per forward pass.
    """
    if method == "qlora":
        bytes_per_param = 0.5  # 4-bit quantized
    else:
        bytes_per_param = 2.0  # fp16

    base_gb = params_b * bytes_per_param

    # MoE models load all experts into VRAM
    if is_moe:
        base_gb *= 1.3  # overhead for routing, expert buffers

    return base_gb


def estimate_training_cost(spec: TrainingSpec, estimated_params_b: float = 0.0, is_moe: bool = False) -> CostEstimate:
    """Deterministic cost estimation based on spec parameters and model size.

    Args:
        spec: Training specification.
        estimated_params_b: Model params in billions (0 = auto-detect from name).
        is_moe: Whether the model is a Mixture-of-Experts architecture.
    """
    # Resolve model size
    params_b = estimated_params_b if estimated_params_b > 0 else _estimate_params_b(spec.model_name)
    if not is_moe:
        is_moe = _is_moe(spec.model_name)

    # Assume 10k samples (small dataset for demo)
    num_samples = 10000

    # Estimate training steps
    effective_batch_size = spec.micro_batch_size * spec.gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * spec.num_epochs

    # Throughput scales inversely with model size
    # ~1000 samples/sec for 272M, ~50 for 7B, ~10 for 70B
    base_throughput = 1000  # samples/sec for a 0.3B model
    size_factor = max(0.3 / params_b, 0.01)  # relative to 300M
    samples_per_sec = base_throughput * size_factor

    # MoE is ~1.5x slower per step due to routing overhead
    if is_moe:
        samples_per_sec *= 0.65

    estimated_seconds = (num_samples * spec.num_epochs) / samples_per_sec
    estimated_seconds *= 1.2  # 20% overhead for data loading, checkpointing

    # Calculate cost
    gpu_price = GPU_PRICING.get(spec.target_gpu, GPU_PRICING["a10g"])
    estimated_cost = estimated_seconds * gpu_price

    # VRAM estimation
    base_model_vram = _estimate_model_vram_gb(params_b, spec.method, is_moe)

    # Optimizer states: AdamW uses 2x model size for momentum + variance
    if spec.method == "qlora":
        optimizer_vram = 0.5  # only adapter params in optimizer
    elif spec.method == "lora":
        optimizer_vram = base_model_vram * 0.1  # ~10% of params trainable
    else:
        optimizer_vram = base_model_vram * 2.0  # full optimizer states

    # Activation memory
    activation_vram = (spec.micro_batch_size * spec.seq_len * params_b * 0.5) / 1e3
    activation_vram = max(activation_vram, 0.5)  # minimum 0.5GB

    total_vram = base_model_vram + optimizer_vram + activation_vram

    return CostEstimate(
        estimated_cost_usd=round(estimated_cost, 4),
        estimated_vram_gb=round(total_vram, 1),
        estimated_duration_min=round(estimated_seconds / 60, 1),
        cost_breakdown={
            "gpu": spec.target_gpu,
            "gpu_price_usd_per_sec": gpu_price,
            "estimated_seconds": round(estimated_seconds, 1),
            "effective_batch_size": effective_batch_size,
            "total_steps": total_steps,
            "model_params_b": params_b,
            "is_moe": is_moe,
            "base_model_vram_gb": round(base_model_vram, 1),
            "optimizer_vram_gb": round(optimizer_vram, 1),
            "activation_vram_gb": round(activation_vram, 1),
        }
    )
