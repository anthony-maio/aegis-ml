"""fitcheck public Python API.

Exposes the core plan() function for programmatic use.
The CLI is a thin wrapper around this module.
"""

from __future__ import annotations

from fitcheck.datasets.analyzer import analyze_local
from fitcheck.hardware.registry import get_hardware
from fitcheck.hub.resolver import resolve_model
from fitcheck.models.profiles import (
    DatasetProfile,
    HardwareSpec,
    LoRAConfig,
    ModelProfile,
    TrainingMethod,
)
from fitcheck.models.results import PlanReport, SolverResult
from fitcheck.profilers.sanity import check_training_sanity
from fitcheck.profilers.vram.components import get_trainable_params
from fitcheck.solver import ConfigSolver

_DEFAULT_SEQ_LEN = 512


def plan(
    model_id: str,
    method: str,
    gpu: str,
    seq_len: int | None = None,
    lora_rank: int = 16,
    batch_size: int | None = None,
    eval_seq_len: int | None = None,
    dataset_path: str | None = None,
) -> PlanReport:
    """Estimate VRAM usage and find optimal training config.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B").
        method: Training method ("full", "lora", or "qlora").
        gpu: GPU name or alias (e.g. "3090", "a100").
        seq_len: Sequence length override. If None, uses dataset p95 or 512.
        lora_rank: LoRA rank (default 16).
        batch_size: Fixed batch size. If None, solver searches for optimal.
        eval_seq_len: Max eval sequence length for KV-cache spike estimation.
        dataset_path: Path to local dataset file (.jsonl/.json) for analysis.

    Returns:
        PlanReport with full VRAM breakdown and recommended config.

    Raises:
        ValueError: If method is unknown or model architecture unsupported.
        KeyError: If GPU name is not recognized.
        FileNotFoundError: If dataset_path doesn't exist.
    """
    # Resolve training method
    training_method = TrainingMethod(method.lower())

    # Resolve hardware
    hardware = get_hardware(gpu)

    # Resolve model from HF Hub
    model_profile = resolve_model(model_id)

    # Analyze dataset if provided
    dataset: DatasetProfile | None = None
    if dataset_path is not None:
        dataset = analyze_local(dataset_path)

    # Resolve sequence length: explicit > dataset p95 > default
    resolved_seq_len, seq_len_reasoning = _resolve_seq_len(seq_len, dataset)

    # Build LoRA config
    lora_config = LoRAConfig(rank=lora_rank)

    # Run solver
    solver = ConfigSolver()
    if batch_size is not None:
        solver_result = solver.estimate_fixed(
            model=model_profile,
            hardware=hardware,
            method=training_method,
            batch_size=batch_size,
            seq_len=resolved_seq_len,
            lora_config=lora_config,
            eval_seq_len=eval_seq_len,
        )
    else:
        solver_result = solver.solve(
            model=model_profile,
            hardware=hardware,
            method=training_method,
            seq_len=resolved_seq_len,
            lora_config=lora_config,
            eval_seq_len=eval_seq_len,
        )

    # Compute trainable params
    trainable = get_trainable_params(model_profile, training_method, lora_config)
    trainable_pct = (trainable / model_profile.total_params) * 100

    # Run sanity checks if dataset available
    if dataset is not None:
        sanity_warnings = check_training_sanity(
            dataset=dataset,
            config=solver_result.recommended,
            trainable_params=trainable,
        )
        for w in sanity_warnings:
            solver_result.warnings.append(f"[{w.severity}] {w.category}: {w.message}")

    # Assemble report
    report = _build_report(
        model_profile=model_profile,
        hardware=hardware,
        training_method=training_method,
        resolved_seq_len=resolved_seq_len,
        seq_len_reasoning=seq_len_reasoning,
        trainable=trainable,
        trainable_pct=trainable_pct,
        solver_result=solver_result,
        dataset=dataset,
    )

    return report


def _resolve_seq_len(
    explicit: int | None,
    dataset: DatasetProfile | None,
) -> tuple[int, str]:
    """Resolve sequence length with reasoning annotation.

    Priority: explicit override > dataset p95 > default 512.
    """
    if explicit is not None:
        if explicit <= 0:
            raise ValueError(f"seq_len must be positive, got {explicit}")
        return explicit, f"--seq-len {explicit}"

    if dataset is not None and dataset.seq_len_stats is not None:
        p95 = dataset.seq_len_stats.p95
        return p95, f"dataset p95 ({p95} tokens)"

    return _DEFAULT_SEQ_LEN, f"default ({_DEFAULT_SEQ_LEN})"


def _build_report(
    model_profile: ModelProfile,
    hardware: HardwareSpec,
    training_method: TrainingMethod,
    resolved_seq_len: int,
    seq_len_reasoning: str,
    trainable: int,
    trainable_pct: float,
    solver_result: SolverResult,
    dataset: DatasetProfile | None,
) -> PlanReport:
    """Assemble a PlanReport from resolved components."""
    # Dataset fields
    dataset_source = "none"
    dataset_rows = 0
    dataset_format = "unknown"
    seq_len_stats = None
    samples_per_epoch = 0.0

    if dataset is not None:
        dataset_source = dataset.source
        dataset_rows = dataset.num_rows
        dataset_format = dataset.detected_format
        if dataset.seq_len_stats is not None:
            seq_len_stats = dataset.seq_len_stats.model_dump()
        samples_per_epoch = float(dataset.num_rows)

    return PlanReport(
        model_id=model_profile.model_id,
        architecture_summary=model_profile.architecture,
        total_params_b=model_profile.total_params_b,
        vocab_size=model_profile.vocab_size,
        num_layers=model_profile.num_layers,
        dataset_source=dataset_source,
        dataset_rows=dataset_rows,
        dataset_format=dataset_format,
        seq_len_stats=seq_len_stats,
        seq_len_used=resolved_seq_len,
        seq_len_reasoning=seq_len_reasoning,
        hardware_name=hardware.name,
        total_vram_gb=hardware.total_vram_gb,
        overhead_gb=hardware.overhead_gb,
        usable_vram_gb=hardware.usable_vram_gb,
        method=training_method.value,
        trainable_params=trainable,
        trainable_pct=trainable_pct,
        samples_per_epoch=samples_per_epoch,
        solver_result=solver_result,
    )
