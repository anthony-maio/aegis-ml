"""Tests for the Rich report formatter.

Validates that format_report produces output containing all expected
sections and that the content reflects the actual data in the report.
"""

from fitcheck.models.results import (
    PlanReport,
    SolverResult,
    TrainingConfig,
    VRAMBreakdown,
    ComponentEstimate,
)
from fitcheck.report.formatter import format_report


def _make_breakdown(steady_state_gb: float = 10.0) -> VRAMBreakdown:
    """Build a realistic VRAMBreakdown for testing."""
    # Distribute across components roughly like QLoRA 8B
    total_bytes = int(steady_state_gb * (1024**3))
    return VRAMBreakdown(
        weights=ComponentEstimate(
            name="Model weights",
            bytes=int(total_bytes * 0.45),
            description="8.03B in NF4 + adapters",
        ),
        optimizer=ComponentEstimate(
            name="Optimizer states",
            bytes=int(total_bytes * 0.01),
            description="8-bit Adam",
        ),
        gradients=ComponentEstimate(
            name="Gradients",
            bytes=int(total_bytes * 0.01),
            description="in bfloat16",
        ),
        activations=ComponentEstimate(
            name="Activations",
            bytes=int(total_bytes * 0.35),
            description="flash attention",
        ),
        logits_buffer=ComponentEstimate(
            name="Logits buffer",
            bytes=int(total_bytes * 0.18),
            description="128k vocab",
        ),
        dynamic_margin_bytes=int(total_bytes * 0.065),
    )


def _make_report(
    *,
    usable_vram_gb: float = 22.8,
    steady_state_gb: float = 10.0,
    does_not_fit: bool = False,
) -> PlanReport:
    breakdown = _make_breakdown(steady_state_gb)

    reasoning = {}
    if does_not_fit:
        reasoning = {
            "verdict": "does_not_fit",
            "detail": "Requires 90.0 GB but only 22.8 GB usable",
        }

    rec = TrainingConfig(
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        effective_batch_size=16,
        seq_len=512,
        gradient_checkpointing=False,
        optimizer="paged_adamw_8bit",
        lora_rank=16,
        lora_targets=["q_proj", "v_proj"],
        vram_breakdown=breakdown,
        reasoning=reasoning,
    )

    aggressive = None
    if not does_not_fit:
        agg_bd = _make_breakdown(steady_state_gb * 1.3)
        aggressive = TrainingConfig(
            micro_batch_size=8,
            gradient_accumulation_steps=2,
            effective_batch_size=16,
            seq_len=512,
            optimizer="paged_adamw_8bit",
            lora_rank=16,
            vram_breakdown=agg_bd,
        )

    return PlanReport(
        model_id="meta-llama/Llama-3.1-8B",
        architecture_summary="LlamaForCausalLM",
        total_params_b=8.03,
        vocab_size=128256,
        num_layers=32,
        seq_len_used=512,
        seq_len_reasoning="--seq-len 512",
        hardware_name="NVIDIA RTX 3090",
        total_vram_gb=24.0,
        overhead_gb=1.2,
        usable_vram_gb=usable_vram_gb,
        method="qlora",
        trainable_params=40_000_000,
        trainable_pct=0.50,
        solver_result=SolverResult(
            recommended=rec,
            aggressive=aggressive,
            warnings=["Logits buffer is large."]
            if not does_not_fit
            else ["This configuration does not fit."],
        ),
    )


class TestFormatterSections:
    """Verify all expected sections appear in output."""

    def test_contains_model_section(self):
        output = format_report(_make_report())
        assert "Model" in output
        assert "meta-llama/Llama-3.1-8B" in output

    def test_contains_hardware_section(self):
        output = format_report(_make_report())
        assert "Hardware" in output
        assert "RTX 3090" in output

    def test_contains_vram_breakdown(self):
        output = format_report(_make_report())
        assert "VRAM Breakdown" in output

    def test_contains_recommended_config(self):
        output = format_report(_make_report())
        assert "Recommended Config" in output

    def test_contains_aggressive_config(self):
        output = format_report(_make_report())
        assert "Aggressive Config" in output

    def test_contains_risks(self):
        output = format_report(_make_report())
        assert "Risks" in output

    def test_contains_training_summary(self):
        output = format_report(_make_report())
        assert "What You're Training" in output
        assert "QLORA" in output


class TestFormatterContent:
    """Verify the content reflects actual report data."""

    def test_components_appear_in_breakdown(self):
        output = format_report(_make_report())
        assert "Model weights" in output
        assert "Optimizer states" in output
        assert "Gradients" in output
        assert "Activations" in output
        assert "Logits buffer" in output

    def test_headroom_reflects_margin(self):
        """With 10 GB used and 22.8 GB usable, headroom should be large."""
        output = format_report(_make_report(usable_vram_gb=22.8, steady_state_gb=10.0))
        # Headroom should show a percentage
        assert "%" in output

    def test_does_not_fit_shows_warning(self):
        output = format_report(_make_report(does_not_fit=True))
        assert "DOES NOT FIT" in output

    def test_params_displayed(self):
        output = format_report(_make_report())
        assert "8.03" in output

    def test_vocab_displayed(self):
        output = format_report(_make_report())
        assert "128" in output  # 128,256 formatted
