"""Tests for the fitcheck public Python API (fitcheck.api.plan).

Validates that plan() produces correct PlanReport structures for
various training methods, sequence lengths, and error conditions.
HF Hub calls are mocked -- these tests exercise the solver and
report assembly, not network resolution.
"""

from unittest.mock import patch

import pytest

from fitcheck.api import plan
from fitcheck.models.results import PlanReport
from tests.fitcheck.conftest import make_llama_8b


class TestPlanBasicBehavior:
    """plan() returns a well-formed PlanReport with expected fields."""

    @patch("fitcheck.api.resolve_model")
    def test_qlora_8b_on_3090_produces_valid_report(self, mock_resolve):
        """QLoRA 8B on a 3090 should fit and return a complete PlanReport."""
        mock_resolve.return_value = make_llama_8b()

        report = plan(model_id="meta-llama/Llama-3.1-8B", method="qlora", gpu="3090")

        assert isinstance(report, PlanReport)
        assert report.model_id == "meta-llama/Llama-3.1-8B"
        assert report.method == "qlora"
        assert report.hardware_name == "NVIDIA RTX 3090"
        assert report.solver_result is not None
        assert report.solver_result.recommended is not None
        assert report.solver_result.recommended.vram_breakdown is not None

    @patch("fitcheck.api.resolve_model")
    def test_explicit_seq_len_is_used(self, mock_resolve):
        """plan(seq_len=1024) should use 1024, not the default 512."""
        mock_resolve.return_value = make_llama_8b()

        report = plan(
            model_id="meta-llama/Llama-3.1-8B",
            method="qlora",
            gpu="3090",
            seq_len=1024,
        )

        assert report.seq_len_used == 1024

    @patch("fitcheck.api.resolve_model")
    def test_default_seq_len_is_512(self, mock_resolve):
        """plan() without seq_len defaults to 512."""
        mock_resolve.return_value = make_llama_8b()

        report = plan(model_id="meta-llama/Llama-3.1-8B", method="qlora", gpu="3090")

        assert report.seq_len_used == 512

    @patch("fitcheck.api.resolve_model")
    def test_explicit_512_still_uses_512(self, mock_resolve):
        """plan(seq_len=512) explicitly should also be 512."""
        mock_resolve.return_value = make_llama_8b()

        report = plan(
            model_id="meta-llama/Llama-3.1-8B",
            method="qlora",
            gpu="3090",
            seq_len=512,
        )

        assert report.seq_len_used == 512


class TestPlanDoesNotFit:
    """plan() correctly reports when a config exceeds VRAM."""

    @patch("fitcheck.api.resolve_model")
    def test_full_ft_8b_does_not_fit_on_3090(self, mock_resolve):
        """Full fine-tuning 8B on a 3090 (24 GB) should not fit.

        8B params in bf16 = ~16 GB weights alone, plus optimizer states
        (32 GB for AdamW fp32), plus activations. Total far exceeds 22.8 GB usable.
        """
        mock_resolve.return_value = make_llama_8b()

        report = plan(model_id="meta-llama/Llama-3.1-8B", method="full", gpu="3090")

        assert report.solver_result.recommended.reasoning.get("verdict") == "does_not_fit"


class TestPlanErrorHandling:
    """plan() raises appropriate errors for invalid inputs."""

    @patch("fitcheck.api.resolve_model")
    def test_unknown_method_raises_value_error(self, mock_resolve):
        """Passing method='banana' should raise ValueError, not silently fail."""
        mock_resolve.return_value = make_llama_8b()

        with pytest.raises(ValueError, match="banana"):
            plan(model_id="test/model", method="banana", gpu="3090")

    @patch("fitcheck.api.resolve_model")
    def test_unknown_gpu_raises_key_error(self, mock_resolve):
        """Passing gpu='potato' should raise KeyError listing available GPUs."""
        mock_resolve.return_value = make_llama_8b()

        with pytest.raises(KeyError, match="potato"):
            plan(model_id="test/model", method="qlora", gpu="potato")


class TestPlanSeqLenReasoning:
    """plan() annotates why it chose a particular sequence length."""

    @patch("fitcheck.api.resolve_model")
    def test_default_reasoning_when_no_dataset(self, mock_resolve):
        """Without a dataset, seq_len_reasoning should say 'default (512)'."""
        mock_resolve.return_value = make_llama_8b()

        report = plan(model_id="meta-llama/Llama-3.1-8B", method="qlora", gpu="3090")

        assert "default" in report.seq_len_reasoning
        assert "512" in report.seq_len_reasoning
