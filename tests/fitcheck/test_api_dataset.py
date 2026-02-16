"""Tests for fitcheck API dataset integration.

Validates that plan() correctly analyzes real dataset files (no mocking
of the dataset analyzer -- it's internal logic), resolves sequence
lengths from dataset statistics, and triggers sanity warnings for
small datasets.

HF Hub calls are mocked. Dataset files are real temporary JSONL files.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from fitcheck.api import plan
from fitcheck.models.results import PlanReport
from tests.fitcheck.conftest import make_llama_8b


def _write_alpaca_jsonl(path: str, num_rows: int) -> None:
    """Write a realistic alpaca-format JSONL file for testing."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_rows):
            row = {
                "instruction": f"Question {i} " * 20,
                "input": "",
                "output": f"Answer {i} " * 30,
            }
            f.write(json.dumps(row) + "\n")


class TestDatasetSeqLenResolution:
    """plan() resolves sequence length from dataset p95 when no explicit override."""

    @patch("fitcheck.api.resolve_model")
    def test_dataset_p95_used_when_no_seq_len(self, mock_resolve):
        """Without --seq-len, plan() should use the dataset's p95 token estimate."""
        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            _write_alpaca_jsonl(path, 500)

            report = plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                dataset_path=path,
            )

            assert "dataset p95" in report.seq_len_reasoning

    @patch("fitcheck.api.resolve_model")
    def test_explicit_seq_len_overrides_dataset(self, mock_resolve):
        """Explicit seq_len=1024 should override dataset p95."""
        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            _write_alpaca_jsonl(path, 500)

            report = plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                seq_len=1024,
                dataset_path=path,
            )

            assert report.seq_len_used == 1024
            assert "--seq-len" in report.seq_len_reasoning


class TestDatasetFieldsPopulated:
    """plan() populates dataset metadata in the report when a file is provided."""

    @patch("fitcheck.api.resolve_model")
    def test_dataset_fields_filled(self, mock_resolve):
        """Dataset source, row count, and format should all be populated."""
        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            _write_alpaca_jsonl(path, 500)

            report = plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                dataset_path=path,
            )

            assert report.dataset_source != "none"
            assert report.dataset_rows > 0
            assert report.dataset_format == "alpaca"

    @patch("fitcheck.api.resolve_model")
    def test_samples_per_epoch_positive(self, mock_resolve):
        """samples_per_epoch should be positive when a dataset is provided."""
        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            _write_alpaca_jsonl(path, 500)

            report = plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                dataset_path=path,
            )

            assert report.samples_per_epoch > 0


class TestDatasetSanityWarnings:
    """plan() triggers sanity warnings for small datasets that risk overfitting."""

    @patch("fitcheck.api.resolve_model")
    def test_tiny_dataset_triggers_overfit_warning(self, mock_resolve):
        """50 rows with ~160M trainable params should trigger an overfit warning.

        LoRA rank=16 on 8B Llama produces ~160M trainable params.
        50 rows / 160M params = 0.3 rows per 1M params, well below the
        threshold of 10 rows/1M for critical overfit risk.
        """
        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            _write_alpaca_jsonl(path, 50)

            report = plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                dataset_path=path,
            )

            overfit_warnings = [
                w for w in report.solver_result.warnings if "overfit" in w.lower()
            ]
            assert len(overfit_warnings) >= 1, (
                f"Expected overfit warning for 50-row dataset, "
                f"got warnings: {report.solver_result.warnings}"
            )

    @patch("fitcheck.api.resolve_model")
    def test_large_dataset_no_overfit_warning(self, mock_resolve):
        """5000 rows should not trigger overfit warnings.

        5000 rows / ~160M trainable params = ~31 rows per 1M params,
        which is above the critical threshold of 10.
        """
        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            _write_alpaca_jsonl(path, 5000)

            report = plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                dataset_path=path,
            )

            overfit_warnings = [
                w for w in report.solver_result.warnings if "overfit" in w.lower()
            ]
            assert len(overfit_warnings) == 0, (
                f"Did not expect overfit warnings for 5000-row dataset, "
                f"got: {overfit_warnings}"
            )


class TestDatasetErrorHandling:
    """plan() raises appropriate errors for missing dataset files."""

    @patch("fitcheck.api.resolve_model")
    def test_invalid_path_raises_file_not_found(self, mock_resolve):
        """Passing a nonexistent dataset_path should raise FileNotFoundError."""
        mock_resolve.return_value = make_llama_8b()

        with pytest.raises(FileNotFoundError):
            plan(
                model_id="meta-llama/Llama-3.1-8B",
                method="qlora",
                gpu="3090",
                dataset_path="/nonexistent/path/to/train.jsonl",
            )
