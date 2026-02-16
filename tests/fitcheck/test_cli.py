"""CLI smoke tests using Typer's CliRunner.

These test the CLI wiring — that flags are parsed, errors are reported,
and output contains the expected sections. They do NOT test VRAM math
(that's test_solver.py and test_estimator.py).

Tests that require HF Hub are marked with @pytest.mark.network and
skipped in CI. The remaining tests use mock model profiles.
"""

from unittest.mock import patch
from typer.testing import CliRunner

from fitcheck.cli import app
from tests.fitcheck.conftest import make_llama_8b

runner = CliRunner()


class TestCLIParsing:
    """Test that CLI flags are parsed correctly."""

    def test_missing_model_flag(self):
        result = runner.invoke(app, ["plan", "--method", "qlora", "--gpu", "3090"])
        assert result.exit_code != 0

    def test_missing_method_flag(self):
        result = runner.invoke(app, ["plan", "--model", "x", "--gpu", "3090"])
        assert result.exit_code != 0

    def test_missing_gpu_flag(self):
        result = runner.invoke(app, ["plan", "--model", "x", "--method", "qlora"])
        assert result.exit_code != 0

    def test_unknown_method(self):
        with patch("fitcheck.api.resolve_model", return_value=make_llama_8b()):
            result = runner.invoke(
                app, ["plan", "--model", "test", "--method", "banana", "--gpu", "3090"]
            )
        assert result.exit_code != 0
        assert "banana" in result.output.lower()

    def test_unknown_gpu(self):
        with patch("fitcheck.api.resolve_model", return_value=make_llama_8b()):
            result = runner.invoke(
                app, ["plan", "--model", "test", "--method", "qlora", "--gpu", "potato"]
            )
        assert result.exit_code != 0


class TestCLIOutput:
    """Test that CLI produces expected output sections."""

    @patch("fitcheck.api.resolve_model")
    def test_qlora_plan_exits_zero(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"

    @patch("fitcheck.api.resolve_model")
    def test_output_contains_model_section(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "Model" in result.output
        assert "8.03" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_output_contains_hardware_section(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "Hardware" in result.output
        assert "24.0" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_output_contains_vram_breakdown(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "VRAM Breakdown" in result.output
        assert "Model weights" in result.output
        assert "Optimizer states" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_output_contains_recommended_config(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "Recommended Config" in result.output
        assert "Micro batch" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_full_ft_shows_does_not_fit(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "full", "--gpu", "3090"]
        )
        assert result.exit_code == 0
        assert "DOES NOT FIT" in result.output or "does not fit" in result.output.lower()

    @patch("fitcheck.api.resolve_model")
    def test_custom_seq_len(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app,
            [
                "plan",
                "--model",
                "test/model",
                "--method",
                "qlora",
                "--gpu",
                "3090",
                "--seq-len",
                "1024",
            ],
        )
        assert result.exit_code == 0
        assert "1024" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_custom_lora_rank(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app,
            [
                "plan",
                "--model",
                "test/model",
                "--method",
                "qlora",
                "--gpu",
                "3090",
                "--lora-rank",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "64" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_fixed_batch_size(self, mock_resolve):
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app,
            [
                "plan",
                "--model",
                "test/model",
                "--method",
                "qlora",
                "--gpu",
                "3090",
                "--batch-size",
                "2",
            ],
        )
        assert result.exit_code == 0


class TestCLINaturalLanguage:
    """Test natural language spec parsing via CLI positional argument.

    The CLI accepts a quoted string like 'qlora test/model on 3090' as
    a positional argument, which is parsed by fitcheck.nlparse.parse_spec()
    into method, model, and GPU fields.
    """

    @patch("fitcheck.api.resolve_model")
    def test_nl_spec_exits_zero_with_model_section(self, mock_resolve):
        """A valid NL spec should produce a successful plan with Model section."""
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(app, ["plan", "qlora test/model on 3090"])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Model" in result.output

    def test_garbage_spec_exits_nonzero_with_parse_error(self):
        """An unparseable NL spec should exit 1 with a parse error message."""
        result = runner.invoke(app, ["plan", "garbage string"])
        assert result.exit_code != 0
        # Typer prints errors to stdout through CliRunner by default
        combined = result.output + (result.stderr or "")
        assert "parse" in combined.lower() or "Could not parse" in combined

    @patch("fitcheck.api.resolve_model")
    def test_nl_spec_with_seq_len_flag_override(self, mock_resolve):
        """--seq-len flag should override the NL spec's default."""
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "qlora test/model on 3090", "--seq-len", "1024"]
        )
        assert result.exit_code == 0
        assert "1024" in result.output

    @patch("fitcheck.api.resolve_model")
    def test_nl_spec_with_nonexistent_dataset_flag(self, mock_resolve):
        """--dataset pointing to a nonexistent file should report an error."""
        mock_resolve.return_value = make_llama_8b()
        result = runner.invoke(
            app, ["plan", "qlora test/model on 3090", "--dataset", "nonexist.jsonl"]
        )
        assert result.exit_code != 0
        combined = result.output + (result.stderr or "")
        assert "not found" in combined.lower() or "Error" in combined

    def test_no_spec_no_flags_exits_with_missing_options(self):
        """Bare 'plan' with no spec and no flags should error about missing options."""
        result = runner.invoke(app, ["plan"])
        assert result.exit_code != 0
        combined = result.output + (result.stderr or "")
        assert "missing" in combined.lower() or "--model" in combined.lower()


class TestCLIDatasetIntegration:
    """Test that --dataset flag flows through CLI to API correctly.

    This specifically tests the CLI → API boundary for dataset-driven
    seq_len resolution, which was broken by a premature default in the
    CLI layer (the critical bug found in code review).
    """

    @patch("fitcheck.api.resolve_model")
    def test_dataset_without_seq_len_uses_p95(self, mock_resolve):
        """--dataset without --seq-len should use dataset p95, not default 512.

        This is the regression test for the critical bug where the CLI
        hardcoded seq_len=512 before passing to api.plan(), bypassing
        the dataset p95 resolution path entirely.
        """
        import json
        import os
        import tempfile

        mock_resolve.return_value = make_llama_8b()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(200):
                    row = {
                        "instruction": f"Question {i} " * 20,
                        "input": "",
                        "output": f"Answer {i} " * 30,
                    }
                    f.write(json.dumps(row) + "\n")

            result = runner.invoke(
                app,
                [
                    "plan",
                    "--model", "test/model",
                    "--method", "qlora",
                    "--gpu", "3090",
                    "--dataset", path,
                ],
            )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # The report should show dataset-derived seq_len, not the default 512
        assert "dataset" in result.output.lower() or "p95" in result.output.lower(), (
            f"Expected dataset/p95 mention in output, got:\n{result.output}"
        )
