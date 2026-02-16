"""Tests for the natural language spec parser (fitcheck.nlparse).

Validates that parse_spec() correctly extracts method, model, GPU,
optional dataset, and optional sequence length from shorthand strings
like "qlora meta-llama/Llama-3.1-8B on 3090".
"""

import pytest

from fitcheck.nlparse import ParsedSpec, parse_spec


class TestBasicParsing:
    """parse_spec() extracts core fields from well-formed specs."""

    def test_qlora_with_slash_model_on_gpu(self):
        """Standard case: method, org/model, GPU."""
        result = parse_spec("qlora meta-llama/Llama-3.1-8B on 3090")

        assert result is not None
        assert result.method == "qlora"
        assert result.model_id == "meta-llama/Llama-3.1-8B"
        assert result.gpu == "3090"
        assert result.dataset_path is None
        assert result.seq_len is None

    def test_lora_with_dataset(self):
        """Spec with 'with data.jsonl' appended."""
        result = parse_spec("lora some/model on a100 with data.jsonl")

        assert result is not None
        assert result.method == "lora"
        assert result.model_id == "some/model"
        assert result.gpu == "a100"
        assert result.dataset_path == "data.jsonl"

    def test_full_with_seq_len(self):
        """Spec with 'at 2048' appended."""
        result = parse_spec("full some/model on h100 at 2048")

        assert result is not None
        assert result.method == "full"
        assert result.seq_len == 2048

    def test_both_dataset_and_seq_len(self):
        """Spec with both 'with' and 'at' clauses."""
        result = parse_spec("qlora model on gpu with data.jsonl at 1024")

        assert result is not None
        assert result.dataset_path == "data.jsonl"
        assert result.seq_len == 1024


class TestCaseHandling:
    """parse_spec() is case-insensitive for method names."""

    def test_uppercase_method(self):
        """QLORA should parse as qlora."""
        result = parse_spec("QLORA model on gpu")

        assert result is not None
        assert result.method == "qlora"

    def test_mixed_case_method(self):
        """QLoRa should parse as qlora."""
        result = parse_spec("QLoRa model on gpu")

        assert result is not None
        assert result.method == "qlora"


class TestNonMatchingInput:
    """parse_spec() returns None for strings that don't match the pattern."""

    def test_garbage_string(self):
        result = parse_spec("garbage string")
        assert result is None

    def test_missing_on_keyword(self):
        """Without 'on', the spec is ambiguous and should not match."""
        result = parse_spec("qlora model 3090")
        assert result is None

    def test_empty_string(self):
        result = parse_spec("")
        assert result is None


class TestEdgeCases:
    """parse_spec() handles model IDs with special characters."""

    def test_model_with_dots_and_hyphens(self):
        """Model IDs like org/model-name.v2 should parse correctly."""
        result = parse_spec("qlora org/model-name.v2 on gpu")

        assert result is not None
        assert result.model_id == "org/model-name.v2"

    def test_whitespace_is_trimmed(self):
        """Leading/trailing whitespace should not affect parsing."""
        result = parse_spec("  qlora model on gpu  ")

        assert result is not None
        assert result.method == "qlora"
