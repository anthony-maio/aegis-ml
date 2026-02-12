"""Tests for the LLM-based intent parser with mocked LLM calls."""

import pytest
from unittest.mock import patch, MagicMock

from aegis.models.state import AegisState, TrainingSpec
from aegis.nodes.intent_llm import parse_intent_llm, ParsedIntent


def test_fallback_without_api_key(monkeypatch):
    """Without OPENROUTER_API_KEY the parser should fall back to rules."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    state = AegisState()
    result = parse_intent_llm(state, user_input="Fine-tune tinyllama with LoRA")
    assert result.spec is not None
    assert result.spec.method == "lora"
    assert "tinyllama" in result.spec.model_name.lower()


def test_structured_output_with_mocked_llm(monkeypatch):
    """With a mocked LLM the parser should produce a spec from structured output."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    mock_parsed = ParsedIntent(
        method="qlora",
        model_name="meta-llama/Meta-Llama-3-8B",
        dataset_path="./data/custom.jsonl",
        num_epochs=5,
        micro_batch_size=8,
        learning_rate=2e-5,
        seq_len=1024,
        target_gpu="a100",
    )

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_parsed

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("aegis.nodes.intent_llm._get_llm", return_value=mock_llm):
        state = AegisState()
        result = parse_intent_llm(state, user_input="qlora llama-3 for 5 epochs")

    assert result.spec is not None
    assert result.spec.method == "qlora"
    assert result.spec.model_name == "meta-llama/Meta-Llama-3-8B"
    assert result.spec.num_epochs == 5
    assert result.spec.micro_batch_size == 8
    assert result.spec.target_gpu == "a100"
    # Should have an event with parser=llm
    assert any(
        e.data and e.data.get("parser") == "llm"
        for e in result.events
    )


def test_fallback_on_llm_error(monkeypatch):
    """If the LLM call raises, the parser should fall back to rule-based."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("API error")
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("aegis.nodes.intent_llm._get_llm", return_value=mock_llm):
        state = AegisState()
        result = parse_intent_llm(state, user_input="Fine-tune tinyllama with LoRA")

    # Should still produce a valid spec via rule-based fallback
    assert result.spec is not None
    assert result.spec.method == "lora"
    # Should log the fallback
    assert any(
        e.data and e.data.get("parser") == "rule_fallback"
        for e in result.events
    )


def test_parsed_intent_schema_defaults():
    """ParsedIntent should have sensible defaults."""
    intent = ParsedIntent(model_name="tinyllama/tinyllama-272m")
    assert intent.method == "lora"
    assert intent.num_epochs == 3
    assert intent.micro_batch_size == 4
    assert intent.learning_rate == 5e-5
    assert intent.seq_len == 512
    assert intent.target_gpu == "a10g"
