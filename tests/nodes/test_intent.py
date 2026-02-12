import pytest
from aegis.models.state import AegisState, RunEvent, TrainingSpec
from aegis.nodes.intent import parse_intent_node


def test_parse_intent_creates_spec():
    state = AegisState()
    user_input = "I want to fine-tune tinyllama-272m on my data using LoRA"
    new_state = parse_intent_node(state, user_input=user_input)
    assert new_state.spec is not None
    assert isinstance(new_state.spec, TrainingSpec)
    assert new_state.spec.method == "lora"
    assert "tinyllama" in new_state.spec.model_name.lower()


def test_parse_intent_adds_event_log():
    state = AegisState()
    user_input = "Train tinyllama with lora"
    new_state = parse_intent_node(state, user_input=user_input)
    assert len(new_state.events) > 0
    assert new_state.events[0].phase == "parse_intent"
    assert new_state.events[0].status in ["completed", "failed"]


def test_parse_intent_returns_immutably():
    state = AegisState()
    user_input = "Train with LoRA"
    new_state = parse_intent_node(state, user_input=user_input)
    assert state.spec is None
    assert len(state.events) == 0
