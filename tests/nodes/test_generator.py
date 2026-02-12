import pytest
from aegis.models.state import AegisState, TrainingSpec
from aegis.nodes.generator import generate_code_node


def test_generate_code_adds_generated_code():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
        )
    )
    new_state = generate_code_node(state)
    assert new_state.generated_code is not None
    assert isinstance(new_state.generated_code, str)
    assert len(new_state.generated_code) > 0


def test_generate_code_includes_imports():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
        )
    )
    new_state = generate_code_node(state)
    assert "from transformers import" in new_state.generated_code


def test_generate_code_requires_spec():
    state = AegisState(spec=None)
    new_state = generate_code_node(state)
    assert new_state.generated_code is None
    assert any(
        e.phase == "generate_code" and e.status == "failed" for e in new_state.events
    )
