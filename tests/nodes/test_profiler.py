import pytest
from aegis.models.state import AegisState, TrainingSpec
from aegis.nodes.profiler import estimate_cost_node


def test_estimate_cost_adds_cost_estimate():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
            target_gpu="a10g",
        )
    )
    new_state = estimate_cost_node(state)
    assert new_state.cost_estimate is not None
    assert new_state.cost_estimate.estimated_cost_usd > 0


def test_estimate_cost_adds_event_log():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
        )
    )
    new_state = estimate_cost_node(state)
    assert any(e.phase == "estimate_cost" for e in new_state.events)


def test_estimate_cost_requires_spec():
    state = AegisState(spec=None)
    new_state = estimate_cost_node(state)
    assert any(
        e.phase == "estimate_cost" and e.status == "failed" for e in new_state.events
    )
