import pytest
from aegis.models.state import (
    TrainingSpec, BudgetPolicy, CostEstimate, RunEvent, AegisState
)
from datetime import datetime


def test_training_spec_creation():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        num_epochs=3,
        micro_batch_size=4,
        target_gpu="a10g"
    )
    assert spec.method == "lora"
    assert spec.model_name == "tinyllama/tinyllama-272m"
    assert spec.micro_batch_size == 4


def test_budget_policy_defaults():
    policy = BudgetPolicy()
    assert policy.max_budget_usd == 5.00
    assert policy.soft_threshold_usd == 2.00
    assert policy.allow_auto_approve is True


def test_cost_estimate_fields():
    estimate = CostEstimate(
        estimated_cost_usd=0.0150,
        estimated_vram_gb=8.5,
        estimated_duration_min=5.2,
        cost_breakdown={"gpu": "a10g", "steps": 1000}
    )
    assert estimate.estimated_cost_usd == 0.0150
    assert estimate.estimated_vram_gb == 8.5


def test_run_event_creation():
    event = RunEvent(
        phase="estimate",
        status="completed",
        message="Cost estimated successfully"
    )
    assert event.phase == "estimate"
    assert isinstance(event.timestamp, datetime)


def test_aegis_state_initialization():
    state = AegisState()
    assert state.spec is None
    assert state.events == []
    assert state.retry_count == 0
    assert state.max_retries == 3


def test_aegis_state_user_input_default():
    """user_input should default to None."""
    state = AegisState()
    assert state.user_input is None


def test_aegis_state_user_input_set():
    """user_input should accept a string."""
    state = AegisState(user_input="Fine-tune llama with LoRA")
    assert state.user_input == "Fine-tune llama with LoRA"


def test_aegis_state_immutability_pattern():
    """Verify that state updates use model_copy for immutability"""
    state = AegisState(retry_count=0)
    new_state = state.model_copy(update={"retry_count": 1})
    assert state.retry_count == 0  # Original unchanged
    assert new_state.retry_count == 1  # New state updated
