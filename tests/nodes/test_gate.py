import pytest
from aegis.models.state import AegisState, CostEstimate
from aegis.nodes.gate import budget_gate_node, check_auto_approve


def test_check_auto_approve_under_threshold():
    state = AegisState(
        budget_policy=AegisState().budget_policy.model_copy(
            update={"soft_threshold_usd": 2.0}
        ),
        cost_estimate=CostEstimate(
            estimated_cost_usd=1.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        ),
    )
    assert check_auto_approve(state) is True


def test_check_auto_approve_over_soft_threshold():
    state = AegisState(
        budget_policy=AegisState().budget_policy.model_copy(
            update={"soft_threshold_usd": 2.0}
        ),
        cost_estimate=CostEstimate(
            estimated_cost_usd=3.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        ),
    )
    assert check_auto_approve(state) is False


def test_check_auto_approve_exceeds_hard_limit():
    state = AegisState(
        budget_policy=AegisState().budget_policy.model_copy(
            update={"max_budget_usd": 5.0}
        ),
        cost_estimate=CostEstimate(
            estimated_cost_usd=6.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        ),
    )
    assert check_auto_approve(state) is False


def test_budget_gate_adds_event():
    state = AegisState(
        cost_estimate=CostEstimate(
            estimated_cost_usd=1.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        )
    )
    new_state = budget_gate_node(state)
    assert any(e.phase == "budget_gate" for e in new_state.events)
