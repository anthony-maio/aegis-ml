import pytest

from aegis.models.state import (
    AegisState,
    BudgetPolicy,
    CostEstimate,
    RunEvent,
    TrainingSpec,
)
from aegis.nodes.reporter import generate_report, write_report_node


def test_generate_report_creates_markdown():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
        ),
        budget_policy=BudgetPolicy(max_budget_usd=5.0),
        cost_estimate=CostEstimate(
            estimated_cost_usd=0.0150,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        ),
        events=[
            RunEvent(
                phase="parse_intent", status="completed", message="Parsed"
            ),
            RunEvent(
                phase="estimate_cost", status="completed", message="Estimated"
            ),
        ],
    )
    report = generate_report(state)
    assert "# " in report
    assert "Aegis-ML" in report
    assert "tinyllama" in report
    assert "lora" in report
    assert "$" in report
    assert "| Time | Phase |" in report


def test_generate_report_includes_executive_summary():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
        ),
        cost_estimate=CostEstimate(
            estimated_cost_usd=0.0150,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        ),
    )
    report = generate_report(state)
    assert "## Executive Summary" in report
    assert "| Metric | Value |" in report


def test_write_report_node_updates_state():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
        )
    )
    new_state = write_report_node(state)
    assert new_state.final_report is not None
    assert len(new_state.final_report) > 0
