"""End-to-end integration tests for the Aegis-ML fine-tuning orchestrator.

These tests invoke the full compiled LangGraph and verify the complete
happy-path flow from ``parse_intent`` through ``write_report``, asserting
on intermediate state fields, event completeness, and report content.
"""

import pytest

from aegis.graph import build_aegis_graph
from aegis.models.state import AegisState


def test_full_graph_flow():
    """Test complete happy path from intent to report."""
    graph = build_aegis_graph()
    initial_state = AegisState()
    config = {"configurable": {"thread_id": "e2e-test"}}
    result = graph.invoke(initial_state, config=config)

    assert result is not None
    # Should have parsed spec
    assert result["spec"] is not None
    assert result["spec"].method in ["lora", "qlora", "full_finetune"]
    # Should have cost estimate
    assert result["cost_estimate"] is not None
    assert result["cost_estimate"].estimated_cost_usd > 0
    # Should have code generated
    assert result["generated_code"] is not None
    assert len(result["generated_code"]) > 0
    # Should have execution completed (mock)
    assert result["execution_result"] is not None
    # Should have evals ran
    assert result["eval_result"] is not None
    # Should have report generated
    assert result["final_report"] is not None
    assert "Aegis-ML Training Report" in result["final_report"]


def test_graph_event_log_completeness():
    """Verify all phases are logged."""
    graph = build_aegis_graph()
    result = graph.invoke(
        AegisState(), config={"configurable": {"thread_id": "log-test"}}
    )

    phases = {e.phase for e in result["events"]}
    expected_phases = {
        "parse_intent",
        "estimate_cost",
        "budget_gate",
        "generate_code",
        "execute",
        "run_evals",
        "write_report",
    }

    assert expected_phases.issubset(phases)


def test_graph_report_contains_cost_info():
    """Verify report includes cost breakdown."""
    graph = build_aegis_graph()
    result = graph.invoke(
        AegisState(), config={"configurable": {"thread_id": "report-test"}}
    )

    report = result["final_report"]
    assert "$" in report
    assert "AUTO-APPROVED" in report


def test_graph_all_events_completed():
    """Verify no events have 'failed' status on happy path."""
    graph = build_aegis_graph()
    result = graph.invoke(
        AegisState(), config={"configurable": {"thread_id": "status-test"}}
    )

    for event in result["events"]:
        assert event.status == "completed", (
            f"Event {event.phase} has status {event.status}: {event.message}"
        )
