import pytest

from aegis.models.state import AegisState
from aegis.nodes.evaluator import run_evals, run_evals_node


def test_run_evals_passing():
    result = run_evals(
        {"metrics": {"train_loss": 1.5, "eval_loss": 1.8}, "model_path": "/tmp/model"}
    )
    assert result["passed"] is True
    assert len(result["failures"]) == 0


def test_run_evals_high_loss_fails():
    result = run_evals(
        {"metrics": {"train_loss": 2.5}, "model_path": "/tmp/model"}
    )
    assert result["passed"] is False
    assert "Loss did not converge" in result["failures"][0]


def test_run_evals_overfitting_fails():
    result = run_evals(
        {"metrics": {"train_loss": 0.5, "eval_loss": 2.0}, "model_path": "/tmp/model"}
    )
    assert result["passed"] is False
    assert "overfitting" in result["failures"][0].lower()


def test_run_evals_node_adds_to_state():
    state = AegisState(
        execution_result={
            "metrics": {"train_loss": 1.5},
            "model_path": "/tmp/model",
        }
    )
    new_state = run_evals_node(state)
    assert new_state.eval_result is not None
    assert new_state.eval_result["passed"] is True
