from aegis.models.state import AegisState, RunEvent


def run_evals(execution_result: dict) -> dict:
    """Evaluate training results against quality and safety gates."""
    metrics = execution_result.get("metrics", {})
    failures: list[str] = []

    final_loss = metrics.get("train_loss", float("inf"))
    if final_loss > 2.0:
        failures.append(f"Loss did not converge: {final_loss:.3f} > 2.0")

    if "eval_loss" in metrics:
        train_eval_gap = metrics["eval_loss"] - metrics["train_loss"]
        if train_eval_gap > 1.0:
            failures.append(f"Severe overfitting: {train_eval_gap:.2f} gap")

    canary_passed = _run_canary_test(execution_result.get("model_path"))
    if not canary_passed:
        failures.append("Canary safety test failed")

    return {"passed": len(failures) == 0, "metrics": metrics, "failures": failures}


def _run_canary_test(model_path: str | None) -> bool:
    """Run canary safety test against the fine-tuned model. MVP stub."""
    return True


def run_evals_node(state: AegisState) -> AegisState:
    """Evaluate training results and update state with eval outcome."""
    if state.execution_result is None:
        event = RunEvent(
            phase="run_evals",
            status="failed",
            message="Cannot run evals: no execution result",
        )
        return state.model_copy(update={"events": state.events + [event]})

    try:
        eval_result = run_evals(state.execution_result)
        status = "completed" if eval_result["passed"] else "failed"
        message = f"Evaluation {'passed' if eval_result['passed'] else 'failed'}"
        if eval_result["failures"]:
            message += f": {len(eval_result['failures'])} failures"

        event = RunEvent(
            phase="run_evals",
            status=status,
            message=message,
            data=eval_result,
        )
        return state.model_copy(
            update={
                "eval_result": eval_result,
                "events": state.events + [event],
            }
        )
    except Exception as e:
        event = RunEvent(
            phase="run_evals",
            status="failed",
            message=f"Eval failed: {str(e)}",
        )
        return state.model_copy(update={"events": state.events + [event]})
