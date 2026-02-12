from aegis.models.state import AegisState, RunEvent
from aegis.profilers.cost import estimate_training_cost


def estimate_cost_node(state: AegisState) -> AegisState:
    """Estimate training cost and VRAM from the current TrainingSpec.

    Appends a RunEvent with phase ``estimate_cost`` regardless of outcome.
    On success the returned state includes a populated ``cost_estimate``.

    Args:
        state: Current graph state; must contain a non-None ``spec``.

    Returns:
        Updated AegisState with cost_estimate and a new event.
    """
    if state.spec is None:
        event = RunEvent(
            phase="estimate_cost",
            status="failed",
            message="Cannot estimate cost: no training spec provided",
        )
        return state.model_copy(update={"events": state.events + [event]})

    try:
        cost_estimate = estimate_training_cost(state.spec)
        event = RunEvent(
            phase="estimate_cost",
            status="completed",
            message=(
                f"Estimated ${cost_estimate.estimated_cost_usd:.4f}, "
                f"{cost_estimate.estimated_vram_gb:.1f}GB VRAM"
            ),
            data=cost_estimate.model_dump(),
        )
        return state.model_copy(
            update={"cost_estimate": cost_estimate, "events": state.events + [event]}
        )
    except Exception as e:
        event = RunEvent(
            phase="estimate_cost",
            status="failed",
            message=f"Cost estimation failed: {str(e)}",
        )
        return state.model_copy(update={"events": state.events + [event]})
