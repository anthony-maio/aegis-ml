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
        # Extract LLM-provided model metadata from parse_intent event
        estimated_params_b = 0.0
        is_moe = False
        for event in state.events:
            if event.phase == "parse_intent" and event.data:
                estimated_params_b = event.data.get("estimated_params_b", 0.0)
                is_moe = event.data.get("is_moe", False)
                break

        cost_estimate = estimate_training_cost(
            state.spec, estimated_params_b=estimated_params_b, is_moe=is_moe
        )
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
