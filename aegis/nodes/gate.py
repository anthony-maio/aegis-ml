from aegis.models.state import AegisState, RunEvent


def check_auto_approve(state: AegisState) -> bool:
    """Determine whether the estimated cost qualifies for auto-approval.

    Auto-approval requires:
    - A cost estimate to be present.
    - The estimated cost to be within the hard budget limit.
    - The estimated cost to be at or below the soft threshold.

    Args:
        state: Current graph state with cost_estimate and budget_policy.

    Returns:
        True if the cost can be auto-approved, False otherwise.
    """
    if state.cost_estimate is None:
        return False
    if state.cost_estimate.estimated_cost_usd > state.budget_policy.max_budget_usd:
        return False
    if state.cost_estimate.estimated_cost_usd <= state.budget_policy.soft_threshold_usd:
        return True
    return False


def budget_gate_node(state: AegisState) -> AegisState:
    """Evaluate the estimated cost against the budget policy.

    Outcomes:
    - **approve**: cost is within the soft threshold (auto-approved).
    - **cancel**: cost exceeds the hard budget limit.
    - **interrupted**: cost is between thresholds; requires human approval.

    Args:
        state: Current graph state; must contain a non-None ``cost_estimate``.

    Returns:
        Updated AegisState with a budget_gate event and optional human_decision.
    """
    if state.cost_estimate is None:
        event = RunEvent(
            phase="budget_gate",
            status="failed",
            message="Cannot evaluate budget: no cost estimate available",
        )
        return state.model_copy(update={"events": state.events + [event]})

    auto_approve = check_auto_approve(state)

    if state.cost_estimate.estimated_cost_usd > state.budget_policy.max_budget_usd:
        status_msg = (
            f"Cost ${state.cost_estimate.estimated_cost_usd:.4f} "
            f"exceeds maximum ${state.budget_policy.max_budget_usd:.2f}"
        )
        status = "failed"
        decision = "cancel"
    elif auto_approve:
        status_msg = f"Cost ${state.cost_estimate.estimated_cost_usd:.4f} auto-approved"
        status = "completed"
        decision = "approve"
    else:
        status_msg = (
            f"Cost ${state.cost_estimate.estimated_cost_usd:.4f} "
            f"requires human approval"
        )
        status = "interrupted"
        decision = None

    event = RunEvent(
        phase="budget_gate",
        status=status,
        message=status_msg,
        data={
            "auto_approve": auto_approve,
            "cost_usd": state.cost_estimate.estimated_cost_usd,
            "soft_threshold": state.budget_policy.soft_threshold_usd,
            "max_budget": state.budget_policy.max_budget_usd,
        },
    )

    updates: dict = {"events": state.events + [event]}
    if decision:
        updates["human_decision"] = decision
    return state.model_copy(update=updates)
