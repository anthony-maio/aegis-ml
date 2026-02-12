from aegis.models.state import AegisState, TrainingSpec, RunEvent


def remediate_spec(spec: TrainingSpec, error_message: str) -> TrainingSpec:
    """Mutate TrainingSpec to fix common issues. Priority order (least invasive first)."""
    error_lower = error_message.lower()

    # 1. Out of memory -> shrink batch size
    if "out of memory" in error_lower or "cuda" in error_lower:
        new_batch = max(1, spec.micro_batch_size // 2)
        if new_batch < spec.micro_batch_size:
            return spec.model_copy(update={"micro_batch_size": new_batch})

    # 2. If batch at 1 -> switch method
    if spec.micro_batch_size == 1:
        if spec.method == "full_finetune":
            return spec.model_copy(update={"method": "lora"})
        elif spec.method == "lora":
            return spec.model_copy(update={"method": "qlora"})

    # 3. NaN or loss spike -> reduce learning rate
    if "nan" in error_lower or "loss spike" in error_lower:
        return spec.model_copy(update={"learning_rate": spec.learning_rate / 2})

    # 4. Generic batch reduction
    new_batch = max(1, spec.micro_batch_size // 2)
    return spec.model_copy(update={"micro_batch_size": new_batch})


def remediate_spec_node(state: AegisState, error_message: str) -> AegisState:
    """Apply remediation and update retry count."""
    if state.spec is None:
        event = RunEvent(
            phase="remediate_spec",
            status="failed",
            message="Cannot remediate: no training spec",
        )
        return state.model_copy(update={"events": state.events + [event]})

    new_spec = remediate_spec(state.spec, error_message)

    changes: list[str] = []
    if new_spec.micro_batch_size != state.spec.micro_batch_size:
        changes.append(
            f"batch_size: {state.spec.micro_batch_size} -> {new_spec.micro_batch_size}"
        )
    if new_spec.method != state.spec.method:
        changes.append(f"method: {state.spec.method} -> {new_spec.method}")
    if new_spec.learning_rate != state.spec.learning_rate:
        changes.append(f"lr: {state.spec.learning_rate} -> {new_spec.learning_rate}")

    change_desc = ", ".join(changes) if changes else "no changes"
    event = RunEvent(
        phase="remediate_spec",
        status="completed",
        message=f"Applied remediation: {change_desc}",
        data={
            "error": error_message,
            "changes": changes,
            "new_spec": new_spec.model_dump(),
        },
    )

    return state.model_copy(
        update={
            "spec": new_spec,
            "events": state.events + [event],
            "retry_count": state.retry_count + 1,
        }
    )
