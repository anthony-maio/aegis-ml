from aegis.models.state import AegisState, RunEvent
from aegis.generators.template import TemplateCodeGenerator


def generate_code_node(state: AegisState) -> AegisState:
    """Generate a HuggingFace training script from the current TrainingSpec.

    Uses :class:`TemplateCodeGenerator` to render a Jinja2 template into a
    complete, runnable Python script.  Appends a RunEvent with phase
    ``generate_code`` regardless of outcome.

    Args:
        state: Current graph state; must contain a non-None ``spec``.

    Returns:
        Updated AegisState with generated_code and a new event.
    """
    if state.spec is None:
        event = RunEvent(
            phase="generate_code",
            status="failed",
            message="Cannot generate code: no training spec provided",
        )
        return state.model_copy(update={"events": state.events + [event]})

    try:
        generator = TemplateCodeGenerator()
        code = generator.generate(state.spec)
        event = RunEvent(
            phase="generate_code",
            status="completed",
            message=f"Generated {len(code)} bytes of training code",
            data={"method": state.spec.method, "code_length": len(code)},
        )
        return state.model_copy(
            update={"generated_code": code, "events": state.events + [event]}
        )
    except Exception as e:
        event = RunEvent(
            phase="generate_code",
            status="failed",
            message=f"Code generation failed: {str(e)}",
        )
        return state.model_copy(update={"events": state.events + [event]})
