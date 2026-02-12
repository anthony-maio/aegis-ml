import os

from jinja2 import Environment, FileSystemLoader

from aegis.generators.base import CodeGenerator
from aegis.models.state import TrainingSpec


class TemplateCodeGenerator(CodeGenerator):
    """Jinja2 template-based code generator for HuggingFace training scripts.

    Renders a Jinja2 template with values from a TrainingSpec to produce
    a complete, runnable HuggingFace training script.

    Args:
        template_dir: Path to the directory containing Jinja2 templates.
            Defaults to the project-level ``templates/`` directory.
    """

    def __init__(self, template_dir: str | None = None) -> None:
        if template_dir is None:
            template_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "templates"
            )
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            keep_trailing_newline=True,
        )
        self.template = self.env.get_template("hf_training.py.j2")

    def generate(self, spec: TrainingSpec) -> str:
        """Generate a HuggingFace training script from a TrainingSpec.

        Args:
            spec: The training specification describing the desired
                fine-tuning configuration.

        Returns:
            A string containing the generated Python training script.
        """
        return self.template.render(spec=spec)
