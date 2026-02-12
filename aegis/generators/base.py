from abc import ABC, abstractmethod

from aegis.models.state import TrainingSpec


class CodeGenerator(ABC):
    """Abstract base class for training script code generators."""

    @abstractmethod
    def generate(self, spec: TrainingSpec) -> str:
        """Generate a training script from the given TrainingSpec.

        Args:
            spec: The training specification describing the desired
                fine-tuning configuration.

        Returns:
            A string containing the generated Python training script.
        """
        pass
