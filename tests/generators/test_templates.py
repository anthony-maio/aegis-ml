import pytest
from aegis.models.state import TrainingSpec
from aegis.generators.template import TemplateCodeGenerator


def test_template_generator_exists():
    generator = TemplateCodeGenerator()
    assert generator is not None


def test_generate_basic_training_script():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        num_epochs=3,
        micro_batch_size=4,
        learning_rate=5e-5,
        target_gpu="a10g",
    )
    generator = TemplateCodeGenerator()
    code = generator.generate(spec)
    assert "from transformers import" in code
    assert "TrainingArguments" in code
    assert "LoraConfig" in code
    assert spec.model_name in code
    assert str(spec.learning_rate) in code


def test_generated_script_has_main_entrypoint():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
    )
    generator = TemplateCodeGenerator()
    code = generator.generate(spec)
    assert "if __name__" in code
    assert "main()" in code


def test_full_finetune_generates_different_code():
    spec = TrainingSpec(
        method="full_finetune",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
    )
    generator = TemplateCodeGenerator()
    code = generator.generate(spec)
    assert "LoraConfig" not in code
