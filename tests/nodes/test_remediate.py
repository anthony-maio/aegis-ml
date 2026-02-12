import pytest

from aegis.models.state import AegisState, TrainingSpec
from aegis.nodes.remediate import remediate_spec, remediate_spec_node


def test_remediate_oom_shrinks_batch_size():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=8,
    )
    new_spec = remediate_spec(spec, "CUDA out of memory")
    assert new_spec.micro_batch_size == 4


def test_remediate_oom_at_batch_1_switches_to_qlora():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=1,
    )
    new_spec = remediate_spec(spec, "CUDA out of memory")
    assert new_spec.method == "qlora"


def test_remediate_full_finetune_at_batch_1_switches_to_lora():
    spec = TrainingSpec(
        method="full_finetune",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=1,
    )
    new_spec = remediate_spec(spec, "CUDA out of memory")
    assert new_spec.method == "lora"


def test_remediate_nan_reduces_lr():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        learning_rate=5e-5,
    )
    new_spec = remediate_spec(spec, "Loss became NaN")
    assert new_spec.learning_rate == 2.5e-5


def test_remediate_node_updates_state():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
            micro_batch_size=8,
        ),
        retry_count=0,
    )
    new_state = remediate_spec_node(state, error_message="CUDA out of memory")
    assert new_state.spec.micro_batch_size == 4
    assert new_state.retry_count == 1
