import pytest

from aegis.executors.modal_runner import ModalExecutor


def test_modal_executor_can_be_created():
    executor = ModalExecutor()
    assert executor is not None


def test_modal_executor_generate_stub():
    executor = ModalExecutor()
    spec_code = executor._generate_stub()
    assert "modal" in spec_code.lower()
    assert "def run_training" in spec_code


def test_modal_executor_execute_returns_result():
    executor = ModalExecutor()
    result = executor.execute("print('hello')", "{}")
    assert result["returncode"] == 0
    assert "metrics" in result
