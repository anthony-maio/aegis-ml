"""Global test fixtures.

Force the Modal executor into mock mode so tests never hit real GPU infra.
"""

import pytest

from aegis.executors.modal_runner import ModalExecutor


@pytest.fixture(autouse=True)
def _force_mock_executor(monkeypatch):
    """Patch ModalExecutor to always use mock, regardless of local auth."""
    monkeypatch.setattr(ModalExecutor, "_has_modal_token", lambda self: False)
