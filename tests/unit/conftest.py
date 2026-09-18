import importlib.util
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_ollama() -> None:
    """Mock ollama module to avoid connection errors in unit tests."""
    mock_ollama = MagicMock()
    mock_response = {"message": {"content": "Test explanation"}}
    mock_ollama.chat.return_value = mock_response
    sys.modules["ollama"] = mock_ollama
    yield
    if "ollama" in sys.modules:
        del sys.modules["ollama"]


@pytest.fixture(autouse=True, scope="session")
def ensure_faiss_mock() -> None:
    """Ensure FAISS can be mocked when unavailable for tests."""
    try:
        spec = importlib.util.find_spec("faiss")
        if spec is not None:
            return
    except ImportError:
        pass

    mock_faiss_module = MagicMock()
    mock_index = MagicMock()
    mock_faiss_module.IndexFlatIP.return_value = mock_index
    mock_faiss_module.IndexHNSWFlat.return_value = mock_index
    mock_faiss_module.__spec__ = type("MockSpec", (), {"name": "faiss"})()

    sys.modules["faiss"] = mock_faiss_module
