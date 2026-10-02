import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402

TEST_ENV = {
    "OPENAI_API_KEY": "test-key",
    "DB_SERVER": "test-server",
    "DB_NAME": "test-db",
    "DB_USER": "test-user",
    "DB_PASSWORD": "test-password",
    "DB_TABLE": "Candidates",
    "LLM_MODEL": "test-model",
}


class FakeResponse:
    """Stand-in for a LlamaIndex Response: str() gives the answer."""

    def __init__(self, answer, metadata=None):
        self.answer = answer
        self.metadata = metadata

    def __str__(self):
        return self.answer


@pytest.fixture
def env(monkeypatch):
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    return TEST_ENV


@pytest.fixture
def query_engine(monkeypatch):
    """Replace the LLM + SQL Server query engine with a mock."""
    engine = MagicMock()
    engine.query.return_value = FakeResponse(
        "Found 1 candidate: Asha (Backend Engineer).",
        {"sql_query": "SELECT Name FROM Candidates WHERE Skills LIKE '%Python%'"},
    )
    monkeypatch.setattr(app_module, "build_query_engine", lambda: engine)
    return engine


@pytest.fixture
def client(env, query_engine):
    with TestClient(app_module.app) as c:
        yield c
