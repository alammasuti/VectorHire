import pytest
from fastapi.testclient import TestClient

import app as app_module
from conftest import FakeResponse


def test_health_reports_config(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db_table": "Candidates", "llm_model": "test-model"}


def test_health_defaults_llm_model(client, monkeypatch):
    monkeypatch.delenv("LLM_MODEL")
    resp = client.get("/health")
    assert resp.json()["llm_model"] == "gpt-4o-mini"


def test_search_returns_answer_and_sql(client, query_engine):
    resp = client.post("/candidates/search", json={"question": "Python backend engineers"})
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": "Found 1 candidate: Asha (Backend Engineer).",
        "sql_query": "SELECT Name FROM Candidates WHERE Skills LIKE '%Python%'",
    }
    query_engine.query.assert_called_once_with("Python backend engineers")


def test_search_handles_missing_metadata(client, query_engine):
    query_engine.query.return_value = FakeResponse("No candidates found.", None)
    resp = client.post("/candidates/search", json={"question": "Rust experts"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "No candidates found.", "sql_query": None}


@pytest.mark.parametrize("question", ["", "   "])
def test_search_rejects_empty_question(client, query_engine, question):
    resp = client.post("/candidates/search", json={"question": question})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Question cannot be empty."
    query_engine.query.assert_not_called()


def test_search_requires_question_field(client):
    resp = client.post("/candidates/search", json={})
    assert resp.status_code == 422


def test_search_returns_500_on_engine_error(client, query_engine):
    query_engine.query.side_effect = RuntimeError("database unreachable")
    resp = client.post("/candidates/search", json={"question": "anyone"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "database unreachable"


def test_startup_fails_without_required_env(env, query_engine, monkeypatch):
    monkeypatch.delenv("DB_PASSWORD")
    with pytest.raises(RuntimeError, match="DB_PASSWORD"):
        with TestClient(app_module.app):
            pass
