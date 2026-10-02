"""API tests: SQL Server and the LLM are replaced with SQLite and a fake query engine."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

import app as app_module
from semantic import SemanticSearch
from tests.test_semantic import ConceptEmbedding


class FakeResponse:
    metadata = {"sql_query": "SELECT 1"}

    def __str__(self):
        return "ok"


class FakeQueryEngine:
    def __init__(self):
        self.questions = []

    def query(self, question):
        self.questions.append(question)
        return FakeResponse()


@pytest.fixture
def client(monkeypatch):
    for k in ["OPENAI_API_KEY", "DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD"]:
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("DB_TABLE", "Candidates")

    # One shared connection, so the table is visible from the app's worker thread.
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE Candidates (Id INTEGER, Name TEXT, PositionAppliedFor TEXT, Skills TEXT)"))
        conn.execute(text(
            "INSERT INTO Candidates VALUES (1, 'Asha', 'DevOps Engineer', 'Kubernetes, Docker'),"
            "(2, 'Meera', 'Backend Engineer', 'Python, Django')"
        ))

    fake = FakeQueryEngine()
    monkeypatch.setattr(app_module, "build_db_engine", lambda: engine)
    monkeypatch.setattr(app_module, "build_query_engine", lambda e: fake)
    monkeypatch.setattr(
        app_module.SemanticSearch, "from_env",
        classmethod(lambda cls, e, t: SemanticSearch(e, t, ConceptEmbedding())),
    )
    with TestClient(app_module.app) as c:
        c.fake = fake
        yield c


def test_health_reports_semantic_search(client):
    assert client.get("/health").json()["semantic_search"] is True


def test_semantic_search_endpoint_finds_synonym(client):
    body = client.post("/candidates/semantic-search", json={"question": "k8s"}).json()
    assert "Kubernetes" in body["related_skills"]
    assert body["matches"][0]["name"] == "Asha"


def test_hybrid_search_passes_related_skills_to_sql_engine(client):
    body = client.post("/candidates/search", json={"question": "k8s experts"}).json()
    assert "Kubernetes" in body["related_skills"]
    assert body["semantic_matches"][0]["name"] == "Asha"
    assert "Kubernetes" in client.fake.questions[-1]


def test_use_semantic_false_is_plain_text_to_sql(client):
    body = client.post("/candidates/search", json={"question": "k8s experts", "use_semantic": False}).json()
    assert body["related_skills"] == [] and body["semantic_matches"] == []
    assert client.fake.questions[-1] == "k8s experts"


def test_reindex(client):
    assert client.post("/candidates/reindex").json() == {"skills_indexed": 4, "candidates_indexed": 2}
