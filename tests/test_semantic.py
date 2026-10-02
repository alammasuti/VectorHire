"""Tests for semantic.py using an in-memory SQLite table and a stand-in embedding model.

The stand-in maps words to a few fixed "concepts", the way a real embedding model
places synonyms near each other, so the tests run offline and without an API key.
"""

import math

import pytest
from sqlalchemy import create_engine, text

from llama_index.core.base.embeddings.base import BaseEmbedding

from semantic import SemanticSearch, expand_question, profile_text, split_skills

CONCEPTS = {
    "kubernetes": 0, "k8s": 0, "container": 0, "orchestration": 0, "docker": 0,
    "react": 1, "reactjs": 1, "frontend": 1, "angular": 1, "ui": 1,
    "python": 2, "django": 2, "backend": 2, "postgresql": 2,
    "flutter": 3, "mobile": 3, "android": 3, "kotlin": 3,
}


class ConceptEmbedding(BaseEmbedding):
    def _vec(self, text: str) -> list[float]:
        v = [0.0] * (len(set(CONCEPTS.values())) + 1)
        for word in text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split():
            v[CONCEPTS.get(word, len(v) - 1)] += 1.0 if word in CONCEPTS else 0.05
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _get_text_embedding(self, text):
        return self._vec(text)

    def _get_query_embedding(self, query):
        return self._vec(query)

    async def _aget_query_embedding(self, query):
        return self._vec(query)


@pytest.fixture
def search():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE Candidates (Id INTEGER, Name TEXT, PositionAppliedFor TEXT, Skills TEXT, "
            "TotalExperience REAL, RelevantExperience REAL, LocationCityState TEXT, ResumeFileData BLOB)"
        ))
        conn.execute(text(
            "INSERT INTO Candidates VALUES "
            "(1, 'Asha', 'DevOps Engineer', 'Kubernetes, Docker', 4, 3, 'Pune, Maharashtra', x'00'),"
            "(2, 'Ravi', 'Frontend Developer', 'ReactJS, Angular', 2, 2, 'Mumbai, Maharashtra', x'00'),"
            "(3, 'Meera', 'Backend Engineer', 'Python, Django, PostgreSQL', 5, 5, 'Delhi, Delhi', x'00'),"
            "(4, 'Kabir', 'Mobile Developer', 'Flutter, kotlin', 1, 1, NULL, NULL)"
        ))
    s = SemanticSearch(engine, "Candidates", ConceptEmbedding(), skill_min_score=0.5, candidate_min_score=0.3)
    s.build()
    return s


def test_build_counts_distinct_skills_and_candidates(search):
    assert search.candidate_count == 4
    assert search.skill_count == 9


def test_synonym_query_expands_to_skills_in_table(search):
    assert set(search.related_skills("k8s")) == {"Kubernetes", "Docker"}


def test_synonym_query_returns_matching_candidate(search):
    matches = search.search_candidates("container orchestration")
    assert matches[0].name == "Asha"
    assert all(m.name != "Meera" for m in matches)


def test_role_level_query_ranks_relevant_candidate_first(search):
    assert search.search_candidates("someone who can build mobile apps")[0].name == "Kabir"


def test_unrelated_question_adds_no_skill_hint(search):
    assert search.related_skills("salary expectations") == []
    assert expand_question("salary expectations", []) == "salary expectations"


def test_expand_question_mentions_related_skills():
    q = expand_question("Find k8s people", ["Kubernetes"])
    assert q.startswith("Find k8s people")
    assert "Kubernetes" in q


def test_helpers():
    assert split_skills(" Python, ,Django ") == ["Python", "Django"]
    assert split_skills(None) == []
    assert "Skills: none listed" in profile_text({"PositionAppliedFor": "QA", "Skills": None})
