"""Embedding-based search over candidate skills and profiles.

Two in-memory vector indexes are built from the candidates table at startup:

- a *skill* index with one entry per distinct skill in the Skills column, used
  to expand a question with related skills ("k8s" -> "Kubernetes") before it is
  handed to the text-to-SQL engine, so LIKE filters catch synonyms;
- a *candidate* index with one entry per candidate (role, skills, experience,
  location), used to return semantically ranked matches directly.
"""

import os
from dataclasses import dataclass

from sqlalchemy import MetaData, Table, select
from sqlalchemy.engine import Engine

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

# Columns used to describe a candidate. ResumeFileData is binary and never read.
PROFILE_COLUMNS = [
    "Id",
    "Name",
    "PositionAppliedFor",
    "Skills",
    "TotalExperience",
    "RelevantExperience",
    "LocationCityState",
    "ExpectedSalary",
    "Availability",
]

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def build_embed_model() -> OpenAIEmbedding:
    """Embedding model from env. Defaults to OpenAI; OpenRouter keys route to OpenRouter."""
    api_key = os.environ.get("EMBED_API_KEY") or os.environ["OPENAI_API_KEY"]
    default_base = OPENROUTER_API_BASE if api_key.startswith("sk-or-") else OPENAI_API_BASE
    api_base = os.environ.get("EMBED_API_BASE", default_base)
    default_model = (
        "openai/text-embedding-3-small" if api_base == OPENROUTER_API_BASE else "text-embedding-3-small"
    )
    model_name = os.environ.get("EMBED_MODEL", default_model)
    # `model` only satisfies OpenAIEmbedding's validation; `model_name` is what is sent to the API.
    return OpenAIEmbedding(
        model="text-embedding-3-small",
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )


def split_skills(skills: str | None) -> list[str]:
    return [s.strip() for s in (skills or "").split(",") if s.strip()]


def profile_text(row: dict) -> str:
    parts = [
        f"Role: {row.get('PositionAppliedFor') or 'unknown'}",
        f"Skills: {row.get('Skills') or 'none listed'}",
    ]
    if row.get("TotalExperience") is not None:
        parts.append(f"Total experience: {row['TotalExperience']} years")
    if row.get("RelevantExperience") is not None:
        parts.append(f"Relevant experience: {row['RelevantExperience']} years")
    if row.get("LocationCityState"):
        parts.append(f"Location: {row['LocationCityState']}")
    return ". ".join(parts)


@dataclass
class CandidateMatch:
    id: str
    name: str | None
    position: str | None
    skills: str | None
    total_experience: float | None
    location: str | None
    score: float


class SemanticSearch:
    def __init__(
        self,
        engine: Engine,
        table_name: str,
        embed_model: BaseEmbedding,
        skill_top_k: int = 8,
        skill_min_score: float = 0.5,
        candidate_top_k: int = 10,
        candidate_min_score: float = 0.3,
    ):
        self.engine = engine
        self.table_name = table_name
        self.embed_model = embed_model
        self.skill_top_k = skill_top_k
        self.skill_min_score = skill_min_score
        self.candidate_top_k = candidate_top_k
        self.candidate_min_score = candidate_min_score
        self.skill_count = 0
        self.candidate_count = 0
        self._skill_index: VectorStoreIndex | None = None
        self._candidate_index: VectorStoreIndex | None = None

    @classmethod
    def from_env(cls, engine: Engine, table_name: str) -> "SemanticSearch":
        env = os.environ
        return cls(
            engine,
            table_name,
            build_embed_model(),
            skill_top_k=int(env.get("SEMANTIC_SKILL_TOP_K", 8)),
            skill_min_score=float(env.get("SEMANTIC_SKILL_MIN_SCORE", 0.5)),
            candidate_top_k=int(env.get("SEMANTIC_CANDIDATE_TOP_K", 10)),
            candidate_min_score=float(env.get("SEMANTIC_CANDIDATE_MIN_SCORE", 0.3)),
        )

    def _load_rows(self) -> list[dict]:
        table = Table(self.table_name, MetaData(), autoload_with=self.engine)
        columns = [table.c[name] for name in PROFILE_COLUMNS if name in table.c]
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(select(*columns))]

    def build(self) -> None:
        """(Re)build both indexes from the current table contents."""
        rows = self._load_rows()

        skills: dict[str, str] = {}  # lowercase -> first spelling seen
        for row in rows:
            for skill in split_skills(row.get("Skills")):
                skills.setdefault(skill.lower(), skill)
        skill_nodes = [TextNode(text=s, id_=f"skill:{k}") for k, s in skills.items()]

        candidate_nodes = [
            TextNode(
                text=profile_text(row),
                id_=f"candidate:{row['Id']}",
                metadata={
                    "id": str(row["Id"]),
                    "name": row.get("Name"),
                    "position": row.get("PositionAppliedFor"),
                    "skills": row.get("Skills"),
                    "total_experience": _to_float(row.get("TotalExperience")),
                    "location": row.get("LocationCityState"),
                },
                excluded_embed_metadata_keys=["id", "name", "position", "skills", "total_experience", "location"],
            )
            for row in rows
        ]

        self._skill_index = VectorStoreIndex(skill_nodes, embed_model=self.embed_model)
        self._candidate_index = VectorStoreIndex(candidate_nodes, embed_model=self.embed_model)
        self.skill_count = len(skill_nodes)
        self.candidate_count = len(candidate_nodes)

    def related_skills(self, question: str) -> list[str]:
        """Skills from the table that are semantically close to the question."""
        if not self._skill_index or not self.skill_count:
            return []
        retriever = self._skill_index.as_retriever(similarity_top_k=self.skill_top_k)
        return [
            n.node.get_content()
            for n in retriever.retrieve(question)
            if (n.score or 0) >= self.skill_min_score
        ]

    def search_candidates(self, question: str, top_k: int | None = None) -> list[CandidateMatch]:
        """Candidates ranked by similarity of their profile to the question."""
        if not self._candidate_index or not self.candidate_count:
            return []
        retriever = self._candidate_index.as_retriever(similarity_top_k=top_k or self.candidate_top_k)
        matches = []
        for n in retriever.retrieve(question):
            if (n.score or 0) < self.candidate_min_score:
                continue
            m = n.node.metadata
            matches.append(
                CandidateMatch(
                    id=m["id"],
                    name=m.get("name"),
                    position=m.get("position"),
                    skills=m.get("skills"),
                    total_experience=m.get("total_experience"),
                    location=m.get("location"),
                    score=round(float(n.score), 4),
                )
            )
        return matches


def expand_question(question: str, related_skills: list[str]) -> str:
    """Append related skills as a hint for the text-to-SQL engine."""
    if not related_skills:
        return question
    return (
        f"{question}\n\n"
        "(Hint from semantic search: these skills exist in the Skills column and are related to "
        f"the question: {', '.join(related_skills)}. If the question asks about skills, "
        "technologies or expertise, match ANY of them with OR'ed Skills LIKE conditions, "
        "alongside any skill named in the question. Ignore this hint otherwise.)"
    )


def _to_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
