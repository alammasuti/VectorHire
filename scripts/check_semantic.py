"""Check semantic search against your real database and embedding API.

Usage:
    uv run python scripts/check_semantic.py "container orchestration"

Prints the related skills and top candidates for the query. No LLM/SQL generation is used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import build_db_engine  # noqa: E402  (also loads .env)
from semantic import SemanticSearch  # noqa: E402
import os  # noqa: E402

query = " ".join(sys.argv[1:]) or "container orchestration"
search = SemanticSearch.from_env(build_db_engine(), os.environ["DB_TABLE"])
search.build()
print(f"Indexed {search.candidate_count} candidates and {search.skill_count} distinct skills.\n")
print(f"Query: {query}")
print("Related skills:", ", ".join(search.related_skills(query)) or "(none above threshold)")
print("\nTop candidates:")
for m in search.search_candidates(query):
    print(f"  {m.score:.3f}  {m.name}  |  {m.position}  |  {m.skills}")
