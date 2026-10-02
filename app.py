import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI

from semantic import SemanticSearch, expand_question

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TABLE_CONTEXT = """
This table contains job candidates who have applied for positions.

Columns:
- Id: unique candidate ID
- Name: candidate's full name
- Email: contact email
- Phone: contact phone number
- Village: village or neighborhood
- LocationCityState: city and state the candidate is in (e.g. "Mumbai, Maharashtra")
- PositionAppliedFor: the job role the candidate applied for (e.g. "Backend Engineer", "Frontend Developer")
- Availability: when the candidate can start
- ResumeFileName: name of the uploaded resume file (text label only)
- AppliedAt: date the candidate applied
- ExpectedSalary: the salary the candidate expects (numeric)
- Skills: comma-separated list of technical skills (e.g. "Python, Django, PostgreSQL")
- TotalExperience: total years of work experience (numeric)
- RelevantExperience: years of experience relevant to the applied role (numeric)

IMPORTANT RULES FOR SQL GENERATION:
1. NEVER include ResumeFileData in any SELECT — it contains raw binary bytes and will cause errors.
2. Use LIKE '%skill%' to search inside the Skills column (e.g. Skills LIKE '%Python%').
3. For experience filters, use TotalExperience or RelevantExperience with numeric comparisons.
4. Always SELECT useful columns: Name, PositionAppliedFor, Skills, TotalExperience, ExpectedSalary, LocationCityState.
"""

# ── Request / Response models (these appear in Swagger UI automatically) ──────

class SearchRequest(BaseModel):
    question: str
    use_semantic: bool = True

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "Find me backend engineers with Python skills"},
                {"question": "Who has more than 3 years of experience?"},
                {"question": "Show candidates in Mumbai expecting less than 80000 salary"},
            ]
        }
    }


class CandidateMatch(BaseModel):
    id: str
    name: str | None = None
    position: str | None = None
    skills: str | None = None
    total_experience: float | None = None
    location: str | None = None
    score: float


class SearchResponse(BaseModel):
    answer: str
    sql_query: str | None = None
    related_skills: list[str] = []
    semantic_matches: list[CandidateMatch] = []


class SemanticSearchRequest(BaseModel):
    question: str
    top_k: int | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "container orchestration"},
                {"question": "someone who can build mobile apps"},
            ]
        }
    }


class SemanticSearchResponse(BaseModel):
    related_skills: list[str]
    matches: list[CandidateMatch]


class ReindexResponse(BaseModel):
    skills_indexed: int
    candidates_indexed: int


class HealthResponse(BaseModel):
    status: str
    db_table: str
    llm_model: str
    semantic_search: bool


# ── Build query engine (called once at startup) ───────────────────────────────

def build_db_engine():
    connection_url = URL.create(
        "mssql+pyodbc",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_SERVER"],
        database=os.environ["DB_NAME"],
        query={
            "driver": "ODBC Driver 17 for SQL Server",
            "TrustServerCertificate": "yes",
        },
    )
    return create_engine(connection_url)


def build_query_engine(engine) -> NLSQLTableQueryEngine:
    table_name = os.environ["DB_TABLE"]
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    llm = OpenAI(
        model=model_name,
        temperature=0,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.llm = llm

    sql_database = SQLDatabase(engine, include_tables=[table_name])

    return NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[table_name],
        context_str_prefix=TABLE_CONTEXT,
        verbose=True,
    )


# ── App lifespan: initialise once on startup ──────────────────────────────────

_query_engine: NLSQLTableQueryEngine | None = None
_semantic: SemanticSearch | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _query_engine, _semantic
    required = ["OPENAI_API_KEY", "DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_TABLE"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    engine = build_db_engine()
    _query_engine = build_query_engine(engine)
    print("Query engine ready.")

    if os.environ.get("SEMANTIC_SEARCH", "true").lower() != "false":
        # Semantic search is an add-on: if embeddings fail, keep serving text-to-SQL.
        try:
            _semantic = SemanticSearch.from_env(engine, os.environ["DB_TABLE"])
            _semantic.build()
            print(f"Semantic index ready: {_semantic.candidate_count} candidates, {_semantic.skill_count} skills.")
        except Exception as e:
            _semantic = None
            print(f"Semantic search disabled: {e}")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Candidate Search API",
    description=(
        "Search job candidates using plain English questions.\n\n"
        "The AI translates your question into SQL, queries SQL Server, "
        "and returns a human-readable answer.\n\n"
        "**Powered by:** LlamaIndex + OpenRouter + SQL Server"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check that the API is running and see the current configuration."""
    return HealthResponse(
        status="ok",
        db_table=os.environ.get("DB_TABLE", ""),
        llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        semantic_search=_semantic is not None,
    )


@app.post("/candidates/search", response_model=SearchResponse, tags=["Candidates"])
def search_candidates(request: SearchRequest):
    """
    Search candidates using a plain English question.

    The API will:
    1. Find skills in the table that are semantically related to your question
       (e.g. "k8s" -> "Kubernetes") and pass them to the LLM as a hint
    2. Translate your question into SQL using an LLM
    3. Run the SQL against your SQL Server candidates table
    4. Return a human-readable answer, plus candidates ranked by embedding similarity

    Set `use_semantic` to false for plain text-to-SQL.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        related, matches = [], []
        if request.use_semantic and _semantic:
            related = _semantic.related_skills(request.question)
            matches = [CandidateMatch(**vars(m)) for m in _semantic.search_candidates(request.question)]

        response = _query_engine.query(expand_question(request.question, related))
        sql = (response.metadata or {}).get("sql_query")
        return SearchResponse(
            answer=str(response),
            sql_query=sql,
            related_skills=related,
            semantic_matches=matches,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/candidates/semantic-search", response_model=SemanticSearchResponse, tags=["Candidates"])
def semantic_search(request: SemanticSearchRequest):
    """
    Rank candidates by meaning rather than exact words, using embeddings only (no SQL generation).

    Finds candidates whose role and skills are similar to the question, so a search for
    "container orchestration" can return someone who lists "Kubernetes".
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not _semantic:
        raise HTTPException(status_code=503, detail="Semantic search is not available. Check the API logs.")

    try:
        return SemanticSearchResponse(
            related_skills=_semantic.related_skills(request.question),
            matches=[CandidateMatch(**vars(m)) for m in _semantic.search_candidates(request.question, request.top_k)],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/candidates/reindex", response_model=ReindexResponse, tags=["System"])
def reindex():
    """Rebuild the semantic index after candidates are added or changed."""
    if not _semantic:
        raise HTTPException(status_code=503, detail="Semantic search is not available. Check the API logs.")
    try:
        _semantic.build()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ReindexResponse(skills_indexed=_semantic.skill_count, candidates_indexed=_semantic.candidate_count)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
