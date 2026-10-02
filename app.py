import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from llama_index.core import Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI

from sql_guard import ReadOnlySQLDatabase, blocked_reason

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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "Find me backend engineers with Python skills"},
                {"question": "Who has more than 3 years of experience?"},
                {"question": "Show candidates in Mumbai expecting less than 80000 salary"},
            ]
        }
    }


class SearchResponse(BaseModel):
    answer: str
    sql_query: str | None = None


class HealthResponse(BaseModel):
    status: str
    db_table: str
    llm_model: str


# ── Build query engine (called once at startup) ───────────────────────────────

def build_query_engine() -> NLSQLTableQueryEngine:
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
    engine = create_engine(connection_url)

    table_name = os.environ["DB_TABLE"]
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    llm = OpenAI(
        model=model_name,
        temperature=0,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.llm = llm

    # Only single SELECT statements are allowed to reach the database.
    sql_database = ReadOnlySQLDatabase(engine, include_tables=[table_name])

    return NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[table_name],
        context_str_prefix=TABLE_CONTEXT,
        verbose=True,
    )


# ── App lifespan: initialise once on startup ──────────────────────────────────

_query_engine: NLSQLTableQueryEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _query_engine
    required = ["OPENAI_API_KEY", "DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_TABLE"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    _query_engine = build_query_engine()
    print("Query engine ready.")
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
    )


@app.post("/candidates/search", response_model=SearchResponse, tags=["Candidates"])
def search_candidates(request: SearchRequest):
    """
    Search candidates using a plain English question.

    The API will:
    1. Translate your question into SQL using an LLM
    2. Run the SQL against your SQL Server candidates table
    3. Return a human-readable answer with the matching candidates
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = _query_engine.query(request.question)
        sql = (response.metadata or {}).get("sql_query")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # The guard already stopped this SQL from running; report it plainly instead
    # of returning the LLM's attempt to explain the error.
    reason = blocked_reason(sql)
    if reason:
        raise HTTPException(status_code=400, detail=reason)
    return SearchResponse(answer=str(response), sql_query=sql)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
