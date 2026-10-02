from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI

from app.config import DB_TABLE, LLM_MODEL, OPENAI_API_KEY
from app.database import get_engine

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

# Singleton — built once at startup, reused for every request
_query_engine: NLSQLTableQueryEngine | None = None


def init_query_engine() -> None:
    global _query_engine

    llm = OpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENAI_API_KEY,
    )
    Settings.llm = llm

    sql_database = SQLDatabase(get_engine(), include_tables=[DB_TABLE])

    _query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[DB_TABLE],
        context_str_prefix=TABLE_CONTEXT,
        verbose=True,
    )


def get_query_engine() -> NLSQLTableQueryEngine:
    return _query_engine
