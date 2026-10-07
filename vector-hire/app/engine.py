from llama_index.core import SQLDatabase, Settings, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_TMPL
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
- ExpectedSalary: the salary the candidate expects (stored as NVARCHAR text, may contain non-numeric values)
- Skills: comma-separated list of technical skills (e.g. "Python, Django, PostgreSQL")
- TotalExperience: total years of work experience (stored as NVARCHAR text, e.g. '3.5')
- RelevantExperience: years of experience relevant to the applied role (stored as NVARCHAR text, e.g. '2')

IMPORTANT RULES FOR SQL GENERATION:
1. NEVER include ResumeFileData in any SELECT — it contains raw binary bytes and will cause errors.
2. Use LIKE '%skill%' to search inside the Skills column (e.g. Skills LIKE '%Python%').
3. ExpectedSalary, TotalExperience and RelevantExperience are NVARCHAR columns. For any numeric comparison,
   filtering or sorting, ALWAYS use TRY_CAST(column AS FLOAT), never CAST or CONVERT to INT
   (e.g. WHERE TRY_CAST(TotalExperience AS FLOAT) > 3 ORDER BY TRY_CAST(TotalExperience AS FLOAT) DESC).
4. Always SELECT useful columns: Name, PositionAppliedFor, Skills, TotalExperience, ExpectedSalary, LocationCityState.
"""

# context_str_prefix is not applied to the SQL prompt, so the rules go into the prompt itself
TEXT_TO_SQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_TMPL.replace("Only use tables listed below.", TABLE_CONTEXT + "\nOnly use tables listed below.")
)

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
        text_to_sql_prompt=TEXT_TO_SQL_PROMPT,
        verbose=True,
    )


def get_query_engine() -> NLSQLTableQueryEngine:
    return _query_engine
