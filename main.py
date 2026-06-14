import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI

load_dotenv()

# ── Table context: tells the LLM what each column means ──────────────────────
# This is "prompt engineering" — we guide the AI so it writes better SQL.
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


def build_connection_url() -> URL:
    # URL.create() handles special characters like backslash in SERVER\INSTANCE correctly
    return URL.create(
        "mssql+pyodbc",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_SERVER"],   # e.g. 10.10.10.9\SQLEXPRESS
        database=os.environ["DB_NAME"],
        query={
            "driver": "ODBC Driver 17 for SQL Server",
            "TrustServerCertificate": "yes",
        },
    )


def main():
    # ── Validate env vars ─────────────────────────────────────────────────────
    required = ["OPENAI_API_KEY", "DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_TABLE"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your values.")
        return

    # ── Connect to SQL Server ─────────────────────────────────────────────────
    print("Connecting to SQL Server...")
    try:
        engine = create_engine(build_connection_url())
        with engine.connect():
            pass  # test the connection
        print("Connected successfully.\n")
    except Exception as e:
        print(f"ERROR: Could not connect to SQL Server.\n{e}")
        return

    table_name = os.environ["DB_TABLE"]
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # ── Set up LlamaIndex with OpenRouter ────────────────────────────────────
    # OpenRouter is OpenAI-compatible but needs a different base URL and model prefix
    llm = OpenAI(
        model=model_name,
        temperature=0,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.llm = llm

    sql_database = SQLDatabase(
        engine,
        include_tables=[table_name],
    )

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[table_name],
        context_str_prefix=TABLE_CONTEXT,
        verbose=True,  # prints the generated SQL so you can learn from it
    )

    # ── Interactive query loop ────────────────────────────────────────────────
    print("Candidate Search  (powered by LlamaIndex + OpenAI)")
    print("Type your question in plain English. Type 'quit' to exit.\n")
    print("Example queries:")
    print("  - Find me backend engineers with Python skills")
    print("  - Who has more than 3 years of experience?")
    print("  - Show candidates in Mumbai expecting less than 80000 salary")
    print("-" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        print("\nSearching...\n")
        try:
            response = query_engine.query(question)
            print("\nAnswer:")
            print(str(response))
        except Exception as e:
            print(f"Error running query: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
