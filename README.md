# Candidate Search — AI POC

Search job candidates using plain English questions. The AI translates your question into SQL, queries SQL Server, and returns a human-readable answer.

**Powered by:** LlamaIndex · OpenRouter · SQL Server · FastAPI · Streamlit

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.13+ | [python.org](https://www.python.org/) |
| uv | `pip install uv` |
| ODBC Driver 17 for SQL Server | Download from Microsoft |
| SQL Server | Remote or local instance |
| OpenRouter API key | [openrouter.ai](https://openrouter.ai/) |

---

## Setup

**1. Install dependencies**
```powershell
uv sync
```

**2. Create your `.env` file** (copy from example and fill in your values)
```powershell
copy .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-or-v1-your-openrouter-key

DB_SERVER=10.10.10.9\SQLEXPRESS
DB_NAME=your_database_name
DB_USER=vectorhire_reader
DB_PASSWORD=your_password
DB_TABLE=SoulsoftJobApplication

LLM_MODEL=gpt-4o-mini
```

**3. Use a read-only database user**

The SQL is written by an LLM, so connect with a login that can only read the candidates table. Never use `sa` or any account that can write. Run this once on SQL Server as an admin (change the names and password to match yours):

```sql
CREATE LOGIN vectorhire_reader WITH PASSWORD = 'choose-a-strong-password';
USE your_database_name;
CREATE USER vectorhire_reader FOR LOGIN vectorhire_reader;
GRANT SELECT ON dbo.SoulsoftJobApplication TO vectorhire_reader;
```

Then set `DB_USER=vectorhire_reader` and its password in `.env`.

The app also checks every generated query before running it and only allows a single `SELECT` (see `sql_guard.py`). Anything else, such as `DELETE`, `DROP`, `EXEC` or a second statement, is refused with a `Blocked query: ...` error (HTTP 400 from the API). That check is a safety net; the read-only user is what actually protects the data.

---

## Running the App

You need **two terminals** running at the same time.

### Terminal 1 — Start the API
```powershell
python app.py
```
API runs at: `http://localhost:8000`

### Terminal 2 — Start the UI
```powershell
uv run python -m streamlit run ui.py
```
UI opens automatically at: `http://localhost:8501`

---

## URLs

| URL | What it is |
|---|---|
| `http://localhost:8501` | Streamlit UI (main interface) |
| `http://localhost:8000/docs` | Swagger API docs (test endpoints directly) |
| `http://localhost:8000/health` | API health check (JSON) |

---

## Example Questions

- `Find me backend engineers with Python skills`
- `Who has more than 3 years of experience?`
- `Show candidates in Mumbai expecting less than 80000 salary`
- `Find candidates who know React or Angular`

---

## CLI (Optional — no UI needed)

Run queries directly in the terminal without starting any server:
```powershell
uv run python main.py
```

---

## Project Structure

```
.
├── app.py          # FastAPI backend (REST API + Swagger)
├── ui.py           # Streamlit frontend
├── main.py         # CLI version (terminal only)
├── sql_guard.py    # Rejects any generated SQL that is not a single SELECT
├── tests/          # Unit tests (uv run python -m unittest)
├── pyproject.toml  # Dependencies (managed by uv)
├── .env            # Your secrets (never commit this)
└── .env.example    # Template for .env
```

---

## How It Works

```
You type:  "Find Python backend engineers with 2+ years"
                        ↓
           LlamaIndex + OpenRouter LLM
                        ↓
           SELECT Name, Skills, ... FROM SoulsoftJobApplication
           WHERE Skills LIKE '%Python%' AND TotalExperience >= 2
                        ↓
           SQL Server returns matching rows
                        ↓
           LLM formats results into a readable answer
```
