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
DB_USER=sa
DB_PASSWORD=your_password
DB_TABLE=SoulsoftJobApplication

LLM_MODEL=gpt-4o-mini
```

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
