# Candidate Search — AI POC

Search job candidates using plain English questions. The AI translates your question into SQL, queries SQL Server, and returns a human-readable answer.

Semantic search (embeddings) makes skill searches synonym-aware: asking for "k8s" or "container orchestration" also finds candidates who list "Kubernetes".

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

**3. Semantic search (optional, on by default)**

Embeddings use your `OPENAI_API_KEY` with no extra setup: an OpenAI key calls OpenAI, an OpenRouter key calls OpenRouter's embeddings endpoint (`openai/text-embedding-3-small`). To use a different key or provider for embeddings, set `EMBED_API_KEY`, `EMBED_API_BASE` and `EMBED_MODEL` (see `.env.example`).

The API embeds every candidate once at startup and keeps the index in memory. If embeddings fail (bad key, provider without an embeddings endpoint), the API logs `Semantic search disabled: ...` and keeps serving plain text-to-SQL; `/health` shows `"semantic_search": false`. Set `SEMANTIC_SEARCH=false` to turn it off.

After adding or editing candidates, rebuild the index with `POST /candidates/reindex` (or restart the API).

Check it against your real data:
```powershell
uv run python scripts/check_semantic.py "container orchestration"
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
- `Who knows container orchestration?` (semantic: finds Kubernetes/Docker)

---

## API Endpoints

| Endpoint | What it does |
|---|---|
| `POST /candidates/search` | Hybrid search: related skills from embeddings are passed to text-to-SQL; response also includes `related_skills` and `semantic_matches`. Send `"use_semantic": false` for plain text-to-SQL. |
| `POST /candidates/semantic-search` | Embedding-only ranking of candidates (no LLM/SQL). Body: `{"question": "...", "top_k": 10}` |
| `POST /candidates/reindex` | Rebuild the semantic index from the table |
| `GET /health` | Status, table, model, and whether semantic search is on |

---

## Tests

```powershell
uv run pytest
```
Tests use SQLite and a stand-in embedding model, so they need no database or API key.

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
├── semantic.py     # Embedding indexes over skills and candidate profiles
├── scripts/        # check_semantic.py: try semantic search on real data
├── tests/          # pytest suite (offline)
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
           Embeddings find related skills in the table (e.g. Django)
           and the closest candidate profiles
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
