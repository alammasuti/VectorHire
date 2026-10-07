import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

EXAMPLES = [
    "Find me backend engineers with Python skills",
    "Who has more than 3 years of experience?",
    "Show candidates expecting less than 80000 salary",
    "Find candidates who know React or Angular",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Candidate Search",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar: health status ────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API Connected")
        st.write(f"**Table:** `{health['db_table']}`")
        st.write(f"**Model:** `{health['llm_model']}`")
    except Exception:
        st.error("API Disconnected")
        st.caption(f"Start the API with: `python api.py`")
        st.caption(f"Expected at: `{API_URL}`")

    st.divider()
    st.caption("How it works:")
    st.caption("1. You type a question in plain English")
    st.caption("2. AI translates it to SQL")
    st.caption("3. SQL runs against SQL Server")
    st.caption("4. Results come back as a readable answer")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Candidate Search")
st.caption("Find candidates by asking questions in plain English — no SQL needed.")

# ── Example buttons ───────────────────────────────────────────────────────────

st.write("**Quick examples — click to use:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["question_input"] = example

st.divider()

# ── Search input ──────────────────────────────────────────────────────────────

question = st.text_input(
    "Your question",
    placeholder="e.g. Find me backend engineers with Python skills",
    key="question_input",
)

search_clicked = st.button("Search", type="primary", use_container_width=False)

# ── Search logic ──────────────────────────────────────────────────────────────

if search_clicked and question.strip():
    st.session_state["question"] = question

    with st.spinner("Searching candidates..."):
        try:
            resp = requests.post(
                f"{API_URL}/candidates/search",
                json={"question": question},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            st.session_state["last_answer"] = data.get("answer", "")
            st.session_state["last_sql"] = data.get("sql_query")
            st.session_state["last_error"] = None

        except requests.exceptions.ConnectionError:
            st.session_state["last_error"] = "Cannot reach the API. Make sure `python api.py` is running."
            st.session_state["last_answer"] = None
        except Exception as e:
            st.session_state["last_error"] = str(e)
            st.session_state["last_answer"] = None

elif search_clicked and not question.strip():
    st.warning("Please enter a question first.")

# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

elif st.session_state.get("last_answer"):
    st.divider()
    st.subheader("Answer")
    st.info(st.session_state["last_answer"])

    if st.session_state.get("last_sql"):
        with st.expander("View generated SQL (what the AI wrote)"):
            st.code(st.session_state["last_sql"], language="sql")
