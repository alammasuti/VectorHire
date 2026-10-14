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

SEMANTIC_EXAMPLES = [
    "someone experienced with container orchestration and CI/CD",
    "backend developer comfortable with relational databases",
    "data analyst with visualization and statistics skills",
]

st.set_page_config(
    page_title="Candidate Search",
    page_icon="🔍",
    layout="wide",
)

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

st.title("Candidate Search")

sql_tab, semantic_tab, qa_tab = st.tabs(
    ["🗄️ Text-to-SQL Search", "🧠 Semantic Search (Manual)", "💬 Resume Q&A (llama)"]
)

with sql_tab:
    st.caption("Find candidates by asking questions in plain English — no SQL needed.")

    st.write("**Quick examples — click to use:**")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True, key=f"sql_example_{example}"):
            st.session_state["question_input"] = example

    st.divider()

    question = st.text_input(
        "Your question",
        placeholder="e.g. Find me backend engineers with Python skills",
        key="question_input",
    )

    search_clicked = st.button("Search", type="primary", use_container_width=False, key="sql_search_btn")

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

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    elif st.session_state.get("last_answer"):
        st.divider()
        st.subheader("Answer")
        st.info(st.session_state["last_answer"])

        if st.session_state.get("last_sql"):
            with st.expander("View generated SQL (what the AI wrote)"):
                st.code(st.session_state["last_sql"], language="sql")

with semantic_tab:
    st.caption("Find candidates by meaning — describe the skills/experience you want, and resumes are matched by content, not just keywords.")

    st.write("**Quick examples — click to use:**")
    semantic_cols = st.columns(len(SEMANTIC_EXAMPLES))
    for col, example in zip(semantic_cols, SEMANTIC_EXAMPLES):
        if col.button(example, use_container_width=True, key=f"semantic_example_{example}"):
            st.session_state["semantic_query_input"] = example

    st.divider()

    semantic_col1, semantic_col2 = st.columns([4, 1])
    with semantic_col1:
        semantic_query = st.text_input(
            "Describe the candidate you're looking for",
            placeholder="e.g. someone experienced with container orchestration and CI/CD",
            key="semantic_query_input",
        )
    with semantic_col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5, key="semantic_top_k")

    semantic_search_clicked = st.button(
        "Search", type="primary", use_container_width=False, key="semantic_search_btn"
    )

    if semantic_search_clicked and semantic_query.strip():
        with st.spinner("Searching resumes..."):
            try:
                resp = requests.post(
                    f"{API_URL}/candidates/semantic-search",
                    json={"query": semantic_query, "top_k": int(top_k)},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                st.session_state["semantic_results"] = data.get("results", [])
                st.session_state["semantic_error"] = None

            except requests.exceptions.ConnectionError:
                st.session_state["semantic_error"] = "Cannot reach the API. Make sure `python api.py` is running."
                st.session_state["semantic_results"] = None
            except Exception as e:
                st.session_state["semantic_error"] = str(e)
                st.session_state["semantic_results"] = None

    elif semantic_search_clicked and not semantic_query.strip():
        st.warning("Please describe who you're looking for first.")

    if st.session_state.get("semantic_error"):
        st.error(st.session_state["semantic_error"])

    elif st.session_state.get("semantic_results") is not None:
        st.divider()
        results = st.session_state["semantic_results"]

        if not results:
            st.info("No matching candidates found.")
        else:
            st.subheader(f"Top {len(results)} matches")
            for rank, result in enumerate(results, start=1):
                with st.container(border=True):
                    header_col, score_col = st.columns([4, 1])
                    with header_col:
                        st.markdown(f"**{rank}. {result['name']}** — {result['position']}")
                    with score_col:
                        st.metric("Match", f"{result['score']:.0%}")
                    st.caption(result["snippet"])

with qa_tab:
    st.caption("Pick a candidate and ask questions about their resume. Answers come from the resume PDF only (LlamaIndex RAG).")

    try:
        cand_resp = requests.get(f"{API_URL}/candidates/resume-candidates", timeout=15)
        cand_resp.raise_for_status()
        candidates = cand_resp.json()
    except Exception as e:
        candidates = []
        st.error(f"Could not load candidates: {e}")

    if candidates:
        labels = {c["candidate_id"]: f"{c['name']} - {c['position']}" for c in candidates}
        candidate_id = st.selectbox(
            "Candidate", options=list(labels), format_func=lambda cid: labels[cid], key="qa_candidate"
        )

        qa_question = st.text_input(
            "Your question",
            placeholder="e.g. What are this candidate's top skills?",
            key="qa_question_input",
        )

        if st.button("Ask", type="primary", key="qa_ask_btn"):
            if not qa_question.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Reading resume..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/candidates/resume-qa",
                            json={"candidate_id": candidate_id, "question": qa_question},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        st.session_state["qa_result"] = resp.json()
                        st.session_state["qa_error"] = None
                    except requests.exceptions.ConnectionError:
                        st.session_state["qa_error"] = "Cannot reach the API. Make sure `python api.py` is running."
                        st.session_state["qa_result"] = None
                    except Exception as e:
                        st.session_state["qa_error"] = str(e)
                        st.session_state["qa_result"] = None

        if st.session_state.get("qa_error"):
            st.error(st.session_state["qa_error"])
        elif st.session_state.get("qa_result"):
            result = st.session_state["qa_result"]
            st.divider()
            st.subheader(f"Answer about {result['name']}")
            st.info(result["answer"])
            with st.expander("⚙️ What happened behind the scenes", expanded=True):
                for s in result.get("steps", []):
                    icon = "⚡" if s["step"] == "Cached index" else "✅"
                    st.markdown(f"{icon} **{s['step']}** — {s['detail']} _({s['seconds']}s)_")
            with st.expander("Resume excerpts the AI used"):
                for src in result["sources"]:
                    score = f"{src['score']:.0%}" if src.get("score") is not None else "n/a"
                    st.caption(f"Match {score}")
                    st.text(src["text"])
