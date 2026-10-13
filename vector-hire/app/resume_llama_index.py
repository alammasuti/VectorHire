"""Resume Q&A: ask a natural-language question about ONE candidate's resume (RAG).

Pipeline (each numbered step below is a function in this file):
    1. load_candidate_resume -> DB gives the file name, PDF is read from resumes/, wrapped in a Document
    2. create_resume_index   -> chunk + embed the Document into an in-memory vector index
    3. get_qa_prompt         -> instructions telling the LLM how to answer
    4. create_qa_engine      -> retriever (find chunks) + LLM (write answer)
    5. ask_resume_question   -> glue: runs steps 1-4 and returns answer + sources
"""

import time

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from sqlalchemy import text

from app.config import LLM_MODEL, OPENAI_API_KEY
from app.database import get_engine
from app.resume_index_manual import EMBED_MODEL, extract_pdf_text, read_resume_file


# Turns text into vectors. OpenRouter speaks the OpenAI API, so we reuse
# LlamaIndex's OpenAIEmbedding and just point api_base at OpenRouter.
_embed_model = OpenAIEmbedding(
    model_name=EMBED_MODEL,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

# Built once and reused. temperature=0 -> deterministic, factual answers.
_llm = OpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

# Building an index costs embedding API calls, so keep one per candidate.
# NOTE: if a candidate re-uploads a resume, restart the app (or clear this
# dict) to pick up the new file.
_index_cache: dict[int, BaseIndex] = {}


# ---------------------------------------------------------------------------
# Step 1: load the resume as a LlamaIndex Document
# ---------------------------------------------------------------------------
def load_candidate_resume(candidate_id: int) -> Document | None:
    """Look up the candidate's resume file name in the DB, read that PDF from
    the resumes/ folder, and return it as a Document.

    Returns None if the candidate does not exist, the file is missing from the
    folder, or the PDF has no extractable text (e.g. a scanned image).
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT Id, Name, PositionAppliedFor, ResumeFileName "
                "FROM JobApplication WHERE Id = :id AND ResumeFileName IS NOT NULL"
            ),
            {"id": candidate_id},
        ).fetchone()

    if row is None:
        return None

    pdf_bytes = read_resume_file(row.ResumeFileName)
    if pdf_bytes is None:
        return None

    resume_text = extract_pdf_text(pdf_bytes)
    if not resume_text:
        return None

    # metadata travels with every chunk, so answers can be traced to a candidate.
    return Document(
        text=resume_text,
        metadata={
            "candidate_id": row.Id,
            "name": row.Name,
            "position": row.PositionAppliedFor or "",
        },
        doc_id=f"candidate-{row.Id}",
    )


# ---------------------------------------------------------------------------
# Step 2: build a vector index from the Document
# ---------------------------------------------------------------------------
def create_resume_index(resume_document: Document) -> BaseIndex:
    """Split the resume into chunks, embed each chunk, and store them in memory.

    from_documents() does the whole ingestion in one call:
    Document -> chunks (default SentenceSplitter) -> embeddings -> index.
    We pass embed_model explicitly so LlamaIndex never falls back to its
    default (real OpenAI) embeddings.
    """
    return VectorStoreIndex.from_documents(
        [resume_document],
        embed_model=_embed_model,
    )


# ---------------------------------------------------------------------------
# Step 3: the prompt the LLM receives
# ---------------------------------------------------------------------------
def get_qa_prompt() -> PromptTemplate:
    """Instructions + placeholders; LlamaIndex fills them in for every question.

    {context_str} -> the resume chunks the retriever found
    {query_str}   -> the user's question
    Telling the model to answer ONLY from the resume reduces hallucination.
    """
    return PromptTemplate(
        "You are a recruitment assistant reviewing a single candidate's resume.\n"
        "Answer the question using ONLY the resume excerpts below.\n"
        "If the answer is not in the excerpts, reply: "
        "\"This is not mentioned in the resume.\"\n"
        "---------------------\n"
        "Resume excerpts:\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer: "
    )


# ---------------------------------------------------------------------------
# Step 4: retriever + LLM = query engine
# ---------------------------------------------------------------------------
def create_qa_engine(resume_index: BaseIndex, top_k: int = 3) -> RetrieverQueryEngine:
    """Combine retrieval and generation.

    Retriever: embeds the question and returns the top_k most similar chunks.
    Query engine: stuffs those chunks into the prompt and asks the LLM.
    """
    retriever = VectorIndexRetriever(index=resume_index, similarity_top_k=top_k)
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=_llm,
        text_qa_template=get_qa_prompt(),
    )


# ---------------------------------------------------------------------------
# Step 5: the function the API calls
# ---------------------------------------------------------------------------
def ask_resume_question(candidate_id: int, question: str) -> dict | None:
    """Answer `question` about one candidate. Returns None if no resume found.

    Return shape:
        {"candidate_id": 1, "name": "...", "question": "...",
         "answer": "...", "sources": [{"score": 0.82, "text": "..."}]}
    """
    # Processing trace shown in the UI: what happened and how long it took.
    steps: list[dict] = []

    def log(step: str, detail: str, started: float) -> None:
        steps.append({"step": step, "detail": detail, "seconds": round(time.perf_counter() - started, 2)})

    index = _index_cache.get(candidate_id)

    if index is None:
        t = time.perf_counter()
        document = load_candidate_resume(candidate_id)
        if document is None:
            return None
        log("Loading resume", f"Read PDF and extracted {len(document.text)} characters", t)

        t = time.perf_counter()
        index = create_resume_index(document)
        _index_cache[candidate_id] = index
        chunk_count = len(index.docstore.docs)
        log("Chunking + indexing", f"Split into {chunk_count} chunk(s), embedded and stored in memory", t)
    else:
        steps.append({"step": "Cached index", "detail": "Reused the index built earlier (no PDF read, no embedding)", "seconds": 0.0})

    t = time.perf_counter()
    response = create_qa_engine(index).query(question)
    log("Retrieval + answer", f"Found {len(response.source_nodes)} matching chunk(s) and asked the LLM", t)

    # source_nodes = the exact chunks the LLM was shown -> lets users verify the answer.
    sources = [
        {"score": node.score, "text": node.get_content()[:300]}
        for node in response.source_nodes
    ]
    first_metadata = response.source_nodes[0].node.metadata if response.source_nodes else {}

    return {
        "candidate_id": candidate_id,
        "name": first_metadata.get("name", ""),
        "question": question,
        "answer": str(response).strip(),
        "sources": sources,
        "steps": steps,
    }


def list_candidates_with_resumes() -> list[dict]:
    """Candidates whose resume file exists in resumes/ (used by the UI dropdown)."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT Id, Name, PositionAppliedFor, ResumeFileName FROM JobApplication "
                "WHERE ResumeFileName IS NOT NULL ORDER BY Name"
            )
        ).fetchall()
    return [
        {"candidate_id": r.Id, "name": r.Name, "position": r.PositionAppliedFor or ""}
        for r in rows
        if read_resume_file(r.ResumeFileName) is not None
    ]
