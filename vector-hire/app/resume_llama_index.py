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


_embed_model = OpenAIEmbedding(
    model_name=EMBED_MODEL,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

_llm = OpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

_index_cache: dict[int, BaseIndex] = {}


def load_candidate_resume(candidate_id: int) -> Document | None:
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

    return Document(
        text=resume_text,
        metadata={
            "candidate_id": row.Id,
            "name": row.Name,
            "position": row.PositionAppliedFor or "",
        },
        doc_id=f"candidate-{row.Id}",
    )


def create_resume_index(resume_document: Document) -> BaseIndex:
    return VectorStoreIndex.from_documents(
        [resume_document],
        embed_model=_embed_model,
    )


def get_qa_prompt() -> PromptTemplate:
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


def create_qa_engine(resume_index: BaseIndex, top_k: int = 3) -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=resume_index, similarity_top_k=top_k)
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=_llm,
        text_qa_template=get_qa_prompt(),
    )


def ask_resume_question(candidate_id: int, question: str) -> dict | None:
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
