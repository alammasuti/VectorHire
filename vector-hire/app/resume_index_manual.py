import hashlib
import io
from pathlib import Path

import chromadb
from openai import OpenAI
from pypdf import PdfReader
from sqlalchemy import text

from app.config import OPENAI_API_KEY
from app.database import get_engine

EMBED_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
RESUMES_DIR = Path(__file__).parent.parent / "resumes"
COLLECTION_NAME = "resumes"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

_chroma_client = None
_collection = None


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def read_resume_file(file_name: str | None) -> bytes | None:
    if not file_name:
        return None
    path = RESUMES_DIR / Path(file_name).name
    return path.read_bytes() if path.is_file() else None


def chunk_text(text_: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text_.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _existing_hash(collection, candidate_id: int) -> str | None:
    existing = collection.get(where={"candidate_id": candidate_id}, limit=1)
    metadatas = existing.get("metadatas") or []
    if metadatas:
        return metadatas[0].get("content_hash")
    return None


def build_index() -> None:
    collection = get_collection()
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT Id, Name, PositionAppliedFor, ResumeFileName FROM JobApplication "
                "WHERE ResumeFileName IS NOT NULL"
            )
        ).fetchall()

    indexed, skipped = 0, 0
    for row in rows:
        pdf_bytes = read_resume_file(row.ResumeFileName)
        if pdf_bytes is None:
            print(f"Skipping candidate {row.Id}: '{row.ResumeFileName}' not found in {RESUMES_DIR}")
            continue
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if _existing_hash(collection, row.Id) == content_hash:
            skipped += 1
            continue

        collection.delete(where={"candidate_id": row.Id})

        resume_text = extract_pdf_text(pdf_bytes)
        chunks = chunk_text(resume_text)
        if not chunks:
            continue

        embeddings = embed_texts(chunks)
        ids = [f"candidate-{row.Id}-chunk-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "candidate_id": row.Id,
                "name": row.Name,
                "position": row.PositionAppliedFor or "",
                "content_hash": content_hash,
            }
            for _ in chunks
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        indexed += 1

    print(f"Resume index: {indexed} candidate(s) (re)indexed, {skipped} already up to date.")


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k * 3, top_k),
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    best_per_candidate: dict[int, dict] = {}
    for distance, document, metadata in zip(distances, documents, metadatas):
        candidate_id = metadata["candidate_id"]
        similarity = 1 - distance
        if candidate_id not in best_per_candidate or similarity > best_per_candidate[candidate_id]["score"]:
            best_per_candidate[candidate_id] = {
                "candidate_id": candidate_id,
                "name": metadata["name"],
                "position": metadata["position"],
                "score": similarity,
                "snippet": document[:300],
            }

    ranked = sorted(best_per_candidate.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]
