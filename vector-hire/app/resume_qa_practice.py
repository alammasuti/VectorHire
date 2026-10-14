from llama_index.core import Document, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from app import config
from app.config import OPENAI_API_KEY, LLM_MODEL
from app.resume_index_manual import extract_pdf_text, read_resume_file

EMBED_MODEL = "openai/text-embedding-3-small"

_embed_model = OpenAIEmbedding(
    model_name=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

_llm = OpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

print("Embedding model initialized:", _embed_model.model_name)
print("LLM initialized:", _llm.model)


def make_document(file_name, name):
    pdf_bytes = read_resume_file(file_name)
    if pdf_bytes is None:
        return None
    extracted_text = extract_pdf_text(pdf_bytes)

    return Document(
        text=extracted_text,
        doc_id=name,
        metadata={"source": file_name},
    )


def build_index(doc: Document) -> VectorStoreIndex:
    return VectorStoreIndex.from_documents([doc], embed_model=_embed_model)


def create_prompt():
    prompt_template = (
        "You are a helpful assistant that answers questions about resumes.\n"
        "Use the following context to answer the question:\n"
        "{context_str}\n"
        "Question: {query_str}\n"
        "Answer:"
    )
    return PromptTemplate(prompt_template)


def _build_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    return RetrieverQueryEngine.from_args(
        retriever=VectorIndexRetriever(index=index),
        llm=_llm,
        text_qa_template=create_prompt(),
    )

_index_cache = {}
_answer_cache = {}

def ask_question_with_doc(question: str, doc_name: str) -> dict | None:
    question = question.strip()
    if not question:
        return None

    key = (doc_name.lower(), question.lower())
    if key in _answer_cache:
        print("Cached answer.")
        return _answer_cache[key]

    index = _index_cache.get(doc_name.lower())
    message = f"Building index for {doc_name}..." if index is None else f"Using cached index for {doc_name}."

    print(message)
    if index is None:
        doc = make_document(doc_name, doc_name)
        if doc is None:
            raise FileNotFoundError(f"Could not read {doc_name}")
        index = build_index(doc)
        _index_cache[doc_name.lower()] = index

    response = _build_query_engine(index).query(question)

    result = {
        "question": question,
        "answer": str(response).strip(),
        "sources": [
            {"score": n.score, "text": n.node.get_content()} for n in response.source_nodes
        ],
    }
    _answer_cache[key] = result
    return result


if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        doc_name = input("Enter the resume file name (e.g., 'amit_resume.pdf'): ")
        try:
            result = ask_question_with_doc(question, doc_name)
            if result:
                print(f"Answer: {result['answer']}")
                for src in result["sources"]:
                    print(f"  source score: {src['score']}")
        except FileNotFoundError as e:
            print(e)

