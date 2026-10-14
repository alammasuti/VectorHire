from fastapi import APIRouter, HTTPException
from app.engine import get_query_engine
from app.models import (
    ResumeQARequest,
    ResumeQAResponse,
    SearchRequest,
    SearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.resume_index_manual import semantic_search
from app.resume_llama_index import ask_resume_question, list_candidates_with_resumes

router = APIRouter()


@router.post("/search", response_model=SearchResponse, tags=["Candidates"])
def search_candidates(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = get_query_engine().query(request.question)
        sql = (response.metadata or {}).get("sql_query")
        return SearchResponse(answer=str(response), sql_query=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic-search", response_model=SemanticSearchResponse, tags=["Candidates"])
def semantic_search_candidates(request: SemanticSearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = semantic_search(request.query, top_k=request.top_k)
        return SemanticSearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume-qa", response_model=ResumeQAResponse, tags=["Candidates"])
def resume_qa(request: ResumeQARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask_resume_question(request.candidate_id, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(status_code=404, detail="No readable resume found for this candidate.")
    return ResumeQAResponse(**result)


@router.get("/resume-candidates", tags=["Candidates"])
def resume_candidates():
    try:
        return list_candidates_with_resumes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
