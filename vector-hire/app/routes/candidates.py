from fastapi import APIRouter, HTTPException
from app.engine import get_query_engine
from app.models import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse, tags=["Candidates"])
def search_candidates(request: SearchRequest):
    """
    Search candidates using a plain English question.

    The API will:
    1. Translate your question into SQL using an LLM
    2. Run the SQL against the SQL Server candidates table
    3. Return a human-readable answer with matching candidates
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = get_query_engine().query(request.question)
        sql = (response.metadata or {}).get("sql_query")
        return SearchResponse(answer=str(response), sql_query=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
