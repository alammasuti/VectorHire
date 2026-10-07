from fastapi import APIRouter
from app.config import DB_TABLE, LLM_MODEL
from app.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", db_table=DB_TABLE, llm_model=LLM_MODEL)
