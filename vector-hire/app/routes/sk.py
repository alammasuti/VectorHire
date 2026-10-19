import uuid

from fastapi import APIRouter, HTTPException

from app.models import AskRequest, AskResponse
from app.sk_agent import ask_agent

router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["Semantic Kernel"])
async def ask_candidates(request: AskRequest):
    """
    Ask a question in plain English. A Semantic Kernel agent decides on its own
    whether to call sql_search (structured filters), semantic_search (resume
    meaning matching), both, or neither, then answers using the real results.

    Pass the same conversation_id in follow-up questions so the agent remembers
    earlier turns. Leave it out to start a new conversation.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    conversation_id = request.conversation_id or uuid.uuid4().hex
    try:
        answer = await ask_agent(request.question, conversation_id)
        return AskResponse(answer=answer, conversation_id=conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
