from pydantic import BaseModel


class SearchRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "Find me backend engineers with Python skills"},
                {"question": "Who has more than 3 years of experience?"},
                {"question": "Show candidates in Mumbai expecting less than 80000 salary"},
            ]
        }
    }


class SearchResponse(BaseModel):
    answer: str
    sql_query: str | None = None


class HealthResponse(BaseModel):
    status: str
    db_table: str
    llm_model: str


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "someone experienced with container orchestration and CI/CD"},
                {"query": "backend developer comfortable with relational databases"},
            ]
        }
    }


class SemanticSearchResult(BaseModel):
    candidate_id: int
    name: str
    position: str
    score: float
    snippet: str


class SemanticSearchResponse(BaseModel):
    results: list[SemanticSearchResult]


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "Who has experience with container orchestration and CI/CD?"},
                {"question": "Now filter those by salary under 80000", "conversation_id": "abc123"},
            ]
        }
    }


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, str]
    seconds: float
    result_preview: str


class AskResponse(BaseModel):
    answer: str
    conversation_id: str
    tool_calls: list[ToolCall] = []
    total_seconds: float | None = None


class ResumeQARequest(BaseModel):
    candidate_id: int
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"candidate_id": 1, "question": "What are this candidate's top skills?"},
                {"candidate_id": 1, "question": "Has the candidate worked with Docker?"},
            ]
        }
    }


class ResumeQASource(BaseModel):
    score: float | None = None
    text: str


class ResumeQAStep(BaseModel):
    step: str
    detail: str
    seconds: float


class ResumeQAResponse(BaseModel):
    candidate_id: int
    name: str
    question: str
    answer: str
    sources: list[ResumeQASource]
    steps: list[ResumeQAStep] = []
