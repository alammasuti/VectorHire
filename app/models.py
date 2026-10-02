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
