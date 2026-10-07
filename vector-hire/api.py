from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import validate
from app.engine import init_query_engine
from app.routes import candidates, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate()
    init_query_engine()
    print("Query engine ready.")
    yield


app = FastAPI(
    title="Candidate Search API",
    description=(
        "Search job candidates using plain English questions.\n\n"
        "The AI translates your question into SQL, queries SQL Server, "
        "and returns a human-readable answer.\n\n"
        "**Powered by:** LlamaIndex + OpenRouter + SQL Server"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(candidates.router, prefix="/candidates")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
