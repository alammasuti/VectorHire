from contextvars import ContextVar
from pathlib import Path
from typing import Annotated
import time

from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.functions import kernel_function

from app.engine import get_query_engine
from app.resume_index_manual import semantic_search
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.contents import ChatHistory
from app.config import LLM_MODEL, OPENAI_API_KEY


PROMPT_PATH = Path(__file__).parent / "prompts" / "candidate_assistant.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

_kernal: Kernel | None = None
SERVICE_ID = "openrouter-chat"


class CandidateSearchPlugin:
    @kernel_function(
        name="search_sql",
        description=(
            "Search candidates using structured/filterable fields: salary,"
            "years of experience, location, position applied for. Use this "
            "for questions involving numbers, filters, or sorting."
        ),
    )
    def search_sql(self, question) -> str:
        response = get_query_engine().query(question)
        return str(response)

    @kernel_function(
        name="search_resume",
        description=(
            "Find candidates by matching the MEANING of their resume text to a "
            "described skill or experience. Use this for open-ended skill/experience "
            "questions, not numeric filters."
        ),
    )
    def search_resume(
        self,
        query: Annotated[
            str,
            "A description of the skills/experience to match, e.g. 'container orchestration and CI/CD'",
        ],
    ) -> str:
        results = semantic_search(query)
        if not results:
            return "No matching candidates found."
        lines = [
            f"- {r['name']}  {r['position']} {r['score']}: {r['snippet']}"
            for r in results
        ]
        return "\n".join(lines)


_request_tool_call: ContextVar[list | None] = ContextVar(
    "request_tool_calls", default=None
)


async def _tool_calling_history(context: AutoFunctionInvocationContext, next):

    log = context.function.plugin_name + " " + context.function.name
    print(f"[total call]{log}")
    start = time.perf_counter()
    await next(context)
    total_time = time.perf_counter() - start

    print(f"time taken by model {context.function.name} is {total_time}")

    calls = _request_tool_call.get()

    if calls is not None:
        calls.append(
            {
                "tool": context.function.name,
                "arguments": {k: str(v) for k, v in context.arguments.items()},
                "seconds": round(total_time, 2),
            }
        )


def build_kernel() -> Kernel:
    global _kernal
    _kernal = Kernel()
    llm = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

    openai_service = OpenAIChatCompletion(
        service_id=SERVICE_ID, ai_model_id=LLM_MODEL, async_client=llm
    )
    _kernal.add_service(openai_service)
    _kernal.add_plugin(CandidateSearchPlugin(), plugin_name="CandidateSearchPlugin")
    _kernal.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, _tool_calling_history)
    return _kernal


history_store = {}


def _get_history(conversation_id) -> ChatHistory:
    if conversation_id not in history_store:
        history = ChatHistory()
        history.add_system_message(SYSTEM_PROMPT)
        history_store[conversation_id] = history
    return history_store[conversation_id]


async def ask_question(question: str, conversation_id: str) -> tuple[str, list[dict]]:
    history = _get_history(conversation_id)
    tool_calls = []
    _request_tool_call.set(tool_calls)

    settings = _kernal.get_prompt_execution_settings_from_service_id(SERVICE_ID)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    service = _kernal.get_service(SERVICE_ID)

    history.add_user_message(question)
    reply = await service.get_chat_message_content(
        settings=settings, kernel=_kernal, chat_history=history
    )
    history.add_assistant_message(str(reply))
    return str(reply), tool_calls


import asyncio

if __name__ == "__main__":
    build_kernel()
    print(
        asyncio.run(
            ask_question("Show candidates expecting less than 80000 salary", "chat1")
        )
    )
    print(
        asyncio.run(
            ask_question("Which of them has the most total experience?", "chat1")
        )
    )
