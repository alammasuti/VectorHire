import time
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.functions import kernel_function

from app.config import LLM_MODEL, OPENAI_API_KEY
from app.engine import get_query_engine
from app.resume_index_manual import semantic_search as run_semantic_search

SERVICE_ID = "openrouter-chat"
PROMPT_PATH = Path(__file__).parent / "prompts" / "candidate_assistant.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


class CandidateSearchPlugin:
    @kernel_function(
        name="sql_search",
        description=(
            "Search candidates using structured/filterable fields: salary, "
            "years of experience, location, position applied for. Use this "
            "for questions involving numbers, filters, or sorting."
        ),
    )
    def sql_search(
        self,
        question: Annotated[str, "A plain-English question about candidate data, e.g. 'Who has more than 3 years experience?'"],
    ) -> str:
        response = get_query_engine().query(question)
        return str(response)

    @kernel_function(
        name="semantic_search",
        description=(
            "Find candidates by matching the MEANING of their resume text to a "
            "described skill or experience. Use this for open-ended skill/experience "
            "questions, not numeric filters."
        ),
    )
    def semantic_search(
        self,
        query: Annotated[str, "A description of the skills/experience to match, e.g. 'container orchestration and CI/CD'"],
    ) -> str:
        results = run_semantic_search(query, top_k=5)
        if not results:
            return "No matching candidates found."
        lines = [
            f"- {r['name']} ({r['position']}), match score {r['score']:.0%}: {r['snippet']}"
            for r in results
        ]
        return "\n".join(lines)


_kernel: Kernel | None = None
_conversations: dict[str, ChatHistory] = {}
_request_tool_calls: ContextVar[list | None] = ContextVar("request_tool_calls", default=None)


async def tool_call_logger(context: AutoFunctionInvocationContext, next):
    name = f"{context.function.plugin_name}.{context.function.name}"
    print(f"\n[TOOL CALL] {name} args={dict(context.arguments)}")

    started = time.perf_counter()
    await next(context)
    elapsed = time.perf_counter() - started

    result = str(context.function_result).replace("\n", " ")
    if len(result) > 150:
        result = result[:150] + "..."
    print(f"[TOOL DONE] {name} in {elapsed:.1f}s -> {result}")

    calls = _request_tool_calls.get()
    if calls is not None:
        calls.append(
            {
                "tool": context.function.name,
                "arguments": {k: str(v) for k, v in context.arguments.items()},
                "seconds": round(elapsed, 2),
                "result_preview": result,
            }
        )


def init_agent() -> None:
    global _kernel

    kernel = Kernel()
    openrouter_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")
    kernel.add_service(
        OpenAIChatCompletion(ai_model_id=LLM_MODEL, async_client=openrouter_client, service_id=SERVICE_ID)
    )
        
    kernel.add_plugin(CandidateSearchPlugin(), plugin_name="CandidateSearch")
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, tool_call_logger)
    _kernel = kernel


def _get_history(conversation_id: str) -> ChatHistory:
    if conversation_id not in _conversations:
        history = ChatHistory()
        history.add_system_message(SYSTEM_PROMPT)
        _conversations[conversation_id] = history
    return _conversations[conversation_id]


def _log_history(label: str, conversation_id: str, history: ChatHistory) -> None:
    print(f"\n===== HISTORY {label} | conversation_id={conversation_id} | {len(history.messages)} messages =====")
    for i, message in enumerate(history.messages):
        text = str(message.content or "").replace("\n", " ")
        if len(text) > 120:
            text = text[:120] + "..."
        print(f"  [{i}] {message.role.value:<9} {text}")
    print("=" * 60)


async def ask_agent(question: str, conversation_id: str) -> tuple[str, list[dict]]:
    tool_calls: list[dict] = []
    _request_tool_calls.set(tool_calls)

    history = _get_history(conversation_id)
    history.add_user_message(question)
    _log_history("BEFORE model call", conversation_id, history)

    settings = _kernel.get_prompt_execution_settings_from_service_id(SERVICE_ID)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    service = _kernel.get_service(SERVICE_ID)
    reply = await service.get_chat_message_content(chat_history=history, settings=settings, kernel=_kernel)
    history.add_assistant_message(str(reply))
    _log_history("AFTER model call", conversation_id, history)
    return str(reply), tool_calls
