"""
Variant of scripts/sk_plugin_demo.py — same idea, but using two plain functions
instead of a Plugin class. Compare the two files side-by-side.

What changed from sk_plugin_demo.py:
  - No `class CandidateSearchPlugin:` wrapper, no `self` parameter.
  - Each function is registered individually with `kernel.add_function(...)`
    instead of one `kernel.add_plugin(...)` call for the whole class.
  - SK still needs a `plugin_name` per function (its internal registry groups
    every function under SOME plugin name, even a single loose function) —
    so "no plugin at all" isn't actually possible, just "no class wrapper."

Run: uv run python scripts/sk_function_demo.py "your question here"
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions import kernel_function

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import LLM_MODEL, OPENAI_API_KEY, validate
from app.engine import get_query_engine, init_query_engine
from app.resume_index_manual import build_index, semantic_search as run_semantic_search


# No class, no `self` — just a plain function with the same decorator as before.
@kernel_function(
    name="sql_search",
    description=(
        "Search candidates using structured/filterable fields: salary, "
        "years of experience, location, position applied for. Use this "
        "for questions involving numbers, filters, or sorting."
    ),
)
def sql_search(
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


async def ask(kernel: Kernel, question: str) -> str:
    settings = kernel.get_prompt_execution_settings_from_service_id("openrouter-chat")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    result = await kernel.invoke_prompt(question, arguments=None, settings=settings)
    return str(result)


async def main(question: str) -> None:
    kernel = Kernel()

    openrouter_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    chat_service = OpenAIChatCompletion(
        ai_model_id=LLM_MODEL,
        async_client=openrouter_client,
        service_id="openrouter-chat",
    )
    kernel.add_service(chat_service)

    # Instead of one add_plugin(SomeClass()) call, register each function on its
    # own. plugin_name is still required — both are grouped under "CandidateSearch"
    # here, same as before, just without a class holding them together.
    kernel.add_function(plugin_name="CandidateSearch", function=sql_search)
    kernel.add_function(plugin_name="CandidateSearch", function=semantic_search)

    print(f"Question: {question}\n")
    print("Answer:")
    print(await ask(kernel, question))


if __name__ == "__main__":
    validate()
    init_query_engine()
    build_index()

    question = " ".join(sys.argv[1:]) or "Who has experience with container orchestration and CI/CD?"
    asyncio.run(main(question))
