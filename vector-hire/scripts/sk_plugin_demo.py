"""
Step 1c + 1d of docs/learning-semantic-kernel.md

Same two real functions as scripts/tool_calling_demo.py (sql_search, semantic_search),
but wrapped as a Semantic Kernel Plugin instead of a hand-written tool schema + loop.
Compare this file side-by-side with tool_calling_demo.py:
  - tool_calling_demo.py -> YOU write the tool schema + the "call model, run tool,
    send result back, call model again" loop by hand.
  - this file            -> SK's `kernel_function` decorator replaces the schema,
    and `FunctionChoiceBehauvior.Auto()` replaces the whole loop.
Same underlying functions do the real work in both — only the "front door" differs.

Run: uv run python scripts/sk_plugin_demo.py "your question here"
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


# 1c. Plugins / Kernel Functions
#
# This class is the SK equivalent of the `TOOLS` list in tool_calling_demo.py.
# Each @kernel_function is one callable tool. The `description=` argument plays
# the exact same role as the "description" field did in the raw tool schema:
# it's the sentence the MODEL reads to decide whether/when to call this function.
# `Annotated[str, "..."]` attaches a per-argument description, same job as the
# "parameters.properties.<name>.description" field did in the raw schema.
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
        print(f"  [called] sql_search(question={question!r})")
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
        print(f"  [called] semantic_search(query={query!r})")
        results = run_semantic_search(query, top_k=5)
        if not results:
            return "No matching candidates found."
        lines = [
            f"- {r['name']} ({r['position']}), match score {r['score']:.0%}: {r['snippet']}"
            for r in results
        ]
        return "\n".join(lines)


async def ask(kernel: Kernel, question: str) -> str:
    # 1d. Automatic Function Calling.
    #
    # FunctionChoiceBehavior.Auto() tells the Kernel: "look at every function from
    # every registered plugin, decide for yourself whether any are needed for this
    # question, call the ones you choose, feed their real results back to the model,
    # and give me the final answer." This one call replaces the entire manual
    # "round 1 -> check tool_calls -> run function -> round 2" loop from
    # tool_calling_demo.py's ask() function.
    settings = kernel.get_prompt_execution_settings_from_service_id("openrouter-chat")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    result = await kernel.invoke_prompt(question, arguments=None, settings=settings)
    return str(result)


async def main(question: str) -> None | str:
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

    # Registering the plugin is what makes its functions visible to the Kernel,
    # the same way `tools=TOOLS` made the raw schema visible to the raw API call.
    kernel.add_plugin(CandidateSearchPlugin(), plugin_name="CandidateSearch")

    result = await ask(kernel, question)
    print(f"Question: {question}\n")
    print("Answer:")
    print(result)
    return str(result)


if __name__ == "__main__":
    validate()
    init_query_engine()
    build_index()

    question = " ".join(sys.argv[1:]) or "Who has experience with container orchestration and CI/CD?"
    asyncio.run(main(question))
