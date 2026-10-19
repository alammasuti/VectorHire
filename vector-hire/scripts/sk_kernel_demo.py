"""
Step 1a + 1b of docs/learning-semantic-kernel.md

Builds the smallest possible Semantic Kernel program: a Kernel, with one
Connector registered (pointed at OpenRouter, same key/model as the rest of
this project), and a single prompt sent through it. No plugins/tools yet —
that's Step 1c/1d.

Run: uv run python scripts/sk_kernel_demo.py "your question here"
"""

import asyncio
import sys
from pathlib import Path

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import LLM_MODEL, OPENAI_API_KEY, validate


async def main(question: str) -> None:
    # 1a. The Kernel is just an empty orchestrator to start — a container we're
    # about to register an AI connection into (and, later, plugins/tools).
    kernel = Kernel()

    # 1b. The Connector is HOW the Kernel reaches an actual AI service. SK's
    # OpenAIChatCompletion connector doesn't take a base_url directly — instead
    # we hand it our own pre-configured AsyncOpenAI client, pointed at OpenRouter,
    # exactly like the raw `openai.OpenAI(...)` client used in resume_index_manual.py
    # and tool_calling_demo.py. Swapping to real OpenAI or Azure OpenAI later would
    # just mean building a different client here — nothing else in this file changes.
    openrouter_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    chat_service = OpenAIChatCompletion(
        ai_model_id=LLM_MODEL,
        async_client=openrouter_client,
        service_id="openrouter-chat",
    )

    # Register the connector with the Kernel under its service_id. A Kernel can
    # hold multiple services (e.g. one for chat, one for embeddings) — this is
    # how it knows which one to use when asked to run a prompt.
    kernel.add_service(chat_service)

    # No plugins/tools registered yet, so this is just "Kernel as a pass-through
    # to the chat model" — proving steps 1a+1b work before we add tool-calling.
    result = await kernel.invoke_prompt(question)
    print(f"Question: {question}\n")
    print("Answer:")
    print(result)


if __name__ == "__main__":
    validate()
    question = " ".join(sys.argv[1:]) or "In one sentence, what is a vector embedding?"
    asyncio.run(main(question))
