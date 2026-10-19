"""
Step 0 of docs/learning-semantic-kernel.md

Raw function/tool calling, using the plain `openai` client — no Semantic Kernel yet.
This is the exact loop that Semantic Kernel automates for you later (Tier 1, item 4).

Run: uv run python scripts/tool_calling_demo.py "your question here"
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import LLM_MODEL, OPENAI_API_KEY, validate
from app.engine import get_query_engine, init_query_engine
from app.resume_index_manual import build_index, semantic_search as run_semantic_search

# base_url picks WHICH provider we talk to, through the same OpenAI-shaped client:
#   - "https://openrouter.ai/api/v1"        -> OpenRouter (what we're using; routes to many providers)
#   - omit base_url entirely                 -> talks to OpenAI directly
#   - an Azure OpenAI resource's endpoint URL -> Azure OpenAI (needs api_version too)
# LLM_MODEL (from .env) picks WHICH model at that provider, e.g. "gpt-4o-mini",
# "gpt-4o", or (via OpenRouter) "anthropic/claude-3.5-sonnet".
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

# This is the "tool schema" — plain-English descriptions the MODEL reads to decide
# whether and how to call each function. Vague descriptions = wrong tool picked.
TOOLS = [
    {
        # "type": "function" marks this as a custom function WE wrote.
        # It's currently the only tool type for your own code. Some AI platforms
        # also offer their OWN built-in tool types you don't implement yourself,
        # e.g. a built-in "web_search" or "code_interpreter" tool — not used here.
        "type": "function",
        "function": {
            "name": "sql_search",
            "description": (
                "Search candidates using structured/filterable fields: salary, "
                "years of experience, location, position applied for. Use this "
                "for questions involving numbers, filters, or sorting."
            ),
            "parameters": {
                # "type": "object" = "this function takes a group of named arguments"
                # (like a dictionary). JSON Schema's other basic types, used for
                # individual fields below, are: "string", "number" (decimals),
                # "integer" (whole numbers), "boolean" (true/false), and "array" (a list).
                "type": "object",
                "properties": {
                    "question": {
                        # This one argument is text. If sql_search instead needed a
                        # number (e.g. a salary cap), we'd write "type": "number" here.
                        "type": "string",
                        "description": "A plain-English question about candidate data, e.g. 'Who has more than 3 years experience?'",
                    }
                },
                # Lists which of the properties above are mandatory. An argument left
                # out of this list would be optional (the model could skip it).
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Find candidates by matching the MEANING of their resume text to a "
                "described skill or experience. Use this for open-ended skill/experience "
                "questions, not numeric filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of the skills/experience to match, e.g. 'container orchestration and CI/CD'",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def call_sql_search(question: str) -> str:
    response = get_query_engine().query(question)
    return str(response)


def call_semantic_search(query: str) -> str:
    results = run_semantic_search(query, top_k=5)
    if not results:
        return "No matching candidates found."
    lines = [
        f"- {r['name']} ({r['position']}), match score {r['score']:.0%}: {r['snippet']}"
        for r in results
    ]
    return "\n".join(lines)


TOOL_IMPLEMENTATIONS = {
    "sql_search": call_sql_search,
    "semantic_search": call_semantic_search,
}


def ask(question: str) -> str:
    # "role" labels who said this message. The 4 roles you'll see:
    #   "system"    -> instructions/persona for the whole conversation (not used here,
    #                  but you could add one at the start, e.g. "You are a recruiting assistant.")
    #   "user"      -> what the human asked (this line)
    #   "assistant" -> what the model replied (added automatically below)
    #   "tool"      -> a real function's result, sent back to the model (added below)
    messages = [{"role": "user", "content": question}]

    # Round 1: send the question + tool schemas, let the model decide what to do.
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        # tool_choice decides how free the model is to use tools. We leave it out,
        # which defaults to "auto" (model decides whether to call a tool at all).
        # Other options: "none" (never call a tool, just answer directly), or
        # "required" (must call some tool), or naming one exact tool to force it.
    )
    # The API can return several alternative replies at once (controlled by an "n"
    # parameter we didn't set, so n=1 by default) — choices[0] takes the first/only one.
    message = response.choices[0].message

    if not message.tool_calls:
        # Model answered directly without needing a tool.
        return message.content

    # The model asked to call one or more tools. Run each one for real,
    # then feed the real results back so the model can write a final answer.
    messages.append(message)

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"  -> model requested tool: {name}({args})")

        result = TOOL_IMPLEMENTATIONS[name](**args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Round 2: model now has the real tool results, writes the final answer.
    final = client.chat.completions.create(model=LLM_MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    validate()
    init_query_engine()
    build_index()

    question = " ".join(sys.argv[1:]) or "Who has experience with container orchestration and CI/CD?"
    print(f"Question: {question}\n")
    print("Answer:")
    print(ask(question))
