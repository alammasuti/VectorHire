# Learning Path: Semantic Kernel (SK) via vector-hire

Study order. Each step builds on the previous one. Check items off as we build and verify them in code.

Stack used throughout: OpenRouter (existing `OPENAI_API_KEY`/base_url) + your existing Chroma-based `semantic_search()` and LlamaIndex-based `sql_search()`. No Azure needed for any of this.

---

## Step 0 — Prerequisite concept: Function/Tool Calling (do this BEFORE touching SK)

**Goal:** understand the raw mechanic SK automates, using the plain `openai` client — no framework.

- [x] What it is: instead of the model only returning text, it can return "please call function X with these arguments"
- [x] The loop: you send the user's question + a list of tool schemas → model replies with a tool call (or a direct answer) → you run the real Python function → you send the result back to the model → model gives the final answer
- [x] Tool schema = name + description + parameters (JSON schema) — the description is what the model reads to decide *whether* and *how* to call it
- [x] Build it raw: `scripts/tool_calling_demo.py` — gives the model `sql_search` and `semantic_search` as tools, manually implements the above loop with the `openai` client
- [x] Verify: ask a SQL-shaped question → confirm the model picks `sql_search`; ask a skills-shaped question → confirm it picks `semantic_search`
  - Tested "Show candidates expecting less than 80000 salary" → picked `sql_search`, generated correct SQL, correct results
  - Tested "Who has experience with container orchestration and CI/CD?" → picked `semantic_search`, correctly surfaced Karan Joshi (DevOps/Docker/Kubernetes/Jenkins)

**Why first:** once this loop is understood, Semantic Kernel's "automatic function calling" will look like "oh, this is just that loop, automated" instead of magic.

---

## Step 1 — Semantic Kernel Tier 1 (Foundations)

### 1a. Kernel
- [x] What it is: the central object holding your AI connection + registered plugins — "the orchestrator"
- [x] Create a `Kernel` instance in the project — `scripts/sk_kernel_demo.py`

### 1b. Connectors
- [x] What it is: how the Kernel talks to an actual AI service (OpenAI-compatible endpoint)
- [x] Register an `OpenAIChatCompletion` connector pointed at OpenRouter (via a pre-built `AsyncOpenAI` client passed as `async_client`, same key/model pattern already used in `resume_index_manual.py` / `engine.py`)
- [x] Confirm: a basic prompt through the Kernel gets a response (no tools yet)
  - Tested "In one sentence, what is a vector embedding?" → got a correct one-sentence answer, no tools involved

### 1c. Plugins / Kernel Functions
- [x] What it is: a class grouping related functions, each marked with `@kernel_function` + a clear description, so the Kernel/model can see and call them
- [x] Wrap existing `sql_search` and `semantic_search` into a `CandidateSearchPlugin` class — `scripts/sk_plugin_demo.py`
- [x] Register the plugin: `kernel.add_plugin(CandidateSearchPlugin(), plugin_name="CandidateSearch")`

### 1d. Automatic Function Calling
- [x] What it is: SK runs the "ask model → get requested function → run it → send result back" loop for you automatically (`FunctionChoiceBehavior.Auto()`)
- [x] Build `/sk/ask` endpoint: one question in, Kernel decides which plugin function(s) to call, returns final answer — `app/sk_agent.py` (Kernel/connector/plugin built once at startup, in `api.py`'s `lifespan`) + `app/routes/sk.py` (POST, Pydantic request/response models, matches the rest of the app's style)
- [x] Verify with 2 test questions (same ones used in Step 0, for direct comparison):
  - [x] A clearly structured/filter question ("Show candidates expecting less than 80000 salary") → called `sql_search`, same SQL + same 8 candidates as the raw version
  - [x] A clearly skills/experience question ("Who has experience with container orchestration and CI/CD?") → called `semantic_search`, same top match (Karan Joshi) as the raw version
  - [x] A mixed question (may call both) — "Find candidates expecting less than 80000 salary who also have container orchestration and CI/CD experience" → called BOTH `sql_search` and `semantic_search` on its own (confirmed via temporary `print()` logging added to each plugin function), then combined both results into one answer. Noted: its final "meets both criteria" reasoning was shaky (picked low-scoring semantic matches) — a good concrete example of why evals matter, not just a one-off bug
- [x] Accept: routing is model-decided, not hardcoded — results can vary slightly run to run (this is expected, not a bug)

**Milestone — DONE:** `/sk/ask` is a real, working endpoint in the running app. Verified live via `curl` with the mixed salary+CI/CD question — correctly called both `sql_search` and `semantic_search`, returned a well-formed `{"answer": "..."}` response.

---

## Step 2 — Tier 2 (Production-shaped concepts)

### 2a. Prompt templates
- [x] What it is: moving prompts out of hardcoded Python strings into reusable template files (`.yaml`/`.txt`) with placeholders like `{{$input}}`
- [x] Convert the agent's instructions into a template file — `app/prompts/candidate_assistant.txt`, loaded in `app/sk_agent.py` and registered as a Kernel function (`CandidateAssistant.answer`) with `{{$question}}` as the placeholder
  - Verified: answers now follow the template (short bullets, no invented candidates, under 150 words), and the endpoint still calls both tools correctly
  - Learning point: edit the `.txt` file to change the assistant's behavior, without touching Python code (the server must be restarted to pick up the change)
  - Note: this SK version has no special chat-message syntax in templates, so the instructions are sent inside the user message. A separate system message comes with chat history (2b)

### 2b. Chat history / context
- [x] What it is: tracking a multi-turn conversation so the agent remembers earlier turns, not just one-shot Q&A
- [x] Add a `ChatHistory` object so `/sk/ask` can handle a follow-up like "now filter those by salary" — `app/sk_agent.py` keeps one history per `conversation_id`; `app/routes/sk.py` accepts an optional `conversation_id` and returns it
  - Instructions moved from a prompt function to the history's system message; the `{{$question}}` placeholder was removed because the question now arrives as a chat message
  - Verified over HTTP: turn 2 with the same `conversation_id` answered about the candidates from turn 1
  - Learning point: the search tools cannot see the conversation. A follow-up only worked after the instructions told the model to put the earlier candidates' names into the tool query. Without that, the SQL tool searched everyone
  - Limitations: histories live in memory (lost on restart) and grow without limit. Answers also vary between runs, which is what evals will measure

### 2c. Filters (SK's hook/middleware system)
- [ ] What it is: intercepting a function call before/after execution — e.g. logging every tool call, or blocking a specific tool
- [ ] Add a simple logging filter that prints which function was called and with what arguments

---

## Step 3 — Tier 3 (Advanced / multi-step) — optional, revisit later

### 3a. Planners
- [ ] What it is: given a goal, auto-chain multiple functions into a multi-step plan (older SK concept), instead of one single function-calling round
- [ ] Example to try: "find backend candidates under 80k, then summarize their top skills" (requires chaining SQL filter → semantic summary)

### 3b. Memory / Vector store connectors
- [ ] What it is: SK's own abstraction for embeddings + vector search — compare against the hand-rolled Chroma code in `resume_index.py`
- [ ] Not required to adopt — useful mainly as a point of comparison to deepen understanding of what you already built manually

### 3c. Agents & multi-agent orchestration (SK's "Agent Framework")
- [ ] What it is: multiple named agents with distinct roles collaborating/handing off work (e.g., a "Recruiter agent" + a "Scheduler agent")
- [ ] Not needed for this project's current scope — revisit only if a multi-role workflow becomes a real need
- [ ] Concept to learn: today we have ONE agent (the model) with TWO tools (`sql_search`, `semantic_search`). Tools are plain functions, not agents. A "third agent" that uses those tools would be a sub-agent (e.g. a "SQL specialist" agent and a "resume specialist" agent, each owning its tools) coordinated by a parent agent. Check the exact SK agent classes in the installed version before coding.

---

## Explicitly out of scope for now

- **MCP (Model Context Protocol):** a way to expose tools across different AI apps/frameworks via a standard protocol. Not needed here — your tools live in the same Python process as the Kernel, so there's no cross-app/cross-process reuse need. Revisit only if you want another AI app (e.g. Claude Desktop) to call your resume-search tool too.
- **Azure OpenAI / Azure AI Search:** SK's connectors are provider-agnostic — OpenRouter + Chroma work fine for all of Tier 1–3. Azure is an infrastructure/ops decision for later, not a requirement for learning SK concepts.

---

## Progress log

| Date | Step completed | Notes |
|---|---|---|
| 2026-10-17 | Step 0 — raw function/tool calling | `scripts/tool_calling_demo.py`, verified both tools route correctly |
| 2026-10-17 | Step 1a/1b — Kernel + Connector | `scripts/sk_kernel_demo.py`, `uv add semantic-kernel`, verified a bare prompt through the Kernel works via OpenRouter |
| 2026-10-18 | Step 1c/1d — Plugin + Automatic Function Calling | `scripts/sk_plugin_demo.py`, results matched Step 0's raw version exactly on both test questions |
| 2026-10-18 | Step 1 complete — `/sk/ask` FastAPI endpoint | `app/sk_agent.py` + `app/routes/sk.py`, verified live via curl with a mixed question (both tools called, correct combined answer) |
| 2026-10-18 | Step 2a — prompt templates | `app/prompts/candidate_assistant.txt` loaded by `app/sk_agent.py`, verified answers follow the template's rules |
| 2026-10-18 | Step 2b — chat history | `ChatHistory` per `conversation_id` in `app/sk_agent.py`, verified a follow-up over HTTP; added a rule so tool queries carry earlier candidate names |
