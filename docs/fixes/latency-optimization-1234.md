# FIX — Latency optimization #1-4 (benchmark-driven)

**Author:** K (spec) → subagent (implement) → K (verify)
**Branch:** feature/langgraph-rewrite
**Guidelines:** karpathy-guidelines.md — think first, simplicity, surgical, goal-driven.

## Context — real benchmark (29 requests, current backend)

| node | p50 | p90 | max |
|---|---|---|---|
| memory | 32ms | 58ms | 12186ms (cold embed load, 1×) |
| planner | 3582ms | **24438ms** | 37757ms |
| retriever/round | 1938ms | 27682ms | 55326ms |
| synthesizer | 6952ms | **30146ms** | 55438ms |
| grader | 0ms | 1ms | 2ms |

100% of cost is LLM (DeepSeek) calls. The tail (p90/max) is DeepSeek server-side latency
variance, confirmed earlier by two back-to-back calls to the SAME model (Flash) taking
21s then 0.3s. Retriever waste observed: `[kb_search, kb_search]` (duplicate same-tool
calls in one round) and `[]` (empty-tool rounds, ~1 wasted LLM call).

Explicitly OUT of scope (Owner decision): lazy embedding load stays as-is (has a purpose,
not being removed). Do not touch `shared/embedding.py` lazy-load behavior.

---

## #1 — Prompt-cache observability (verify M.7 is actually paying off)

**Finding:** `_PLANNER_SYSTEM_PROMPT` (`nodes/planner.py`) is already a 100% static string
(no f-string interpolation) — optimal for DeepSeek's automatic prefix caching. DeepSeek's
OpenAI-compatible API returns `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` in the
response `usage` object, but **nothing logs it today** — so there is currently no way to
confirm caching is working or measure its savings.

**Fix:** surface these fields in `node_complete` logs for `planner`, `retriever_agent`,
`synthesizer` (all three call `get_chat_model(...)`).
- After each `ainvoke`/`astream` call, read cache fields from the response. With
  `langchain_openai.ChatOpenAI`, check `ai_msg.response_metadata.get("token_usage", {})`
  for `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` (DeepSeek extension fields;
  they may be absent for streaming chunks — check the final aggregated chunk / use
  `ai_msg.usage_metadata` too, whichever LangChain populates for streaming). If absent,
  log `null` — don't fail the request over missing telemetry.
- Add `cache_hit_tokens` / `cache_miss_tokens` keys next to the existing `tokens` field in
  each node's `node_complete` log line.
- Do NOT change prompt structure/order in planner (already optimal). For synthesizer,
  leave `tool_results`/`resolved_query` interpolation as-is — reordering would only help
  if evidence were stable across requests, which it isn't; not worth the readability cost.

**Acceptance:** fire a chat request against the live backend (Docker + real DeepSeek key
running), tail `eca.log`, confirm `node_complete` for planner/retriever/synthesizer include
`cache_hit_tokens`/`cache_miss_tokens` (values may be `null` if DeepSeek doesn't return them
for this call — that's an acceptable, informative outcome, not a bug).

---

## #2 — Timeout + retry + Gemini fallback for DeepSeek LLM calls

**Finding:** `ChatOpenAI(...)` in `llm.py:get_chat_model()` has no `timeout` and no
`max_retries` — a slow-but-non-erroring DeepSeek call can hang far longer than the observed
55s tail (bounded only by the underlying httpx/OS default, effectively unbounded from the
app's perspective). `GEMINI_API_KEYS` is already present in `.env` and
`langchain_google_genai` is already installed (leftover from the pre-LangGraph
architecture) but wired to nothing.

**Fix (`llm.py`):**
1. Add `timeout=` to the `ChatOpenAI(...)` construction in `get_chat_model()`:
   - fast roles (planner, retriever, conversation): `timeout=20.0`
   - heavy role (synthesizer): `timeout=35.0`
   (Use the existing `_HEAVY_ROLES` set / `_model_for_role` pattern to pick the value —
   don't hardcode per-call-site, keep it centralized in `get_chat_model`.)
2. Add `max_retries=1` (langchain-openai's built-in retry — cheap, handles transient
   network blips; do NOT set this high, a slow-but-succeeding call should not be retried
   3x and make things worse).
3. Add `get_fallback_chat_model(role: str)` — builds a `ChatGoogleGenerativeAI` (model
   `gemini-2.0-flash` for fast roles, `gemini-2.5-pro` or similar available flash/pro model
   for heavy — check what's reasonable/available) reading the FIRST key from
   `GEMINI_API_KEYS` (comma-separated). Return `None` if `GEMINI_API_KEYS` is unset/empty
   (graceful — fallback simply unavailable, not a startup error). Cache with
   `@lru_cache` like `get_chat_model`.

**Fix (call sites — `planner.py`, `synthesizer.py`):** both already wrap their primary LLM
call in `try/except Exception`. On exception (this now includes `TimeoutError` /
`APITimeoutError` from the bounded timeout above), BEFORE falling through to the existing
safe-fallback/error-message logic:
- Attempt ONE call to `get_fallback_chat_model(role)` if it returns non-`None`.
- If the fallback call succeeds, use its result and continue normally (log
  `llm_fallback_used: true` in `node_complete`/a dedicated log line with `request_id`).
- If the fallback also raises, or is unavailable, fall through to the EXISTING error
  handling (don't change that path — it's already correct: RECOVERABLE error + safe
  default for planner, CRITICAL error + Vietnamese fallback message for synthesizer).
- For synthesizer specifically: the primary path streams via `llm.astream(msgs)` +
  `writer(...)`. The fallback call does not need to stream — a single `ainvoke` is fine;
  emit the whole fallback answer as one `writer({"content": final})` chunk so the SSE
  contract (`token` events) is preserved for the frontend.

**Acceptance:** unit-test the fallback wiring with a mocked primary model that raises
`TimeoutError` and a mocked fallback that returns a canned response — assert the node
returns the fallback's content and logs `llm_fallback_used`. Also unit-test: primary
succeeds → fallback is never invoked (no wasted call). Do NOT make a real Gemini API call
in tests — mock `get_fallback_chat_model`.

---

## #3 — Bound synthesizer output size

**Finding:** No `max_tokens` is set anywhere in `llm.py`; synthesizer instructs "under 500
words" in the prompt but nothing enforces it, and synthesizer is the most expensive node
(p50 6952ms, up to 55438ms) partly from unbounded generation length.

**Fix:**
1. In `llm.py get_chat_model()`, add `max_tokens=` to the `ChatOpenAI` construction:
   heavy role (synthesizer) → `1024`; fast roles → `512` (plenty for planner's JSON output
   / retriever's tool-call decision — these rarely generate long prose anyway, but bound
   them too as a safety net).
2. Tighten the synthesizer prompt instruction from "Keep under 500 words unless the topic
   requires detail" to a firmer "Keep under 350 words. Do not pad or repeat safety
   disclaimers — state each once." — the 500-word ceiling was loose enough for
   ~1024-token cutoff to sometimes truncate mid-sentence for longer clinical answers with
   citations; 350 words leaves headroom.

**Acceptance:** fire a synthesize-mode request (e.g. an exercise-recommendation query)
against the live backend; confirm `output_chars` in the synthesizer `node_complete` log
is well under the old baseline for similar queries and the response is NOT truncated
mid-sentence (read the actual streamed text). Run existing synthesizer unit tests — must
still pass (they test mode derivation / safety prefix logic, not token limits, so should
be unaffected — but verify).

---

## #4 — Fix retriever tool-call waste (duplicate same-tool calls in one round)

**Finding:** benchmark shows retriever rounds like `["kb_search", "kb_search"]` (8×) and
`["kb_search", "kb_search", "kb_search"]` (1×) — the LLM emits multiple tool_calls to the
SAME tool in one round (often with similar/identical queries), each executed by `ToolNode`,
wasting execution time + tokens for redundant results.

**Fix (`retriever_agent.py`, in `retriever_agent_node`, right after `tool_calls =
getattr(ai_msg, "tool_calls", []) or []`):**
- Deduplicate `tool_calls` by `(name, frozenset(args.items()))` — if two tool_calls have
  the exact same tool name AND identical arguments, keep only the first; if names match
  but *args differ* (e.g. two different `kb_search` queries), keep both — that's a
  legitimate parallel search, not waste.
- After deduping, if the count changed, mutate `ai_msg.tool_calls` to the deduped list
  (check whether `AIMessage.tool_calls` allows direct assignment — langchain's `AIMessage`
  is a pydantic model; if frozen/validated, use `ai_msg.tool_calls = deduped` normally
  first, and only reach for `object.__setattr__` if that raises) BEFORE returning
  `{"messages": [ai_msg], ...}`, so `ToolNode` (downstream) only executes the deduped set.
  Log `tool_calls_deduped: <removed_count>` in `node_complete` when >0.
- Do NOT touch the "empty tool_calls round" behavior — `route_after_retriever` already
  routes straight to synthesizer when there are no pending tool_calls (P2 logic), so an
  empty round already costs exactly one bounded LLM call and no wasted tool execution;
  that's an acceptable "the model decided it's done" signal, not a bug to fix here.

**Acceptance:** unit test with a mocked `ai_msg.tool_calls` containing two identical
`kb_search(query="X")` calls + one distinct `kb_search(query="Y")` — assert the returned
message's `tool_calls` has 2 entries (X once, Y once), not 3. Existing retriever tests
must still pass.

---

## Cross-cutting

- Run full unit suite `python -m pytest tests/langgraph_agents/ -m unit -q` — must stay
  green, no regressions.
- Add new tests alongside existing test files (`test_planner*.py`, `test_synthesizer*.py`,
  `test_retriever*.py`, or a new `test_fix_latency_1234.py` if that's cleaner — your call).
- Append a worklog entry to `docs/worklogs/DD-MM-YYYY.md` (today) summarizing the 4 changes.
- Do NOT touch: grader, memory node, graph.py routing (beyond what #4 requires — none),
  frontend, embedding lazy-load (#5, explicitly deferred by Owner), Docker/ports.
- Backend runs on **:8000** (never 8080 — reserved). Conda env `firstconda`. Python path
  root is `agenticRAG` (`import langgraph_agents...`).
