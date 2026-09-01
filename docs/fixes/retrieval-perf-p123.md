# FIX — Retrieval perf & web_search toggle (P1 + P2 + P3)

**Author:** K (spec) → subagent (implement) → K (verify)
**Branch:** feature/langgraph-rewrite
**Guidelines:** karpathy-guidelines.md — think first, simplicity, surgical, goal-driven. No scope creep.

Diagnosed live via eca.log (request `f75557b7`, query "thời tiết hôm nay ở Hồ Chí Minh").
Backend runs on **:8000** (8080 = Owner's Spring, do NOT use). Docker PG(5433)/Redis/SearXNG up.

---

## P1 — Embedding loads offline (kill HF-Hub round-trips)

**Problem:** `SentenceTransformer(model_name)` re-validates the model against huggingface.co
on load (dozens of HTTP calls, some 404 / rate-limited / "unauthenticated"), even though
`intfloat/multilingual-e5-small` is already cached at `~/.cache/huggingface/hub/`. This makes
the **first `kb_search` after each restart** slow and internet-dependent.

**Load sites (both):**
- `agenticRAG/langgraph_agents/shared/embedding.py:45` — `SentenceTransformer(self.model_name)`
- `agenticRAG/langgraph_agents/core/resource_guard.py:142` — `SentenceTransformer(model_name)`

**Fix:** load offline-first. Add `local_files_only=True` at BOTH sites, gated by an env so a
fresh machine (no cache) can still download once:
```python
_offline = os.getenv("EMBEDDING_ALLOW_DOWNLOAD") != "1"   # default: offline (local cache only)
SentenceTransformer(model_name, local_files_only=_offline)
```
(Use `os` import already present, or add it.) Keep both sites consistent.

**Acceptance:**
- Backend starts; first `kb_search` produces **no** `huggingface.co` requests in eca.log.
- Existing embedding unit tests still pass.
- RUNBOOK note (§ environment): model must be pre-cached; set `EMBEDDING_ALLOW_DOWNLOAD=1`
  for the one-time download on a clean machine.

---

## P2 — Hard-cap retriever ⇄ tools loop at 2 rounds

**Problem:** "max 2 rounds" is only a line in the retriever **prompt**
(`retriever_agent.py:109`); the LLM ignored it and ran **3 rounds** (search_medical, then
kb_search ×2 — wasting ~25s). The only hard cap today is the graph-wide `recursion_limit`.

**Fix (hard cap in graph, don't trust the prompt):**
- Add a round counter to `AgentState` (find it under `langgraph_agents/` — likely `state.py`),
  e.g. `retriever_rounds: int` (default 0).
- Increment it in `retriever_agent_node` each time the node runs.
- In `route_after_retriever` (the conditional edge fn used at `graph.py:138`): if
  `retriever_rounds >= MAX_RETRIEVER_ROUNDS` (constant = 2), force route → `synthesizer`
  (never → `tools`), regardless of pending tool_calls. Below the cap, keep current logic.

**Acceptance:**
- A query that would loop now runs **at most 2** `retriever_agent` node executions before
  synthesizer (verify via eca.log `node_start`/`node_complete` count for one request_id).
- Grader retry path (`route_after_grader` → retriever_agent) still works; the counter must not
  permanently wedge a legitimate grader-triggered retry — decide whether the counter resets on
  grader retry or is a per-turn hard ceiling; document the choice in the worklog. (Simplest:
  hard per-turn ceiling covering retries too.)
- Existing graph/routing tests pass.

---

## P3 — Enforce web_search toggle at ALL layers (D27 actually honored)

**Problem (confirmed, not cosmetic — cost + privacy):** turning web search OFF does NOT stop
web searches. `_build_tools(False)` correctly removes `search_medical` from `bind_tools`
(verified), BUT:
1. `graph.py:111` — `ToolNode(all_tools)` is built **once at compile** with `all_tools`
   INCLUDING `search_medical` (only `generate_motion` filtered). It ignores the per-request
   toggle → if the LLM emits a `search_medical` call, the ToolNode **executes it**.
2. `_RETRIEVER_SYSTEM_PROMPT` (`retriever_agent.py:~78, ~92`) **always** advertises
   `search_medical` + decision-rule "real-time → search_medical", so the LLM still calls it
   even when it's absent from `bind_tools`.

Net: log showed `web_search: false` for all 3 rounds yet round 1 ran `search_medical`.

**Fix (defense-in-depth, mirrors how ChatGPT/Gemini enforce "off"):**
1. **Conditional prompt** — build `_RETRIEVER_SYSTEM_PROMPT` so the `search_medical` tool
   description AND decision-rule line are included **only when `web_search_enabled`**. When
   off, the LLM is never told web search exists. (Template it; don't leave a static string
   that always lists it.)
2. **Execution guard** — even if the LLM still emits a `search_medical` call, it must NOT run
   when `web_search=false`. Since `ToolNode` is compile-time static, wrap the "tools" node in a
   small custom async node that reads `config["configurable"].get("web_search", ...)`:
   - if web search is OFF and the last AIMessage has a `search_medical` tool_call, short-circuit
     it with a `ToolMessage` like `{"blocked": "web_search_disabled"}` (so the graph keeps
     flowing) instead of executing; delegate all other tool_calls to the underlying `ToolNode`.
   - if ON, delegate everything to `ToolNode` unchanged.
3. **Default:** confirm the intended default. Product intent = search only when user opts in →
   default should be **false**. `schemas.py:12` already `web_search: bool = False`; make the
   retriever's `config...get("web_search", <default>)` consistent (currently defaults `True` at
   `retriever_agent.py:136` — align to the product intent; if defaulting False, ensure planner
   still allows retrieval for KB — kb_search is unaffected by the web toggle).

**Acceptance:**
- With `web_search=false`: no `search_medical` in `tool_names`, no SearXNG (:6666) call in
  eca.log for that request. Verify with a weather query (which previously triggered it).
- With `web_search=true`: `search_medical` available and used for a real-time query; SearXNG
  hit; results reach synthesizer.
- kb_search / memory_search / youtube_transcript unaffected by the toggle.

---

## Cross-cutting

- **Tests:** add/adjust unit tests: (P2) round-cap forces synthesizer after 2; (P3) prompt omits
  search_medical when off + guard blocks execution when off. Run full suite
  `python -m pytest tests/langgraph_agents/ -q` — must stay green (no regressions).
- **Worklog:** append to `docs/worklogs/DD-MM-YYYY.md` (today), note "K spec + subagent impl".
- **Do NOT:** touch synthesizer/grader logic, frontend, Docker, ports, or the planner LLM
  latency (that's DeepSeek server-side, separate). Keep diffs surgical.
