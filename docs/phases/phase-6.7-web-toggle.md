# PHASE 6.7 — Web Search Toggle (UI opt-in)

> Architect: K | Developer: N | Date: 2026-05-25
> Branch: `feature/langgraph-rewrite`
> Scope: **~30 LOC, ~30 min N time.** UI toggle + retriever conditional tool binding.

---

## Why

Academic project advisory scope. Default behavior should be fast (no SearXNG call) using
pgvector + DeepSeek training knowledge. User opts in for web search when they want
external evidence. Disclaimer footer appended to every response.

---

## Decisions (Owner confirmed)

| | Choice |
|---|---|
| Default toggle state | **OFF** — user opts in |
| Disclaimer location | **UI append** (frontend only, not backend) |
| Persistence | **Session-only** — no localStorage, resets on tab close |
| Visual cue | **🔍 icon** on message bubble when web search was used |
| Architecture | Retriever drops MCP web tool from bindings when toggle off — planner untouched |

---

## Tasks

### Task 1 — Backend schema + config passthrough

**Files**: `api/schemas.py`, `api/main.py`

**`api/schemas.py`** — add field to `ChatRequest`:

```python
class ChatRequest(BaseModel):
    # ...existing fields...
    web_search: bool = False    # NEW — UI opt-in for SearXNG
```

**`api/main.py`** — pass into RunnableConfig:

```python
config = {"configurable": {
    # ...existing keys...
    "web_search": req.web_search,    # NEW
}}
```

**Done when**:
```powershell
# Backend running
'{"query":"Xin chao","web_search":true}' | curl -s -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '@-' | head
# Should NOT 422 (Pydantic accepts the field). Should stream SSE normally.
```

---

### Task 2 — Retriever conditional tool binding

**File**: `agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retriever_agent.py`

**Find** the place where tools are bound to LLM (likely `bind_tools(tools)`). Filter out
MCP web search tool when `web_search` is False.

```python
async def retriever_agent_node(state, config):
    web_search_enabled = config["configurable"].get("web_search", False)
    
    all_tools = await get_all_tools()    # pgvector + MCP (kimodo + web_search)
    if not web_search_enabled:
        # Strip MCP web search tool — keep pgvector + kimodo
        tools = [t for t in all_tools if t.name != "search_medical"]
        logger.info("web_search_disabled", extra={"node": "retriever_agent"})
    else:
        tools = all_tools
        logger.info("web_search_enabled", extra={"node": "retriever_agent"})
    
    # ...rest of existing logic with `tools` instead of `all_tools`...
```

**Important**: planner stays untouched. It may still suggest `search_strategy=[..., "web_search_if_low_quality"]` —
retriever simply doesn't have the tool available, so LLM can't call it. Graceful.

**Done when**:
- With `web_search=false`: `docker logs vva-searxng --tail 5` shows NO new requests after chat call
- With `web_search=true`: SearXNG logs show GET /search request
- Both cases return non-empty `final_answer`

---

### Task 3 — Frontend toggle UI

**File**: `ECA_UI/index.html`

Add toggle in chat input bar, near persona selector. Use plain checkbox or switch:

```html
<div class="chat-controls">
  <select id="persona-select">...</select>
  <label class="toggle">
    <input type="checkbox" id="web-search-toggle">
    <span>🔍 Web search</span>
  </label>
</div>
```

Style as inline toggle (match existing UI). Default unchecked.

**State**: in-memory variable in the React/JS component. **Do NOT** localStorage.

```javascript
let webSearchEnabled = false;
document.getElementById('web-search-toggle').addEventListener('change', (e) => {
  webSearchEnabled = e.target.checked;
});
```

**Done when**: toggle clickable, state reflected in JS variable.

---

### Task 4 — Pass toggle to backend

**File**: `ECA_UI/api.js`

Update `streamChat` signature to accept + send `web_search`:

```javascript
async function streamChat({ query, userId, sessionId, personaId, outputMode, webSearch, onEvent }) {
  // ...existing fetch setup...
  body: JSON.stringify({
    query,
    user_id: userId,
    session_id: sessionId,
    persona_id: personaId || "eca_default",
    output_mode: outputMode || "text",
    web_search: webSearch || false,    // NEW
  }),
  // ...rest unchanged...
}
```

In `index.html` where `streamChat` is called, pass `webSearch: webSearchEnabled`.

**Done when**: browser DevTools Network tab shows `web_search: true` in request body when toggle on.

---

### Task 5 — Disclaimer footer + 🔍 visual cue

**File**: `ECA_UI/index.html` (message render logic)

When SSE `done` event fires, append disclaimer to the message:

```javascript
function onDoneEvent(messageElement, requestUsedWebSearch) {
  const disclaimer = document.createElement('div');
  disclaimer.className = 'message-disclaimer';
  disclaimer.innerHTML = '<small><em>*Câu trả lời chỉ mang tính chất tham khảo.*</em></small>';
  messageElement.appendChild(disclaimer);
  
  if (requestUsedWebSearch) {
    const icon = document.createElement('span');
    icon.className = 'web-search-badge';
    icon.title = 'Trả lời có dùng web search';
    icon.textContent = '🔍';
    messageElement.querySelector('.message-header').appendChild(icon);
  }
}
```

Style `.message-disclaimer` muted gray small. Style `.web-search-badge` as inline icon
in message header (next to timestamp or persona name).

**Done when**:
- Every response has disclaimer line at bottom (gray small italic)
- Messages where toggle was ON have 🔍 icon in header; OFF messages no icon

---

## Acceptance gate

- [ ] Backend smoke: 5 chat requests (toggle off) → SearXNG logs show 0 new requests
- [ ] Backend smoke: 5 chat requests (toggle on) → SearXNG logs show 5 new requests
- [ ] UI smoke: toggle off + send greeting → reply appears + disclaimer footer + NO 🔍
- [ ] UI smoke: toggle on + send "đau lưng dưới triệu chứng gì" → reply + disclaimer + 🔍 in header
- [ ] Session-only persistence: refresh page → toggle resets to OFF (no localStorage)
- [ ] Existing unit tests pass: `pytest tests/langgraph_agents/ -m unit` (102/102)
- [ ] 1 new test: `test_retriever_drops_web_tool_when_disabled` — mock config with `web_search=False`,
      verify `search_medical` not in bound tools list

---

## Commits

| # | Commit | Files |
|---|--------|-------|
| 1 | `feat(api): web_search field in ChatRequest + config passthrough` | schemas.py, main.py |
| 2 | `feat(retriever): conditional tool binding based on web_search flag` | retriever_agent.py + 1 test |
| 3 | `feat(ui): web search toggle + disclaimer footer + 🔍 badge` | index.html, api.js |

3 commits sequential. Smoke output paste into worklog per commit.

---

## Out of scope

- Persistent toggle preference (localStorage) — defer if users complain about resetting
- Toggle for pgvector / Kimodo independently — single toggle covers web search only
- Disclaimer text translation EN/VN — keep VN only for academic scope
- Visual indicator while web search is in-flight (spinner) — existing stage SSE events suffice
- Cost meter ("Web search used X queries") — defer until billing matters
