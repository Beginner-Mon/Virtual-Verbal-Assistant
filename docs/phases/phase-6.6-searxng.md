# PHASE 6.6 — Replace DDG with self-hosted SearXNG

> Architect: K | Developer: N | Date: 2026-05-25
> Branch: `feature/langgraph-rewrite` (or sub-branch off it)
> Scope: **1 file rewrite + 1 docker service + settings + tests + docs**. ~1h N time.
> Process: per-commit smoke-test paste into worklog before moving on.

---

## Why

Gemini Grounding is paid in production (Google AI Studio free tier ≠ production tier).
DDG (`ddgs` lib) works but quality < SearXNG (SearXNG aggregates Google + Bing + DDG + Wikipedia,
returns merged ranked results). Owner prefers self-hosted to avoid API costs + lock-in.

SearXNG: open-source metasearch, Docker official image, no API key, JSON API mode.

---

## Decisions (already confirmed by Owner)

| | Choice |
|---|---|
| Port | **6666** (no conflict with backend 8080, frontend 3000, PG 5432, Redis 6379) |
| Cache via Redis | **No** — keep simple, defer until traffic data justifies |
| Engines | **SearXNG default** (Google, Bing, DDG, Wikipedia) — tune later if quality issues |
| Tool output shape | **Keep as-is**: `list[{title, snippet, url, source_domain}]` — minimal disruption to retriever/synthesizer |
| Auto-start | Add to `docker-compose.langgraph.yml` so `docker compose up -d` brings it up with PG + Redis |

---

## Tasks

### Task 1 — Docker service + settings file

**Files**:
- Edit `docker-compose.langgraph.yml`
- Create `config/searxng/settings.yml`
- Create `config/searxng/.gitignore` (ignore generated `uwsgi.ini`, `limiter.toml` etc.)

**Compose service**:

```yaml
  searxng:
    image: searxng/searxng:latest
    container_name: eca-searxng
    ports:
      - "6666:8080"           # internal SearXNG always on 8080
    volumes:
      - ./config/searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:6666/
      - INSTANCE_NAME=eca-search
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
```

**`config/searxng/settings.yml`** (~20 lines) — uses env var for secret to keep file safe to commit:

```yaml
use_default_settings: true

server:
  bind_address: "0.0.0.0"     # explicit (image default, but documents intent)
  secret_key: "${SEARXNG_SECRET_KEY}"   # read from env, set in docker-compose
  limiter: false              # disable rate limit for self-host local dev
  image_proxy: false          # we use JSON API only, no HTML rendering

search:
  safe_search: 0              # 0=off (medical/anatomy content not filtered)
  formats:
    - html
    - json                    # CRITICAL: default off — must enable for our MCP server

ui:
  default_locale: ""
  query_in_title: false
```

**Pass secret via docker-compose** (env-driven, no hardcoded keys in git):

```yaml
  searxng:
    # ...other fields above...
    environment:
      - SEARXNG_BASE_URL=http://localhost:6666/
      - INSTANCE_NAME=eca-search
      - SEARXNG_SECRET_KEY=${SEARXNG_SECRET_KEY:?must be set — see RUNBOOK}
```

The `:?` syntax fails `docker compose up` early if env not set — better than SearXNG starting
with empty key and failing later.

**Generate secret_key** (cross-platform, Python only):
```bash
python -c "import secrets; print(secrets.token_hex(32))"
# → 64-char hex string, ~32 bytes entropy
```

> ⚠️ **Đổi 10/08/2026.** Đoạn dưới từng bảo đặt biến này vào `.env` ở **gốc repo**.
> File đó đã xoá — cả một file chỉ để giữ một biến, tức thêm một thứ phải trao tay
> cho member mới và thêm một thứ để quên.

Đặt vào **`agenticRAG/.env`**, chung với phần còn lại:
```bash
# agenticRAG/.env
SEARXNG_SECRET_KEY=<paste_64_char_hex_here>
```

`docker-compose.langgraph.yml` đọc file đó qua `env_file` ở service `searxng` —
**không** phải qua interpolation `${...}`, vì interpolation chỉ đọc `.env` ở thư
mục chứa compose file hoặc shell, nên trỏ sang chỗ khác sẽ bắt mọi lệnh
`docker compose` phải kèm `--env-file`.

**Reference**: official SearXNG container template at
`github.com/searxng/searxng/tree/master/container` uses a separate compose file.
We inline the service in our existing `docker-compose.langgraph.yml` so one
`docker compose up -d` brings up the full stack (PG + Redis + SearXNG together).

**Done when**:
```powershell
docker compose -f docker-compose.langgraph.yml up -d
docker ps --format "{{.Names}}: {{.Status}}"            # eca-searxng shows Up
curl "http://localhost:6666/search?q=physical+therapy&format=json" | jq '.results[0]'
# Expected: JSON object with title, url, content fields. If 403 → settings.yml not picked up.
```

Paste this output into worklog.

---

### Task 2 — Rewrite MCP server backend

**File**: `agenticRAG/agentic_rag_gemini/langgraph_agents/mcp/web_search_server.py`

**Current**: uses `ddgs.DDGS().text()` synchronously.
**New**: uses `httpx.AsyncClient` to query SearXNG.

**Spec**:

```python
import os
import httpx

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:6666")
DEFAULT_TIMEOUT = 10.0


async def _search_searxng(query: str, max_results: int = 3) -> list[dict]:
    """Query self-hosted SearXNG, return normalized result list.

    Output shape preserved from DDG version so retriever/synthesizer
    code stays unchanged.
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general,science",
            },
        )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
            "source_domain": r.get("engine", "unknown"),
        }
        for r in data.get("results", [])[:max_results]
    ]
```

Replace the existing DDG-based search function with this. Keep the MCP server boilerplate
(tool name `search_medical`, schema, server registration) unchanged.

**Error handling**: if SearXNG unreachable / 5xx → return `[]` + log structured warning
(`search_failed` event). Graceful — retriever continues, synthesizer falls back to
pgvector-only or empty-context prompt.

**Done when**:
```powershell
# With SearXNG container up + backend NOT running
python -c "import asyncio; from langgraph_agents.mcp.web_search_server import _search_searxng; print(asyncio.run(_search_searxng('lower back pain exercises', 3)))"
# Expected: list of 3 dicts with title/snippet/url/source_domain.
```

Paste output into worklog.

---

### Task 3 — Pass SEARXNG_URL into MCP subprocess

**File**: `agenticRAG/agentic_rag_gemini/langgraph_agents/mcp/client.py`

**Change**: add `SEARXNG_URL` to `_ENV_PASSTHROUGH` tuple so when MCP subprocess spawns,
it inherits the env var.

```python
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "SYSTEMROOT", "APPDATA",
    "LOCALAPPDATA", "HF_TOKEN",
    "SEARXNG_URL",            # ← ADD
)
```

**Done when**: existing MCP unit tests still pass (`pytest tests/langgraph_agents/test_phase3_mcp_web_search.py -v`).

---

### Task 4 — Update requirements

**File**: `requirements-langgraph.txt`

Remove `ddgs>=9.0.0`. Add nothing — `httpx` already pinned.

**Done when**: `pip install -r requirements-langgraph.txt` runs clean in a fresh conda env.
(N can verify in existing env: `pip uninstall ddgs -y` then verify import error in old code path,
then re-`pip install -r requirements-langgraph.txt` confirms no re-install.)

---

### Task 5 — Update tests

**File**: `tests/langgraph_agents/test_phase3_mcp_web_search.py`

Replace DDG mocks with httpx mocks. Use `respx` (httpx mock lib) OR plain `unittest.mock.patch`
on the `_search_searxng` call inside the MCP handler.

**Minimum 2 tests to update/add**:

1. `test_web_search_returns_results` — patch SearXNG response with 3 fake hits, verify tool
   returns properly shaped list. Use `respx.mock` or `httpx.MockTransport`.

2. `test_web_search_handles_searxng_down` (NEW) — patch httpx to raise `httpx.ConnectError`,
   verify tool returns `[]` and logs `search_failed`.

**Existing**: `test_web_search_list_tools` should still pass unchanged — MCP tool registration
is independent of backend.

**Done when**:
```powershell
pytest tests/langgraph_agents/test_phase3_mcp_web_search.py -v
# Expected: 3 passed (list_tools + returns_results + handles_searxng_down).
```

Paste output into worklog.

---

### Task 6 — Update docs

**File**: `docs/RUNBOOK.md`

Add to §3 (Database setup → rename to "§3 Containers setup" since now PG + Redis + SearXNG):

```markdown
The compose file creates:
- PostgreSQL: DB `eca`, user `eca`, password `eca_dev`, port 5432
- Redis: port 6379, 512MB maxmemory, LRU eviction
- SearXNG: port 6666, web search aggregator (Google + Bing + DDG + Wikipedia)

**First-time setup**: generate SearXNG secret and put in repo-root `.env`:

```bash
python -c "import secrets; print('SEARXNG_SECRET_KEY=' + secrets.token_hex(32))" >> .env
```

Verify `.env` is in `.gitignore`. Then `docker compose -f docker-compose.langgraph.yml up -d`
will auto-read `.env` and pass `SEARXNG_SECRET_KEY` to the container. If not set,
compose fails immediately with a clear error (intended — better than silent SearXNG crash).
```

Add to §7 Common errors:

```markdown
### SearXNG returns 403 / empty results
- `limiter: true` in settings.yml → set to `false` for local dev
- `formats:` doesn't include `json` → add it, restart container
- Secret key still placeholder → regenerate as above

### `httpx.ConnectError` from search_medical tool
SearXNG container down or wrong port. Verify:
```powershell
docker ps --format "{{.Names}}: {{.Status}}" | grep searxng
curl -s "http://localhost:6666/search?q=test&format=json" | head
```
```

Add to "Port map" table:
```
| SearXNG | 6666 | Self-hosted metasearch (Google+Bing+DDG+Wikipedia) |
```

**File**: `README.md`

Update Architecture diagram + Quick Start `docker compose up -d` description to mention
"PostgreSQL + Redis + SearXNG". Update Common Errors table to add SearXNG row.

**Done when**: `cat docs/RUNBOOK.md | grep -i searxng | wc -l` returns ≥ 5 lines.

---

### Task 7 — Smoke test end-to-end

After all 6 tasks:

```powershell
# 1. Containers up
docker compose -f docker-compose.langgraph.yml up -d
docker ps                                       # 3 containers Up

# 2. Backend up
conda activate firstconda    # or vva
cd agenticRAG/agentic_rag_gemini
python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8080

# 3. In another terminal — verify SearXNG is used by retriever
curl -s -N -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"query":"đau lưng dưới là gì"}'
# Expected SSE events: stage(memory) → stage(planner intent=knowledge_query) → stage(retriever_agent) → stage(synthesizer) → stage(grader) → stage(conversation) → token(...) → done

# 4. Verify SearXNG actually hit
docker logs eca-searxng --tail 20    # should show GET /search?q=... entries
```

**Done when**: chat returns non-empty `final_answer` referencing knowledge sources (paste 1 sample answer + SearXNG access log line into worklog).

---

## Order + commits

| # | Commit | Files | Smoke verify before next |
|---|--------|-------|-------------------------|
| 1 | `feat(searxng): docker service + settings file` | compose + config/searxng/ | Task 1 done-when curl |
| 2 | `feat(searxng): MCP server uses SearXNG backend` | web_search_server.py + client.py | Task 2 done-when CLI |
| 3 | `chore(searxng): drop ddgs from requirements` | requirements-langgraph.txt | `pip install` clean |
| 4 | `test(searxng): update mocks for httpx + SearXNG` | test_phase3_mcp_web_search.py | Task 5 done-when pytest |
| 5 | `docs(searxng): RUNBOOK + README updates` | docs/RUNBOOK.md, README.md | Task 6 done-when grep |

5 commits, sequential. Each commit's smoke output pasted into `docs/worklogs/DD-MM-YYYY.md`
before moving on.

---

## Acceptance gate (K review)

- [ ] All 5 commits land, individual smoke tests in worklog
- [ ] `pytest tests/langgraph_agents/ -m unit` — still 100+/100+ pass (3 web_search tests now use SearXNG mocks)
- [ ] `pytest tests/langgraph_agents/ -m integration -v -k web_search` — passes against live SearXNG container (no internet block on Google/Bing aggregation)
- [ ] Manual: stop SearXNG container → chat with knowledge query → still returns answer (graceful, synthesizer prompt without web context)
- [ ] Manual: real query in browser → answer mentions actual sources (citations or domain names from SearXNG aggregated results)

Pass → merge to `feature/langgraph-rewrite`, no tag (Phase 6.6 is incremental within Phase 6).

---

## Out of scope (defer)

- Tuning SearXNG engines (which to include/exclude) — defer until quality complaints
- Redis-cached SearXNG results — defer until traffic > 100 req/min
- SearXNG fallback to DDG when down — defer; graceful empty result handling is enough for MVP
- Auth on SearXNG endpoint — localhost only, not exposed externally
- HTTPS/TLS on SearXNG — same reason
