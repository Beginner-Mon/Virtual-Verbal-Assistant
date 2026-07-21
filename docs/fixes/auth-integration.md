# FEATURE — Auth Integration: frontend ↔ backend ↔ Cognito JWT

> Author: K | Date: 2026-06-18 | Audience: implementer (subagent) + N review
> Owner decision 18/06: demo runs on NEW React UI, auth can be SKIPPED for demo,
> but wire all 3 pieces now (frontend API client + JWT + backend verify).
> Closes IDOR 🔴 when `REQUIRE_AUTH=true`.

---

## Quyết định kiến trúc (K) — auth ENFORCED-WHEN-PRESENT, không bắt buộc

The crux that satisfies "demo skips auth" AND "close IDOR": a flag `REQUIRE_AUTH`
(default **false**).
- Token present + valid → `user_id = sub` from the JWT (client-supplied user_id IGNORED).
- No/invalid token + `REQUIRE_AUTH=false` (dev/demo) → fall back to client-supplied
  user_id (current behavior — demo works without login).
- No/invalid token + `REQUIRE_AUTH=true` (network deploy) → 401. ← this is the IDOR close.

So the machinery ships now; flipping the flag closes IDOR. No demo blockage.

**Contract (both sides MUST agree):**
- Header: `Authorization: Bearer <Cognito ID token>` (idToken, not accessToken).
- `/chat` body keeps existing schema; when authenticated the backend IGNORES body
  `user_id` and uses `sub`.
- SSE events unchanged: `stage` / `token` / `done`.

---

## PART A — Backend (Python / FastAPI)

### A1. `requirements-langgraph.txt`
Add `PyJWT[crypto]` (JWKS RS256 verify). Nothing heavier.

### A2. New `api/auth.py`
```python
# Config from env (all optional; empty in demo):
#   REQUIRE_AUTH = "false" | "true"   (default false)
#   COGNITO_REGION, COGNITO_USER_POOL_ID, COGNITO_APP_CLIENT_ID
#
# async def resolve_user_id(request: Request, fallback_user_id: str | None) -> str
```
Behavior:
- Read `Authorization: Bearer <jwt>`.
- If present → verify:
  - fetch + cache JWKS from
    `https://cognito-idp.{REGION}.amazonaws.com/{POOL_ID}/.well-known/jwks.json`
    (module-level cache; refetch on unknown `kid`).
  - `jwt.decode(token, key, algorithms=["RS256"], audience=APP_CLIENT_ID,
    issuer=f"https://cognito-idp.{REGION}.amazonaws.com/{POOL_ID}")`
  - assert `claims["token_use"] == "id"`.
  - return `_to_uuid(claims["sub"])` (reuse `db/session_store._to_uuid`; Cognito sub is
    already a UUID → unchanged).
  - any verify failure → treat as "no valid token" (→ 401 or fallback per flag).
- If absent/invalid:
  - `REQUIRE_AUTH` true → `raise HTTPException(401, "authentication required")`.
  - else → return `_to_uuid(fallback_user_id)` (fallback_user_id from body/query/path).
- If `REQUIRE_AUTH=true` but Cognito config missing → fail loud at startup (don't
  silently allow). In demo (REQUIRE_AUTH=false) missing config is fine.

### A3. Wire into endpoints (`api/main.py`)
Replace every spot that trusts a client user_id with `resolve_user_id`:
- `POST /chat`: `uid = await resolve_user_id(request, req.user_id)`; use `uid` in
  config/`write_session_turn` instead of `req.user_id`.
- `GET /sessions`, `GET /sessions/{session_id}`: `uid = await resolve_user_id(request, user_id_query)`.
- `DELETE /sessions/{user_id}/{session_id}`, `DELETE /sessions/{session_id}/messages/...`,
  `DELETE /users/{user_id}`, `POST|GET|DELETE /users/{user_id}/memory...`:
  `uid = await resolve_user_id(request, path_user_id)`. When authenticated and the path
  user_id ≠ sub → the resolved `uid` (=sub) wins (so a user can only act on their own data).
- Keep behavior identical when `REQUIRE_AUTH=false` and no token (demo).

### A4. CORS — add Vite dev origin
`api/main.py` default `ALLOWED_ORIGINS` currently `localhost:3000,localhost:8080`.
Add `http://localhost:5173` (Vite dev) to the default list.

### A5. Tests `tests/langgraph_agents/test_auth.py` (mock JWKS/decode)
- [ ] valid id token (mock `jwt.decode` → `{"sub": "...", "token_use":"id"}`) → resolve_user_id returns that sub
- [ ] no token + REQUIRE_AUTH=false → returns `_to_uuid(fallback)`
- [ ] no token + REQUIRE_AUTH=true → HTTPException 401
- [ ] invalid/expired token (decode raises) + REQUIRE_AUTH=true → 401; + false → fallback
- [ ] `token_use != "id"` → rejected
- [ ] wrong audience/issuer (decode raises InvalidAudience/Issuer) → rejected
Mock JWKS fetch + `jwt.decode`; do NOT hit network. Full suite must stay green.

---

## PART B — Frontend (React/TS, `ECA_UI/frontend/`)

No test runner exists (no vitest) — do NOT add one. Acceptance = `npm run build`
(tsc typecheck) passes + manual chat smoke against a running backend.

### B1. `src/lib/api.ts` (new)
- `const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080'`
- `authHeader()`: `try { const s = await fetchAuthSession(); const t =
  s.tokens?.idToken?.toString(); return t ? {Authorization:'Bearer '+t} : {} } catch { return {} }`
- `currentUserId()`: if a session/sub exists → use `sub`; else a stable demo id
  (localStorage `vva_demo_user`, generate `crypto.randomUUID()` once). Demo works w/o login.
- `streamChat({query, sessionId, personaId='eca_default', outputMode='text', webSearch=false},
  onEvent, signal)`: `POST {API_BASE}/chat` with `{...authHeader()}` + JSON body
  `{query, user_id, session_id, persona_id, output_mode, web_search}`; parse SSE with
  `resp.body.getReader()` + buffer split on blank line, lines `event:`/`data:`.
  **Reference implementation: `ECA_UI/test-ui/sse-test/api.js` `streamChat` (~line 526)** —
  copy its SSE-parsing logic, add the auth header + AbortController signal.
- `listSessions()`, `getSession(id)`, `deleteSession(id)`, user_memory CRUD — all with
  `authHeader()`; user_id from `currentUserId()` (only used when not authenticated;
  backend overrides with sub when authed).

### B2. Replace ChatPanel mock with real streaming
`src/components/ChatPanel.tsx`: remove `DEMO_RESPONSES` + `setInterval` mock. `handleSend`
calls `streamChat(...)`, appends `token` event content into the streaming assistant message,
finishes on `done`. `handleStop` aborts via `AbortController`. Keep the existing UI/markup.

### B3. Wire chat into the page
Inspect `App.tsx` / `MainLayout.tsx` / `pages/ChatPage.tsx` routing. ChatPage currently
renders only `<CharacterViewer/>`. Wire `ChatPanel` into the actual chat surface so the
running app shows a working chat (minimal — match how the layout intends panels to sit;
don't redesign).

### B4. Env + README
Add `VITE_API_BASE_URL` to a `.env.example` or README note (default 8080). Note demo mode
needs no Amplify (`amplify_outputs.json` absent → no auth header → backend dev fallback).

---

## Definition of DONE
1. Backend: `resolve_user_id` per A2; wired A3; CORS A4; PyJWT added; tests A5 pass;
   full suite green (was 237) with the new auth tests, 0 regression.
2. Frontend: `api.ts` with streamChat (real SSE) + auth header; ChatPanel real; chat wired
   into page; `npm run build` typechecks clean.
3. Manual smoke (report it): with backend on :8080 and `REQUIRE_AUTH=false`, the new UI
   (`npm run dev`, :5173) sends a message and streams a real DeepSeek reply.
4. IDOR close path verified by a test: `REQUIRE_AUTH=true` + no token → 401.
5. Worklog → K review. Do NOT commit (K reviews first).

## Out of scope (don't do)
- Deploying Cognito (`ampx sandbox` / AWS) — demo mode + mocked-token tests are enough.
- Making auth mandatory by default (keep `REQUIRE_AUTH=false`).
- Voice/TTS, avatar pipeline, persona picker — untouched.
- Frontend test runner / vitest.
- Redesigning the UI layout.

## STOP-and-report (karpathy #1) if:
- The frontend layout makes "wire ChatPanel into the page" ambiguous (report how App
  routes, propose minimal wiring, don't guess-redesign).
- `npm install` / `npm run build` fails for environment reasons (report, don't hack around).
- PyJWT API differs from the spec snippet — report, don't silently swap libs.
