# FIX — Wire ChatPanel to real backend (surgical, no UI change)

> Author: K | Date: 2026-06-18 | Audience: subagent + K self-check
> Owner: keep Tri's UI EXACTLY; only make the chat talk to the backend.
> Lesson from last attempt: it changed ChatPage layout + greeting → "UI khác quá nhiều".
> This time: touch ONE function in ONE file. Nothing visual changes.

---

## Kiến trúc đã xác minh (K)
`ChatPanel.tsx` is the ONE chat component. It is rendered in 3 places UNCHANGED:
- Desktop: `FloatingNavBar` → `PanelContent('chat')` → `<ChatPanel/>` (floating panel on "Chat" click).
- Mobile: `MainLayout` bottom panel + `MobileNavBar`.
So fixing `ChatPanel.handleSend` fixes chat everywhere with ZERO layout/markup change.

`streamChat` already exists in `ECA_UI/frontend/src/lib/api.ts` (real SSE, auth header,
demo user id). Backend `/chat` is up on :8080 (REQUIRE_AUTH=false → demo, no login needed).

---

## SCOPE — edit ONLY `ECA_UI/frontend/src/components/ChatPanel.tsx`

Allowed changes (and NOTHING else):
1. Add `import { streamChat } from '../lib/api'`.
2. Replace `handleSend`'s mock body (the `setTimeout` + `setInterval` + `DEMO_RESPONSES`
   word-by-word fake) with a real call:
   - append the user message (as now);
   - create an empty assistant message;
   - `await streamChat({ query: text, sessionId: sessionIdRef.current }, onEvent, signal)`;
   - in `onEvent(type, data)`: on `type==='token'` append `data.content` to the assistant
     message; on `type==='done'` stop generating; ignore other event types (stage etc.).
   - on error (catch): set the assistant message to a short error line, stop generating.
3. `handleStop`: call `abortControllerRef.current?.abort()` + reset flags.
4. Replace the two mock refs (`networkTimeoutRef`, `streamIntervalRef`) with
   `abortControllerRef` (AbortController) + a stable `sessionIdRef` (`useRef(crypto.randomUUID())`).
5. Remove `DEMO_RESPONSES` (orphan of this change — karpathy: clean up your own orphans).
6. `handleSend` becomes `async`; guard `if (!text || isGenerating) return`.

## MUST NOT change (this is the whole point)
- `INITIAL_MESSAGES` greeting text — keep Tri's EXACT string (incl. the 🎙️ + markdown note).
- ANY JSX / markup / className / the typing indicator / the add menu / mic+send buttons.
- ANY other file: NOT ChatPage.tsx, NOT MainLayout.tsx, NOT FloatingNavBar.tsx, NOT api.ts,
  NOT amplify.ts, NOT styles. If you think another file needs touching, STOP and report.

## Karpathy guidelines (mandatory)
1. THINK FIRST — if `streamChat`'s signature or `Message` type doesn't fit the plan, STOP and
   report; don't redesign or touch other files.
2. SIMPLICITY — minimal swap; no new components, no markup, no abstractions.
3. SURGICAL — every changed line traces to "make handleSend call the backend". Match Tri's
   existing code style in the file. Remove only the orphans YOUR change creates (DEMO_RESPONSES,
   the two old refs).
4. GOAL-DRIVEN — success:
   - `cd ECA_UI/frontend && npm run build` → tsc clean (run it, report result).
   - `git diff ECA_UI/frontend/src/components/ChatPanel.tsx` shows changes ONLY in imports +
     handleSend + handleStop + the refs; greeting + all JSX byte-identical.
   - `git status` shows ONLY ChatPanel.tsx modified (no other file).

## Report back
Files changed (must be only ChatPanel.tsx), the handleSend/onEvent design, the build result,
and confirm greeting + JSX untouched. Do NOT run playwright, do NOT commit, do NOT write a
worklog — K does the live verification + review.
