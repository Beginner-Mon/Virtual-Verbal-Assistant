# 31-08-2026 (2) — Session Switch Flicker Fix

> Plan: `docs/plans/session-switch-fix.md`

## Vấn đề
Bấm session khác -> `ChatPanel` vẫn hiện message của session cũ tới khi `getSession` về (flash).

## Nguyên nhân
`ChatContext.switchToSession` chỉ `setMessages` sau `await getSession`, `switchingRef` là ref không re-render, `ChatPanel` render cũ, `FloatingNavBar` đóng panel ngay (`onSessionSelected` fire-and-forget).

## Sửa
- `contexts/ChatContext.tsx`: thêm `isSwitching` state, trong `switchToSession` set `isSwitching=true`, `setActiveSessionId(id)` + `setMessages([])` + clear input/typing **trước** `await getSession(id)`, sau xong `setMessages(history)` + `isSwitching=false` với race-check `sessionId !== sessionIdRef.current`.
- `hooks/useChat.ts`: expose `isSwitching`.
- `components/ChatPanel.tsx`: gate `isSwitching` -> skeleton `Loader2 Đang tải hội thoại...`, không map `messages` cũ.
- `components/panels/ChatSessionsPanel.tsx`: `await switchToSession` trước `onSessionSelected`, thêm `switchingId` spinner, `disabled={isSwitching}`.

## Verify
- `tsc --noEmit` pass
- `vite build` pass (3481 modules)
- Manual: click A->B -> ChatPanel hiện skeleton ngay, không flash A; highlight B đổi ngay; bấm liên tiếp không nhầm session.
