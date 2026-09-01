# Fix: Session Switch Flicker — Không hiện session cũ khi đang load

> Branch: `feature/langgraph-rewrite` | Ngày: 31-08-2026 | K: Senior Solution Architect
> Vấn đề: bấm session khác -> route sang chat nhưng vẫn thấy message của session cũ 1 khoảng rồi mới nhảy sang session mới

## 1. Nguyên nhân gốc (đã điều tra)

**`ChatContext.tsx:326-362` `switchToSession` giữ `messages` cũ trong suốt `await getSession()`:**
```ts
const switchToSession = async (id) => {
  if (id === sessionIdRef.current || switchingRef.current) return
  switchingRef.current = true  // ref, không gây re-render
  abortControllerRef.current?.abort()
  const data = await getSession(id) // 100-500ms + Neon cold start, messages cũ vẫn trong state
  // mới setMessages(...history) ở đây
}
```
- `switchingRef` là `useRef(false)` (`:97`), không phải `useState` -> UI không biết đang switch
- `isRestoring` chỉ cho lần mount đầu (`:91`), không dùng cho switch
- `setActiveSessionId` cũng chỉ set *sau* fetch -> highlight trong `ChatSessionsPanel.tsx:100` `isActive === activeSessionId` cũng lag

**`ChatPanel.tsx:94` render `messages` vô điều kiện:**
```tsx
const { messages, isRestoring } = useChat()
{messages.map(...)} // luôn hiện session cũ trong lúc await
```
Không có `isSwitching` để gate.

**`ChatSessionsPanel.tsx:104` fire-and-forget + `FloatingNavBar.tsx:489/554` đóng panel ngay:**
```ts
onClick={() => {
  switchToSession(id) // không await
  onSessionSelected?.() // -> setActivePanel('chat') ngay -> ChatPanel hiện stale
}}
```

**Timeline hiện tại:**
1. Click -> `switchingRef=true`, `await getSession`
2. `onSessionSelected` -> `FloatingNavBar` đổi sang `chat` ngay
3. `ChatPanel` vẫn giữ `messages` cũ -> flash
4. Network về -> `setMessages(new)` -> nhảy

`startNewSession` (`:291`) thì làm đúng: `setMessages(buildInitialMessages())` ngay *trước* khi có network, nên không flash.

## 2. Mục tiêu

- Trong lúc `getSession` đang chạy, **không hiện message của session cũ**
- Hiện skeleton/loading thay thế, không nhảy 2 lần
- `activeSessionId` highlight đổi ngay để user biết đang chọn cái nào
- Vẫn giữ abort + chống double-click

## 3. Phương án

### A. Immediate clear + loading flag (Khuyên dùng — đúng yêu cầu Owner)

**Trước `await getSession`:** `setIsSwitching(true)`, `setMessages([])` hoặc `setMessages(buildInitialMessages)` + `setActiveSessionId(id)` ngay, `ChatPanel` hiện skeleton.

**Sau `await`:** `setMessages(history)`, `setIsSwitching(false)`.

Pros: đúng yêu cầu "không hiện session cũ", đơn giản, giống `startNewSession`. Cons: mất context cũ trong lúc load (chính là mục tiêu).

### B. Overlay loading giữ session cũ mờ

Giữ `messages` cũ nhưng phủ `backdrop` + spinner. Không đạt yêu cầu Owner (vẫn thấy cũ).

### C. Prefetch + transition

Fetch trước, chỉ switch panel khi xong. Đỡ flash nhưng panel đóng trễ, cảm giác lag.

**Chọn A.**

## 4. Thiết kế chi tiết (A)

### 4.1 `ChatContext.tsx`

Thêm state:
```ts
const [isSwitching, setIsSwitching] = useState(false)
```

Sửa `switchToSession`:
```ts
const switchToSession = useCallback(async (sessionId: string) => {
  if (sessionId === sessionIdRef.current || switchingRef.current) return
  switchingRef.current = true
  abortControllerRef.current?.abort()
  // 1. Đổi UI ngay
  setIsSwitching(true)
  setActiveSessionId(sessionId) // highlight đổi ngay
  sessionIdRef.current = sessionId // để sendMessage dùng đúng id nếu user gõ ngay
  localStorage.setItem(SESSION_KEY, sessionId)
  setMessages([]) // hoặc buildInitialMessages(uiRef.current) + skeleton riêng — chọn [] để ChatPanel hiện skeleton rõ
  setInput('')
  setIsTyping(false)
  setStageLabel(null)
  setIsGenerating(false)
  endThinking()

  try {
    const data = await getSession(sessionId)
    if (sessionId !== sessionIdRef.current) return // race: user đã bấm session khác trong lúc await
    const history = (data?.messages ?? []) as SessionMessage[]
    if (history.length > 0) {
      setMessages([...buildInitialMessages(uiRef.current), ...history.map(...)])
    } else {
      setMessages(buildInitialMessages(uiRef.current))
    }
  } catch (e) {
    if ((e as Error)?.name === 'AbortError' || (e as any)?.code === 'ERR_CANCELED') return
    console.warn('[session] switch failed:', e)
    // giữ skeleton + có thể toast, không restore cũ
  } finally {
    setIsSwitching(false)
    switchingRef.current = false
  }
}, [endThinking])
```

Expose trong `value` và `ChatContextType` (`hooks/useChat.ts`):
```ts
isSwitching: boolean
switchToSession: (id: string) => Promise<void>
```

`isRestoring` giữ nguyên cho mount đầu.

### 4.2 `ChatPanel.tsx`

```tsx
const { messages, isSwitching, isRestoring } = useChat()

if (isSwitching) {
  return <div className="flex-1 flex items-center justify-center"><Loader2 className="animate-spin" /> Đang tải hội thoại...</div>
}
// hoặc skeleton 3 dòng message
```

Hoặc nếu muốn giữ header: render skeleton thay vì `messages.map`.

Cần import `Loader2` từ `lucide-react` (đã có `isRestoring` dùng chưa có spinner, thêm vào).

### 4.3 `ChatSessionsPanel.tsx`

- Disable nút khi `isSwitching`, hiện spinner trên dòng đang switch
- `onClick` thành `async` và `await switchToSession` trước khi `onSessionSelected`:

```tsx
const { activeSessionId, switchToSession, isSwitching } = useChat()
const [switchingId, setSwitchingId] = useState<string | null>(null)

onClick={async () => {
  if (isSwitching) return
  setSwitchingId(s.session_id)
  await switchToSession(s.session_id)
  onSessionSelected?.()
  setSwitchingId(null)
}}
className={cn(switchingId === s.session_id && 'opacity-60')}
{switchingId === s.session_id && <Loader2 className="w-3 h-3 animate-spin" />}
```

Hoặc đơn giản chỉ `await` — `ChatPanel` đã có `isSwitching` nên không cần `switchingId`, nhưng có thì UX rõ hơn.

### 4.4 `FloatingNavBar.tsx`

Không cần đổi nhiều, vì `onSessionSelected` giờ được await trong `ChatSessionsPanel`. Nếu giữ fire-and-forget thì `ChatPanel` vẫn hiện skeleton (do `isSwitching`), cũng đạt nhưng highlight sẽ đúng hơn nếu await.

### 4.5 Cân nhắc khác

- `startNewSession` đã đúng, không đổi
- `isSwitching` khác `isRestoring`: restoring là mount đầu load từ localStorage, switching là click đổi session — tách riêng để `ChatPanel` hiện chữ khác
- Abort: đã có `abortControllerRef`, giữ
- Race: check `sessionId !== sessionIdRef.current` sau await để bỏ kết quả cũ nếu user bấm liên tiếp

## 5. Rủi ro & Mitigation

| Rủi ro | Giảm |
|--------|------|
| Clear `messages` làm mất scroll position | Skeleton thay thế, `bottomRef` không scroll tới cũ |
| Bấm liên tiếp 2 session nhanh | `switchingRef` + race check `sessionIdRef.current` |
| `getSession` fail -> màn trống | Giữ skeleton + toast, không restore cũ (tránh quay lại flash cũ) |
| `localStorage` id đổi trước khi fetch xong | Đổi trước để `sendMessage` dùng đúng, nhưng nếu fail có thể để lại id rỗng — chấp nhận vì user đã chọn |

## 6. Tiêu chí nghiệm thu

- [ ] Bấm session B khi đang ở session A -> `ChatPanel` **không** hiện message của A, hiện `Đang tải hội thoại...` / skeleton ngay
- [ ] Highlight `activeSessionId` đổi ngay khi bấm (không lag)
- [ ] `await` xong mới hiện message của B, không nhảy 2 lần
- [ ] Bấm liên tiếp A->B->C nhanh không hiện nhầm session giữa
- [ ] `isRestoring` (mount) vẫn hoạt động, không lẫn với `isSwitching`
- [ ] `tsc --noEmit` + `vite build` pass

## 7. Slice triển khai (N)

**Slice 1 — ChatContext (1h):** thêm `isSwitching` state, sửa `switchToSession` clear trước await, expose, thêm race check

**Slice 2 — ChatPanel (30m):** gate `isSwitching` -> skeleton/Loader2

**Slice 3 — ChatSessionsPanel (30m):** `await switchToSession`, disable + spinner, dùng `activeSessionId` mới (đã đổi ngay)

**Slice 4 — Verify (30m):** manual switch A<->B với throttle 3G, log `docs/worklogs/31-08-2026.md`

Tổng ~2.5h, chỉ FE, không đụng BE.

## 8. Thay đổi file

- Mới đổi: `contexts/ChatContext.tsx`, `hooks/useChat.ts`, `components/ChatPanel.tsx`, `components/panels/ChatSessionsPanel.tsx`
- Không đụng: `FloatingNavBar.tsx` (optional), `api.ts`, `MotionContext`

> Sau duyệt, N làm Slice 1-3, K review worklog trước merge.
