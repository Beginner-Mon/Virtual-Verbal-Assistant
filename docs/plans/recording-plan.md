# Plan — Ghi âm trong Chat (Frontend only, không đụng backend)

> Branch: `feature/frontend-fixes`. Chỉ `ECA_UI/frontend`, không sửa `agenticRAG` hay infra.

## Hiện trạng

- `ChatPanel.tsx:222-228` nút Mic câm: `disabled={isGenerating}`, không `onClick`.
- `ChatContext.tsx` chỉ gửi `text` qua `streamChat`, chưa có `audio`.
- `ChatMessage.tsx` chưa render `audio`.
- Chưa có `MediaRecorder` hay permission handling.

## Mục tiêu (chỉ frontend)

Bấm Mic → ghi âm → preview → gửi như **message audio local** (play được ngay), không cần backend, không STT, không upload.

## Plan — 5 bước frontend only

### 1. Hook `src/hooks/useAudioRecorder.ts` (mới)
- `start()`: `navigator.mediaDevices.getUserMedia({audio:true})` → `MediaRecorder` (check `isTypeSupported('audio/webm')` fallback `audio/mp4` cho Safari) → `chunks[]` via `ondataavailable`.
- `stop()`: `recorder.stop()` → `new Blob(chunks, {type})` → `URL.createObjectURL(blob)` → `audioUrl` + `audioBlob`.
- `cancel()`: `revokeObjectURL`, clear chunks, `track.stop()`.
- State: `isRecording: boolean`, `duration: number` (setInterval 100ms), `error: string | null` (NotAllowedError → "Microphone permission denied").
- Cleanup: `useEffect` revoke URL on unmount, `mediaStream.getTracks().forEach(t=>t.stop())`.

### 2. ChatContext — thêm audio local
- Thêm: `audioUrl: string | null`, `audioBlob: Blob | null`, `isRecording`, `recordingDuration`, `recordingError`, `startRecord()`, `stopRecord()`, `cancelRecord()`, `sendAudio()`.
- `sendAudio()`: nếu `audioBlob` có → `const url = URL.createObjectURL(audioBlob)` → `Message {id, role:'user', content:'', audioUrl: url, audioBlob, timestamp: new Date()}` → `setMessages(prev=>[...prev, msg])` → `cancelRecord()` để clear preview. Không gọi `streamChat`, không đụng backend.
- Guard: `isGenerating` → Mic disabled (đã có), `isRecording` → Send text disabled.

### 3. ChatPanel — UI Mic
- `ChatPanel.tsx:222` Mic button: `onClick={isRecording ? stopRecord : startRecord}` + icon `Mic`/`Square` + `animate-pulse` khi recording.
- Khi `isRecording`: show bar `● 00:05` + `Stop` + `Cancel` (hủy).
- Khi `audioUrl` preview có: show `<audio controls src={previewUrl} />` + `Gửi` / `Hủy` (Gửi gọi `sendAudio`).
- `recordingError` → toast inline `text-destructive`.

### 4. ChatMessage — render audio
- Nếu `message.audioUrl`: render `<audio controls preload="metadata" src={audioUrl} className="w-full" />` + duration. Nếu có `content` thì render cả text + audio.
- `URL.revokeObjectURL` khi message bị xóa/unmount (dùng `useEffect` cleanup).

### 5. Edge case & test (chỉ frontend)
- Permission deny → `recordingError`, Mic về idle, không crash.
- `isGenerating` → Mic disabled.
- Safari: mime fallback, test `isTypeSupported`.
- Bấm Mic khi đang ghi → stop, không start chồng.
- Ghi xong `Gửi` → message xuất hiện, play được, revoke preview URL.
- `npm run build` + `tsc -b` pass.

## Không đụng

- Không sửa `agenticRAG` (`streamChat`, `api.ts`, `crud_app.py`, `main.py`), không thêm endpoint.
- Không đụng `FloatingNavBar`, `MotionContext`, `MotionControlPanel`.
- Không upload S3, không STT.

## Effort

- **★☆☆ Low-Medium**, ~1.5-2 ngày (hook 0.5 + context 0.5 + panel/message 0.5 + test 0.5).

## Acceptance

- Bấm Mic idle → xin permission → `● 00:05` chạy
- Bấm lại → preview audio + `Gửi`/`Hủy`
- `Gửi` → message audio trong list, `<audio>` play được
- `Hủy` → revoke, không gửi
- Deny permission → báo lỗi, Mic idle
- `isGenerating` → Mic disabled
- `npm run build` xanh

## Duyệt?

Nếu ok thì tách 1 session làm `useAudioRecorder` + `ChatContext` + `ChatPanel`/`ChatMessage` (chỉ frontend).
