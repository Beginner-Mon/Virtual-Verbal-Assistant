---
title: "API Contract"
description: "Unified gateway endpoints, request/response schemas, and task lifecycle."
tags:
  - api
  - contract
  - gateway
  - endpoints
  - json
  - polling
---

# API Contract

> All clients route through the **Unified Gateway** on port 8000. Internal ports (5001, 8080) are not exposed to the UI.

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/process_query` | `POST` | Submit an async query task |
| `/tasks/{task_id}` | `GET` | Poll task progress and results |
| `/download/{file}` | `GET` | Proxy DART motion artifacts |
| `/history/{user_id}` | `GET` | Retrieve chat history |
| `/health` | `GET` | Per-service health status |

## POST /process_query

### Request

```json
{
  "query": "Show me exercises for neck pain",
  "user_id": "web_user",
  "conversation_history": []
}
```

### Response

```json
{
  "task_id": "abc123",
  "status": "processing",
  "progress_stage": "rag_processing",
  "language": "en",
  "text_answer": "Neck pain can often be alleviated...",
  "exercises": [
    {"name": "Chin tuck"},
    {"name": "Shoulder roll"}
  ],
  "motion": {
    "motion_file_url": "/download/motion_abc123.npz",
    "num_frames": 160,
    "fps": 30
  },
  "request_id": "abc123def456",
  "errors": null
}
```

## Task Lifecycle

Stages returned by `GET /tasks/{task_id}`:

1. `queued` — Task accepted
2. `rag_processing` — AgenticRAG generating text
3. `motion_generation` — DART synthesizing 3D sequence
4. `voice_synthesis` — TTS generation (optional)
5. `completed` — All enrichment tasks done

## DART Contract (Port 5001)

### POST /generate

```json
{
  "text_prompt": "jump",
  "duration_seconds": 12,
  "guidance_scale": 5.0,
  "num_steps": 50,
  "respacing": "",
  "seed": null
}
```

### Response

```json
{
  "request_id": "a1b2c3d4e5f6",
  "motion_file_url": "/download/motion_a1b2c3d4e5f6.npz",
  "num_frames": 480,
  "fps": 30,
  "duration_seconds": 16.0,
  "text_prompt": "jump"
}
```

## Motion Prompt Key Evolution

- **Legacy**: `motion_prompt` (used by `pipeline_orchestrator.py`)
- **Current**: `exercise_motion_prompt` (used by `api_orchestrator.py` and `main_api_downstream.py`)

The pipeline orchestrator was patched in D8c to accept both keys for backward compatibility.

## Headers

- `X-Request-ID`: Correlation ID echoed in responses (generated if not provided).
- `AGENTIC_TRACE=1` (env): JSON responses include `agent_trace` field with per-stage timings.

## Avatar Animation SSE Events (Phase D contract)

> Source of truth for the facial-animation handoff — see [[facial-animation-plan]] §7, §9.
> These events ride the **existing per-turn `/chat` SSE stream** (opens on POST, closes on
> `done`). There is NO persistent connection; avatar events are interaction-scoped. Idle
> behavior is client-autonomous and needs no backend traffic.

Frontend consumes every event through `streamChat(options, onEvent)` — the same callback that
already handles `stage` / `token` / `done`. Adding avatar handling is a callback branch, not new
transport (the named-event parser already works).

### `avatar.emotion` (backend → client)

Emitted by the Conversation node when a response carries an emotional intent.

```
event: avatar.emotion
data: {"emotion":"happy","intensity":0.8,"duration":1000}
```

| Field | Type | Notes |
|-------|------|-------|
| `emotion` | enum | One of `neutral \| happy \| sad \| angry \| relaxed \| surprised` (canonical set). Any other value → client warns + drops. |
| `intensity` | number | Clamped to `[0,1]` client-side. |
| `duration` | number (ms) | How long to hold before the client auto-fades to neutral. A hint; the client owns easing. Also extends the ENGAGED window. |

Client action: `avatarController.setEmotion(emotion, intensity, duration)`.

### `tts.audio` — timing resolution (was open question)

The plan's original `tts.audio {audioUrl}` assumed the URL is known during the turn. It is **not**:
TTS runs as an async Celery task; the `/chat` stream emits `speech_pending` + a `speech_task_id`
and the client polls for the result **after** the turn stream closes.

**Resolved contract**: lip sync is driven by a separate ready signal, not an in-stream event.

- During the turn: `speech_pending` / `speech_task_id` (unchanged, already emitted).
- When the poll resolves the audio, the client fetches the clip and calls
  `avatarController.startLipSync(analyser)` (analyser built from the `<audio>` element via
  `analyserFromElement`). On `ended`, `avatarController.stopLipSync()`.
- No new backend event is strictly required — the client's existing poll completion is the
  trigger. If a push is preferred later, add `event: speech_ready {task_id, audioUrl}`; the
  contract above stays identical on the client.

Engagement: active lip sync holds the avatar ENGAGED; a `TTS_GRACE` (~1.5s) after audio ends
before falling back to IDLE.

## Related Notes

- [[system-overview]] — Architecture and service map
- [[troubleshooting]] — Common API errors
- [[facial-animation-plan]] — Avatar animation phases + module design

---

#api #contract #gateway #endpoints #json #polling
