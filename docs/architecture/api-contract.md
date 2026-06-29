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

## Related Notes

- [[system_overview]] — Architecture and service map
- [[troubleshooting]] — Common API errors

---

#api #contract #gateway #endpoints #json #polling
