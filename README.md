# Virtual Verbal Assistant

Unified gateway architecture for KineticChat Official UI and Streamlit parity.

## Default Frontend

- Official UI 2.0: http://localhost:3000
- Streamlit parity UI: http://localhost:8501

`run_stack.py` launches both, with Official UI as default.

## Gateway Architecture (Port 8000)

All browser traffic should go through Port 8000:

- `POST /process_query` -> async task submission
- `GET /tasks/{task_id}` -> polling with `progress_stage`
- `GET /download/{file}` -> DART artifact proxy via gateway
- `GET /history/{user_id}` -> file-based chat history for UI restore

Motion URLs returned to the UI are gateway-safe (Port 8000), so remote use works behind a single public API tunnel.

## Async Contract

Unified task envelope:

```json
{
  "task_id": "...",
  "status": "processing|completed|failed",
  "progress_stage": "queued|motion_generation|...|completed",
  "result": {},
  "error": null
}
```

## Remote Access (Ngrok)

Recommended tunnels:

1. UI tunnel for Port 3000
2. API tunnel for Port 8000

Open:

- `https://<ui-tunnel>.ngrok-free.app/?api_base=https://<api-tunnel>.ngrok-free.app`

Do not expose internal service ports directly (`5001`, `8080`) for normal UI usage.

## See Also

- [QUICKSTART.md](QUICKSTART.md)
- [README_DEV.md](README_DEV.md)
- [agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md](agenticRAG/agentic_rag_gemini/README_DEVELOPERS.md)
