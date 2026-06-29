---
title: "Troubleshooting"
description: "Common failures, port conflicts, slow responses, and diagnostic commands."
tags:
  - troubleshooting
  - errors
  - ports
  - chromadb
  - redis
  - ffmpeg
  - rate-limits
  - latency
---

# Troubleshooting

## Check Stack Status

```powershell
# Check ports
python check_ports.py --ports 3000 5001 6379 8000 8080

# Check ChromaDB
docker compose ps

# Check ffmpeg
Get-Command ffmpeg

# Health check
curl http://localhost:8000/health
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| API Key Exhaustion | `MAX_TOKENS` misidentified as retryable | Fixed: removed `2` from `_RETRYABLE_FINISH_REASONS` in `gemini_client.py`; use `2048` tokens |
| "AgenticRAG unavailable" timeout | Pipeline latency > 10s | Increase `DOWNSTREAM_TIMEOUT` in `main_api.py` to `90.0` |
| Fallback infinite loop | `max_tokens=256` too small | Set orchestrator `max_tokens` to `1024` |
| Malformed JSON from LLM | Large prompts break JSON enforcement | `clean_json_response` regex in `rag_pipeline.py` |
| Documents not persisting | Wrong ChromaDB client | Use `PersistentClient` (`vector_store.py`) |
| Negative similarity scores | Wrong distance formula | Use `max(0.0, 1.0 - distance/2)` |
| 404 model errors | Invalid model name | Run `python list_available_models.py` |
| 429 rate limits | Free Gemini tier (20 req/day) | `RateLimiter` auto-throttles; reduce retries or upgrade key |
| Documents not retrieved | Threshold too high | Lower `similarity_threshold` in `config.yaml` |
| Slow responses (>20s) | 5 serial Gemini calls + 10s web search | Disable `enable_query_expansion` + `enable_iterative_reflection`; web search now parallel |
| `KeyError` in RAG results | Missing `source_type` key | Use `.get()` for all result dict access |
| RateLimiter deadlock | Lock held while sleeping | Lock released before `time.sleep()` in `acquire()` |

## Kill a Stuck Port

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
taskkill /PID <PID> /F
```

## Debugging Agent Decisions

Set environment variable:

```powershell
$env:AGENTIC_TRACE=1
```

Then inspect `agent_trace` field in `/query` JSON response for per-stage timings and decisions.

## Related Notes

- [[setup-guide]] — Re-run setup if env is broken
- [[api-contract]] — Verify expected request/response shapes
- [[agentic-rag-refactor]] — Recent changes that may affect behavior

---

#troubleshooting #errors #ports #chromadb #redis #ffmpeg #rate-limits #latency
