# VVA Architecture — Full Flow + Optimization Inventory (pre-deploy)

> Status: **Living review doc** before Phase 7 deployment
> Date: 2026-05-26
> Reflects: Phase 6.7 (Web toggle) + 6.8 (skip styling, persona in synthesizer, model per role, TTS via asyncio.create_task)

---

## 0. TL;DR

Current latency after Phase 6.x optimizations:

| Scenario | Total | Bottleneck |
|---|---|---|
| Greeting "Xin chào" | 6-10s | planner (5s) + conversation (3-5s) — both flash now |
| Knowledge query, web OFF | 15-25s | retriever (4-9s) + synthesizer (10-20s) |
| Knowledge query, web ON | 25-45s | + SearXNG (3-5s) + retriever extra round (4-5s) |
| Speech mode adds | +3-5s | parallel TTS (asyncio.create_task) — non-blocking after fix |

**Realistic deploy SLA**: text response p50 ~15s, p95 ~40s (web on). Below ChatGPT (~3-8s) due to:
- DeepSeek API latency (Vietnam → API region)
- No model caching
- Sequential graph (no parallel synthesizer)
- No prompt caching

---

## 1. Full request flow (current state)

### 1.1 Entry — POST /chat with SSE

```
Browser fetch + custom SSE parser (api.js)
   │
   ▼  POST http://localhost:8080/chat (CORS preflight then POST)
FastAPI lifespan loaded:
   - _graph (LangGraph StateGraph instance)
   - _redis (sync client, socket_timeout=5)
   - MCP discovery cached (2 tools: kimodo_motion, search_medical)
   - Persona MD files loaded on demand
   │
   ▼  request_id = uuid4(), with_request_id() ContextVar
event_generator() starts streaming
```

### 1.2 Graph execution — 7 nodes

```
START
  │
  ▼  intent=conversation → planner → conversation (LLM call) → END
                                           │
                                           ▼ SSE: token events
  ▼  intent=clarify    → planner → conversation (LLM, style clarify) → END
  │
  ▼  intent=knowledge/exercise/motion
       │
       ▼ memory (Redis STM + conditional LTM pgvector)
       │     ├─ Always: read Redis stm:{session_id} (3 Q&A FIFO)
       │     ├─ Conditional: if recall keywords → pgvector search past sessions
       │     └─ User profile from PG users table
       │
       ▼ planner (LLM call, flash model, JSON mode structured output)
       │     ├─ Intent classification (5 types)
       │     ├─ Expand query with anatomical synonyms
       │     ├─ Build plan: required_outputs, search_strategy, constraints
       │     └─ Detect needs_clarification (LTM ambiguous OR low confidence)
       │
       ▼ retriever_agent (LLM call 1, flash model, bind_tools)
       │     ├─ Decide which tools: pgvector_search, search_medical (web), generate_motion
       │     ├─ if web_search=false → drop search_medical from tools list
       │     ├─ LLM may call multiple tools in parallel (ToolNode batches)
       │     │
       │     ├─ ToolNode executes:
       │     │   ├─ pgvector_search → in-process VectorBackend (Postgres, MiniLM embedding)
       │     │   ├─ search_medical → MCP stdio subprocess → SearXNG (localhost:6666) → Google/Bing/DDG/Wikipedia
       │     │   └─ generate_motion → MCP stdio subprocess → Kimodo (mock, returns mock://...)
       │     │
       │     └─ LLM call 2: consume tool results, decide "done" or "more tools"
       │
       ▼ synthesizer (LLM call, PRO model, persona injected, streams via writer)
       │     ├─ Build system prompt = persona_md + clinical task instructions
       │     ├─ User msg = tool_results + plan + memory + query
       │     ├─ llm.astream() → writer({"content": ...}) per chunk
       │     └─ reasoning_output = final text, total_tokens accumulated
       │
       ▼ grader (rule-based, 0ms)
       │     ├─ Check word count, intent-specific markers (sets/reps for exercise, ...)
       │     ├─ pass → conversation (skip via fast path)
       │     ├─ retry → back to retriever_agent (only once, retry_count++)
       │     └─ pass_with_warning → conversation (skip + append warning)
       │
       ▼ conversation (fast-path skip for styling mode — see Phase 6.8)
       │     ├─ Styling mode (intent has reasoning_output): NO LLM call, just propagate
       │     ├─ Generation mode (greeting/clarify only): LLM call, flash model, streams
       │     └─ final_answer set
       │
       ▼ END
```

### 1.3 Post-graph — TTS + session persist

```
event_generator continues:
  │
  ▼ write_session_turn(user_id, session_id, query, answer, intent, tokens)
  │     ├─ INSERT users ON CONFLICT DO NOTHING (auto-create user)
  │     └─ INSERT conversations ON CONFLICT (session_id) DO UPDATE
  │
  ▼ SSE: session_persisted event
  │
  ▼ if output_mode in ("speech", "both"):
  │     ├─ asyncio.create_task(synthesize_speech_async(...)) — fire immediately
  │     ├─ Strong ref via _pending_tts_tasks set (prevent GC)
  │     ├─ SSE: speech_pending
  │     │
  │     └─ _poll_speech_result loop (max 15s, 250ms interval):
  │         ├─ Check Redis task_result:{task_id}
  │         ├─ If exists → yield speech_ready (URL of audio file)
  │         └─ Timeout → speech_failed
  │
  ▼ SSE: done event
```

**Parallel TTS task** (asyncio):
```
synthesize_speech_async (in tasks.py):
  ├─ httpx POST localhost:5000/synthesize → VieNeu service
  ├─ Receive {audio_file: "vieneu_en_xxx.wav"}
  ├─ Build URL: http://localhost:5000/audio/{filename}
  └─ Redis SETEX task_result:{task_id} (TTL 1h)
```

---

## 2. Where time is spent (measured)

### 2.1 Per-node (typical knowledge query, web off)

| Node | Time | LLM model | Why this much |
|---|---|---|---|
| memory | ~10ms | none | Redis read fast, LTM skipped (no recall keyword) |
| planner | 3-5s | flash | JSON mode + structured output, network RTT to DeepSeek |
| retriever_agent LLM call 1 | 2-3s | flash | Decide tools |
| Tool execution (pgvector) | 0.1-0.5s | none | In-process, MiniLM embed + IVFFlat scan |
| Tool execution (SearXNG, if web) | 3-5s | none | SearXNG aggregates 4 engines, throttled |
| retriever_agent LLM call 2 | 2-3s | flash | Consume tool results |
| synthesizer | 10-20s | pro | Heavy reasoning, 1500-2500 chars output |
| grader | 0ms | none | Rules |
| conversation (skipped for content) | 0ms | none | Fast-path |
| write_session_turn | 30-80ms | none | PG INSERT |
| TTS parallel (if speech) | 3-5s (parallel) | none | VieNeu CPU inference, non-blocking |

**Sequential critical path** (text mode, web off): 5 + 3 + 0.3 + 3 + 15 = **~26s** typical, but DeepSeek API latency volatility makes p95 ~40s.

### 2.2 Per-LLM-call breakdown (DeepSeek API)

| | Time per call | Notes |
|---|---|---|
| Network RTT (Vietnam → DeepSeek) | 200-500ms | Variable, sometimes 1s+ |
| Model thinking (v4-pro) | 5-25s | Thinking tokens hidden, output streaming |
| Model thinking (v4-flash) | 1-5s | Non-thinking, much faster |
| Output streaming | depends on output size | ~30-60 tokens/sec typical |

---

## 3. Optimization inventory (ranked by impact)

### 3.1 ALREADY APPLIED in Phase 6.x

| # | Optimization | Save | Status |
|---|---|---|---|
| 1 | Skip conversation styling (fast-path) | -20-35s | ✅ |
| 2 | Synthesizer streams tokens via writer | UX +20s feel | ✅ |
| 3 | Per-role model: planner/retriever/conversation use flash | -10-15s | ✅ |
| 4 | Persona injected into synthesizer system prompt | quality + 0s cost | ✅ |
| 5 | Default web_search OFF | -10s when off | ✅ |
| 6 | asyncio.create_task for TTS (non-blocking) | TTS doesn't block response | ✅ |
| 7 | UI fetch+text() fallback for AV-buffered SSE | streaming works in AV env | ✅ |

### 3.2 HIGH impact, LOW risk (next wave — Phase 6.9?)

| # | Optimization | Est save | Effort | Risk |
|---|---|---|---|---|
| H1 | **Merge retriever + synthesizer** (single agent that calls tools AND writes response) | -10s (skip retriever's "consume tool results" LLM call 2) | Medium (rewrite retriever_agent prompt + remove synthesizer call) | Medium (LLM might skip pgvector if it thinks training data is enough) |
| H2 | **Memory cache for embeddings** (LRU on query → embedding) | -50-100ms per query | Easy (~20 LOC) | Low |
| H3 | **MCP tool result caching** (Redis cache `tool_name:hash(args)` → result, TTL 1h) | -3-5s when query repeats | Easy (~30 LOC) | Low |
| H4 | **Parallelize planner + memory** (currently sequential; memory has no LLM, can run alongside planner) | -10ms (memory is fast — minor) | Easy | Low |
| H5 | **Reduce planner prompt** (currently has 6 few-shot examples, ~1k tokens) | -1-2s | Easy | Medium (intent accuracy regression) |
| H6 | **Streaming SearXNG results** (return first 3 results immediately, append more) | -2s perceived | Hard (rewrite search wrapper) | Low |

### 3.3 MEDIUM impact, MEDIUM risk

| # | Optimization | Est save | Effort | Risk |
|---|---|---|---|---|
| M1 | **Prompt caching** (DeepSeek supports — system prompt cache hit = 50% off + faster) | -2-5s per call after first | Easy (~10 LOC env config) | Low (need DeepSeek API support verified) |
| M2 | **Skip grader for short responses** (current always runs; if <50 chars likely greeting echo) | -negligible (grader is 0ms anyway) | Trivial | Low |
| M3 | **Pre-warm models on lifespan startup** (do 1 dummy call to each model, prime DeepSeek cache + network) | -2-3s first request only | Easy | Low |
| M4 | **Switch retriever to single-pass (no consume LLM call)** — use plan directly to call all tools, then to synthesizer | -3s | Medium | Medium (lose LLM reasoning about tool combination) |
| M5 | **Quantize embedding model** (MiniLM-L6-v2 → ONNX int8) | -20-50ms per embed | Medium | Low |
| M6 | **Connection pooling for httpx clients** (currently new client per call for SearXNG/VieNeu) | -50-100ms per external call | Easy (~10 LOC) | Low |

### 3.4 HIGH impact, HIGH risk (deferred to post-deploy)

| # | Optimization | Est save | Effort | Risk |
|---|---|---|---|---|
| R1 | **Switch synthesizer to flash too** (for cases where pro overkill) | -10s | Easy | High (quality regression for clinical) |
| R2 | **Bypass LangGraph for greeting** (direct LLM call from FastAPI for intent=conversation) | -2s graph overhead | Hard | High (breaks architecture) |
| R3 | **GPU-accelerated embedding** (CUDA MiniLM) | -10ms per embed | Medium | Low — but needs GPU on edge |
| R4 | **Multi-region deploy** (Asia VPS closer to DeepSeek API endpoint) | -200-500ms RTT per call × N calls = save 1-2s | Hard (Phase 7) | Low |

### 3.5 UX optimizations (no actual speedup, but perceived faster)

| # | Optimization | Effect |
|---|---|---|
| U1 | Show stage indicator with descriptive labels per node | User knows progress, less impatient |
| U2 | Optimistic UI: show user message immediately before backend echo | Faster send experience |
| U3 | Show partial sources as they arrive (citation pills update during synthesizer) | Trust signal |
| U4 | Skeleton response shape while waiting (e.g. "Bài tập 1: ___\nBài tập 2: ___") | Frames expectation |
| U5 | Audio playback button instead of autoplay (some browsers block autoplay) | Reliability |

---

## 4. Architecture concerns for deploy

### 4.1 Production hardening still needed

| Concern | Current | Deploy needs |
|---|---|---|
| Backend log file | stdout redirect via `*> vva.log` | Proper rotation (logrotate or python RotatingFileHandler) |
| CORS allow_origins=["*"] | dev only | Lock to known origin in prod |
| _to_uuid auto-create user | Anonymous "user_123" maps to deterministic UUID | Need real auth — anonymous mode OK for demo, not multi-tenant |
| Redis no auth | localhost trust | Add password if exposed |
| SpeechLLm on port 5000 no auth | localhost trust | Same |
| MCP subprocess crashes | Graceful fallback (0 tools) | Add restart policy + alerting |
| Circuit breaker state in-memory | resets on restart | OK for single instance, fail open in multi-instance |
| TTS audio files growing | VieNeu writes to disk | Need cleanup cron or TTL on storage |

### 4.2 Scalability bottlenecks

| Bottleneck | When hits | Mitigation |
|---|---|---|
| DeepSeek rate limit | ~50 req/s on standard tier | Queue + retry, or scale plan |
| Sync Redis client | OK for <1000 req/s | Switch to redis.asyncio for high concurrency |
| SearXNG self-host | 1 instance limit | Multi-instance behind LB if heavy use |
| pgvector single index | scales to ~1M docs IVFFlat | HNSW + partitioning beyond |
| VieNeu CPU inference | ~1 concurrent | Pool of TTS workers if needed |
| asyncio.create_task tracking | unbounded set growth on long-running | Add max-size + LRU eviction |

### 4.3 Observability gaps

| Gap | Priority |
|---|---|
| No metrics endpoint (Prometheus /metrics) | High for prod |
| No distributed tracing (OpenTelemetry/LangSmith) | Medium |
| No request log retention strategy | High |
| No alert on circuit breaker open | Medium |
| No cost tracker (total_tokens × price = bill) | Medium |

---

## 5. Recommended pre-deploy sequence

### Phase 6.9 — "Final polish before deploy" (~3-4h)

Pick from §3.2 high-impact-low-risk:

1. **H3 — MCP tool result caching** (~30 LOC) — biggest win for repeat queries, very low risk
2. **M3 — Pre-warm models in lifespan** (~10 LOC) — eliminates "cold start" feel
3. **M6 — Connection pooling for SearXNG/VieNeu httpx clients** (~10 LOC) — small but free
4. **M1 — DeepSeek prompt caching env** (~5 LOC + verify API support) — 50% cost reduction + 2-5s faster

Total: ~60 LOC, ~3h work, save 5-10s per request typically, 30%+ on repeat queries.

### Phase 6.10 — "Production hardening" (~1 day)

From §4.1:
- Log rotation
- Lock CORS to allowed origins
- TTS audio cleanup (cron delete files older than 1h)
- Health probe for SpeechLLm + SearXNG inside `/health/detailed`
- Document `Owner-deploy-checklist.md`

### Phase 7 — "Deploy"

Per existing `docs/DEPLOYMENT.md` stub: VPS + Supabase + edge worker. Out of scope here.

---

## 6. Things NOT to do before deploy

Anti-patterns or premature optimizations to skip:

1. ❌ **Don't merge retriever+synthesizer (H1) unless you can A/B test** — quality risk for clinical responses
2. ❌ **Don't switch synthesizer to flash (R1)** — Owner's product is healthcare advisory; quality matters
3. ❌ **Don't add complex multi-instance state sharing** for circuit breakers/cache — single instance is fine for academic deploy
4. ❌ **Don't add LangSmith tracing** until you have real traffic — adds ~100ms per call, only worth it with volume
5. ❌ **Don't rewrite to native EventSource** (GET endpoint) — POST + fetch+text() fallback works fine
6. ❌ **Don't try to bundle SpeechLLm into main backend** — keep as separate service, lighter coupling

---

## 7. Decision template

Owner fill after research:

| Item | Decide? | Notes |
|---|---|---|
| Phase 6.9 picks (which of H3/M1/M3/M6) | [ ] H3 [ ] M1 [ ] M3 [ ] M6 | |
| Phase 6.10 hardening scope | [ ] Full [ ] Skip [ ] Partial | |
| Acceptable p50 latency for deploy | ___s | |
| Acceptable p95 latency for deploy | ___s | |
| Skip H1 (merge retriever+synthesizer) — yes/no | [ ] Yes [ ] No | Risk vs save |
| Defer M-tier (M1-M6) entirely — yes/no | [ ] Yes [ ] No | |
| Deploy target | VPS / Supabase / Local-only | |

---

## 8. References

- Plan v2.4 source: `.claude/plans/purrfect-herding-kahn.md`
- Phase 6 P0 spec: `phases/phase-6-p0.md`
- Phase 6.6 SearXNG: `phases/phase-6.6-searxng.md`
- Phase 6.7 Web toggle: `phases/phase-6.7-web-toggle.md`
- Architecture options doc: `architecture/decision-response-nodes.md`
- Runbook: `docs/RUNBOOK.md`
- Deployment stub: `docs/DEPLOYMENT.md`
- DeepSeek prompt caching: <https://api-docs.deepseek.com/guides/kv_cache>
- Anthropic "Building effective agents": <https://www.anthropic.com/research/building-effective-agents>
- LangGraph agent patterns: <https://langchain-ai.github.io/langgraph/tutorials/introduction/>
