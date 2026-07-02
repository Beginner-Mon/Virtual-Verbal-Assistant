# Architecture Decision — Response Generation Pipeline

> Status: **Open question — Owner researching**
> Context: Phase 6.x optimization pass. Triggered by 80-145s response latency for knowledge queries.
> Date: 2026-05-26

---

## 1. Current architecture (v2.4.1)

For a knowledge/exercise query, the response pipeline is:

```
planner ──► retriever_agent ──► synthesizer ──► grader ──► conversation ──► END
   ↓             ↓                   ↓             ↓            ↓
1 LLM call   2 LLM calls         1 LLM call   0 (rules)    1 LLM call
            (decide tools +
             consume results)
```

For greeting (`intent=conversation`) or `clarify`:

```
planner ──► conversation ──► END
   ↓             ↓
1 LLM call   1 LLM call
```

**Total LLM calls per request type:**

| Intent | LLM calls | Bottleneck |
|---|---|---|
| Greeting | 2 | planner + conversation |
| Clarify | 2 | planner + conversation |
| Knowledge query | 4 | retriever (×2) + synthesizer + conversation |
| Exercise rec | 4 | same as knowledge |
| Visualize motion | 4 | same |

**Measured timing** (knowledge query, web_search=on, DeepSeek v4-pro):
- planner: ~5s
- retriever_agent (2 LLM rounds + tools): ~9s
- synthesizer: ~23s
- grader: 0s (rule-based)
- conversation: ~35s
- **Total: ~72-85s**

---

## 2. The core question — node responsibilities

Currently each node has a distinct role:

| Node | Role | Inputs | Outputs |
|---|---|---|---|
| `planner` | Intent classification + structured plan | query + memory | intent, plan, expanded_query |
| `retriever_agent` | Execute tools per plan | plan + tools (pgvector + MCP) | ToolMessage(s) in state.messages |
| `synthesizer` | Generate clinical content from tool results | tool_results + plan + memory | reasoning_output (clinical text) |
| `conversation` | Apply persona styling to content | reasoning_output + persona | final_answer (styled text) |

The duplication concern:
- **synthesizer** writes 1500 chars of clinical content
- **conversation** rewrites those 1500 chars in persona voice (~1600 chars, same info)
- Both LLM calls take ~20-35s each
- **The information is the same, only the tone changes**

---

## 3. Sub-question — does retriever already write the response?

`retriever_agent` is built with `bind_tools()` + ToolNode loop:
1. LLM call 1: "Given the plan, which tools should I call?"
2. ToolNode executes (parallel where possible)
3. LLM call 2: "Tool results received. Do I need more tools, or am I done?"
4. If done, returns a final message

**Open question for research:**
- Does LLM call 2's output already contain a usable response, or just a "tool decision" message?
- If usable → synthesizer is a 3rd duplicate
- If just a decision → synthesizer is needed for proper response generation

Reference: read [LangGraph ToolNode + bind_tools pattern](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/) to understand what LLM call 2 produces.

Check `agenticRAG/agentic_rag_gemini/langgraph_agents/nodes/retriever_agent.py` for actual prompt sent to retriever LLM.

---

## 4. Three options on the table

### L1 — Conservative (Owner already accepts as baseline)
Add persona to synthesizer system prompt. Keep conversation node for greeting/clarify only. Skip conversation styling for knowledge/exercise (already done in Phase 6.8 task 1).

```
planner ─► retriever ─► synthesizer (persona) ─► grader ─► END
planner ─► conversation (persona, greeting/clarify) ─► END
```

| | Detail |
|---|---|
| LLM calls (knowledge) | 4 (planner + retriever ×2 + synthesizer) |
| LLM calls (greeting) | 2 (planner + conversation) |
| Code change | ~15 LOC in synthesizer.py |
| Test impact | None |
| Risk | Low |
| Save vs current | ~30s (skip conversation styling) |

### L2 — Merge response generation (Option A in chat)
Delete conversation node. Synthesizer becomes universal response generator with 3 modes:
- Mode A: clinical (has tool results)
- Mode B: greeting (no tools)
- Mode C: clarify (no tools, has clarification_question)

```
planner ─►
  knowledge/exercise/motion ─► retriever ─► synthesizer ─► grader ─► END
  conversation/clarify       ─► synthesizer ─► END
error_handler ─► sets final_answer ─► END
```

| | Detail |
|---|---|
| LLM calls (knowledge) | 4 (same as L1) |
| LLM calls (greeting) | 2 (planner + synthesizer) |
| Code change | ~80 LOC across 4-5 files |
| Test impact | Need update `test_phase2_5_integration.py` (3 tests) |
| Risk | Medium |
| Save vs current | ~30s same as L1, plus cleaner architecture |
| Naming concern | "synthesizer" misleading for greeting — could rename to `responder` |

### L3 — Merge retriever + synthesizer (radical)
Retriever_agent's LLM directly generates final response (no separate synthesizer). The LLM that decided which tools to call + saw tool results writes the final answer.

```
planner ─►
  knowledge/exercise/motion ─► agent (tools + response) ─► grader ─► END
  conversation/clarify       ─► responder (no tools)   ─► END
```

| | Detail |
|---|---|
| LLM calls (knowledge) | 3 (planner + agent ×2) — saved 1 call |
| LLM calls (greeting) | 2 |
| Code change | ~120 LOC |
| Test impact | Multiple test files affected |
| Risk | High |
| Save vs current | ~50s (-30 conversation -20 separate synthesizer) |
| Concerns | Agent prompt becomes complex; LLM might skip tools and answer from training data; harder to debug |

### L4 — Switch model per role (cross-cutting)
Independent of L1/L2/L3. Use different LLM tier per node:
- `planner` — fast model (gemini-flash, deepseek-v3-light, gpt-4o-mini)
- `retriever_agent` — fast model
- `synthesizer` — quality model (current deepseek-v4-pro)
- `conversation` — fast model (only restyling)

| | Detail |
|---|---|
| LLM calls | Same count, different speed/cost per call |
| Code change | ~5 LOC in `llm.py` (per-role model mapping) |
| Test impact | Need re-baseline quality |
| Risk | Low |
| Save | 30-60% latency reduction if fast model is 2-3x faster |
| Concern | Fast models may be less accurate for medical reasoning |

L4 stacks with L1/L2/L3 — they're independent optimizations.

---

## 5. Research questions for Owner

To inform the decision, look into:

1. **Does retriever_agent's last LLM message already contain a usable response?**
   - Read `nodes/retriever_agent.py` — what's the system prompt?
   - Run integration test, check log + state.messages content after retriever_agent done
   - Tutorial: [LangGraph ToolNode agentic pattern](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools)

2. **What does "persona styling" actually add vs synthesizer output?**
   - Run 1 query with current architecture, capture both `reasoning_output` (synthesizer) and `final_answer` (conversation styled)
   - Diff them — is the change worth 35s?
   - If diff is mostly tone tweaks → conversation node is low value

3. **Single Responsibility vs Performance trade-off**
   - Plan v2.4 chose 7 nodes for clarity + debuggability
   - 80s response is below typical product expectation (<10s for chat)
   - At what point does "clean architecture" stop being worth the latency cost?
   - Reference: [Andrej Karpathy on agent architectures](https://x.com/karpathy/status/1822848857477632175) (search for similar threads)

4. **DeepSeek v4-pro thinking model behavior**
   - "Thinking" models always do internal CoT before responding → slow
   - Does conversation styling actually need a thinking model? Could use deepseek-v3 (non-thinking) for 3-5x speedup
   - DeepSeek docs: [API reference](https://api-docs.deepseek.com/), see model comparison

5. **Streaming UX vs total time**
   - Currently silent during retriever + synthesizer = 30s no UI feedback
   - User perception of speed != actual speed
   - Could synthesizer stream tokens like conversation does? Code: `get_stream_writer()` in `nodes/conversation.py` lines 80-83

6. **What other production agents do (research)**
   - ChatGPT plugins, Claude tools, Perplexity — how many LLM hops per response?
   - Read [Anthropic's "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — they advocate fewer nodes when possible
   - LangGraph examples: [pre-built agents](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

## 6. Decision matrix template (fill after research)

| Criterion | Weight | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| Latency reduction | ? | ~30s | ~30s | ~50s | ~50%+ |
| Code complexity | ? | Low | Medium | High | Low |
| Test impact | ? | 0 | 3 tests | 5+ tests | Re-baseline |
| Architecture cleanliness | ? | OK | Good | Best | Orthogonal |
| Risk of regression | ? | Low | Medium | High | Medium |
| Reversibility | ? | Easy | Medium | Hard | Easy |
| Owner's effort | ? | 15min | 1h | 2-3h | 30min |
| **Owner's score** | | | | | |

---

## 7. K's recommendation (advisory only)

If Owner picks one path now without research, **L1 + L4 combined**:
- L1 gets the speedup (30s save) with lowest risk
- L4 (switch conversation/planner to fast model) compounds savings (~15-20s extra)
- Total: ~45s save, ~1h work, 0 test breakage
- Defer L2/L3 until production data justifies the rewrite

**L2** is right if Owner values architectural purity over short-term risk.
**L3** is right if Owner is willing to test deeply + accept "agent does too much" risk.
**L4 alone** doesn't change architecture, just speeds it up — always safe.

Decision is Owner's. Research first, then come back.

---

## 8. Out of scope (separate decisions)

- Streaming UX for synthesizer (token-level streaming) — independent of L1/L2/L3
- Model switching per role (L4) — orthogonal, can combine with any
- Caching identical queries — different concern (memo layer)
- Voice (TTS) — already wired, just needs VieNeu service running
- Web search latency (SearXNG) — separate optimization (engine tuning, caching)
