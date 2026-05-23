# Phase 4 — Conversation Agent + Persona System + Voice Identity

**Architect**: K | **Developer**: N | **Date**: 2026-05-21
**Branch**: `feature/langgraph-rewrite` (continue from Phase 3)
**Estimated time**: ~4-6h

---

## Overview

Implement the Conversation Agent (persona-aware styling) and the full persona system. Each persona defines both **text personality** (tone, rules, formatting) and **voice identity** (voice sample for VieNeu-TTS cloning). ConversationAgent loads the selected persona, applies it as LLM system prompt to style `raw_answer`, and passes `voice_path` downstream to dispatch.

**Key insight**: Owner has already built VieNeu-TTS with zero-shot voice cloning support (`voice_path` parameter in `SpeechLLm/api_server.py`). Phase 4 wires persona voice identity into that existing infrastructure.

**Critical rule**: Do NOT modify `SpeechLLm/` code. Persona voice integration flows through dispatch → Celery task → existing VieNeu-TTS API.

---

## New Files to Create

```
langgraph_agents/
  nodes/
    conversation.py              # Stub → real: persona loading + LLM styling
  personas/
    eca_default.md               # Update: add Voice Identity section
    eca_friendly.md              # NEW: casual, approachable persona
    eca_clinical.md              # NEW: formal, clinical persona

tests/langgraph_agents/
  test_phase4_conversation.py    # Tests for persona loading + styling + voice
```

## Files to Modify

```
langgraph_agents/nodes/conversation.py    # Stub → real implementation
langgraph_agents/nodes/dispatch.py        # Pass voice_path to TTS task
langgraph_agents/personas/eca_default.md  # Add Voice Identity section
langgraph_agents/state.py                 # Add voice_path field
langgraph_agents/services/vieneu_tts/tasks.py  # Accept voice_path parameter
config/langgraph.yaml                     # Add persona section
```

---

## Task 1: Update persona MD format — add Voice Identity section

### eca_default.md (update existing)

```markdown
# ECA Default

## Identity
Name: ECA | Role: Physical therapy AI assistant | Avatar: eca_default.png

## Voice Identity
voice_path: "voices/eca_default.wav"
language: vi

## Personality
Tone: Warm, professional, encouraging | Formality: Semi-formal

## Behavioral Rules
- Acknowledge pain before suggesting exercises
- Use anatomical terms with plain-language explanations
- End exercise recs with safety reminders
- Refer to medical professionals for anything beyond wellness
- Use Vietnamese by default, switch to English if user writes in English

## Response Formatting
- Bullet points for exercise lists, include rep/set counts
- Bold safety warnings
- Keep under 300 words
```

### eca_friendly.md (new)

```markdown
# ECA Friendly

## Identity
Name: ECA Buddy | Role: Fitness & wellness companion | Avatar: eca_friendly.png

## Voice Identity
voice_path: "voices/eca_friendly.wav"
language: vi

## Personality
Tone: Casual, cheerful, motivating | Formality: Informal

## Behavioral Rules
- Use casual Vietnamese ("ban" instead of formal "quý khách")
- Add encouragement ("Giỏi lắm!", "Cố lên nha!")
- Simplify medical terms to everyday language
- Still include safety warnings but in friendly tone
- Use emoji sparingly in text responses

## Response Formatting
- Short paragraphs, conversational style
- Use "→" for exercise steps instead of bullet points
- Keep under 200 words
```

### eca_clinical.md (new)

```markdown
# ECA Clinical

## Identity
Name: Dr. ECA | Role: Clinical rehabilitation advisor | Avatar: eca_clinical.png

## Voice Identity
voice_path: "voices/eca_clinical.wav"
language: vi

## Personality
Tone: Authoritative, precise, measured | Formality: Formal

## Behavioral Rules
- Use proper anatomical terminology with Vietnamese medical terms
- Cite evidence basis when available ("Theo nghiên cứu...")
- Always recommend professional consultation for complex cases
- Include contraindications and precautions prominently
- Structure response as clinical assessment

## Response Formatting
- Structured sections: Assessment → Recommendation → Precautions
- Medical terminology with explanations in parentheses
- Keep under 400 words for thorough clinical responses
```

**Note to N**: Voice sample `.wav` files don't exist yet — Owner will record them separately using `SpeechLLm/record_voice.py`. Code must handle missing voice files gracefully (fall back to VieNeu default/preset voice).

---

## Task 2: Update `config/langgraph.yaml` — add persona section

Append under `langgraph:`:

```yaml
  persona:
    default: "eca_default"
    personas_dir: "langgraph_agents/personas"
    styling_model: "gemini-2.5-flash"     # model for persona styling LLM call
    styling_temperature: 0.7
    styling_max_tokens: 4096
    voices_dir: "voices"                   # base directory for voice .wav files
```

---

## Task 3: Implement persona loader — helper functions in `conversation.py`

### Persona file parser

```python
def _load_persona(persona_id: str) -> dict:
```

**Logic**:
1. Build path: `{personas_dir}/{persona_id}.md`
2. Read file content as string
3. Parse sections by `## Header` markers into a dict:
   - `identity` → raw text under `## Identity`
   - `voice_identity` → parse `voice_path:` and `language:` values
   - `personality` → raw text under `## Personality`
   - `behavioral_rules` → raw text under `## Behavioral Rules`
   - `response_formatting` → raw text under `## Response Formatting`
4. Return dict: `{"persona_id": ..., "identity": ..., "voice_identity": {"voice_path": ..., "language": ...}, "personality": ..., "behavioral_rules": ..., "response_formatting": ...}`
5. If file not found → log warning, return default fallback dict (hardcoded minimal persona)

**Parser approach**: Simple regex/split on `## ` headers. No need for a markdown library. Key-value fields like `voice_path: "..."` use basic string parsing.

### Persona cache

Module-level dict cache to avoid re-reading files every call:

```python
_persona_cache: dict[str, dict] = {}

def _get_persona(persona_id: str) -> dict:
    if persona_id not in _persona_cache:
        _persona_cache[persona_id] = _load_persona(persona_id)
    return _persona_cache[persona_id]
```

### Build system prompt from persona

```python
def _build_persona_prompt(persona: dict, intent: str) -> str:
```

Construct a system prompt string that instructs the LLM to restyle `raw_answer` according to persona rules. Template:

```
You are {identity}.

## Your Personality
{personality}

## Rules
{behavioral_rules}

## Formatting
{response_formatting}

## Task
Restyle the following clinical response to match your personality and formatting rules.
Do NOT add new medical information — only restyle what is given.
Do NOT remove safety warnings — rephrase them in your tone.
Preserve all factual content.
If the response is empty or just a greeting, respond naturally in character.
Respond in the same language as the original response.
```

---

## Task 4: Implement `conversation_node` — real LLM persona styling

Replace the stub in `conversation.py`:

```python
async def conversation_node(state: AgentState) -> dict:
```

**Logic**:
1. Read `persona_id` from state (default: config default)
2. Load persona via `_get_persona(persona_id)`
3. Read `raw_answer` from state
4. If `raw_answer` is empty → return `{"final_answer": ""}` immediately (no LLM call)
5. Build system prompt via `_build_persona_prompt(persona, intent)`
6. Call LLM via `LLMGateway`:
   ```python
   gateway = LLMGateway(model=_STYLING_MODEL)
   styled = await gateway.chat(
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": raw_answer},
       ],
       temperature=_STYLING_TEMP,
       max_tokens=_STYLING_MAX_TOKENS,
   )
   ```
7. Extract `voice_path` from persona's `voice_identity` section
8. Return:
   ```python
   {
       "final_answer": styled,
       "voice_path": voice_path,  # passed to dispatch for TTS
   }
   ```

**Error handling**: If LLM call fails → RECOVERABLE error, fall back to `raw_answer` as `final_answer` (unstyled but functional).

**Config** (module-level, from langgraph.yaml):
```python
_STYLING_MODEL = _CFG.get("persona", {}).get("styling_model", "gemini-2.5-flash")
_STYLING_TEMP = _CFG.get("persona", {}).get("styling_temperature", 0.7)
_STYLING_MAX_TOKENS = _CFG.get("persona", {}).get("styling_max_tokens", 4096)
```

---

## Task 5: Update `langgraph_agents/state.py` — add `voice_path`

Add field to `AgentState`:

```python
# Conversation output (post-persona)
final_answer: str
voice_path: Optional[str]          # NEW — persona voice sample path for TTS
```

This field is set by `conversation_node` and read by `dispatch_node`.

---

## Task 6: Update `dispatch.py` — pass `voice_path` to TTS task

Current speech dispatch (line 60-63):

```python
task = celery_app.send_task(
    "langgraph.synthesize_speech",
    args=(final_answer, request_id, session_id),
)
```

**Change to**:

```python
voice_path = state.get("voice_path")  # from conversation_node
task = celery_app.send_task(
    "langgraph.synthesize_speech",
    args=(final_answer, request_id, session_id),
    kwargs={"voice_path": voice_path},
)
```

Using `kwargs` so existing task signature doesn't break if `voice_path` is None.

---

## Task 7: Update `vieneu_tts/tasks.py` — accept and forward `voice_path`

Update task signature:

```python
def synthesize_speech(self, text, request_id, session_id, voice_path=None):
```

Update the client call:

```python
# Build payload with optional voice_path
call_kwargs = {"text": text}
if voice_path:
    call_kwargs["voice_path"] = voice_path

client = get_vieneu_tts_client()
result = client.synthesize_sync(**call_kwargs)
```

Update `VieNeuTTSClient.synthesize_sync()` to accept `voice_path`:

```python
def synthesize_sync(self, text: str, voice_path: str = None) -> dict:
    payload = {"text": text}
    if voice_path:
        payload["voice_path"] = voice_path
    # ... rest unchanged
```

Same for async `synthesize()` method.

**Owner's VieNeu-TTS API** already accepts `voice_path` in `TTSRequest` — no server changes needed.

---

## Task 8: Write tests — `tests/langgraph_agents/test_phase4_conversation.py`

### Unit tests (no services needed, 10 tests)

1. **`test_load_persona_default`** — load `eca_default.md` → verify all sections parsed correctly
2. **`test_load_persona_friendly`** — load `eca_friendly.md` → verify different tone/rules
3. **`test_load_persona_clinical`** — load `eca_clinical.md` → verify clinical sections
4. **`test_load_persona_missing`** — load nonexistent persona → returns fallback dict, no crash
5. **`test_parse_voice_identity`** — verify `voice_path` and `language` extracted from `## Voice Identity`
6. **`test_parse_voice_identity_missing`** — persona without `## Voice Identity` → `voice_path=None`
7. **`test_build_persona_prompt`** — verify prompt contains identity, rules, formatting sections
8. **`test_conversation_empty_raw_answer`** — `raw_answer=""` → `final_answer=""`, no LLM call
9. **`test_persona_cache`** — second call to `_get_persona` returns cached (same object identity)
10. **`test_conversation_fallback_on_error`** — LLM fails → `final_answer = raw_answer` (unstyled)

### Integration tests (require GEMINI_API_KEY, 4 tests)

11. **`test_conversation_styling_default`** — real LLM with eca_default persona → output differs from raw_answer, retains medical facts
12. **`test_conversation_styling_friendly`** — eca_friendly persona → output more casual than default
13. **`test_conversation_different_personas_different_output`** — same raw_answer through 2 different personas → different styled outputs
14. **`test_voice_path_in_state`** — conversation_node returns `voice_path` matching persona's voice_identity

### Full graph test (1 test)

15. **`test_full_graph_persona_styling`** — full graph invocation → `final_answer` is styled (not identical to `raw_answer`), `voice_path` present in output state

### Mocking strategy

For unit tests: mock `LLMGateway.chat` to return a fixed styled string. Verify the system prompt passed to mock contains persona sections.

For integration tests: use real Gemini API. Compare outputs qualitatively (styled output should contain same facts as raw_answer but different wording).

---

## Task 9: Verify dispatch integration

After all changes, verify the full chain manually:

1. `conversation_node` loads persona → LLM styles `raw_answer` → sets `final_answer` + `voice_path`
2. `dispatch_node` reads `voice_path` from state → passes to `synthesize_speech` task
3. `synthesize_speech` task forwards `voice_path` to VieNeu-TTS client
4. VieNeu-TTS client includes `voice_path` in POST payload to server

Run existing Phase 3 tests to confirm no regressions:
```
pytest tests/langgraph_agents/test_phase3_dispatch.py -m unit -v
```

---

## Acceptance Criteria

1. `conversation_node` loads persona from MD file by `persona_id`
2. Persona MD files contain `## Voice Identity` with `voice_path` field
3. ConversationAgent calls LLM to restyle `raw_answer` per persona rules
4. If LLM fails → fallback to unstyled `raw_answer` (RECOVERABLE, not CRITICAL)
5. If persona file missing → fallback to hardcoded minimal persona, warning logged
6. `voice_path` flows: persona MD → conversation_node → state → dispatch → Celery task → VieNeu-TTS API
7. Missing voice `.wav` files → `voice_path=None` passed → VieNeu uses default/preset voice
8. Persona cache avoids re-reading MD files on every request
9. 3 personas created: `eca_default`, `eca_friendly`, `eca_clinical`
10. All unit tests pass: `pytest tests/langgraph_agents/test_phase4_conversation.py -m unit -v`
11. Phase 3 tests still pass (no regressions)

---

## Execution Order

| Step | Task | Est. |
|------|------|------|
| 1 | Task 1: Update persona MD files (3 files) | 15m |
| 2 | Task 2: Update config/langgraph.yaml | 5m |
| 3 | Task 5: Update state.py (add voice_path) | 5m |
| 4 | Task 3: Implement persona loader helpers | 45m |
| 5 | Task 4: Implement conversation_node | 45m |
| 6 | Task 7: Update vieneu_tts client + task (voice_path) | 20m |
| 7 | Task 6: Update dispatch.py (voice_path) | 10m |
| 8 | Task 8: Write tests | 1.5h |
| 9 | Task 9: Verify dispatch integration + regression | 30m |
| | **Total** | **~4-5h** |

---

## Architecture Note

```
User query → ... → Validator (raw_answer)
                        ↓
              Conversation Node
              ├─ Load persona MD (cached)
              ├─ Build system prompt from persona sections
              ├─ LLM call: restyle raw_answer → final_answer
              └─ Extract voice_path from persona
                        ↓
              Dispatch Node
              ├─ Speech: send_task("synthesize_speech", text, ..., voice_path=...)
              └─ Motion: store pending (unchanged)
                        ↓
              Celery Worker
              └─ VieNeu-TTS: POST /synthesize {text, voice_path}
                             → voice cloning with persona voice sample
```
