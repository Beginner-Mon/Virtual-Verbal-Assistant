---
title: "SpeechLLm Overview"
description: "Voice I/O pipeline: STT, emotion detection, local SLM reasoning, TTS."
tags:
  - speechllm
  - voice
  - stt
  - tts
  - emotion
  - ollama
---

# SpeechLLm Overview

> Location: `SpeechLLm/`  
> Port: 5000 (optional — Orchestrator degrades gracefully if unavailable)

## Pipeline

```
Voice / Text Input
    ↓
Speech-to-Text (Whisper / wav2vec2)
    ↓
Emotion Detection (audio + text features)
    ↓
Context Formatting (JSON prompt)
    ↓
Local SLM Reasoning (Phi-3 / Mistral / Gemma via Ollama)
    ↓
Text-to-Speech + Avatar control
```

## Key Modules

| Module | Path | Responsibility |
|--------|------|--------------|
| Orchestrator | `src/core/orchestrator.py` | Pipeline loop coordination |
| STT Stage | `src/stages/stt_stage.py` | Whisper / wav2vec2 transcription |
| LLM Stage | `src/stages/llm_stage.py` | Ollama SLM inference |
| TTS Stage | `src/stages/tts_stage.py` | Text-to-speech synthesis |
| Emotion Stage | `src/stages/emotion_stage.py` | Audio/text emotion classification |
| Services | `src/services/` | Whisper client, Phi-3 client, TTS engine |

## Notes

- This service is **optional**. If port 5000 is unreachable, the Orchestrator skips TTS and emotion enrichment.
- Runs its own conda environment (typically `tts`).

## Related Notes

- [[system-overview]] — Service topology
- [[api-contract]] — Gateway polling contract (includes optional `tts` stage)
- [[troubleshooting]] — If SpeechLLm fails to start

---

#speechllm #voice #stt #tts #emotion #ollama
