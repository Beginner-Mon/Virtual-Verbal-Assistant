---
title: "Virtual Verbal Assistant — Docs Vault"
description: "Obsidian-style hub for all project documentation."
tags:
  - index
  - vault
  - overview
  - agentic-rag
  - dart
  - speechllm
  - api
  - architecture
date: 2026-05-09
---

# Virtual Verbal Assistant — Docs Vault

> This folder is treated as an **Obsidian vault**. Use `[[...]]` wiki links to navigate between notes, and rely on YAML frontmatter tags for filtering.

---

## Quick Links

| Note | What it covers |
|------|--------------|
| [[system_overview]] | End-to-end system architecture, ports, services |
| [[agentic_rag_internals]] | Deep dive into AgenticRAG: agents, pipeline, data flow |
| [[agents_catalog]] | Complete inventory of all agents and tools in the system |
| [[ARCHITECTURE_REVIEW]] | Senior-architect critique, P0 issues, proposed architecture, D1-D9 roadmap |
| [[agentic_rag_refactor]] | May 2026 refactoring — new modules, tracing, circuit breaker, resource guard |
| [[api_contract]] | Unified gateway endpoints, request/response schemas, task lifecycle |
| [[setup_guide]] | Prerequisites, conda envs, Docker, Redis, one-command launch |
| [[troubleshooting]] | Common failures, port checks, ffmpeg, ChromaDB, rate limits |
| [[dart_architecture]] | DART motion synthesis — MVAE, diffusion, inference pipeline |
| [[speechllm_overview]] | Voice I/O, STT, emotion detection, TTS |

---

## Service Map

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  ECA UI      │────▶│ Unified Gateway │────▶│ AgenticRAG   │
│  (port 3000) │     │   (port 8000)   │     │  (port 8080) │
└──────────────┘     └────────┬────────┘     └──────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌────────┐     ┌────────┐     ┌──────────┐
         │  DART  │     │SpeechLL│     │  Redis   │
         │(5001)  │     │(5000)  │     │ (6379)   │
         └────────┘     └────────┘     └──────────┘
```

---

## Tags

#index #vault #overview #agentic-rag #dart #speechllm #api #architecture
