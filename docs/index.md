---
title: "Embodied Conversational Agent — Docs Vault"
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

# Embodied Conversational Agent — Docs Vault

> This folder is treated as an **Obsidian vault**. Use `[[...]]` wiki links to navigate between notes, and rely on YAML frontmatter tags for filtering.

---

## Architecture

| Note | What it covers |
|------|--------------|
| [[system-overview]] | End-to-end system architecture, ports, services |
| [[architecture-review]] | Senior-architect critique, P0 issues, proposed architecture, D1-D9 roadmap |
| [[decision-response-nodes]] | Architecture decision — response generation pipeline |
| [[full-flow-predeploy]] | Full flow + optimization inventory (pre-deploy) |
| [[schema-redesign]] | PostgreSQL schema redesign plan |
| [[agentic-rag-internals]] | Deep dive into AgenticRAG: agents, pipeline, data flow |
| [[agentic-rag-refactor]] | May 2026 refactoring — new modules, tracing, circuit breaker |
| [[agents-catalog]] | Complete inventory of all agents and tools in the system |
| [[api-contract]] | Unified gateway endpoints, request/response schemas, task lifecycle |
| [[dart-architecture]] | DART motion synthesis — MVAE, diffusion, inference pipeline |

## Operations

| Note | What it covers |
|------|--------------|
| [[setup-guide]] | Prerequisites, conda envs, Docker, Redis, one-command launch |
| [[troubleshooting]] | Common failures, port checks, ffmpeg, ChromaDB, rate limits |
| [[speechllm-overview]] | Voice I/O, STT, emotion detection, TTS |
| [[deployment]] | Deployment guide |
| [[runbook]] | Operational runbook |

## Phases

| Note | What it covers |
|------|--------------|
| [[phase-2.5]] | Architecture refactor v2.2 → v2.4 |
| [[phase-3]] | MCP Servers + TTS at FastAPI layer |
| [[phase-3.5]] | Phase 3 finalize + simplification |
| [[phase-5]] | SSE streaming + session reopen + frontend refactor |
| [[phase-6-p0]] | Production hardening (P0) |
| [[phase-6.6-searxng]] | Replace DDG with self-hosted SearXNG |
| [[phase-6.7-web-toggle]] | Web search toggle (UI opt-in) |
| [[phase-6.10-predeploy]] | Pre-deploy hardening |

## Fixes & Features

| Note | What it covers |
|------|--------------|
| [[m9-closeout]] | M.9 closeout + security patch |
| [[memory-layer]] | Memory layer gaps fix |
| [[r1-gdpr-resummarize]] | R1 GDPR re-summarize + R2 tests |
| [[youtube-paste]] | YouTube paste-link Q&A |

## Tracking

| Note | What it covers |
|------|--------------|
| [[status]] | Status & roadmap |
| [[tech-debt]] | Tech debt & pending tasks |
| [[predeploy-audit]] | Pre-deploy audit |

## Plans

| Note | What it covers |
|------|--------------|
| [[v2.4-plan]] | Re-architecture plan v2.4.1 |
| [[reupdate-plan]] | Re-architecture plan (update) |
| [[mobile-ui-polish]] | Mobile UI polish plan |

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
