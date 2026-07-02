# VVA — Deployment Guide (Phase 7+)

> **Status**: Stub — Phase 7 Hybrid Edge-Cloud deployment is not yet implemented.
> For single-machine deployment, see [RUNBOOK.md](./RUNBOOK.md).

## Target Architecture (Phase 7)

```
[CloudFront CDN] → [VPS: FastAPI + PostgreSQL + Redis]
                         ↓
              [Edge: HP ProDesk 48GB RAM, RTX 3060]
              (Kimodo GPU render, VieNeu-TTS, Celery worker)
```

## Planned Sections

- **§1 VPS provisioning** — Ubuntu 22.04, Docker, firewall (ufw)
- **§2 Supabase migration** — PostgreSQL + pgvector, connection string
- **§3 CloudFront setup** — CDN distribution, SSL cert, ECA_UI static hosting
- **§4 Edge worker setup** — Windows/WSL, GPU driver, Celery worker service
- **§5 CI/CD** — GitHub Actions deploy on merge to `main`
- **§6 Monitoring** — Prometheus metrics, Grafana dashboards, alerting

## Current (Phase 6)

Deploy is single-machine only. See [RUNBOOK.md](./RUNBOOK.md) for step-by-step instructions.

## TODO

- [ ] VPS provider selection (Hetzner / Vultr / DigitalOcean)
- [ ] Supabase project setup + migration
- [ ] CloudFront distribution + SSL
- [ ] Celery worker registration on Edge machine
- [ ] Zero-downtime deploy script
