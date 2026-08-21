#!/bin/bash
# Entry point for vva-agent under the AWS Lambda Web Adapter.
#
# Same reasoning as infra/lambda/crud_api/run.sh: LWA runs a real HTTP server in
# the sandbox and proxies each invocation to it, so the app keeps ONE process,
# ONE event loop and ONE asyncpg pool for the life of the container. An ASGI
# adapter that drives the app per invocation (Mangum) manages the loop itself,
# and a pool cached in a module global then belongs to a loop that has already
# closed by the second warm request — a failure that only appears in production,
# under reuse.
#
# Unlike the CRUD function this is a CONTAINER, so there is no
# AWS_LAMBDA_EXEC_WRAPPER and no /opt/bootstrap: LWA is copied into
# /opt/extensions in the Dockerfile and starts as an extension. Setting the
# wrapper here would be the zip recipe applied to the wrong packaging.
set -euo pipefail

# Mirrors the repo layout: several modules resolve config/ as
# `Path(__file__).parents[3]`, so langgraph_agents must sit one directory deep
# inside /var/task. See the Dockerfile.
export PYTHONPATH="/var/task/agenticRAG:${PYTHONPATH:-}"

# Access logging off: Lambda already writes a REPORT line per invocation, and
# CloudWatch ingestion is billed per GB. The service's own structured logs and
# errors still go out.
exec python -m uvicorn langgraph_agents.api.main:create_app \
    --factory \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --no-access-log
