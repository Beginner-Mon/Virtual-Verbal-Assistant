#!/bin/bash
# Entry point for the CRUD API under the AWS Lambda Web Adapter.
#
# LWA runs this instead of a Lambda handler: it starts a real HTTP server in the
# sandbox and proxies each invocation to it as an ordinary request. That is the
# point — the app keeps one process, one event loop and one asyncpg pool for the
# life of the container. An ASGI adapter that drives the app per invocation
# (Mangum) manages the loop itself, and the pool cached in a module global then
# belongs to a loop that has already closed by the second warm request; the
# failure only appears in production, under reuse.
#
# Handler is set to "run.sh" and AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap in the
# stack. See infra/infra/crud_api_stack.py.

set -euo pipefail

# /var/task is on sys.path for the managed Python runtime, but the wrapper
# starts us through bash, so say it explicitly rather than rely on inheritance.
export PYTHONPATH="/var/task:${PYTHONPATH:-}"

# Access logging off: Lambda already writes a REPORT line per invocation with
# duration and billed time, and CloudWatch ingestion is charged per GB against a
# $10/month budget. Errors and the service's own structured logs still go out.
exec python -m uvicorn langgraph_agents.api.crud_app:create_crud_app \
    --factory \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --no-access-log
