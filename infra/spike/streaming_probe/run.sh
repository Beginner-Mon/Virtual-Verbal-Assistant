#!/bin/bash
# Entry point for the streaming probe under the AWS Lambda Web Adapter.
#
# Same shape as infra/lambda/crud_api/run.sh — the handler is the script that
# starts an HTTP server, and /opt/bootstrap (from the LWA layer) execs it.
set -euo pipefail

export PYTHONPATH="/var/task:${PYTHONPATH:-}"

exec python -m uvicorn app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --no-access-log
