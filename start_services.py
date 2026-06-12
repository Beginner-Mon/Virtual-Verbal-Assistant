"""Launch all VVA dev services, each in its own PowerShell window.

Usage:
    python start_services.py                # start everything
    python start_services.py --only backend frontend
    python start_services.py --skip tts     # start all except TTS
    python start_services.py --list         # show services and exit

Notes:
- Kimodo + web_search MCP servers are NOT here — the backend spawns them as
  stdio subprocesses automatically (see config/mcp_servers.yaml).
- Docker container deps (PostgreSQL/Redis/SearXNG) come up together via the one
  `docker` service below. Docker Desktop must already be running.
- Each window stays open after the process exits (-NoExit) so you can read logs
  / errors. Close the window to stop that service.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Service registry ─────────────────────────────────────────────────────────
# Each service: a shell command to run inside a fresh PowerShell window.
#   cwd        : working dir (relative to repo root)
#   conda_env  : conda env to activate first (None = no activation)
#   command    : the actual command to run
#   note       : printed in --list
SERVICES: dict[str, dict] = {
    "docker": {
        "cwd": ".",
        "conda_env": None,
        "command": "docker compose -f docker-compose.langgraph.yml up",
        "note": "PostgreSQL :5432 + Redis :6379 + SearXNG :6666 (needs Docker Desktop)",
    },
    "backend": {
        "cwd": "agenticRAG/agentic_rag_gemini",
        "conda_env": "firstconda",
        "command": "python -m uvicorn langgraph_agents.api.main:create_app --factory --port 8080",
        "note": "LangGraph FastAPI :8080 (spawns Kimodo + web_search MCP itself)",
    },
    "frontend": {
        "cwd": "ECA_UI",
        "conda_env": None,
        "command": "python -m http.server 3000",
        "note": "ECA UI :3000  →  http://localhost:3000/?api_base=http://localhost:8080  (test-ui at /test-ui/)",
    },
    "tts": {
        "cwd": "SpeechLLm",
        "conda_env": "tts",
        "command": "python api_server.py",
        "note": "VieNeu TTS :5000",
    },
}

# Order matters: docker first (deps), then backend, then the rest.
DEFAULT_ORDER = ["docker", "backend", "frontend", "tts"]


def build_ps_command(svc: dict) -> str:
    """Build the -Command string run inside the new PowerShell window."""
    cwd = (ROOT / svc["cwd"]).resolve()
    parts = [f"Set-Location '{cwd}'"]
    if svc["conda_env"]:
        parts.append(f"conda activate {svc['conda_env']}")
    parts.append(svc["command"])
    return "; ".join(parts)


def launch(name: str, svc: dict) -> None:
    title = f"VVA · {name}"
    inner = build_ps_command(svc)
    # Set window title, then run the service command. -NoExit keeps the window
    # open so logs/errors stay visible after the process ends.
    ps = f"$host.UI.RawUI.WindowTitle = '{title}'; {inner}"
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command", ps],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    print(f"  ✓ launched {name:<9} → {svc['command']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="SVC", help="start only these services")
    ap.add_argument("--skip", nargs="+", metavar="SVC", help="start all except these")
    ap.add_argument("--list", action="store_true", help="list services and exit")
    args = ap.parse_args()

    if args.list:
        print("Available services:\n")
        for name in DEFAULT_ORDER:
            print(f"  {name:<9} {SERVICES[name]['note']}")
        return 0

    if sys.platform != "win32":
        print("This launcher uses PowerShell windows — Windows only.", file=sys.stderr)
        return 1

    selected = list(DEFAULT_ORDER)
    if args.only:
        selected = [s for s in DEFAULT_ORDER if s in args.only]
    if args.skip:
        selected = [s for s in selected if s not in args.skip]

    unknown = set((args.only or []) + (args.skip or [])) - set(SERVICES)
    if unknown:
        print(f"Unknown service(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        print(f"Valid: {', '.join(DEFAULT_ORDER)}", file=sys.stderr)
        return 1

    if not selected:
        print("Nothing to start.", file=sys.stderr)
        return 1

    print(f"Starting {len(selected)} service(s), each in its own PowerShell window:\n")
    for name in selected:
        launch(name, SERVICES[name])

    print("\nDone. Close a window to stop that service.")
    print("Health check once backend is up:  curl http://localhost:8080/health/detailed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
