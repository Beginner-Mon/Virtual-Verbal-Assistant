#!/usr/bin/env python3
"""Stack Status Monitor -- quick health dashboard for all services + Ngrok."""

import json
import socket
import sys
import urllib.request

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Services to check --
SERVICES = [
    ("AgenticRAG API", "127.0.0.1", 8000, "/health"),
    ("Streamlit UI",   "127.0.0.1", 8501, None),
    ("ECA UI 2.0",     "127.0.0.1", 3000, None),
    ("Orchestrator",   "127.0.0.1", 8080, "/health"),
    ("DART",           "127.0.0.1", 5001, "/health"),
    ("ChromaDB",       "127.0.0.1", 8100, "/api/v1/heartbeat"),
    ("Redis",          "127.0.0.1", 6379, None),
]

NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def tcp_check(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
        return True
    except Exception:
        return False


def http_check(host: str, port: int, path: str, timeout: float = 2.0):
    try:
        url = f"http://{host}:{port}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def get_ngrok_tunnels() -> list:
    try:
        with urllib.request.urlopen(NGROK_API, timeout=2) as resp:
            data = json.loads(resp.read().decode())
        return data.get("tunnels", [])
    except Exception:
        return []


def main() -> int:
    W = 64
    bar = "=" * W
    print()
    print(bar)
    print("  AgenticRAG Stack Monitor".center(W))
    print(bar)

    # -- Service Checks --
    all_ok = True
    for name, host, port, health_path in SERVICES:
        port_open = tcp_check(host, port)
        icon = "[OK]" if port_open else "[--]"
        detail = ""

        if port_open and health_path:
            data = http_check(host, port, health_path)
            if data and "checks" in data:
                parts = []
                for svc, info in data["checks"].items():
                    s = info.get("status", "?")
                    parts.append(f"{svc}:{'ok' if 'ok' in s else 'FAIL'}")
                detail = f"  ({', '.join(parts)})"
            elif data and "status" in data:
                detail = f"  ({data['status']})"

        if not port_open:
            all_ok = False

        print(f"  {icon}  {name:<20s}  :{port:<5d}{detail}")

    # -- Ngrok Tunnels --
    print(bar)
    tunnels = get_ngrok_tunnels()
    if tunnels:
        print("  Ngrok Tunnels:")
        for t in tunnels:
            public = t.get("public_url", "?")
            local = t.get("config", {}).get("addr", "?")
            print(f"    {public}")
            print(f"      -> {local}")
    else:
        print("  Ngrok: not running")

    # -- Summary --
    print(bar)
    if all_ok:
        print("  >>> All services operational <<<")
    else:
        print("  >>> WARNING: Some services are down <<<")
    print(bar)
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())