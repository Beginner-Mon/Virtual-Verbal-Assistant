import argparse
import os
import re
import subprocess
from typing import Dict, Iterable, List, Set

import psutil


DEFAULT_TARGET_PORTS = [8000, 8501, 3000, 8080, 5001, 8100, 5000]


def _terminate_windows_pid(pid: int, reason: str, dry_run: bool = False) -> bool:
    try:
        proc = psutil.Process(pid)
        name = proc.name()
    except psutil.NoSuchProcess:
        return False

    print(f"[WIN] {'Would stop' if dry_run else 'Stopping'} {name} (PID {pid}) | {reason}")
    if dry_run:
        return True

    try:
        proc.terminate()
        proc.wait(timeout=3)
        return True
    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
        pass
    except psutil.AccessDenied:
        pass

    try:
        proc.kill()
        proc.wait(timeout=3)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
        return False


def kill_windows_processes_on_ports(ports: Iterable[int], dry_run: bool = False) -> Set[int]:
    target_ports = set(int(p) for p in ports)
    pid_to_ports: Dict[int, Set[int]] = {}

    for conn in psutil.net_connections(kind="inet"):
        if not conn.laddr or conn.pid is None:
            continue
        if conn.laddr.port not in target_ports:
            continue
        pid_to_ports.setdefault(conn.pid, set()).add(conn.laddr.port)

    killed: Set[int] = set()
    for pid, bound_ports in sorted(pid_to_ports.items()):
        if pid <= 4 or pid == os.getpid():
            continue
        reason = f"bound to ports {sorted(bound_ports)}"
        if _terminate_windows_pid(pid, reason, dry_run=dry_run):
            killed.add(pid)

    return killed


def kill_windows_processes_in_cwd(target_cwd: str, dry_run: bool = False, exclude_pids: Set[int] | None = None) -> Set[int]:
    exclude = exclude_pids or set()
    normalized_target = os.path.normcase(os.path.abspath(target_cwd))
    killed: Set[int] = set()

    for proc in psutil.process_iter(["pid", "name", "cwd"]):
        try:
            pid = proc.info["pid"]
            if pid in exclude:
                continue
            if pid <= 4 or pid == os.getpid():
                continue
            cwd = proc.info.get("cwd")
            if not cwd:
                continue
            normalized_cwd = os.path.normcase(os.path.abspath(cwd))
            if normalized_cwd != normalized_target:
                continue
            name = str(proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            reason = f"running from {normalized_cwd}"
            if _terminate_windows_pid(pid, reason, dry_run=dry_run):
                killed.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue

    return killed


def kill_windows_processes_by_name(name_tokens: Iterable[str], dry_run: bool = False, exclude_pids: Set[int] | None = None) -> Set[int]:
    exclude = exclude_pids or set()
    lowered_tokens = [str(token).lower() for token in name_tokens if str(token).strip()]
    killed: Set[int] = set()

    if not lowered_tokens:
        return killed

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid in exclude:
                continue
            if pid <= 4 or pid == os.getpid():
                continue

            name = str(proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            haystack = f"{name} {cmdline}".strip()

            if not any(token in haystack for token in lowered_tokens):
                continue

            reason = f"matched token(s) {lowered_tokens}"
            if _terminate_windows_pid(pid, reason, dry_run=dry_run):
                killed.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
            continue

    return killed


def find_wsl_pids_by_port(port: int) -> List[int]:
    try:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "ss -ltnp"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    pids: Set[int] = set()
    port_token = f":{port}"
    for line in result.stdout.splitlines():
        if port_token not in line:
            continue
        for match in re.finditer(r"pid=(\d+)", line):
            pids.add(int(match.group(1)))

    return sorted(pids)


def kill_wsl_processes_on_ports(ports: Iterable[int], dry_run: bool = False) -> int:
    kill_count = 0
    for port in ports:
        pids = find_wsl_pids_by_port(int(port))
        if not pids:
            continue
        for pid in pids:
            print(f"[WSL] {'Would stop' if dry_run else 'Stopping'} PID {pid} on port {port}")
            if dry_run:
                kill_count += 1
                continue
            result = subprocess.run(["wsl", "-e", "bash", "-lc", f"kill -9 {pid}"], check=False)
            if result.returncode == 0:
                kill_count += 1
    return kill_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup stack services on Windows + WSL")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be killed without terminating processes")
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        default=DEFAULT_TARGET_PORTS,
        help="Target ports to clean up",
    )
    args = parser.parse_args()

    print(f"[Cleanup] Starting cleanup for ports: {args.ports}")

    # 1) Kill Windows listeners on target ports (includes SpeechLLM when on :5000).
    killed_windows = kill_windows_processes_on_ports(args.ports, dry_run=args.dry_run)

    # 2) Extra guard: kill python processes started from SpeechLLm folder.
    speech_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpeechLLm")
    killed_speech = kill_windows_processes_in_cwd(speech_root, dry_run=args.dry_run, exclude_pids=killed_windows)

    # 3) Kill WSL listeners on target ports (DART on :5001).
    killed_ngrok = kill_windows_processes_by_name(
        ["ngrok.exe", "ngrok"],
        dry_run=args.dry_run,
        exclude_pids=(killed_windows | killed_speech),
    )

    # 4) Kill WSL listeners on target ports (DART on :5001).
    killed_wsl_count = kill_wsl_processes_on_ports(args.ports, dry_run=args.dry_run)

    total_windows = len(killed_windows | killed_speech | killed_ngrok)
    print(f"[Cleanup] Windows processes stopped: {total_windows}")
    print(f"[Cleanup] WSL processes stopped: {killed_wsl_count}")
    print("[Cleanup] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())