import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = REPO_ROOT / "agenticRAG" / "agentic_rag_gemini"
SPEECH_LLM_ROOT = REPO_ROOT / "SpeechLLm"
DART_ROOT = REPO_ROOT / "text-to-motion" / "DART"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = 8000
MAIN_API_PORT = 8080
SPEECH_LLM_PORT = int(os.getenv("SPEECH_LLM_PORT", "5000"))
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
DART_HOST = "127.0.0.1"
DART_PORT = 5001
DART_START_TIMEOUT_SECONDS = 240
UI_PORT = int(os.getenv("AGENTICRAG_UI_PORT", "8501"))
ECA_UI_PORT = int(os.getenv("ECA_UI_PORT", "3000"))
ECA_UI_ROOT = REPO_ROOT / "ECA_UI"
REQUIRED_PY_MODULES = ("celery", "fastapi", "redis")


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def kill_ports(ports: list[int]) -> None:
    """Kill processes on multiple ports in parallel."""
    if not ports:
        return

    def _kill_one(port: int) -> None:
        try:
            if os.name == "nt":
                cmd = (
                    f'powershell -NoProfile -Command "'
                    f'Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue '
                    f'| Select-Object -ExpandProperty OwningProcess '
                    f'| ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"'
                )
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(
                    f"fuser -k {port}/tcp",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    # ── Kill all ports in parallel ──────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        list(pool.map(_kill_one, ports))


def cleanup_everything() -> None:
    """Hard-reset all relevant ports and WSL processes before startup."""
    print("[Stack] Performing global cleanup phase...")

    # Port kills + WSL pkill run fully in parallel
    cleanup_futures: list = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        cleanup_futures.append(
            pool.submit(kill_ports, [8000, 8080, 5000, 5001, 3000, 8501, 6379])
        )
        cleanup_futures.append(
            pool.submit(
                subprocess.run,
                ["wsl", "-e", "pkill", "-f", "python"],
                dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
            )
        )
        cleanup_futures.append(
            pool.submit(
                subprocess.run,
                ["wsl", "-e", "pkill", "-f", "api_server.py"],
                dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
            )
        )
        for f in as_completed(cleanup_futures):
            try:
                f.result()
            except Exception:
                pass

    print("[Stack] Global cleanup complete.")


def probe_host(host: str) -> str:
    if host in {"0.0.0.0", "::", "*", ""}:
        return "127.0.0.1"
    return host


def reader_thread(
    service_name: str,
    process: subprocess.Popen,
    log_tail: dict[str, deque[str]],
    log_lock: threading.Lock,
) -> None:
    if process.stdout is None:
        return
    try:
        for line in process.stdout:
            if not line:
                break
            cleaned = line.rstrip()
            with log_lock:
                log_tail[service_name].append(cleaned)
            try:
                print(f"[{service_name}] {cleaned}")
            except UnicodeEncodeError:
                print(f"[{service_name}] {cleaned.encode('ascii', 'replace').decode('ascii')}")
    except Exception:
        pass


def start_process(name: str, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    print(f"[Stack] starting {name}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


def windows_to_wsl_path(path: Path) -> str:
    """Convert Windows path to WSL path without spawning a subprocess."""
    resolved = str(path.resolve())
    # Fast pure-Python conversion: C:\foo\bar → /mnt/c/foo/bar
    if len(resolved) >= 2 and resolved[1] == ":":
        drive = resolved[0].lower()
        rest = resolved[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    # Fallback for UNC / already-Unix paths
    try:
        out = subprocess.check_output(
            ["wsl", "wslpath", "-a", resolved],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return resolved.replace("\\", "/")


def resolve_wsl_ipv4() -> str | None:
    """Return the first IPv4 reported by WSL, if available."""
    try:
        out = subprocess.check_output(
            ["wsl", "-e", "bash", "-lc", "hostname -I"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception:
        return None

    match = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", out)
    return match.group(0) if match else None


def wait_for_service_ready(
    host: str, port: int, timeout_seconds: int, label: str, path: str = "/health"
) -> bool:
    """Poll until the service responds HTTP 200 on `path`."""
    import http.client

    deadline = time.time() + timeout_seconds
    probe = "127.0.0.1" if host in ("0.0.0.0", "::", "*", "") else host

    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(probe, port, timeout=1.0)
            conn.request("GET", path)
            resp = conn.getresponse()
            if resp.status == 200:
                print(f"[Stack] {label} is READY on {probe}:{port}")
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def wait_for_port(host: str, port: int, timeout_seconds: int, label: str) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_port_open(host, port):
            print(f"[Stack] {label} is ready on {host}:{port}")
            return True
        time.sleep(1.0)
    return False


def python_has_modules(python_exe: str, modules: tuple[str, ...]) -> bool:
    probe = "import " + ", ".join(modules)
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve_python_executable() -> str:
    """
    Resolve the best Python executable.

    Optimisation: check whether sys.executable already has the required
    modules first (zero extra subprocess spawns in the common case where
    the user runs the script from the right environment).
    """
    # Fast-path: current interpreter already has everything we need
    if python_has_modules(sys.executable, REQUIRED_PY_MODULES):
        return sys.executable

    candidates: list[str] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "python.exe"))

    default_firstconda = Path.home() / "miniconda3" / "envs" / "firstconda" / "python.exe"
    candidates.append(str(default_firstconda))

    seen: set[str] = set()
    for exe in candidates:
        if not exe or exe in seen or exe == sys.executable:
            continue
        seen.add(exe)
        if Path(exe).exists() and python_has_modules(exe, REQUIRED_PY_MODULES):
            return exe

    return sys.executable


def enrich_path_for_ffmpeg(env: dict[str, str]) -> dict[str, str]:
    if shutil.which("ffmpeg", path=env.get("PATH")):
        return env

    candidate_dirs = [
        Path.home() / "scoop" / "shims",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
    ]

    found_bin = None
    for base in candidate_dirs:
        if not base.exists():
            continue
        if base.name.lower() == "packages":
            for ffmpeg_exe in base.rglob("ffmpeg.exe"):
                found_bin = ffmpeg_exe.parent
                break
        else:
            ffmpeg_exe = base / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                found_bin = ffmpeg_exe.parent
        if found_bin is not None:
            break

    if found_bin is not None:
        env = env.copy()
        current_path = env.get("PATH", "")
        env["PATH"] = f"{current_path}{os.pathsep}{str(found_bin)}"
        print(f"[Stack] ffmpeg not in PATH, added fallback: {found_bin}")

    return env


def resolve_conda_executable() -> str | None:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "Scripts" / "conda.exe"),
        r"C:\Miniconda\Scripts\conda.exe",
    ]

    seen: set[str] = set()
    for exe in candidates:
        if not exe or exe in seen:
            continue
        seen.add(exe)
        if Path(exe).exists():
            return exe

    return None


def terminate_process(name: str, process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print(f"[Stack] terminating {name} (pid={process.pid})")
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"[Stack] force-killing {name} (pid={process.pid})")
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    health_probe_host = probe_host(API_HOST)

    # ── Resolve slow one-time things in parallel ────────────────────────────
    print("[Stack] Resolving environment (parallel)...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_py    = pool.submit(resolve_python_executable)
        f_conda = pool.submit(resolve_conda_executable)
        f_wsl   = pool.submit(resolve_wsl_ipv4)

    py_exe   = f_py.result()
    conda_exe = f_conda.result()
    dart_wsl_ipv4 = f_wsl.result()

    base_env = os.environ.copy()
    speech_llm_conda_env = os.getenv("SPEECH_LLM_CONDA_ENV", "tts")
    speech_llm_python = os.getenv("SPEECH_LLM_PYTHON", "python")
    base_env["MOTION_ASYNC_ENABLED"] = "true"
    base_env["API_PUBLIC_BASE_URL"] = os.getenv(
        "API_PUBLIC_BASE_URL",
        f"http://{API_HOST}:{API_PORT}",
    )
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env["PYTHONUTF8"] = "1"
    base_env = enrich_path_for_ffmpeg(base_env)

    cleanup_everything()  # already parallelised internally

    dart_probe_hosts = [DART_HOST]
    if dart_wsl_ipv4 and dart_wsl_ipv4 not in dart_probe_hosts:
        dart_probe_hosts.append(dart_wsl_ipv4)
    dart_client_host = DART_HOST

    print(f"[Stack] python executable: {py_exe}")
    if not python_has_modules(py_exe, REQUIRED_PY_MODULES):
        print(
            "[Stack] warning: selected python may be missing required modules "
            f"{REQUIRED_PY_MODULES}. Consider running with firstconda explicitly."
        )

    processes: list[tuple[str, subprocess.Popen]] = []
    log_threads: list[threading.Thread] = []
    stop_event = threading.Event()
    log_lock = threading.Lock()
    log_tail: dict[str, deque[str]] = {}
    exit_code = 0

    def request_shutdown(signum=None, frame=None) -> None:
        if not stop_event.is_set():
            print("\n[Stack] shutdown requested, stopping child processes...")
            stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    try:
        # ── DART ────────────────────────────────────────────────────────────
        existing_dart_host = next(
            (h for h in dart_probe_hosts if is_port_open(h, DART_PORT)), None
        )

        if existing_dart_host is not None:
            dart_client_host = existing_dart_host
            print(f"[Stack] DART already available at {existing_dart_host}:{DART_PORT}, skipping.")
        else:
            dart_wsl_root = windows_to_wsl_path(DART_ROOT)
            dart_cmd = (
                "source ~/miniconda3/etc/profile.d/conda.sh; "
                "conda activate DART; "
                f"cd '{dart_wsl_root}'; "
                "python api_server.py"
            )
            dart_proc = start_process("DART", ["wsl", "-e", "bash", "-lc", dart_cmd], REPO_ROOT, base_env)
            processes.append(("DART", dart_proc))
            print("[Stack] DART loading model in background (2-5 min)…")

            dart_client_host = dart_wsl_ipv4 or DART_HOST

            def _dart_watcher():
                if wait_for_service_ready(dart_client_host, DART_PORT, DART_START_TIMEOUT_SECONDS, "DART", "/health"):
                    print("[Stack] OK: DART is ready - motion generation enabled.")
                else:
                    print(f"[Stack] ⚠ DART did not start within {DART_START_TIMEOUT_SECONDS}s.")

            threading.Thread(target=_dart_watcher, daemon=True).start()

        if dart_client_host != DART_HOST:
            print(f"[Stack] DART via WSL IP: {dart_client_host}:{DART_PORT}")

        base_env["DART_HOST"] = dart_client_host
        base_env["DART_URL"] = f"http://{dart_client_host}:{DART_PORT}"

        # ── Redis ────────────────────────────────────────────────────────────
        if is_port_open(REDIS_HOST, REDIS_PORT):
            print(f"[Stack] redis already up at {REDIS_HOST}:{REDIS_PORT}, skipping.")
        else:
            redis_proc = start_process("Redis", ["redis-server"], REPO_ROOT, base_env)
            processes.append(("Redis", redis_proc))

        # ── Services that can ALL start in parallel ──────────────────────────
        # Worker, Beat, API, SpeechLLM, ECA_UI, Streamlit UI
        # We launch them all immediately; only Orchestrator must wait for API.

        worker_proc = start_process(
            "Worker",
            [py_exe, "-m", "celery", "-A", "celery_app", "worker", "--loglevel=info", "-P", "solo"],
            SERVICE_ROOT,
            base_env,
        )
        processes.append(("Worker", worker_proc))

        beat_proc = start_process(
            "Beat",
            [py_exe, "-m", "celery", "-A", "celery_app", "beat", "--loglevel=info"],
            SERVICE_ROOT,
            base_env,
        )
        processes.append(("Beat", beat_proc))

        if is_port_open(health_probe_host, API_PORT):
            print(f"[Stack] API already up at {API_HOST}:{API_PORT}, skipping.")
        else:
            api_proc = start_process("API", [py_exe, "api_server.py"], SERVICE_ROOT, base_env)
            processes.append(("API", api_proc))

        # SpeechLLM – starts in parallel while we wait for the API below
        if is_port_open(health_probe_host, SPEECH_LLM_PORT):
            print(f"[Stack] SpeechLLM already up at {API_HOST}:{SPEECH_LLM_PORT}, skipping.")
        else:
            if conda_exe is not None:
                speech_cmd = [conda_exe, "run", "-n", speech_llm_conda_env, speech_llm_python, "api_server.py"]
                print(f"[Stack] SpeechLLM conda env: {speech_llm_conda_env}")
            else:
                speech_cmd = [py_exe, "api_server.py"]
                print(f"[Stack] warning: conda not found; SpeechLLM using shared python: {py_exe}")

            speech_proc = start_process("SpeechLLM", speech_cmd, SPEECH_LLM_ROOT, base_env)
            processes.append(("SpeechLLM", speech_proc))

        # ECA UI – starts in parallel too
        if is_port_open(health_probe_host, ECA_UI_PORT):
            print(f"[Stack] ECA UI already up at {API_HOST}:{ECA_UI_PORT}.")
        else:
            eca_proc = start_process(
                "ECA_UI",
                [py_exe, "-m", "http.server", str(ECA_UI_PORT), "--bind", API_HOST],
                ECA_UI_ROOT,
                base_env,
            )
            processes.append(("ECA_UI", eca_proc))

        # Streamlit UI – starts in parallel
        if is_port_open(health_probe_host, UI_PORT):
            print(f"[Stack] Streamlit UI already up at {API_HOST}:{UI_PORT}, skipping.")
        else:
            ui_proc = start_process(
                "UI",
                [
                    py_exe, "-m", "streamlit", "run", "ui.py",
                    "--server.port", str(UI_PORT),
                    "--server.address", API_HOST,
                    "--server.headless", "true",
                    "--logger.level=info",
                ],
                SERVICE_ROOT,
                base_env,
            )
            processes.append(("UI", ui_proc))

        # ── Attach log readers for all services started so far ───────────────
        for name, proc in processes:
            if name not in log_tail:
                log_tail[name] = deque(maxlen=40)
                t = threading.Thread(target=reader_thread, args=(name, proc, log_tail, log_lock), daemon=True)
                t.start()
                log_threads.append(t)

        # ── Wait for AgenticRAG API health, THEN start Orchestrator ─────────
        # This is the only hard ordering constraint.
        api_ready = wait_for_service_ready(health_probe_host, API_PORT, 60, "AgenticRAG API")
        if not api_ready:
            print("[Stack] ⚠ AgenticRAG API did not become healthy within 60 s. Starting Orchestrator anyway.")

        if is_port_open(health_probe_host, MAIN_API_PORT):
            print(f"[Stack] Orchestrator already up at {API_HOST}:{MAIN_API_PORT}, skipping.")
        else:
            orch_proc = start_process("Orchestrator", [py_exe, "main_api.py"], SERVICE_ROOT, base_env)
            processes.append(("Orchestrator", orch_proc))
            log_tail["Orchestrator"] = deque(maxlen=40)
            t = threading.Thread(
                target=reader_thread, args=("Orchestrator", orch_proc, log_tail, log_lock), daemon=True
            )
            t.start()
            log_threads.append(t)

        print(f"\n[Stack] ✓ All services launched.")
        print(f"[Stack]   Default frontend : http://{API_HOST}:{ECA_UI_PORT}")
        print(f"[Stack]   Streamlit UI     : http://{API_HOST}:{UI_PORT}")
        print(f"[Stack]   SpeechLLM API    : http://{API_HOST}:{SPEECH_LLM_PORT}")
        print("[Stack] Press Ctrl+C to stop.\n")

        while not stop_event.is_set():
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"[Stack] {name} exited with code {code}. Triggering shutdown.")
                    if code != 0:
                        exit_code = 1
                        with log_lock:
                            tail_lines = list(log_tail.get(name, []))
                        if tail_lines:
                            print(f"[Stack] last logs from {name}:")
                            for entry in tail_lines[-20:]:
                                print(f"[{name}] {entry}")
                    stop_event.set()
                    break
            time.sleep(0.5)

    except FileNotFoundError as exc:
        print(f"[Stack] command not found: {exc}")
        return 1
    except Exception as exc:
        print(f"[Stack] unexpected error: {exc}")
        return 1
    finally:
        for name, proc in reversed(processes):
            terminate_process(name, proc)
        for thread in log_threads:
            thread.join(timeout=2.0)
        print("[Stack] cleanup complete")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())