import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = REPO_ROOT / "agenticRAG" / "agentic_rag_gemini"
DART_ROOT = REPO_ROOT / "text-to-motion" / "DART"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = 8000
MAIN_API_PORT = 8080
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
DART_HOST = "127.0.0.1"
DART_PORT = 5001
DART_START_TIMEOUT_SECONDS = 240
UI_PORT = int(os.getenv("AGENTICRAG_UI_PORT", "8501"))
REQUIRED_PY_MODULES = ("celery", "fastapi", "redis")


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def reader_thread(
    service_name: str,
    process: subprocess.Popen,
    log_tail: dict[str, deque[str]],
    log_lock: threading.Lock,
) -> None:
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            cleaned = line.rstrip()
            with log_lock:
                log_tail[service_name].append(cleaned)
            print(f"[{service_name}] {cleaned}")
    except Exception as exc:
        print(f"[{service_name}] log reader error: {exc}")


def start_process(name: str, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    print(f"[Stack] starting {name}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def windows_to_wsl_path(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["wsl", "wslpath", "-a", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip()
    except Exception:
        resolved = str(path.resolve())
        drive = resolved[0].lower()
        rest = resolved[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"


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
    candidates: list[str] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "python.exe"))

    candidates.append(sys.executable)

    default_firstconda = Path.home() / "miniconda3" / "envs" / "firstconda" / "python.exe"
    candidates.append(str(default_firstconda))

    seen = set()
    for exe in candidates:
        if not exe or exe in seen:
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
    py_exe = resolve_python_executable()
    base_env = os.environ.copy()
    base_env["MOTION_ASYNC_ENABLED"] = "true"
    base_env["API_PUBLIC_BASE_URL"] = os.getenv(
        "API_PUBLIC_BASE_URL",
        f"http://{API_HOST}:{API_PORT}",
    )
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env = enrich_path_for_ffmpeg(base_env)

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

    def request_shutdown(signum: int | None = None, frame=None) -> None:
        if not stop_event.is_set():
            print("\n[Stack] shutdown requested, stopping child processes...")
            stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    try:
        if is_port_open(DART_HOST, DART_PORT):
            print(f"[Stack] DART already available at {DART_HOST}:{DART_PORT}, skip starting WSL DART")
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

            if not wait_for_port(DART_HOST, DART_PORT, DART_START_TIMEOUT_SECONDS, "DART"):
                raise RuntimeError(
                    f"DART did not become ready on {DART_HOST}:{DART_PORT} within {DART_START_TIMEOUT_SECONDS}s"
                )

        if is_port_open(REDIS_HOST, REDIS_PORT):
            print(f"[Stack] redis already available at {REDIS_HOST}:{REDIS_PORT}, skip starting redis-server")
        else:
            redis_proc = start_process("Redis", ["redis-server"], REPO_ROOT, base_env)
            processes.append(("Redis", redis_proc))

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

        if is_port_open(API_HOST, API_PORT):
            print(f"[Stack] API already available at {API_HOST}:{API_PORT}, skip starting api_server.py")
        else:
            api_proc = start_process("API", [py_exe, "api_server.py"], SERVICE_ROOT, base_env)
            processes.append(("API", api_proc))

        if is_port_open(API_HOST, MAIN_API_PORT):
            print(f"[Stack] Orchestrator already available at {API_HOST}:{MAIN_API_PORT}, skip starting main_api.py")
        else:
            main_api_proc = start_process("Orchestrator", [py_exe, "main_api.py"], SERVICE_ROOT, base_env)
            processes.append(("Orchestrator", main_api_proc))

        # --- Streamlit Chat UI ---
        if is_port_open(API_HOST, UI_PORT):
            print(f"[Stack] UI already available at {API_HOST}:{UI_PORT}, skip starting streamlit")
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

        for name, proc in processes:
            log_tail[name] = deque(maxlen=40)
            thread = threading.Thread(target=reader_thread, args=(name, proc, log_tail, log_lock), daemon=True)
            thread.start()
            log_threads.append(thread)

        print("[Stack] all services launched. Press Ctrl+C to stop.")

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
                            print(f"[Stack] last logs from {name} before exit:")
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