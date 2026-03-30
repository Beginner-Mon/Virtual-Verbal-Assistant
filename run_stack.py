import os
import re
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


def probe_host(host: str) -> str:
    # 0.0.0.0/:: are bind addresses, not routable connect targets.
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


def resolve_conda_executable() -> str | None:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "Scripts" / "conda.exe"),
        r"C:\Miniconda\Scripts\conda.exe",
    ]

    seen = set()
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
    py_exe = resolve_python_executable()
    base_env = os.environ.copy()
    speech_llm_conda_env = os.getenv("SPEECH_LLM_CONDA_ENV", "tts")
    speech_llm_python = os.getenv("SPEECH_LLM_PYTHON", "python")
    conda_exe = resolve_conda_executable()
    base_env["MOTION_ASYNC_ENABLED"] = "true"
    base_env["API_PUBLIC_BASE_URL"] = os.getenv(
        "API_PUBLIC_BASE_URL",
        f"http://{API_HOST}:{API_PORT}",
    )
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env = enrich_path_for_ffmpeg(base_env)

    dart_wsl_ipv4 = resolve_wsl_ipv4()
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

    def request_shutdown(signum: int | None = None, frame=None) -> None:
        if not stop_event.is_set():
            print("\n[Stack] shutdown requested, stopping child processes...")
            stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    try:
        existing_dart_host = next(
            (host for host in dart_probe_hosts if is_port_open(host, DART_PORT)),
            None,
        )

        if existing_dart_host is not None:
            dart_client_host = existing_dart_host
            print(f"[Stack] DART already available at {existing_dart_host}:{DART_PORT}, skip starting WSL DART")
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
            print(f"[Stack] DART is loading model in background (may take 2-5 min)... other services starting now.")

            # On some Windows+WSL setups localhost forwarding is disabled;
            # in that case DART is reachable through the WSL VM IP.
            dart_client_host = dart_wsl_ipv4 or DART_HOST

            def _dart_watcher():
                if wait_for_port(dart_client_host, DART_PORT, DART_START_TIMEOUT_SECONDS, "DART"):
                    print("[Stack] ✓ DART is now ready — motion generation enabled.")
                else:
                    print(f"[Stack] ⚠ DART did not start within {DART_START_TIMEOUT_SECONDS}s. Motion generation will be unavailable.")

            dart_thread = threading.Thread(target=_dart_watcher, daemon=True)
            dart_thread.start()

        if dart_client_host != DART_HOST:
            print(
                f"[Stack] DART localhost forwarding unavailable; "
                f"using WSL IP for clients: {dart_client_host}:{DART_PORT}"
            )

        base_env["DART_HOST"] = dart_client_host
        base_env["DART_URL"] = f"http://{dart_client_host}:{DART_PORT}"

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

        if is_port_open(health_probe_host, API_PORT):
            print(f"[Stack] API already available at {API_HOST}:{API_PORT}, skip starting api_server.py")
        else:
            api_proc = start_process("API", [py_exe, "api_server.py"], SERVICE_ROOT, base_env)
            processes.append(("API", api_proc))

        if is_port_open(health_probe_host, MAIN_API_PORT):
            print(f"[Stack] Orchestrator already available at {API_HOST}:{MAIN_API_PORT}, skip starting main_api.py")
        else:
            main_api_proc = start_process("Orchestrator", [py_exe, "main_api.py"], SERVICE_ROOT, base_env)
            processes.append(("Orchestrator", main_api_proc))

        if is_port_open(health_probe_host, SPEECH_LLM_PORT):
            print(
                f"[Stack] SpeechLLM API already available at {API_HOST}:{SPEECH_LLM_PORT}, "
                "skip starting SpeechLLm/api_server.py"
            )
        else:
            if conda_exe is not None:
                speech_cmd = [
                    conda_exe,
                    "run",
                    "-n",
                    speech_llm_conda_env,
                    speech_llm_python,
                    "api_server.py",
                ]
                print(f"[Stack] SpeechLLM will run in conda env: {speech_llm_conda_env}")
            else:
                speech_cmd = [py_exe, "api_server.py"]
                print(
                    "[Stack] warning: conda executable not found; "
                    f"starting SpeechLLM with shared python: {py_exe}"
                )

            speech_proc = start_process("SpeechLLM", speech_cmd, SPEECH_LLM_ROOT, base_env)
            processes.append(("SpeechLLM", speech_proc))

        # --- ECA Official UI 2.0 (Default Frontend) ---
        if is_port_open(health_probe_host, ECA_UI_PORT):
            print(f"[Stack] ECA UI already available at {API_HOST}:{ECA_UI_PORT}, using as default frontend")
        else:
            eca_proc = start_process(
                "ECA_UI",
                [py_exe, "-m", "http.server", str(ECA_UI_PORT), "--bind", API_HOST],
                ECA_UI_ROOT,
                base_env,
            )
            processes.append(("ECA_UI", eca_proc))

        # --- Streamlit Chat UI (Parity/Testing Fallback) ---
        if is_port_open(health_probe_host, UI_PORT):
            print(f"[Stack] Streamlit UI already available at {API_HOST}:{UI_PORT}, skip starting streamlit")
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

        print(f"[Stack] all services launched. Default frontend: http://{API_HOST}:{ECA_UI_PORT}")
        print(f"[Stack] Streamlit parity UI remains available at: http://{API_HOST}:{UI_PORT}")
        print(f"[Stack] SpeechLLM TTS API available at: http://{API_HOST}:{SPEECH_LLM_PORT}")
        print("[Stack] Press Ctrl+C to stop.")

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