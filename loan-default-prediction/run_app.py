"""Single command launcher for the LoanRisk credit risk assessment platform."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

# Force unbuffered output so stdout flushes immediately in terminals and pipes
os.environ["PYTHONUNBUFFERED"] = "1"

# Configure stdout and stderr for UTF-8 on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "ui" / "app.py").is_file() and (PROJECT_ROOT / "loan-default-prediction").is_dir():
    PROJECT_ROOT = PROJECT_ROOT / "loan-default-prediction"


def find_python_executable() -> str:
    """Find a Python executable equipped with project dependencies."""
    try:
        import fastapi  # noqa: F401
        import streamlit  # noqa: F401
        import uvicorn  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return sys.executable


# Auto-relaunch inside virtualenv if current python lacks required packages
CURRENT_PYTHON = find_python_executable()
if Path(CURRENT_PYTHON).resolve() != Path(sys.executable).resolve():
    target_script = str(PROJECT_ROOT / "run_app.py")
    args = [CURRENT_PYTHON, target_script] + sys.argv[1:]
    try:
        result = subprocess.run(args, cwd=str(PROJECT_ROOT), env=os.environ)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)

# Add project root to sys.path
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
STREAMLIT_URL = "http://localhost:8501"
STREAMLIT_HEALTH_URL = "http://127.0.0.1:8501/_stcore/health"
READINESS_TIMEOUT_SECONDS = 90


def _get_json(url: str, timeout: float = 2.0) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _api_ready() -> bool:
    health = _get_json(f"{API_BASE_URL}/health")
    return bool(health and health.get("status") == "healthy" and health.get("model_loaded"))


def _streamlit_ready() -> bool:
    try:
        with urllib.request.urlopen(STREAMLIT_HEALTH_URL, timeout=2.0) as response:
            return response.status == 200 and response.read().decode("utf-8").strip() == "ok"
    except (OSError, urllib.error.URLError):
        return False


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc.terminate()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _api_command(python_bin: str) -> list[str]:
    parsed = urllib.parse.urlparse(API_BASE_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API_BASE_URL must be an HTTP(S) URL with a hostname.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return [
        python_bin,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        parsed.hostname,
        "--port",
        str(port),
    ]


def _streamlit_command(python_bin: str) -> list[str]:
    return [
        python_bin,
        "-m",
        "streamlit",
        "run",
        "ui/app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def run_checks(python_bin: str) -> bool:
    """Run project integrity and pre-flight checks."""
    print("Checking dependencies, files, model artifacts, and configuration...", flush=True)
    check_script = PROJECT_ROOT / "scripts" / "check_project.py"
    if not check_script.is_file():
        print(f"[ERROR] check_project.py not found at {check_script}", file=sys.stderr, flush=True)
        return False

    result = subprocess.run(
        [python_bin, str(check_script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print("\n[ERROR] Project verification failed:", file=sys.stderr, flush=True)
        if result.stdout:
            print(result.stdout, file=sys.stderr, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        return False

    print("✓ Dependencies, model artifacts, and configuration verified.", flush=True)
    return True


def print_banner() -> None:
    """Print the exact product URLs and status banner."""
    banner = """==================================================
              LOANRISK
        CREDIT RISK PLATFORM
==================================================

Application:
http://localhost:8501

API:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

Status:
✓ FastAPI running
✓ Streamlit running
✓ Model loaded
✓ Application ready

Open the Application URL in your browser.
=================================================="""
    print(banner, flush=True)


def main() -> int:
    python_bin = CURRENT_PYTHON

    # Step 1: Checks required dependencies, files, model artifacts, and configuration
    if not run_checks(python_bin):
        return 1

    managed_processes: list[tuple[str, subprocess.Popen, Path]] = []
    log_dir = PROJECT_ROOT / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fastapi_log_path = log_dir / "fastapi.log"
    streamlit_log_path = log_dir / "streamlit.log"

    environment = os.environ.copy()
    environment["API_BASE_URL"] = API_BASE_URL
    environment["PYTHONUNBUFFERED"] = "1"

    try:
        # Step 2 & 3: Start FastAPI backend and wait until /health is available
        if _api_ready():
            print(f"Reusing healthy FastAPI backend at {API_BASE_URL}", flush=True)
        else:
            print("Starting FastAPI backend...", flush=True)
            fastapi_out = open(fastapi_log_path, "w", encoding="utf-8", errors="replace")
            api_proc = subprocess.Popen(
                _api_command(python_bin),
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=fastapi_out,
                stderr=subprocess.STDOUT,
            )
            managed_processes.append(("FastAPI", api_proc, fastapi_log_path))

            # Wait until FastAPI /health is successfully available
            deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if _api_ready():
                    break
                if api_proc.poll() is not None:
                    fastapi_out.flush()
                    with open(fastapi_log_path, "r", encoding="utf-8", errors="replace") as f:
                        err = f.read()
                    print(f"\n[ERROR] FastAPI failed to start (exit code {api_proc.returncode}):\n{err}", file=sys.stderr, flush=True)
                    return 1
                time.sleep(0.5)
            else:
                print(f"\n[ERROR] FastAPI did not become ready within {READINESS_TIMEOUT_SECONDS}s.", file=sys.stderr, flush=True)
                return 1

        # Step 4 & 5: Start Streamlit frontend and verify server is running
        if _streamlit_ready():
            print(f"Reusing Streamlit UI at {STREAMLIT_URL}", flush=True)
        else:
            print("Starting Streamlit frontend...", flush=True)
            streamlit_out = open(streamlit_log_path, "w", encoding="utf-8", errors="replace")
            ui_proc = subprocess.Popen(
                _streamlit_command(python_bin),
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=streamlit_out,
                stderr=subprocess.STDOUT,
            )
            managed_processes.append(("Streamlit", ui_proc, streamlit_log_path))

            # Wait until Streamlit server is running
            deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if _streamlit_ready():
                    break
                if ui_proc.poll() is not None:
                    streamlit_out.flush()
                    with open(streamlit_log_path, "r", encoding="utf-8", errors="replace") as f:
                        err = f.read()
                    print(f"\n[ERROR] Streamlit failed to start (exit code {ui_proc.returncode}):\n{err}", file=sys.stderr, flush=True)
                    return 1
                time.sleep(0.5)
            else:
                print(f"\n[ERROR] Streamlit did not become ready within {READINESS_TIMEOUT_SECONDS}s.", file=sys.stderr, flush=True)
                return 1

        # Step 6: Automatically open browser to working UI
        try:
            webbrowser.open(STREAMLIT_URL)
        except Exception:
            pass

        # Step 7: Print exact required banner
        print(flush=True)
        print_banner()
        print("\nPress Ctrl+C to stop services started by this launcher.\n", flush=True)

        # Keep running until user stops
        while True:
            for name, proc, log_path in managed_processes:
                if proc.poll() is not None:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        err = f.read()
                    print(f"\n[ERROR] {name} stopped unexpectedly (exit code {proc.returncode}):\n{err}", file=sys.stderr, flush=True)
                    return 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping services started by run_app.py...", flush=True)
        return 0
    finally:
        for name, proc, _ in reversed(managed_processes):
            _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())