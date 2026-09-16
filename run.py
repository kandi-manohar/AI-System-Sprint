"""
Single-command launcher: starts the FastAPI backend and the Streamlit UI
together, satisfying the assessment's "must start with a single command"
constraint without requiring Docker.

Usage:
    python run.py

Stops both processes on Ctrl+C.
"""

import signal
import subprocess
import sys
import time

API_CMD = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
UI_CMD = [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.headless", "true"]


def main():
    print("Starting backend API on http://127.0.0.1:8000 ...")
    api_proc = subprocess.Popen(API_CMD)

    time.sleep(2)  # give the API a moment to come up before the UI starts querying it

    print("Starting Streamlit UI on http://127.0.0.1:8501 ...")
    ui_proc = subprocess.Popen(UI_CMD)

    def shutdown(*_):
        print("\nShutting down...")
        for proc in (ui_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            if api_proc.poll() is not None:
                print("Backend API process exited unexpectedly.")
                shutdown()
            if ui_proc.poll() is not None:
                print("Streamlit UI process exited unexpectedly.")
                shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
