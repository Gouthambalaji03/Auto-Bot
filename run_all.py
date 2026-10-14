"""
run_all.py — Development convenience launcher.

Starts both the Gradio UI and the FastAPI API as separate, independent
subprocesses from a single command:

    python run_all.py

Both servers run in genuinely separate OS processes (no shared event loop,
no shared in-memory state). Press Ctrl+C to stop both.

For production: use Docker Compose or a Procfile instead of this script.

    Gradio UI  → http://localhost:7860
    FastAPI    → http://localhost:8000  (docs: http://localhost:8000/docs)
"""

import subprocess
import sys
import time


def main() -> None:
    print("=" * 60, flush=True)
    print("  AutoBot — Starting all services", flush=True)
    print("  Gradio UI  →  http://localhost:7860", flush=True)
    print("  FastAPI    →  http://localhost:8000", flush=True)
    print("  API docs   →  http://localhost:8000/docs", flush=True)
    print("  Press Ctrl+C to stop both servers", flush=True)
    print("=" * 60, flush=True)

    processes: list[subprocess.Popen] = []

    try:
        gradio_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(gradio_proc)
        print("[RUN_ALL] Gradio process started (PID %d)" % gradio_proc.pid, flush=True)

        # Small delay so Gradio's startup messages don't interleave badly
        time.sleep(1)

        api_proc = subprocess.Popen(
            [sys.executable, "api_server.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(api_proc)
        print("[RUN_ALL] FastAPI process started (PID %d)" % api_proc.pid, flush=True)

        # Wait for either process to exit (should run until Ctrl+C)
        while all(p.poll() is None for p in processes):
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[RUN_ALL] Ctrl+C received — stopping all servers...", flush=True)

    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[RUN_ALL] All servers stopped.", flush=True)


if __name__ == "__main__":
    main()
