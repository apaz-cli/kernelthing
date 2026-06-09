"""Thread-safe control/status bus shared between the loop and the web UI.

The orchestrator runs the loop in the main thread and publishes status here; the
embedded web server (a background thread) reads status and writes control
(parallelism, stop). The loop reads control at dispatch boundaries, so N can be
changed live and a stop takes effect after the in-flight candidates settle.
"""
from __future__ import annotations

import threading
import time


class LoopBus:
    def __init__(self, parallelism: int, wall_clock_s: int = 0):
        self._lock = threading.Lock()
        self._parallelism = max(1, int(parallelism))
        self._wall_clock_s = max(0, int(wall_clock_s))
        self._stop = False
        self._status: dict = {}
        self._log: list[str] = []
        self._log_cap = 400

    # --- control: read by the loop ---
    def parallelism(self) -> int:
        with self._lock:
            return self._parallelism

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop

    def wall_clock(self) -> int:
        """Live wall-clock budget in seconds (0 = off). Tunable from the UI."""
        with self._lock:
            return self._wall_clock_s

    # --- control: written by the UI ---
    def set_parallelism(self, n: int) -> None:
        with self._lock:
            self._parallelism = max(1, int(n))

    def set_wall_clock(self, seconds: int) -> None:
        with self._lock:
            self._wall_clock_s = max(0, int(seconds))

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    # --- status: written by the loop ---
    def publish(self, **kw) -> None:
        with self._lock:
            self._status.update(kw)

    def loop_dir(self) -> str:
        """Cheap single-key read (avoids copying the whole snapshot/log).

        Returns "" until the loop publishes one -- the web server can be polled
        before the first publish."""
        with self._lock:
            return self._status.get("loop_dir", "")

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(f"{time.strftime('%H:%M:%S')} {line}")
            if len(self._log) > self._log_cap:
                self._log = self._log[-self._log_cap:]

    # --- snapshot: read by the UI ---
    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self._status)
            snap["control"] = {
                "parallelism": self._parallelism,
                "wall_clock_s": self._wall_clock_s,
                "stop": self._stop,
            }
            snap["log"] = list(self._log[-120:])
            return snap
