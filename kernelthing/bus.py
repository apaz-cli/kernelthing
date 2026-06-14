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
    def __init__(self, parallelism: int, wall_clock_s: int = 0, max_candidates: int = 0):
        self._lock = threading.Lock()
        self._parallelism = max(1, int(parallelism))
        self._wall_clock_s = max(0, int(wall_clock_s))
        self._max_candidates = max(0, int(max_candidates))
        self._explore_bias = 50     # 0-100: 0=all exploit, 100=all explore
        self._explore_auto = True   # when True, the orchestrator applies a schedule
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

    def max_candidates(self) -> int:
        """Live candidate budget (0 = unlimited). Tunable from the UI."""
        with self._lock:
            return self._max_candidates

    def explore_bias(self) -> int:
        """0-100: 0 = all exploit, 100 = all explore."""
        with self._lock:
            return self._explore_bias

    def explore_auto(self) -> bool:
        """Whether the auto-schedule is active."""
        with self._lock:
            return self._explore_auto

    # --- control: written by the UI ---
    def set_parallelism(self, n: int) -> None:
        with self._lock:
            self._parallelism = max(1, int(n))

    def set_wall_clock(self, seconds: int) -> None:
        with self._lock:
            self._wall_clock_s = max(0, int(seconds))

    def set_max_candidates(self, n: int) -> None:
        """Set live candidate budget (0 = unlimited)."""
        with self._lock:
            self._max_candidates = max(0, int(n))

    def set_explore_bias(self, bias: int) -> None:
        """Set manual explore bias (0-100). Disables auto-schedule."""
        with self._lock:
            self._explore_bias = max(0, min(100, int(bias or 50)))
            self._explore_auto = False

    def set_explore_auto(self) -> None:
        """Re-enable the auto-schedule."""
        with self._lock:
            self._explore_auto = True

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
                "max_candidates": self._max_candidates,
                "stop": self._stop,
                "explore_bias": self._explore_bias,
                "explore_auto": self._explore_auto,
            }
            snap["log"] = list(self._log[-120:])
            return snap
