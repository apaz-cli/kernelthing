"""Append-only run journal + file-based control channel.

Every run directory is self-describing on disk, so the web UI (or an agent
debugging a run) never needs the run process to be alive:

  * ``run.json``      -- immutable metadata written once at setup
  * ``events.ndjson`` -- the journal: one JSON event per line, in order.
                         UI state is a pure fold over this file; a live run is
                         just a journal that is still growing.
  * ``control.json``  -- live-tunable knobs. The web UI writes it (atomic
                         replace); the loop re-reads it at dispatch boundaries,
                         exactly where the old in-memory LoopBus was consulted.
  * ``live.lock``     -- a flock held by the run process for its lifetime.
                         Liveness is "the lock is held", so a crashed run reads
                         as dead with no cleanup step.

NDJSON rather than SQLite on purpose: a run can be debugged with tail/grep/jq
(or by an agent reading the file), and appends are crash-safe without a schema.

Every event carries ``seq`` (1-based, dense) and ``t`` (epoch seconds). Event
types are produced by the orchestrator; the journal itself is type-agnostic.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import IO, Any

RUN_JSON = "run.json"
EVENTS_NDJSON = "events.ndjson"
CONTROL_JSON = "control.json"
LIVE_LOCK = "live.lock"

# Hard ceiling for live -j: the orchestrator sizes its worker pool to this, so
# parallelism can be raised as well as lowered mid-run (threads spawn lazily).
MAX_PARALLELISM = 64

# The knobs the UI may set, with clamps. ``stop`` is one-way (never un-set by
# the UI); ``explore_auto`` is turned off implicitly by setting a manual bias.
CONTROL_DEFAULTS: dict[str, Any] = {
    "parallelism": 1,  # -j: agents editing at once
    "elite_k": 4,  # -k: size of the top-K frontier (the exploit pool)
    "wall_clock_s": 0,  # -w: 0 = off
    "max_candidates": 0,  # -m: 0 = unlimited
    "explore_bias": 50,  # 0-100: 0 = all exploit, 100 = all explore
    "explore_auto": True,  # when True, the orchestrator applies a schedule
    "stop": False,
}


def _clamp_control(raw: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a control dict to known keys with in-range values."""
    out = dict(CONTROL_DEFAULTS)
    try:
        out["parallelism"] = max(
            1, min(MAX_PARALLELISM, int(raw.get("parallelism", out["parallelism"])))
        )
        out["elite_k"] = max(1, int(raw.get("elite_k", out["elite_k"])))
        out["wall_clock_s"] = max(0, int(raw.get("wall_clock_s", out["wall_clock_s"])))
        out["max_candidates"] = max(0, int(raw.get("max_candidates", out["max_candidates"])))
        out["explore_bias"] = max(0, min(100, int(raw.get("explore_bias", out["explore_bias"]))))
        out["explore_auto"] = bool(raw.get("explore_auto", out["explore_auto"]))
        out["stop"] = bool(raw.get("stop", out["stop"]))
    except (TypeError, ValueError):
        pass  # keep whatever parsed before the bad key
    return out


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# --- journal ------------------------------------------------------------------


class Journal:
    """Append-only writer for a run's ``events.ndjson``. Thread-safe; each emit
    is one flushed line, so readers only ever see whole events."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0
        # Held open for the run's lifetime; close() ends the journal.
        self._fh: IO[str] | None = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, type: str, **fields: Any) -> None:
        with self._lock:
            if self._fh is None:
                return
            self._seq += 1
            rec = {"seq": self._seq, "t": round(time.time(), 3), "type": type, **fields}
            self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def read_events(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read whole events from byte ``offset``; returns ``(events, new_offset)``.

    Only complete lines are consumed (a partially-written tail line is left for
    the next poll), so a reader can incrementally follow a live journal by
    passing back the returned offset. Unparseable lines are skipped but their
    bytes are still consumed."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    events: list[dict[str, Any]] = []
    for line in data[: end + 1].splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events, offset + end + 1


# --- control channel ----------------------------------------------------------


def update_control(path: Path, changes: dict[str, Any]) -> dict[str, Any]:
    """UI-side read-modify-write of ``control.json`` (atomic replace).

    Unknown keys are dropped and values clamped, so a hostile/buggy POST body
    cannot corrupt the channel. Returns the new control dict."""
    try:
        cur = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cur = {}
    cur = cur if isinstance(cur, dict) else {}
    merged = _clamp_control({**cur, **changes})
    _write_json_atomic(Path(path), merged)
    return merged


class LoopControl:
    """Loop-side view of ``control.json`` -- the replacement for LoopBus.

    The orchestrator constructs it with the run's starting knobs (which seeds
    the file so the UI sees them) and calls the accessors at dispatch
    boundaries. Each accessor re-reads the file only when its mtime changed;
    observed changes are recorded to the journal as ``control_changed`` events,
    so the run's history includes every mid-run tuning action."""

    def __init__(self, path: Path, journal: Journal | None = None, **initial: Any):
        self.path = Path(path)
        self._journal = journal
        self._cur = _clamp_control({**CONTROL_DEFAULTS, **initial})
        _write_json_atomic(self.path, self._cur)
        self._stamp = self._stat_stamp()

    def _stat_stamp(self) -> tuple[int, int, int]:
        """Change detector: ``os.replace`` gives every update a fresh inode, so
        (ino, mtime, size) changes even when writes land within one clock tick."""
        try:
            st = self.path.stat()
            return (st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            return (0, 0, 0)

    def _refresh(self) -> None:
        stamp = self._stat_stamp()
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return  # torn/absent file: keep current values, retry next poll
        new = _clamp_control(raw if isinstance(raw, dict) else {})
        changes = {k: v for k, v in new.items() if self._cur.get(k) != v}
        if changes and self._journal is not None:
            self._journal.emit("control_changed", changes=changes)
        self._cur = new

    # --- accessors (read by the loop at dispatch boundaries) ---
    def parallelism(self) -> int:
        self._refresh()
        return int(self._cur["parallelism"])

    def elite_k(self) -> int:
        self._refresh()
        return int(self._cur["elite_k"])

    def wall_clock(self) -> int:
        self._refresh()
        return int(self._cur["wall_clock_s"])

    def max_candidates(self) -> int:
        self._refresh()
        return int(self._cur["max_candidates"])

    def explore_bias(self) -> int:
        self._refresh()
        return int(self._cur["explore_bias"])

    def explore_auto(self) -> bool:
        self._refresh()
        return bool(self._cur["explore_auto"])

    def stop_requested(self) -> bool:
        self._refresh()
        return bool(self._cur["stop"])


# --- liveness ------------------------------------------------------------------


class LiveLock:
    """Exclusive flock on ``live.lock``, held by the run process for its
    lifetime. The kernel releases it on process death, so liveness probes are
    accurate even after a crash or SIGKILL."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh: IO[str] | None = None

    def acquire(self) -> None:
        # Held open for the run's lifetime; the flock lives on the open fd.
        self._fh = open(self.path, "w")  # noqa: SIM115
        fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if self._fh is not None:
            self._fh.close()  # closing drops the flock
            self._fh = None


def is_live(run_dir: Path) -> bool:
    """True when some process currently holds the run's live lock."""
    try:
        with open(Path(run_dir) / LIVE_LOCK) as fh:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False  # we got it, so nobody holds it
    except FileNotFoundError:
        return False
    except OSError:
        return True


# --- run discovery --------------------------------------------------------------


def run_dir_of(root: Path, run_id: str) -> Path | None:
    """Resolve a run id (the run dir's path relative to ``root``) safely.

    Rejects anything that escapes ``root`` or does not look like a run dir
    (must sit at ``.../.humanize/rlcr/<ts>`` and contain ``run.json``)."""
    root = Path(root).resolve()
    if not run_id or run_id.startswith(("/", "\\")):
        return None
    p = (root / run_id).resolve()
    if not p.is_relative_to(root):
        return None
    if p.parent.name != "rlcr" or p.parent.parent.name != ".humanize":
        return None
    if not (p / RUN_JSON).is_file():
        return None
    return p


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """All runs under ``root``, newest first.

    Scans ``<root>/.humanize/rlcr/*`` and ``<root>/*/.humanize/rlcr/*`` (the
    managed problem-repo layout). Each entry carries the id (relative path,
    usable with the API), the ``run.json`` metadata, and a liveness flag."""
    root = Path(root).resolve()
    rlcr_dirs = [root / ".humanize" / "rlcr"]
    with contextlib.suppress(OSError):
        rlcr_dirs += sorted(root.glob("*/.humanize/rlcr"))
    runs: list[dict[str, Any]] = []
    for rlcr in rlcr_dirs:
        if not rlcr.is_dir():
            continue
        for d in sorted(rlcr.iterdir()):
            meta_file = d / RUN_JSON
            if not meta_file.is_file():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs.append(
                {
                    "id": str(d.relative_to(root)),
                    "live": is_live(d),
                    "run": meta,
                }
            )
    runs.sort(key=lambda r: str(r["run"].get("timestamp", "")), reverse=True)
    return runs
