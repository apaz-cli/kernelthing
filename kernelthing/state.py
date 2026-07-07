"""Run-directory layout + in-memory run state.

A run's artifacts live in ``<working_dir>/.humanize/rlcr/<timestamp>/`` and are
fully self-describing (see journal.py for the journal/control/liveness files):

    run.json            immutable run metadata (problem, config, git baseline)
    events.ndjson       append-only journal of everything that happened
    control.json        live-tunable knobs (web UI writes, loop reads)
    live.lock           flock held by the run process (liveness probe)
    loop.log            human-readable controller narrative
    plan.md             backup of the plan file
    members/<id>/       everything about one candidate:
        prompt.md           the exact rendered prompt the agent received
        opencode.ndjson     the agent's full NDJSON event transcript
        stderr.log          the opencode process's stderr (crash traces)
        summary.md          the agent's candidate-summary.md
        diff.patch          git diff parent..commit (survives ref cleanup)
        result.json         score verdict + raw bench timings + cost/tokens
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .journal import CONTROL_JSON, EVENTS_NDJSON, LIVE_LOCK, RUN_JSON


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class State:
    """In-memory run bookkeeping; persisted (with problem/config metadata) as
    part of ``run.json`` -- see ``save_run``."""

    timestamp: str
    plan_file: str
    model: str
    start_branch: str
    base_branch: str
    base_commit: str
    current_round: int = 0
    methodology: bool = False
    started_at: str = field(default_factory=utcnow)


class LoopDirs:
    """Resolves artifact paths under ``.humanize/rlcr/<timestamp>/``."""

    def __init__(self, working_dir: Path, timestamp: str):
        self.working_dir = Path(working_dir)
        self.base = self.working_dir / ".humanize" / "rlcr" / timestamp

    def ensure(self) -> LoopDirs:
        self.base.mkdir(parents=True, exist_ok=True)
        return self

    # --- run-level files ---
    @property
    def run_json(self) -> Path:
        return self.base / RUN_JSON

    @property
    def events_file(self) -> Path:
        return self.base / EVENTS_NDJSON

    @property
    def control_file(self) -> Path:
        return self.base / CONTROL_JSON

    @property
    def live_lock(self) -> Path:
        return self.base / LIVE_LOCK

    @property
    def logfile(self) -> Path:
        return self.base / "loop.log"

    @property
    def plan_backup(self) -> Path:
        return self.base / "plan.md"

    # --- per-member files ---
    def member_dir(self, member_id: int) -> Path:
        return self.base / "members" / str(int(member_id))

    def ensure_member(self, member_id: int) -> Path:
        d = self.member_dir(member_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def member_prompt(self, member_id: int) -> Path:
        return self.member_dir(member_id) / "prompt.md"

    def member_log(self, member_id: int) -> Path:
        return self.member_dir(member_id) / "opencode.ndjson"

    def member_stderr(self, member_id: int) -> Path:
        return self.member_dir(member_id) / "stderr.log"

    def member_summary(self, member_id: int) -> Path:
        return self.member_dir(member_id) / "summary.md"

    def member_diff(self, member_id: int) -> Path:
        return self.member_dir(member_id) / "diff.patch"

    def member_result(self, member_id: int) -> Path:
        return self.member_dir(member_id) / "result.json"


def save_run(dirs: LoopDirs, state: State, **extra: Any) -> None:
    """Write ``run.json``: the State fields plus caller-provided metadata blocks
    (problem/config), so a run dir is interpretable with no other context."""
    dirs.ensure()
    data = {**dataclasses.asdict(state), **extra}
    dirs.run_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
