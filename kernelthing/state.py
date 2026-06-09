"""Loop state + per-round artifact layout.

State lives in ``<working_dir>/.humanize/rlcr/<timestamp>/`` mirroring Humanize's
on-disk layout, but the structured state is a single ``state.json`` (the
orchestrator owns it, so there is no YAML-frontmatter schema to reconstruct).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import DRIFT_NORMAL, VERDICT_UNKNOWN


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class State:
    """Serializable loop state (the JSON analogue of Humanize's state.md)."""

    timestamp: str
    plan_file: str
    model: str
    start_branch: str
    base_branch: str
    base_commit: str
    current_round: int = 0
    mainline_stall_count: int = 0
    last_mainline_verdict: str = VERDICT_UNKNOWN
    drift_status: str = DRIFT_NORMAL
    bitlesson_required: bool = True
    bitlesson_file: str = ".humanize/bitlesson.md"
    methodology: bool = False
    started_at: str = field(default_factory=_utcnow)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "State":
        data = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class LoopDirs:
    """Resolves artifact paths under ``.humanize/rlcr/<timestamp>/``."""

    def __init__(self, working_dir: Path, timestamp: str):
        self.working_dir = Path(working_dir)
        self.base = self.working_dir / ".humanize" / "rlcr" / timestamp

    def ensure(self) -> "LoopDirs":
        self.base.mkdir(parents=True, exist_ok=True)
        return self

    # --- core files ---
    @property
    def state_file(self) -> Path:
        return self.base / "state.json"

    @property
    def finalize_state_file(self) -> Path:
        # The state snapshot for the finalize phase (mirrors Humanize's
        # finalize-state.md, which marks and records the post-COMPLETE phase).
        return self.base / "finalize-state.json"

    @property
    def goal_tracker(self) -> Path:
        return self.base / "goal-tracker.md"

    @property
    def plan_backup(self) -> Path:
        return self.base / "plan.md"

    # --- per-round files (names mirror Humanize) ---
    def prompt(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-prompt.md"

    def contract(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-contract.md"

    def summary(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-summary.md"

    def review_prompt(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-review-prompt.md"

    def review_result(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-review-result.md"

    def finalize_summary(self) -> Path:
        return self.base / "finalize-summary.md"


def save_state(dirs: LoopDirs, state: State) -> None:
    dirs.ensure()
    dirs.state_file.write_text(state.to_json(), encoding="utf-8")


def load_state(dirs: LoopDirs) -> State:
    return State.from_json(dirs.state_file.read_text(encoding="utf-8"))


def find_latest_loop(working_dir: Path) -> Path | None:
    """Return the most recent loop directory under .humanize/rlcr, if any."""
    base = Path(working_dir) / ".humanize" / "rlcr"
    if not base.is_dir():
        return None
    candidates = sorted((d for d in base.iterdir() if (d / "state.json").is_file()))
    return candidates[-1] if candidates else None
