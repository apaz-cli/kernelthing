"""Loop state + per-round artifact layout.

State lives in ``<working_dir>/.humanize/rlcr/<timestamp>/`` mirroring Humanize's
on-disk layout, but the structured state is a single ``state.json`` (the
orchestrator owns it, so there is no YAML-frontmatter schema to reconstruct).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")


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
    methodology: bool = False
    started_at: str = field(default_factory=utcnow)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> State:
        data = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class LoopDirs:
    """Resolves artifact paths under ``.humanize/rlcr/<timestamp>/``."""

    def __init__(self, working_dir: Path, timestamp: str):
        self.working_dir = Path(working_dir)
        self.base = self.working_dir / ".humanize" / "rlcr" / timestamp

    def ensure(self) -> LoopDirs:
        self.base.mkdir(parents=True, exist_ok=True)
        return self

    # --- core files ---
    @property
    def state_file(self) -> Path:
        return self.base / "state.json"

    @property
    def plan_backup(self) -> Path:
        return self.base / "plan.md"

    # --- per-round files ---
    def prompt(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-prompt.md"

    def summary(self, rnd: int) -> Path:
        return self.base / f"round-{rnd}-summary.md"


def save_state(dirs: LoopDirs, state: State) -> None:
    dirs.ensure()
    dirs.state_file.write_text(state.to_json(), encoding="utf-8")
