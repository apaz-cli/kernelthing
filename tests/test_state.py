import json
from pathlib import Path

from kernelthing.state import LoopDirs, State, new_timestamp, save_run


def _make_state(ts: str) -> State:
    return State(
        timestamp=ts,
        plan_file="examples/gemm/plan.md",
        model="deepseek/deepseek-v4-pro",
        start_branch="main",
        base_branch="main",
        base_commit="abc1234",
    )


def test_save_run_writes_run_json(tmp_path: Path):
    ts = new_timestamp()
    dirs = LoopDirs(tmp_path, ts).ensure()
    save_run(dirs, _make_state(ts), problem={"name": "gemm", "unit": "us"})
    data = json.loads(dirs.run_json.read_text(encoding="utf-8"))
    assert data["timestamp"] == ts
    assert data["base_commit"] == "abc1234"
    assert data["problem"] == {"name": "gemm", "unit": "us"}


def test_loopdirs_layout(tmp_path: Path):
    ts = "2026-06-04_00-00-00"
    d = LoopDirs(tmp_path, ts)
    assert str(d.base).endswith(f".humanize/rlcr/{ts}")
    assert d.run_json.name == "run.json"
    assert d.events_file.name == "events.ndjson"
    assert d.control_file.name == "control.json"
    assert d.live_lock.name == "live.lock"
    assert d.logfile.name == "loop.log"
    assert d.member_dir(3) == d.base / "members" / "3"
    assert d.member_prompt(3).name == "prompt.md"
    assert d.member_log(3).name == "opencode.ndjson"
    assert d.member_summary(3).name == "summary.md"
    assert d.member_diff(3).name == "diff.patch"
    assert d.member_result(3).name == "result.json"


def test_ensure_member_creates_dir(tmp_path: Path):
    d = LoopDirs(tmp_path, new_timestamp()).ensure()
    md = d.ensure_member(7)
    assert md.is_dir() and md == d.member_dir(7)
