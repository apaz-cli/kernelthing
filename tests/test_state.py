from pathlib import Path

from kernelthing.state import LoopDirs, State, new_timestamp, save_state


def _make_state(ts: str) -> State:
    return State(
        timestamp=ts,
        plan_file="examples/gemm/plan.md",
        model="deepseek/deepseek-v4-pro",
        start_branch="main",
        base_branch="main",
        base_commit="abc1234",
    )


def test_state_json_round_trip():
    ts = new_timestamp()
    s = _make_state(ts)
    s2 = State.from_json(s.to_json())
    assert s2 == s


def test_state_from_json_ignores_unknown_keys():
    ts = new_timestamp()
    s = _make_state(ts)
    blob = s.to_json().rstrip().rstrip("}") + ', "future_field": 99}'
    s2 = State.from_json(blob)
    assert s2.start_branch == "main"


def test_save_and_load(tmp_path: Path):
    ts = new_timestamp()
    dirs = LoopDirs(tmp_path, ts).ensure()
    s = _make_state(ts)
    save_state(dirs, s)
    assert dirs.state_file.is_file()
    loaded = State.from_json(dirs.state_file.read_text(encoding="utf-8"))
    assert loaded == s


def test_loopdirs_artifact_names(tmp_path: Path):
    ts = "2026-06-04_00-00-00"
    d = LoopDirs(tmp_path, ts)
    assert d.summary(2).name == "round-2-summary.md"
    assert d.prompt(3).name == "round-3-prompt.md"
    assert str(d.base).endswith(f".humanize/rlcr/{ts}")
