from pathlib import Path

from kernelthing.state import LoopDirs, State, load_state, new_timestamp, save_state


def _make_state(ts: str) -> State:
    return State(
        timestamp=ts, plan_file="examples/gemm/plan.md",
        model="deepseek/deepseek-v4-pro",
        start_branch="main", base_branch="main", base_commit="abc1234",
        current_round=3, mainline_stall_count=2, last_mainline_verdict="stalled",
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
    assert s2.current_round == 3


def test_save_and_load(tmp_path: Path):
    ts = new_timestamp()
    dirs = LoopDirs(tmp_path, ts).ensure()
    s = _make_state(ts)
    save_state(dirs, s)
    assert dirs.state_file.is_file()
    assert load_state(dirs) == s


def test_loopdirs_artifact_names(tmp_path: Path):
    ts = "2026-06-04_00-00-00"
    d = LoopDirs(tmp_path, ts)
    assert d.summary(2).name == "round-2-summary.md"
    assert d.review_result(0).name == "round-0-review-result.md"
    assert d.contract(5).name == "round-5-contract.md"
    assert str(d.base).endswith(f".humanize/rlcr/{ts}")
