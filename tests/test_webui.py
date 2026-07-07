"""Journal + web UI tests: the append-only event journal, the file-based
control channel, run discovery/liveness, and the HTTP endpoints (including the
run-id and member-id restrictions)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kernelthing import journal, webui
from kernelthing.state import LoopDirs

TS = "2026-01-02_03-04-05"


def _ndjson(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


# --- journal ------------------------------------------------------------------


def test_journal_emit_and_read(tmp_path: Path):
    j = journal.Journal(tmp_path / "events.ndjson")
    j.emit("run_start", problem={"name": "gemm"})
    j.emit("dispatch", member=1, op="explore")
    events, offset = journal.read_events(tmp_path / "events.ndjson")
    assert [e["type"] for e in events] == ["run_start", "dispatch"]
    assert [e["seq"] for e in events] == [1, 2]
    assert all("t" in e for e in events)
    # incremental read: nothing new at the returned offset
    more, offset2 = journal.read_events(tmp_path / "events.ndjson", offset)
    assert more == [] and offset2 == offset
    # ... until a new event lands
    j.emit("run_end", reason="maxiter")
    more, _ = journal.read_events(tmp_path / "events.ndjson", offset)
    assert [e["type"] for e in more] == ["run_end"]
    j.close()


def test_read_events_ignores_partial_tail(tmp_path: Path):
    p = tmp_path / "events.ndjson"
    p.write_text('{"seq":1,"type":"a"}\n{"seq":2,"ty', encoding="utf-8")
    events, offset = journal.read_events(p)
    assert len(events) == 1
    # the partial line was not consumed; completing it makes it readable
    with open(p, "a", encoding="utf-8") as f:
        f.write('pe":"b"}\n')
    more, _ = journal.read_events(p, offset)
    assert len(more) == 1 and more[0]["seq"] == 2


def test_read_events_missing_file(tmp_path: Path):
    assert journal.read_events(tmp_path / "nope.ndjson") == ([], 0)


# --- control channel ----------------------------------------------------------


def test_loop_control_seeds_file_and_reads_updates(tmp_path: Path):
    ctl_path = tmp_path / "control.json"
    ctl = journal.LoopControl(
        ctl_path, parallelism=4, elite_k=8, wall_clock_s=60, max_candidates=10
    )
    assert ctl.parallelism() == 4
    assert ctl.elite_k() == 8
    assert ctl.wall_clock() == 60
    assert ctl.max_candidates() == 10
    assert ctl.stop_requested() is False
    # the UI-side write is observed on the next accessor call
    journal.update_control(ctl_path, {"parallelism": 2, "elite_k": 3, "stop": True})
    assert ctl.parallelism() == 2
    assert ctl.elite_k() == 3
    assert ctl.stop_requested() is True


def test_control_changes_are_journaled(tmp_path: Path):
    j = journal.Journal(tmp_path / "events.ndjson")
    ctl = journal.LoopControl(tmp_path / "control.json", j, parallelism=4)
    journal.update_control(tmp_path / "control.json", {"explore_bias": 80, "explore_auto": False})
    ctl.explore_bias()  # first read after the change emits the event
    events, _ = journal.read_events(tmp_path / "events.ndjson")
    changed = [e for e in events if e["type"] == "control_changed"]
    assert len(changed) == 1
    assert changed[0]["changes"] == {"explore_bias": 80, "explore_auto": False}
    j.close()


def test_update_control_clamps_and_drops_unknown_keys(tmp_path: Path):
    p = tmp_path / "control.json"
    new = journal.update_control(
        p, {"parallelism": -3, "elite_k": 0, "explore_bias": 999, "evil": "x"}
    )
    assert new["parallelism"] == 1
    assert new["elite_k"] == 1
    assert new["explore_bias"] == 100
    assert "evil" not in new
    assert "evil" not in json.loads(p.read_text(encoding="utf-8"))
    # -j is capped so a live raise can never exceed the worker pool size
    assert (
        journal.update_control(p, {"parallelism": 9999})["parallelism"] == journal.MAX_PARALLELISM
    )


# --- liveness -----------------------------------------------------------------


def test_live_lock_probe(tmp_path: Path):
    assert journal.is_live(tmp_path) is False  # no lock file at all
    lock = journal.LiveLock(tmp_path / journal.LIVE_LOCK)
    lock.acquire()
    assert journal.is_live(tmp_path) is True
    lock.release()
    assert journal.is_live(tmp_path) is False


# --- run discovery --------------------------------------------------------------


def _make_run(root: Path, problem: str = "gemm", ts: str = TS) -> LoopDirs:
    dirs = LoopDirs(root / problem, ts).ensure()
    dirs.run_json.write_text(
        json.dumps({"timestamp": ts, "problem": {"name": problem, "unit": "us"}}),
        encoding="utf-8",
    )
    return dirs


def test_discover_and_resolve_runs(tmp_path: Path):
    dirs = _make_run(tmp_path)
    runs = journal.discover_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r["run"]["problem"]["name"] == "gemm"
    assert r["live"] is False
    assert journal.run_dir_of(tmp_path, r["id"]) == dirs.base.resolve()


def test_run_dir_of_rejects_escapes(tmp_path: Path):
    _make_run(tmp_path)
    outside = tmp_path.parent / "evil" / ".humanize" / "rlcr" / TS
    outside.mkdir(parents=True)
    (outside / "run.json").write_text("{}", encoding="utf-8")
    assert journal.run_dir_of(tmp_path, f"../evil/.humanize/rlcr/{TS}") is None
    assert journal.run_dir_of(tmp_path, "/etc") is None
    assert journal.run_dir_of(tmp_path, "gemm") is None  # not a run dir shape
    assert journal.run_dir_of(tmp_path, "") is None


# --- HTTP endpoints -------------------------------------------------------------


def _get(port: int, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, r.read().decode()


def _post(port: int, path: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@pytest.fixture()
def server(tmp_path: Path):
    dirs = _make_run(tmp_path)
    httpd, port = webui.start_server(tmp_path, port=0)
    try:
        yield dirs, port
    finally:
        httpd.shutdown()


def _run_id(dirs: LoopDirs, root: Path) -> str:
    return str(dirs.base.relative_to(root))


def test_page_and_runs_endpoint(server, tmp_path: Path):
    dirs, port = server
    code, page = _get(port, "/")
    assert code == 200 and "kernelthing" in page

    code, body = _get(port, "/api/runs")
    runs = json.loads(body)
    assert code == 200 and len(runs) == 1
    assert runs[0]["id"] == _run_id(dirs, tmp_path)


def test_events_endpoint_incremental(server, tmp_path: Path):
    dirs, port = server
    j = journal.Journal(dirs.events_file)
    j.emit("run_start", problem={"name": "gemm"})
    rid = _run_id(dirs, tmp_path)

    _, body = _get(port, f"/api/events?run={rid}&offset=0")
    r = json.loads(body)
    assert [e["type"] for e in r["events"]] == ["run_start"]
    assert r["live"] is False

    j.emit("dispatch", member=1, op="explore")
    _, body = _get(port, f"/api/events?run={rid}&offset={r['offset']}")
    r2 = json.loads(body)
    assert [e["type"] for e in r2["events"]] == ["dispatch"]
    j.close()


def test_events_unknown_run_404(server):
    _dirs, port = server
    try:
        _get(port, "/api/events?run=nope&offset=0")
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_log_endpoint(server, tmp_path: Path):
    dirs, port = server
    dirs.logfile.write_text("hello narrative\n", encoding="utf-8")
    _, log = _get(port, f"/api/log?run={_run_id(dirs, tmp_path)}")
    assert "hello narrative" in log


def test_member_transcript_and_files(server, tmp_path: Path):
    dirs, port = server
    dirs.ensure_member(3)
    dirs.member_log(3).write_text(
        _ndjson(
            {"type": "text", "part": {"text": "trying cp.async staging"}},
            {"type": "tool_use", "part": {"name": "bash", "input": {"command": "nvcc a.cu"}}},
        ),
        encoding="utf-8",
    )
    dirs.member_prompt(3).write_text("do the thing", encoding="utf-8")
    rid = _run_id(dirs, tmp_path)

    _, tx = _get(port, f"/api/member?run={rid}&id=3&file=transcript")
    assert "trying cp.async staging" in tx and "$ bash nvcc a.cu" in tx

    _, prompt = _get(port, f"/api/member?run={rid}&id=3&file=prompt")
    assert prompt == "do the thing"

    # ids and file names are strictly validated: no traversal possible
    for bad in ("id=../3&file=prompt", "id=3&file=../../run.json"):
        try:
            _get(port, f"/api/member?run={rid}&{bad}")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404


def test_agents_endpoint_summarizes(server, tmp_path: Path):
    dirs, port = server
    dirs.ensure_member(1)
    dirs.member_log(1).write_text(
        _ndjson(
            {"type": "tool_use", "part": {"name": "bash", "input": {"command": "ncu ./a.out"}}},
            {"type": "step_finish", "part": {"cost": 0.0123}},
        ),
        encoding="utf-8",
    )
    _, body = _get(port, f"/api/agents?run={_run_id(dirs, tmp_path)}&ids=1")
    a = json.loads(body)["1"]
    assert a["tools"] == 1
    assert a["last_tool"] == "bash ncu ./a.out"
    assert a["cost"] == 0.0123


def test_control_post_requires_live_run(server, tmp_path: Path):
    dirs, port = server
    rid = _run_id(dirs, tmp_path)
    code, _ = _post(port, f"/api/control?run={rid}", {"stop": True})
    assert code == 409  # nobody holds the live lock

    lock = journal.LiveLock(dirs.live_lock)
    lock.acquire()
    try:
        code, body = _post(
            port, f"/api/control?run={rid}", {"parallelism": 5, "wall_clock": "10m", "stop": True}
        )
        assert code == 200
        ctl = json.loads(body)["control"]
        assert ctl["parallelism"] == 5
        assert ctl["wall_clock_s"] == 600
        assert ctl["stop"] is True
        # and the loop-side reader sees it
        loop_ctl = json.loads(dirs.control_file.read_text(encoding="utf-8"))
        assert loop_ctl["stop"] is True
    finally:
        lock.release()


def test_summarize_agent_log_extracts_tools_text_cost(tmp_path: Path):
    log = tmp_path / "opencode.ndjson"
    log.write_text(
        _ndjson(
            {"type": "text", "part": {"text": "trying cp.async   staging"}},
            {"type": "tool_use", "part": {"name": "bash", "input": {"command": "nvcc -O3 a.cu"}}},
            {"type": "tool_use", "part": {"name": "edit", "input": {"filePath": "submission.py"}}},
            {"type": "step_finish", "part": {"cost": 0.0123}},
        ),
        encoding="utf-8",
    )
    s = webui.summarize_agent_log(log)
    assert s["tools"] == 2
    assert s["last_tool"] == "edit submission.py"
    assert s["last_text"] == "trying cp.async staging"
    assert s["cost"] == 0.0123
