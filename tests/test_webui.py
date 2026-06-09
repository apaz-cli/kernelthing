"""Web UI + bus tests: status flow, the lightweight loop_dir accessor, and the
HTTP endpoints (including the candlog basename restriction)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from kernelthing import webui
from kernelthing.bus import LoopBus


def test_bus_loop_dir_and_snapshot():
    bus = LoopBus(parallelism=3)
    assert bus.loop_dir() == ""
    bus.publish(loop_dir="/tmp/x", round=2)
    assert bus.loop_dir() == "/tmp/x"
    snap = bus.snapshot()
    assert snap["round"] == 2
    assert snap["control"] == {"parallelism": 3, "stop": False}
    assert snap["log"] == []


def test_bus_control_roundtrip():
    bus = LoopBus(1)
    bus.set_parallelism(4)
    bus.request_stop()
    assert bus.parallelism() == 4
    assert bus.stop_requested() is True
    assert bus.snapshot()["control"]["stop"] is True


def _get(port: int, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, r.read().decode()


@pytest.fixture()
def server():
    bus = LoopBus(2)
    httpd, port = webui.start_server(bus, port=0)
    try:
        yield bus, port
    finally:
        httpd.shutdown()


def test_endpoints_serve(server, tmp_path: Path):
    bus, port = server
    loop_dir = tmp_path
    (loop_dir / "loop.log").write_text("hello narrative\n", encoding="utf-8")
    bus.publish(loop_dir=str(loop_dir), problem="gemm", round=0)

    code, page = _get(port, "/")
    assert code == 200 and "kernelthing" in page

    code, body = _get(port, "/api/status")
    assert code == 200
    assert json.loads(body)["problem"] == "gemm"

    code, log = _get(port, "/api/log")
    assert "hello narrative" in log


def test_candlog_is_basename_restricted(server, tmp_path: Path):
    bus, port = server
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    bus.publish(loop_dir=str(tmp_path))
    # A traversal attempt is reduced to its basename inside the loop dir, so the
    # secret one level up is never read.
    _, body = _get(port, "/api/candlog?file=../secret.txt")
    assert "TOPSECRET" not in body


def _ndjson(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_summarize_agent_log_extracts_tools_text_cost(tmp_path: Path):
    log = tmp_path / "mem-3-explore-opencode.log"
    log.write_text(_ndjson(
        {"type": "text", "part": {"text": "trying cp.async   staging"}},
        {"type": "tool_use", "part": {"name": "bash", "input": {"command": "nvcc -O3 a.cu"}}},
        {"type": "tool_use", "part": {"name": "edit", "input": {"filePath": "submission.py"}}},
        {"type": "step_finish", "part": {"cost": 0.0123}},
    ), encoding="utf-8")
    s = webui._summarize_agent_log(log)
    assert s["tools"] == 2
    assert s["last_tool"] == "edit submission.py"
    assert s["last_text"] == "trying cp.async staging"
    assert s["cost"] == 0.0123


def test_status_enriches_inflight_agents(server, tmp_path: Path):
    bus, port = server
    (tmp_path / "mem-1-exploit-opencode.log").write_text(_ndjson(
        {"type": "tool_use", "part": {"name": "bash", "input": {"command": "ncu ./a.out"}}},
    ), encoding="utf-8")
    bus.publish(loop_dir=str(tmp_path), mode="evolve",
                agents=[{"id": 1, "op": "exploit", "parent": 0,
                         "log_file": "mem-1-exploit-opencode.log"}])
    _, body = _get(port, "/api/status")
    agent = json.loads(body)["agents"][0]
    assert agent["tools"] == 1
    assert agent["last_tool"] == "bash ncu ./a.out"


def test_control_post(server):
    bus, port = server
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/control",
        data=json.dumps({"parallelism": 5, "stop": True}).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        assert json.loads(r.read())["ok"] is True
    assert bus.parallelism() == 5
    assert bus.stop_requested() is True
