"""Embedded stdlib web UI for watching/controlling a running loop.

Zero dependencies: a ThreadingHTTPServer in a daemon thread serves a single HTML
page that polls ``/api/status`` and renders the live evolutionary search -- a
fitness chart (best vs. kernels submitted), the in-flight agents (each streaming
its exact tool calls), the MAP-Elites niches, the lineage tree, and a
leaderboard. Shares a :class:`LoopBus` with the loop.

The agents stream for free: every opencode turn writes its NDJSON event log to a
per-agent file *as it arrives*. The controller can't see inside a running turn
(it blocks on the subprocess), but this server can tail the file -- so
``/api/status`` enriches each in-flight agent with the live tool/text summary of
its log, and ``/api/candlog`` returns the full readable transcript on demand.

Endpoints:
  * ``GET  /            `` -- the page (served from disk, editable live)
  * ``GET  /app.js      `` -- client-side JavaScript (served from disk)
  * ``GET  /api/status  `` -- the snapshot, agents enriched with live log tails
  * ``GET  /api/log     `` -- the controller narrative (loop.log)
  * ``GET  /api/candlog?file=`` -- one agent's full transcript (basename-restricted)
  * ``POST /api/control `` -- stop / set parallelism / set turn cap
"""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..bus import LoopBus  # re-exported for callers that import from kernelthing.webui

_WEBUI_DIR = Path(__file__).resolve().parent


def tail_text(path: Path, nbytes: int = 262144) -> str:
    """Read the last ``nbytes`` of a file as text, dropping a partial first line."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            raw = f.read()
    except OSError:
        return ""
    txt = raw.decode("utf-8", errors="replace")
    if size > nbytes and "\n" in txt:
        txt = txt.split("\n", 1)[1]
    return txt


def tool_line(part: dict[str, Any]) -> str:
    """One readable line for a tool event: name + its salient argument.

    opencode nests the call args under ``part.state.input`` (e.g. a read tool is
    ``{"tool":"read","state":{"input":{"filePath":...}}}``); only some shapes put
    them at ``part.input``. Check both, else the line is just the bare tool name."""
    name = part.get("tool") or part.get("name", "tool")
    state = part["state"] if isinstance(part.get("state"), dict) else {}
    inp = (
        state["input"]
        if isinstance(state.get("input"), dict)
        else (part["input"] if isinstance(part.get("input"), dict) else {})
    )
    arg = ""
    for key in (
        "command",
        "filePath",
        "file_path",
        "path",
        "pattern",
        "url",
        "query",
        "description",
        "prompt",
    ):
        if inp.get(key):
            arg = inp[key]
            break
    return " ".join((str(name) + " " + str(arg)).split())


def is_tool(d: dict[str, Any], part: dict[str, Any]) -> bool:
    return d["type"] in ("tool", "tool_use") or (
        "type" in part and part["type"] in ("tool", "tool-invocation")
    )


def summarize_agent_log(path: Path) -> dict[str, Any]:
    """Live summary of an agent's NDJSON log for its card: tool count, latest
    tool call, latest reasoning line, and accumulated cost."""
    out: dict[str, Any] = {"tools": 0, "cost": 0.0, "last_tool": "", "last_text": ""}
    if not path.is_file():
        return out
    for line in tail_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = d["part"] if "part" in d and isinstance(d["part"], dict) else {}
        if d["type"] == "text" and "text" in part and part["text"]:
            out["last_text"] = " ".join(part["text"].split())[:160]
        elif is_tool(d, part):
            out["tools"] += 1
            line_txt = tool_line(part)
            if line_txt:
                out["last_tool"] = line_txt[:160]
        elif d["type"] == "step_finish":
            if part.get("cost"):
                out["cost"] = part["cost"]
    out["cost"] = round(float(out["cost"] or 0.0), 4)
    return out


def candlog_text(path: Path, lines: int = 400) -> str:
    """Full readable transcript of an agent's NDJSON log (text + tool lines)."""
    if not path.is_file():
        return "(no log yet)"
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines()[-4000:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        part = d["part"] if "part" in d and isinstance(d["part"], dict) else {}
        if d["type"] == "text" and "text" in part and part["text"]:
            out.append("· " + " ".join(part["text"].split()))
        elif is_tool(d, part):
            out.append("$ " + (tool_line(part) or "tool"))
    return "\n".join(out[-lines:]) or "(no text yet)"


def make_handler(bus: LoopBus) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            if args and len(args) >= 2 and isinstance(args[1], int) and args[1] >= 400:
                super().log_message(format, *args)

        def _send(self, code: int, body: str | bytes, ctype: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                self._send(
                    200,
                    (_WEBUI_DIR / "index.html").read_text(encoding="utf-8"),
                    "text/html; charset=utf-8",
                )
            elif u.path == "/app.js":
                self._send(
                    200,
                    (_WEBUI_DIR / "app.js").read_text(encoding="utf-8"),
                    "application/javascript; charset=utf-8",
                )
            elif u.path == "/api/status":
                snap = bus.snapshot()
                loop_dir = Path(bus.loop_dir())
                if str(loop_dir):
                    for a in snap.get("agents", []):
                        lf = a["log_file"]
                        if lf:
                            a.update(summarize_agent_log(loop_dir / Path(lf).name))
                self._send(200, json.dumps(snap))
            elif u.path == "/api/log":
                loop_dir = Path(bus.loop_dir())
                lf = loop_dir / "loop.log"
                txt = (
                    lf.read_text(encoding="utf-8", errors="replace")
                    if lf.is_file()
                    else "(no log yet)"
                )
                self._send(200, txt, "text/plain; charset=utf-8")
            elif u.path == "/api/candlog":
                q = parse_qs(u.query)
                loop_dir = Path(bus.loop_dir())
                name = (q.get("file") or [""])[0]
                target = loop_dir / Path(name).name if str(loop_dir) and name else None
                self._send(
                    200,
                    candlog_text(target) if target else "(no file)",
                    "text/plain; charset=utf-8",
                )
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/control":
                self._send(404, "not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                body = {}
            if "parallelism" in body:
                bus.set_parallelism(body["parallelism"])
            if "wall_clock" in body:
                from ..config import parse_duration

                with contextlib.suppress(ValueError, TypeError):
                    bus.set_wall_clock(parse_duration(body["wall_clock"]))
            if "explore_bias" in body:
                bus.set_explore_bias(body["explore_bias"])
            if "explore_auto" in body:
                bus.set_explore_auto()
            if "max_candidates" in body:
                bus.set_max_candidates(body["max_candidates"])
            if body.get("stop"):
                bus.request_stop()
            self._send(200, json.dumps({"ok": True}))

    return Handler


def start_server(
    bus: LoopBus, port: int = 8765, host: str = "127.0.0.1"
) -> tuple[ThreadingHTTPServer, int]:
    """Start the web UI in a daemon thread; returns (httpd, actual_port)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(bus))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]
