"""Standalone stdlib web UI: a pure reader of run directories.

Zero dependencies: a ThreadingHTTPServer serving a single HTML page. The server
is pointed at a *root* directory (the managed problem root by default) and
discovers every run under it -- live or finished -- by scanning for
``.humanize/rlcr/<timestamp>/run.json``. Nothing is shared with the run process:
the loop writes its journal/artifacts to the run dir (see journal.py/state.py)
and this server only reads them, so the same UI replays an old run or follows a
live one. It runs embedded in ``kernelthing`` (a daemon thread) and standalone
as ``kernelthing web``.

The one write path is control: POST /api/control updates the run's
``control.json`` (rejected with 409 when the run is not live), which the loop
re-reads at dispatch boundaries.

Endpoints (run ids are the run dir's path relative to the root):
  * ``GET  /                 `` -- the page (served from disk, editable live)
  * ``GET  /app.js           `` -- client-side JavaScript (served from disk)
  * ``GET  /api/runs         `` -- all discovered runs, newest first
  * ``GET  /api/events?run=&offset=`` -- journal events from a byte offset
  * ``GET  /api/log?run=     `` -- the controller narrative (loop.log)
  * ``GET  /api/agents?run=&ids=`` -- live tool/text/cost summary per member
  * ``GET  /api/member?run=&id=&file=`` -- one member artifact (transcript/...)
  * ``POST /api/control?run= `` -- stop / budgets / parallelism / explore bias
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import journal

_WEBUI_DIR = Path(__file__).resolve().parent

# Member artifacts a client may fetch verbatim; "transcript" is the structured
# JSON view of opencode.ndjson and handled separately.
MEMBER_FILES = {
    "prompt": "prompt.md",
    "summary": "summary.md",
    "diff": "diff.patch",
    "result": "result.json",
    "log": "opencode.ndjson",
    "stderr": "stderr.log",
}


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


def clip(text: str, head: int = 16000, tail: int = 8000) -> str:
    """Clip huge tool output, keeping head and tail (errors usually sit at the end)."""
    if len(text) <= head + tail + 64:
        return text
    omitted = len(text) - head - tail
    return text[:head] + f"\n… [{omitted} chars omitted] …\n" + text[-tail:]


def transcript_items(path: Path, lines: int = 8000) -> list[dict[str, Any]]:
    """The agent's whole context as structured items for the transcript pane:
    assistant prose (``text``), reasoning (``think`` -- the client renders it
    collapsed), and tool calls (``tool``) with their complete input/output.
    One item per NDJSON part, in stream order."""
    items: list[dict[str, Any]] = []
    if not path.is_file():
        return items
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        part = d["part"] if "part" in d and isinstance(d["part"], dict) else {}
        ptype = part.get("type")
        if d["type"] in ("reasoning", "thinking") or ptype in ("reasoning", "thinking"):
            if part.get("text"):
                items.append({"kind": "think", "text": part["text"]})
        elif d["type"] == "text" and part.get("text"):
            items.append({"kind": "text", "text": part["text"]})
        elif is_tool(d, part):
            state = part["state"] if isinstance(part.get("state"), dict) else {}
            inp = (
                state["input"]
                if isinstance(state.get("input"), dict)
                else (part["input"] if isinstance(part.get("input"), dict) else {})
            )
            out = state.get("output") or state.get("error") or part.get("output") or ""
            items.append(
                {
                    "kind": "tool",
                    "line": tool_line(part) or "tool",
                    "input": json.dumps(inp, indent=2) if inp else "",
                    "output": clip(str(out)),
                    "status": state.get("status") or "",
                }
            )
    return items


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    root = Path(root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            if args and len(args) >= 2 and isinstance(args[1], int) and args[1] >= 400:
                super().log_message(format, *args)

        def _send(
            self, code: int, body: str | bytes, ctype: str = "application/json",
            etag: str | None = None,
        ) -> None:
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if etag:
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _run_dir(self, q: dict[str, list[str]]) -> Path | None:
            """The validated run dir for ?run=<id>, or None (404 already sent)."""
            run_id = (q.get("run") or [""])[0]
            d = journal.run_dir_of(root, run_id)
            if d is None:
                self._send(404, json.dumps({"error": "unknown run"}))
            return d

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
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
            elif u.path == "/api/runs":
                self._send(200, json.dumps(journal.discover_runs(root)))
            elif u.path == "/api/events":
                d = self._run_dir(q)
                if d is None:
                    return
                try:
                    offset = max(0, int((q.get("offset") or ["0"])[0]))
                except ValueError:
                    offset = 0
                events, new_offset = journal.read_events(d / journal.EVENTS_NDJSON, offset)
                self._send(
                    200,
                    json.dumps(
                        {"events": events, "offset": new_offset, "live": journal.is_live(d)}
                    ),
                )
            elif u.path == "/api/log":
                d = self._run_dir(q)
                if d is None:
                    return
                lf = d / "loop.log"
                txt = (
                    lf.read_text(encoding="utf-8", errors="replace")
                    if lf.is_file()
                    else "(no log yet)"
                )
                self._send(200, txt, "text/plain; charset=utf-8")
            elif u.path == "/api/agents":
                d = self._run_dir(q)
                if d is None:
                    return
                ids = []
                for part in (q.get("ids") or [""])[0].split(","):
                    if part.strip().isdigit():
                        ids.append(int(part))
                out = {
                    str(i): summarize_agent_log(d / "members" / str(i) / "opencode.ndjson")
                    for i in ids[:64]
                }
                self._send(200, json.dumps(out))
            elif u.path == "/api/member":
                d = self._run_dir(q)
                if d is None:
                    return
                mid = (q.get("id") or [""])[0]
                which = (q.get("file") or ["transcript"])[0]
                if not mid.isdigit():
                    self._send(404, "bad member id", "text/plain")
                    return
                mdir = d / "members" / str(int(mid))
                if which == "transcript":
                    # The NDJSON log is append-only, so (mtime, size) identifies
                    # its content; a 304 spares re-parsing and re-sending the
                    # whole transcript on every poll while it's unchanged.
                    f = mdir / "opencode.ndjson"
                    try:
                        st = f.stat()
                        etag = f'"{st.st_mtime_ns}-{st.st_size}"'
                    except OSError:
                        etag = None
                    if etag and self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.end_headers()
                        return
                    self._send(200, json.dumps(transcript_items(f)), etag=etag)
                elif which in MEMBER_FILES:
                    f = mdir / MEMBER_FILES[which]
                    txt = (
                        f.read_text(encoding="utf-8", errors="replace")
                        if f.is_file()
                        else f"(no {MEMBER_FILES[which]})"
                    )
                    self._send(200, txt, "text/plain; charset=utf-8")
                else:
                    self._send(404, "unknown file", "text/plain")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            u = urlparse(self.path)
            if u.path != "/api/control":
                self._send(404, "not found", "text/plain")
                return
            q = parse_qs(u.query)
            d = self._run_dir(q)
            if d is None:
                return
            if not journal.is_live(d):
                self._send(409, json.dumps({"error": "run is not live"}))
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                body = {}
            body = body if isinstance(body, dict) else {}
            if "wall_clock" in body:  # human form ("10m") from the UI input
                from ..config import parse_duration

                try:
                    body["wall_clock_s"] = parse_duration(body.pop("wall_clock"))
                except (ValueError, TypeError):
                    body.pop("wall_clock", None)
            new = journal.update_control(d / journal.CONTROL_JSON, body)
            self._send(200, json.dumps({"ok": True, "control": new}))

    return Handler


def make_server(root: Path, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(root))


def start_server(
    root: Path, port: int = 8765, host: str = "127.0.0.1"
) -> tuple[ThreadingHTTPServer, int]:
    """Start the web UI in a daemon thread; returns (httpd, actual_port)."""
    httpd = make_server(root, port, host)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]
