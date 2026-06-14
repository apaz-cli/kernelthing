"""Subprocess wrapper around ``opencode run`` (headless, ``--format json``).

opencode 1.15.13 facts this relies on (see README):
- the prompt is read from **stdin**;
- ``--format json`` emits **newline-delimited JSON events** (step_start / text /
  tool / step_finish); the assistant reply is the concatenation of ``text``
  parts, and every event carries ``sessionID``;
- ``-s <id>`` continues a session; ``--dangerously-skip-permissions`` auto-approves
  tool use (safe only because the run is bwrap-sandboxed).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import sandbox

# The PreToolUse guard plugin and the block-message templates it renders.
_GUARD_PLUGIN = Path(__file__).resolve().parent / "oc_guard" / "guard.js"
_GUARD_BLOCK_DIR = Path(__file__).resolve().parent.parent / "prompts" / "block"


def _opencode_state_dirs() -> list[Path]:
    """opencode's own writable state dirs (kept writable inside the sandbox)."""
    home = Path(os.path.expanduser("~"))
    xdg_data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    xdg_state = Path(os.environ.get("XDG_STATE_HOME", home / ".local" / "state"))
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    return [
        xdg_data / "opencode",
        xdg_state / "opencode",
        xdg_cache / "opencode",
    ]


@dataclass
class OpencodeResult:
    text: str
    session_id: str | None
    cost: float
    tokens: dict
    exit_code: int
    tool_calls: int = 0
    raw_lines: list[str] = field(default_factory=list)


def _parse_ndjson(stdout: str) -> tuple[str, str | None, float, dict, int]:
    text_parts: list[str] = []
    session_id: str | None = None
    cost = 0.0
    tokens: dict = {}
    tool_calls = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = ev["sessionID"]
        etype = ev["type"]
        part = ev["part"] if "part" in ev and isinstance(ev["part"], dict) else {}
        if etype == "text" and "text" in part:
            text_parts.append(part["text"])
        elif etype in ("tool", "tool_use") or ("type" in part and part["type"] in ("tool", "tool-invocation")):
            tool_calls += 1
        elif etype == "step_finish":
            if "cost" in part and part["cost"]:
                cost = part["cost"]
            if "tokens" in part and part["tokens"]:
                tokens = part["tokens"]
    return "".join(text_parts), session_id, cost, tokens, tool_calls


def run_interactive(
    *,
    working_dir: Path,
    model: str,
    session: str | None = None,
    continue_last: bool = False,
    prompt: str | None = None,
    gpu_index: int = 0,
    writable: bool = True,
    sandboxed: bool = True,
    data_dir: Path | None = None,
    extra_writable: list[Path] | tuple[Path, ...] = (),
    ncu: bool = True,
    gpu_lock: Path | None = None,
) -> int:
    """Launch opencode's interactive TUI attached to the terminal, return exit code.

    Used by the problem-bootstrap flow: launch the TUI with an initial ``--prompt``
    (the bootstrap instructions) so the operator can describe the objective and
    converse with the agent live as it authors the problem dir. ``-s session`` /
    ``-c`` (continue_last) resume an existing session for follow-up edits. Unlike
    ``run`` this inherits the real stdin/stdout/stderr (a tty) instead of
    redirecting them, and uses no ``--format json`` (the human drives the TUI).
    Still bwrap-sandboxed.
    """
    opencode = shutil.which("opencode")
    if opencode is None:
        raise FileNotFoundError("opencode not found on PATH")

    inner = [opencode, "-m", model]
    if session:
        inner += ["-s", session]
    elif continue_last:
        inner += ["-c"]
    if prompt:
        inner += ["--prompt", prompt]

    # GPU mutex wrapper: bound after --tmpfs /tmp so it reaches through the mask
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["KERNELTHING_GPU_INDEX"] = str(gpu_index)
    if gpu_lock is not None:
        env["KERNELTHING_GPU_LOCK"] = str(gpu_lock)
    # Disable opencode's snapshot feature (see ``run`` for why it is so costly here).
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps({"snapshot": False})
    if data_dir is not None:
        data_dir = Path(data_dir)
        oc_state = [data_dir / "share", data_dir / "state", data_dir / "cache"]
        env["XDG_DATA_HOME"] = str(oc_state[0])
        env["XDG_STATE_HOME"] = str(oc_state[1])
        env["XDG_CACHE_HOME"] = str(oc_state[2])
    else:
        oc_state = _opencode_state_dirs()

    argv = sandbox.wrap(
        inner,
        project_dir=Path(working_dir),
        writable=writable,
        writable_extra=[*oc_state, *extra_writable],
        enabled=sandboxed and sandbox.available(),
        ncu=ncu,
        gpu_lock=gpu_lock,
    )
    # Inherit the tty (stdin/stdout/stderr = None) so the user interacts directly.
    return subprocess.run(argv, env=env).returncode


def run(
    prompt: str,
    *,
    working_dir: Path,
    model: str,
    session: str | None = None,
    timeout: int = 5400,
    gpu_index: int = 0,
    writable: bool = True,
    sandboxed: bool = True,
    log_path: Path | None = None,
    data_dir: Path | None = None,
    extra_writable: list[Path] | tuple[Path, ...] = (),
    guard: dict | None = None,
    ncu: bool = True,
    gpu_lock: Path | None = None,
) -> OpencodeResult:
    """Run one opencode turn and return the parsed result.

    The NDJSON event stream is written to ``log_path`` *as it arrives* (stdout is
    redirected straight to that file), so a long turn can be watched live with
    ``tail -f``. The prompt is fed from a temp file on stdin; because both stdin
    and stdout are real files (not pipes), there is no pipe-buffer deadlock.

    ``data_dir``: isolate opencode's XDG data/state/cache here (its own SQLite
    DB) instead of the shared default -- required when running several sessions
    in parallel (the evolve workers) so they don't contend on one DB. ``extra_writable``
    adds further read-write binds in the sandbox (e.g. the main repo's .git so a
    worktree can commit).

    ``guard``: loop context (loopDir, projectRoot, planFile, currentRound, phase)
    for the opencode PreToolUse guard plugin. When provided, the guard is loaded
    via ``OPENCODE_CONFIG_CONTENT`` and the context is passed in ``KERNELTHING_GUARD``
    so the plugin can reject writes/edits/reads/bash that would corrupt loop state
    (see oc_guard/guard.js). Omit to run without enforcement.
    """
    opencode = shutil.which("opencode")
    if opencode is None:
        raise FileNotFoundError("opencode not found on PATH")

    inner = [
        opencode, "run",
        "--format", "json",
        "-m", model,
        "--dangerously-skip-permissions",
        "--dir", str(Path(working_dir).resolve()),
    ]
    if session:
        inner += ["-s", session]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["KERNELTHING_GPU_INDEX"] = str(gpu_index)
    if gpu_lock is not None:
        env["KERNELTHING_GPU_LOCK"] = str(gpu_lock)

    # Force opencode's snapshot/checkpoint feature OFF. It git-adds the ENTIRE
    # working dir on every edit into a repo under .humanize/oc-data -- which lives
    # *inside* the worktree, so it recursively snapshots its own growing store plus
    # the multi-MB _build/_build_ref artifacts. That pegs a CPU and dominates wall
    # time. kernelthing manages its own git commits and discards the worktree, so the
    # snapshots are pure overhead. The repo's opencode.json (snapshot:false) can't
    # reach the agent -- it runs in the per-worktree problem repo and we override
    # config via OPENCODE_CONFIG_CONTENT here -- so set it inline.
    oc_config: dict = {"snapshot": False}
    if guard is not None:
        guard_cfg = dict(guard)
        guard_cfg.setdefault("blockDir", str(_GUARD_BLOCK_DIR))
        env["KERNELTHING_GUARD"] = json.dumps(guard_cfg)
        oc_config["plugin"] = [str(_GUARD_PLUGIN)]
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(oc_config)

    if data_dir is not None:
        data_dir = Path(data_dir)
        oc_state = [data_dir / "share", data_dir / "state", data_dir / "cache"]
        env["XDG_DATA_HOME"] = str(oc_state[0])
        env["XDG_STATE_HOME"] = str(oc_state[1])
        env["XDG_CACHE_HOME"] = str(oc_state[2])
    else:
        oc_state = _opencode_state_dirs()

    argv = sandbox.wrap(
        inner,
        project_dir=Path(working_dir),
        writable=writable,
        writable_extra=[*oc_state, *extra_writable],
        enabled=sandboxed and sandbox.available(),
        ncu=ncu,
        gpu_lock=gpu_lock,
    )

    import tempfile

    prompt_fd, prompt_name = tempfile.mkstemp(suffix=".md", prefix="kt_prompt_")
    with os.fdopen(prompt_fd, "w") as pf:
        pf.write(prompt)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        out_name = str(log_path)
    else:
        out_fd, out_name = tempfile.mkstemp(suffix=".ndjson", prefix="kt_oc_")
        os.close(out_fd)

    returncode = 0
    try:
        with open(prompt_name) as stdin_fh, open(out_name, "w") as out_fh:
            proc = subprocess.Popen(
                argv, stdin=stdin_fh, stdout=out_fh,
                stderr=subprocess.PIPE, text=True, env=env,
            )
            try:
                proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                returncode = 124
        if returncode == 0:
            returncode = proc.returncode
    finally:
        os.unlink(prompt_name)

    stdout = Path(out_name).read_text(encoding="utf-8", errors="replace")
    if log_path is None:
        os.unlink(out_name)

    text, sid, cost, tokens, tool_calls = _parse_ndjson(stdout)
    return OpencodeResult(
        text=text,
        session_id=sid,
        cost=cost,
        tokens=tokens,
        exit_code=returncode,
        tool_calls=tool_calls,
        raw_lines=stdout.splitlines(),
    )
