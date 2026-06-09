"""Between-turn gates and review-output sentinel parsing.

These replace the bulk of Humanize's 2200-line stop hook. They stay small
because the orchestrator owns the artifacts (summaries, contracts, state)
instead of reconstructing the agent's intent from a transcript.
"""
from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import NamedTuple

from .config import (
    MARKER_COMPLETE,
    MARKER_SETUP_BLOCKED,
    MARKER_STOP,
    MAX_FILE_LINES,
    VERDICT_ADVANCED,
    VERDICT_REGRESSED,
    VERDICT_STALLED,
    VERDICT_UNKNOWN,
)

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".cc", ".cxx",
    ".h", ".hpp", ".cu", ".cuh", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash",
}
DOC_EXTS = {".md", ".rst", ".txt", ".adoc"}

_P_MARKER_RE = re.compile(r"\[P[0-9]\]")
_VERDICT_RE = re.compile(
    r"Mainline Progress Verdict:\s*(ADVANCED|STALLED|REGRESSED)", re.IGNORECASE
)


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=False,
    )


def git_status_porcelain(cwd: Path) -> str:
    return git(["status", "--porcelain"], cwd).stdout


def tree_is_clean(cwd: Path) -> tuple[bool, str]:
    """Clean if no changes outside untracked ``.humanize/`` paths.

    Returns (clean, offending_status_text).
    """
    status = git_status_porcelain(cwd)
    offending = [
        line for line in status.splitlines()
        if line and not re.match(r"^\?\? \.humanize[-/]", line)
    ]
    return (not offending), "\n".join(offending)


def git_not_clean_note_templates(offending: str) -> list[str]:
    """Block-message stems to append to git-not-clean for the offending changes.

    Faithful to Humanize's git-not-clean-untracked / git-not-clean-humanize-local
    variants, which are conditional notes appended to the base message. The caller
    renders these ``block/<stem>.md`` files (single source of truth for wording).
    """
    stems: list[str] = []
    if any(line.startswith("?? ") for line in offending.splitlines()):
        stems.append("git-not-clean-untracked")
    if ".humanize" in offending:
        stems.append("git-not-clean-humanize-local")
    return stems


# --- goal tracker initialization (port of stop-hook round-0 check) ---

def _section(text: str, heading: str) -> str:
    """Return the body under ``heading`` up to the next same-or-higher heading."""
    lines = text.splitlines()
    depth = len(heading) - len(heading.lstrip("#"))
    out: list[str] = []
    capture = False
    for line in lines:
        if line.strip() == heading.strip():
            capture = True
            continue
        if capture and line.startswith("#"):
            d = len(line) - len(line.lstrip("#"))
            if d <= depth:
                break
        if capture:
            out.append(line)
    return "\n".join(out)


_PLACEHOLDER_RE = re.compile(r"\[To be [a-z]", re.IGNORECASE)


def goal_tracker_uninitialized(goal_tracker_path: Path) -> str:
    """Return a markdown list of still-placeholder sections, or "" if initialized.

    The immutable section (Ultimate Goal, Acceptance Criteria) and the initial
    Active Tasks must be filled in Round 0; leftover ``[To be ...]`` placeholders
    mean the tracker was never initialized.
    """
    try:
        text = goal_tracker_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    checks = [
        ("### Ultimate Goal", "**Ultimate Goal**: Still contains placeholder text"),
        ("### Acceptance Criteria", "**Acceptance Criteria**: Still contains placeholder text"),
        ("#### Active Tasks", "**Active Tasks**: Still contains placeholder text"),
    ]
    missing = [f"- {label}" for heading, label in checks
               if _PLACEHOLDER_RE.search(_section(text, heading))]
    return "\n".join(missing)


def bitlesson_kb_empty(kb_text: str) -> bool:
    """True if the BitLesson KB has no concrete lesson entries (only the seed)."""
    body = re.sub(r"(?m)^#.*$", "", kb_text)          # drop headings
    body = re.sub(r"(?m)^\s*Project-specific.*$", "", body)  # drop seed blurb
    return not body.strip()


def large_files(cwd: Path, max_lines: int = MAX_FILE_LINES) -> list[str]:
    """Return tracked/changed code or doc files exceeding ``max_lines``."""
    out: list[str] = []
    for line in git_status_porcelain(cwd).splitlines():
        if not line:
            continue
        name = line[3:]
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        path = cwd / name
        if not path.is_file():
            continue
        if path.suffix.lower() not in (CODE_EXTS | DOC_EXTS):
            continue
        try:
            n = sum(1 for _ in path.open("rb"))
        except OSError:
            continue
        if n > max_lines:
            out.append(f"{name}: {n} lines")
    return out


# --- incomplete-todo gating (port of check-todos-from-transcript.py) ---
#
# Humanize blocks Stop while the agent has incomplete native tasks, classifying
# each by a leading [mainline]/[blocking]/[queued] tag (queued never blocks).
# opencode has no Claude task tools but ships an equivalent ``todowrite`` tool
# whose authoritative state lives in its SQLite store (``opencode.db``, table
# ``todo``, keyed by session_id) -- the analog of Humanize's per-session task
# files. We read that table directly rather than reconstructing it from the
# NDJSON turn stream (the orchestrator already knows the implementer session id),
# then apply the same lane rules.

_LANE_PREFIX_RE = re.compile(r"^\s*\[(mainline|blocking|queued)\](?:\s|$)", re.IGNORECASE)
# opencode todo statuses are pending / in_progress / completed (+ cancelled);
# only the terminal ones count as resolved.
_TODO_DONE = {"completed", "cancelled"}


def classify_lane(*parts: str) -> str:
    """Infer a todo's lane from its text, defaulting to blocking for safety.

    Only a *leading* ``[mainline]``/``[blocking]``/``[queued]`` tag counts; an
    inline mention (e.g. "fix docs that mention [queued] work") must not
    downgrade an otherwise-blocking task. Faithful to Humanize's classifier.
    """
    for part in parts:
        if not part:
            continue
        m = _LANE_PREFIX_RE.match(part)
        if m:
            return m.group(1).lower()
    return "blocking"


def opencode_db_file(data_home: Path | None = None) -> Path:
    """Path to opencode's SQLite store (holds the ``todo`` table, among others)."""
    if data_home is None:
        data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return Path(data_home) / "opencode" / "opencode.db"


def incomplete_todos(session_id: str | None, *, data_home: Path | None = None) -> list[dict]:
    """Return incomplete, lane-blocking todos for ``session_id`` from opencode's DB.

    Each returned item is ``{"status", "content", "lane"}``. Items whose status
    is terminal (completed/cancelled) are skipped, as are ``[queued]`` items
    (documented follow-ups that must not block finishing). A missing or
    unreadable store means "no incomplete todos" -- the gate fails open, exactly
    like Humanize when no task data is available. The DB is opened read-only so
    we never perturb opencode's state.
    """
    if not session_id:
        return []
    db = opencode_db_file(data_home)
    if not db.is_file():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        try:
            rows = con.execute(
                "SELECT content, status FROM todo WHERE session_id = ? ORDER BY position",
                (session_id,)).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []
    out: list[dict] = []
    for content, status in rows:
        if str(status or "pending").lower() in _TODO_DONE:
            continue
        content = content or "todo"
        lane = classify_lane(content)
        if lane == "queued":
            continue
        out.append({"status": status or "pending", "content": content, "lane": lane})
    return out


def format_incomplete_todos(items: list[dict]) -> str:
    """Render the ``{{INCOMPLETE_LIST}}`` body for block/incomplete-todos.md."""
    return "\n".join(f"  - [{i['status']}] [{i['lane']}] {i['content']}" for i in items)


# --- kernelguard cheat detection ---
#
# The loop's only fitness signal is the problem's score command, so the agent is
# strongly incentivised to game the benchmark instead of genuinely optimising:
# monkeypatching timers, caching/replaying results, hardcoding shapes, CUDA-graph
# replay, silent fallback to the reference, ... kernelguard is a rule-based
# detector for exactly these patterns. We only act on its high-confidence verdict
# (should_filter / classification == "hacked"); softer signals are ignored to keep
# false positives from stalling legitimate rounds. Like every other gate this
# fails open: if kernelguard is unimportable or errors, we report no violation.


def kernelguard_violations(
    edit_files: list[str], repo_root: Path, *,
    profile: str = "default", metadata: dict | None = None,
) -> list[dict]:
    """High-confidence kernelguard cheat hits across the problem's edit files.

    Each item is ``{"file", "classification", "reason", "patterns"}``. Returns
    ``[]`` (fails open) when kernelguard is unavailable, errors, or finds nothing.
    """
    try:
        import kernelguard
    except Exception:
        return []
    try:
        kernelguard.configure_runtime(profile=profile)
    except Exception:
        pass
    out: list[dict] = []
    for rel in edit_files:
        path = Path(repo_root) / rel
        if not path.is_file():
            continue
        try:
            code = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            r = kernelguard.analyze_code(code, metadata=metadata)
        except Exception:
            continue
        if not (r["should_filter"] or r["classification"] == "hacked"):
            continue
        patterns = sorted({p["pattern"] for p in r["matched_patterns"]})
        out.append({
            "file": rel,
            "classification": r["classification"],
            "reason": r["filter_reason"] if r["filter_reason"] else "benchmark gaming detected",
            "patterns": patterns,
        })
    return out


def format_kernelguard_violations(items: list[dict]) -> str:
    """Render the ``{{VIOLATIONS}}`` body for block/kernelguard-cheat.md."""
    blocks = []
    for it in items:
        pats = ", ".join(it["patterns"]) or "(unnamed rules)"
        blocks.append(
            f"- {it['file']} [{it['classification']}]: {it['reason']}\n"
            f"    rules: {pats}")
    return "\n".join(blocks)


# --- review-output sentinel parsing ---

def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def has_complete(review_text: str) -> bool:
    return _last_nonempty_line(review_text) == MARKER_COMPLETE


def has_stop(review_text: str) -> bool:
    return _last_nonempty_line(review_text) == MARKER_STOP


def has_setup_blocked(text: str) -> bool:
    """True if the setup agent gave up (last non-empty line is SETUP_BLOCKED)."""
    return _last_nonempty_line(text) == MARKER_SETUP_BLOCKED


def parse_verdict(review_text: str) -> str:
    """Extract the (last) Mainline Progress Verdict, normalized to lowercase."""
    matches = _VERDICT_RE.findall(review_text)
    if not matches:
        return VERDICT_UNKNOWN
    verdict = matches[-1].lower()
    return {
        "advanced": VERDICT_ADVANCED,
        "stalled": VERDICT_STALLED,
        "regressed": VERDICT_REGRESSED,
    }.get(verdict, VERDICT_UNKNOWN)


def verdict_is_present(review_text: str) -> bool:
    return bool(_VERDICT_RE.search(review_text))


def scan_p_markers(review_text: str) -> list[str]:
    """Return sorted-unique ``[P0-9]`` severity markers found in the review."""
    return sorted(set(_P_MARKER_RE.findall(review_text)))


# --- bitlesson delta (minimal port of bitlesson-validate-delta.sh) ---

_ACTION_RE = re.compile(r"^- *Action:\s*(none|add|update)\s*$", re.IGNORECASE | re.MULTILINE)
_IDS_RE = re.compile(r"^- *Lesson ID\(s\):\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NOTES_RE = re.compile(r"^- *Notes:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# Accept any reasonable lesson-id token (e.g. "BL-20260604-tiling" or
# "round-0-register-blocking"). We deliberately do NOT require Humanize's strict
# BL-YYYYMMDD- prefix: the prompt does not teach that format, so enforcing it
# just causes false rejects. We only require a non-empty, well-formed slug.
_BL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class BitlessonError(NamedTuple):
    """A BitLesson Delta validation failure.

    ``template`` is the ``block/<template>.md`` stem that explains the specific
    problem to the agent (faithful to Humanize's per-error message routing).
    ``action`` fills ``{{ACTION}}`` for the missing-notes template; ``reason`` is
    the human-readable fallback / log line.
    """
    template: str
    action: str
    reason: str


def bitlesson_delta_error(summary_text: str, *, kb_text: str | None = None) -> BitlessonError | None:
    """Validate the summary's ``## BitLesson Delta`` section.

    Returns a BitlessonError (naming the specific block message) or None.
    When ``kb_text`` is given and the KB has no concrete entries yet, ``Action:
    none`` is rejected (port of bitlesson-delta-empty-kb): the loop must record
    at least one reusable lesson before ``none`` becomes acceptable.
    """
    if "## BitLesson Delta" not in summary_text:
        return BitlessonError("bitlesson-delta-missing", "",
                              "Summary is missing the '## BitLesson Delta' section.")
    section = summary_text.split("## BitLesson Delta", 1)[1]
    action_m = _ACTION_RE.search(section)
    if not action_m:
        return BitlessonError("bitlesson-delta-invalid", "",
                              "BitLesson Delta missing a valid 'Action: none|add|update' line.")
    action = action_m.group(1).lower()
    ids_m = _IDS_RE.search(section)
    ids_raw = ids_m.group(1).strip() if ids_m else ""
    ids = [] if ids_raw.upper() in ("", "NONE") else [s.strip() for s in re.split(r"[,\s]+", ids_raw) if s.strip()]

    if action == "none":
        if ids:
            return BitlessonError("bitlesson-delta-inconsistent", "",
                                  "Action 'none' but Lesson ID(s) provided.")
        if kb_text is not None and bitlesson_kb_empty(kb_text):
            return BitlessonError("bitlesson-delta-empty-kb", "",
                                  "Action 'none' not allowed: the BitLesson KB has no concrete entries.")
        return None

    # add / update
    if not ids:
        return BitlessonError("bitlesson-delta-inconsistent", "",
                              f"Action '{action}' requires concrete Lesson ID(s).")
    bad = [i for i in ids if not _BL_ID_RE.match(i)]
    if bad:
        return BitlessonError("bitlesson-delta-inconsistent", "",
                              f"Invalid BitLesson Lesson ID(s): {', '.join(bad)}")
    notes_m = _NOTES_RE.search(section)
    notes = notes_m.group(1).strip() if notes_m else ""
    if not notes or notes.startswith("[") or notes.startswith("<"):
        return BitlessonError("bitlesson-delta-missing-notes", action,
                              f"Action '{action}' requires non-placeholder Notes.")
    return None
