"""Between-turn gates and review-output sentinel parsing.

These replace the bulk of Humanize's 2200-line stop hook. They stay small
because the orchestrator owns the artifacts (summaries, contracts, state)
instead of reconstructing the agent's intent from a transcript.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import Any

from .config import MARKER_SETUP_BLOCKED


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )


# --- review-output sentinel parsing ---


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def has_setup_blocked(text: str) -> bool:
    """True if the setup agent gave up (last non-empty line is SETUP_BLOCKED)."""
    return last_nonempty_line(text) == MARKER_SETUP_BLOCKED


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
    edit_files: list[str],
    repo_root: Path,
    *,
    profile: str = "default",
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """High-confidence kernelguard cheat hits across the problem's edit files.

    Each item is ``{"file", "classification", "reason", "patterns"}``. Returns
    ``[]`` (fails open) when kernelguard is unavailable, errors, or finds nothing.
    """
    try:
        import kernelguard
    except Exception:
        return []
    with contextlib.suppress(Exception):
        kernelguard.configure_runtime(profile=profile)
    out: list[dict[str, Any]] = []
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
        out.append(
            {
                "file": rel,
                "classification": r["classification"],
                "reason": r["filter_reason"] if r["filter_reason"] else "benchmark gaming detected",
                "patterns": patterns,
            }
        )
    return out


def format_kernelguard_violations(items: list[dict[str, Any]]) -> str:
    """Render the ``{{VIOLATIONS}}`` body for block/kernelguard-cheat.md."""
    blocks = []
    for it in items:
        pats = ", ".join(it["patterns"]) or "(unnamed rules)"
        blocks.append(f"- {it['file']} [{it['classification']}]: {it['reason']}\n    rules: {pats}")
    return "\n".join(blocks)
