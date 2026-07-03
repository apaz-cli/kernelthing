"""Prompt template loader.

Faithful port of Humanize's hooks/lib/template-loader.sh: single-pass
``{{VAR}}`` substitution where replacement values are NOT re-scanned (so a
``{{OTHER}}`` appearing inside a value is left intact, preventing placeholder
injection). Missing variables keep their ``{{NAME}}`` placeholder unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import PROMPTS_DIR

VAR_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def render(content: str, **variables: object) -> str:
    """Single-pass ``{{VAR}}`` substitution."""

    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        return match.group(0)

    # re.sub with a function replaces in a single left-to-right pass and does
    # not rescan the inserted text -- matching the awk single-pass design.
    return VAR_RE.sub(_repl, content)


def load(rel_path: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Return the raw contents of a prompt file, or '' if missing."""
    path = prompts_dir / rel_path
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def load_and_render_safe(
    rel_path: str,
    fallback: str,
    *,
    prompts_dir: Path = PROMPTS_DIR,
    **variables: object,
) -> str:
    """Load + render a prompt, falling back to ``fallback`` if it is missing/empty."""
    content = load(rel_path, prompts_dir)
    if not content.strip():
        content = fallback
    return render(content, **variables)
