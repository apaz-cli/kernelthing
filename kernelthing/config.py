"""Static configuration and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"

# Sentinels (ported verbatim from Humanize loop semantics).
MARKER_COMPLETE = "COMPLETE"
# Emitted by the setup agent when it cannot derive the scoring objective.
MARKER_SETUP_BLOCKED = "SETUP_BLOCKED"

# Duration units for the wall-clock budget (``-w``), seconds per unit.
DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> int:
    """Parse a duration like ``10m``/``2h``/``90s``/``1d``/``1w`` into whole seconds.

    A bare number is seconds; ``0`` means 'off'. Raises ``ValueError`` on garbage
    so callers (the CLI arg type, the web UI control) can report it their own way."""
    s = str(text).strip().lower()
    unit = 1
    if s and s[-1] in DURATION_UNITS:
        unit, s = DURATION_UNITS[s[-1]], s[:-1]
    value = float(s)
    if value < 0:
        raise ValueError(f"duration must be >= 0, got '{text}'")
    return round(value * unit)


def format_duration(seconds: int) -> str:
    """Render a seconds count back into the compact ``s/m/h/d/w`` form used on the
    CLI (e.g. 600 -> '10m', 5400 -> '90m', 7200 -> '2h'). Picks the largest unit
    that divides evenly; falls back to seconds when nothing else is exact."""
    if seconds <= 0:
        return "0s"
    for suffix, size in (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{suffix}"
    return f"{seconds}s"


@dataclass
class Config:
    """Run-time knobs for one loop invocation."""

    model: str = "deepseek/deepseek-v4-pro"
    opencode_timeout: int = 1200

    # --- search budget & shape ---
    # ``-j``: max agents editing at once (the GPU benchmark stage stays serialized).
    parallelism: int = 4
    max_candidates: int = 24  # dispatch budget; 0 = until wall-clock / stop
    wall_clock_s: int = 0  # wall-clock budget in seconds; 0 = off
    elite_k: int = 4  # size of the top-K frontier (the exploit pool)
    min_niches: int = 4  # below this many strategy niches, bias to explore
    evolve_seed: int = 0  # RNG seed for operator/parent sampling

    # --- GPU pool ---
    # Pool of CUDA device indices the loop may use (--gpu 0 --gpu 1 / --gpu 0,1);
    # tasks go to the least-busy device. Exclusivity is a per-device flock
    # (kernelthing/gpulock.py), not in-process state, so agents and the bench never
    # share a card -- even across separate kernelthing processes on the same GPU.
    gpu_indices: list[int] = field(default_factory=lambda: [0])

    # --- agent tools & sandboxing ---
    sandbox: bool = True
    # kernelguard: static cheat detection (timer monkeypatching, result/CUDA-graph
    # replay, shape hardcoding, ...). On by default; disqualifies cheaters.
    kernelguard: bool = True
    kernelguard_profile: str = "default"
    # Vendored KDA skills injected into prompts; ncu also binds the GPU perf-counter
    # capability nodes in the sandbox.
    ncu: bool = True
    wiki: bool = True

    # --- bootstrap / phases ---
    # auto_setup: author the problem non-interactively, accept on validation pass.
    auto_setup: bool = False
    methodology: bool = False

    # --- paths ---
    # Managed repo root: problem dirs are copied into standalone git repos here so
    # worktrees always branch from committed state.
    problem_root: Path = Path.home() / ".cache" / "kernelthing"
