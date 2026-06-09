"""Static configuration and constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"

# Sentinels (ported verbatim from Humanize loop semantics).
MARKER_COMPLETE = "COMPLETE"
MARKER_STOP = "STOP"
# Emitted by the setup agent when it cannot derive the scoring objective.
MARKER_SETUP_BLOCKED = "SETUP_BLOCKED"
VERDICT_ADVANCED = "advanced"
VERDICT_STALLED = "stalled"
VERDICT_REGRESSED = "regressed"
VERDICT_UNKNOWN = "unknown"
DRIFT_NORMAL = "normal"
DRIFT_REPLAN = "replan_required"

# Consecutive STALLED/REGRESSED verdicts before forcing drift-recovery.
DRIFT_STALL_THRESHOLD = 3
MAX_FILE_LINES = 2000

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
    return int(round(value * unit))


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
    # ``-j``: max agents editing at once in the evolutionary search (the GPU
    # benchmark stage stays serialized). Live-tunable down via the web UI.
    parallelism: int = 4
    max_candidates: int = 24      # dispatch budget; 0 = until wall-clock / stop
    wall_clock_s: int = 0         # wall-clock budget in seconds; 0 = off
    elite_k: int = 4              # size of the top-K frontier (the exploit pool)
    min_niches: int = 4           # below this many strategy niches, bias to explore
    # bench.score is NOT concurrency-safe (in-process import + global GPU env), so
    # this MUST stay 1 until benchmarking is made process-isolated per device.
    bench_concurrency: int = 1
    evolve_seed: int = 0          # RNG seed for operator/parent sampling
    op_explore: float = 0.4       # base operator-selection weights
    op_exploit: float = 0.4
    op_salvage: float = 0.2
    opencode_timeout: int = 5400
    gpu_index: int = 0
    bitlesson_required: bool = True
    methodology: bool = False
    sandbox: bool = True
    # pygpubench is the default benchmarking engine: every kernel is scored through
    # its adversarial, sandboxed subprocess (kernelthing/bench.py). Turn it off to
    # fall back to the problem's plain ``score_command`` (debug / no-torch boxes).
    pygpubench: bool = True
    # Bootstrap phase (kernelthing/bootstrap.py): when launched without an existing
    # problem dir, an agent authors one from the objective before the loop starts.
    # auto_setup: build non-interactively and accept on validation pass (no review).
    auto_setup: bool = False
    # kernelguard cheat detection: scan the problem's edit files for
    # benchmark-gaming patterns (timer monkeypatching, result/CUDA-graph replay,
    # shape hardcoding, ...). On by default; rolls back / disqualifies cheaters.
    kernelguard: bool = True
    kernelguard_profile: str = "default"
    # Kernel-domain agent tooling (vendored KDA skills), injected into implementer
    # prompts. ncu also binds the GPU perf-counter capability nodes in the sandbox.
    ncu: bool = True
    wiki: bool = True
    # Docs path handed to review prompts (relative to working dir).
    docs_path: str = "docs"
    # Managed problem repo root: problem dirs are copied into standalone git repos
    # under this path so worktrees always branch from committed state.
    problem_root: Path = Path.home() / ".cache" / "kernelthing"
