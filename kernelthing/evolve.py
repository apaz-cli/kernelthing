"""Asynchronous evolutionary kernel search: pure data model + selection logic.

The controller (``Orchestrator.run``) owns one :class:`Population` and
continuously dispatches mutation tasks to a worker pool, serializing **only** the
GPU benchmark stage. (Benchmarking is not concurrency-safe: ``bench.score``
imports the candidate module into this process and pins the GPU through a
process-global env var, so two benches at once would corrupt each other's timing
and module state -- agent *editing*, being API-bound, runs many-at-once.)

Everything in this module is pure / side-effect-free so it can be unit tested;
the side-effecting orchestration (worktrees, agents, git) lives in
orchestrator.py.

Operators -- one per dispatched task:
  * ``explore``  -- fork the base, take a strategy not yet in the archive (breadth)
  * ``exploit``  -- refine a top-scoring elite along the same approach (depth)
  * ``salvage``  -- re-target a viable-but-stalled near-miss the agent judged
    one *fixable* step short of a higher ceiling

Selection keeps a top-K elite set plus a per-strategy niche map (MAP-Elites), so
the search cannot collapse onto one lineage. Compute is steered toward lineages
with the best metric and recent headroom via a UCB-style bandit whose visit count
includes in-flight children (virtual loss), so concurrent dispatch does not pile
onto a single arm before its results land.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

# operators
OP_EXPLORE = "explore"
OP_EXPLOIT = "exploit"
OP_SALVAGE = "salvage"

# member status (for display / classification)
ST_ELITE = "elite"    # viable and on the top-K frontier
ST_LIVE = "live"      # viable, retained, parent-able (salvage pool)
ST_DEAD = "dead"      # not viable (incorrect / no commit / no metric)

# agent triage of an approach's headroom
WALL_FIXABLE = "fixable"
WALL_FUNDAMENTAL = "fundamental"
WALL_UNKNOWN = "unknown"


@dataclass
class Member:
    """One scored kernel attempt -- a node in the search."""

    id: int
    operator: str
    parent_id: int | None = None
    commit: str | None = None
    metric: float | None = None
    correct: bool = False
    strategy: str = "uncategorized"   # niche descriptor (agent-reported)
    wall: str = WALL_UNKNOWN          # fixable | fundamental | unknown
    ceiling: float | None = None      # agent's estimated ceiling for this approach
    next_lever: str = ""              # the agent's noted next step
    summary_text: str = ""
    error: str | None = None
    children: int = 0                 # tasks dispatched from this member (bandit visits)
    status: str = ST_DEAD

    @property
    def viable(self) -> bool:
        return self.correct and self.commit is not None and self.metric is not None


@dataclass
class Task:
    """A unit of work handed to a worker: a fully-rendered prompt + where to fork."""

    member_id: int
    operator: str
    parent_id: int | None
    parent_commit: str | None    # None -> fork the base
    prompt: str


@dataclass
class Descriptor:
    strategy: str = "uncategorized"
    wall: str = WALL_UNKNOWN
    ceiling: float | None = None
    next_lever: str = ""


_STRATEGY_RE = re.compile(r"^\s*Strategy:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_WALL_RE = re.compile(r"^\s*Wall:\s*(fixable|fundamental)\s*$", re.IGNORECASE | re.MULTILINE)
_CEILING_RE = re.compile(r"^\s*Ceiling:\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE | re.MULTILINE)
_NEXT_RE = re.compile(r"^\s*Next:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_descriptor(summary_text: str) -> Descriptor:
    """Extract the ``## Strategy Descriptor`` fields from a candidate summary.

    Tolerant: any field may be missing. ``strategy`` is lowercased and clamped so
    it forms a stable niche key; ``ceiling`` is parsed as a float when present.
    """
    d = Descriptor()
    m = _STRATEGY_RE.search(summary_text)
    if m:
        label = re.sub(r"\s+", " ", m.group(1).strip().lower())
        d.strategy = label[:60] or "uncategorized"
    m = _WALL_RE.search(summary_text)
    if m:
        d.wall = m.group(1).lower()
    m = _CEILING_RE.search(summary_text)
    if m:
        try:
            d.ceiling = float(m.group(1))
        except ValueError:
            d.ceiling = None
    m = _NEXT_RE.search(summary_text)
    if m:
        d.next_lever = m.group(1).strip()
    return d


def _better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "maximize" else a < b


def _norm(values: list[float], direction: str) -> list[float]:
    """Scale ``values`` to [0, 1] with higher = better (direction-aware)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    out = []
    for v in values:
        x = (v - lo) / (hi - lo)
        out.append(x if direction == "maximize" else 1.0 - x)
    return out


def _bandit_weights(pool: list[Member], direction: str, in_flight: dict[int, int],
                    value_fn, c: float) -> dict[int, float]:
    """UCB-style weight per member: exploitation (normalized value) + exploration
    (``c * sqrt(ln(total_visits+1)/(visits+1))``), divided by ``1 + in_flight`` so
    concurrent dispatch spreads across arms (virtual loss)."""
    vals = [float(value_fn(m)) for m in pool]
    norms = _norm(vals, direction)
    total = sum(m.children for m in pool)
    weights: dict[int, float] = {}
    for m, nv in zip(pool, norms):
        explore = c * math.sqrt(math.log(total + 1) / (m.children + 1))
        w = (nv + explore) / (1 + in_flight.get(m.id, 0))
        weights[m.id] = max(w, 1e-9)
    return weights


def _weighted_pick(pool: list[Member], weights: dict[int, float], rng: random.Random) -> Member:
    total = sum(weights[m.id] for m in pool)
    r = rng.random() * total
    acc = 0.0
    for m in pool:
        acc += weights[m.id]
        if r <= acc:
            return m
    return pool[-1]


class Population:
    """The live archive of scored kernels + frontier/niche bookkeeping."""

    def __init__(self, *, direction: str = "maximize",
                 elite_k: int = 4, ucb_c: float = 0.7):
        self.direction = direction
        self.elite_k = elite_k
        self.ucb_c = ucb_c
        self.members: list[Member] = []
        self._seq = 0

    def next_id(self) -> int:
        i = self._seq
        self._seq += 1
        return i

    def insert(self, m: Member) -> None:
        self.members.append(m)
        self._classify()

    # --- frontier views ---
    def _viable_sorted(self) -> list[Member]:
        viable = [m for m in self.members if m.viable]
        viable.sort(key=lambda m: m.metric or 0.0, reverse=(self.direction == "maximize"))
        return viable

    def elites(self) -> list[Member]:
        """Top-K viable members (the exploit pool)."""
        return self._viable_sorted()[: self.elite_k]

    def best(self) -> Member | None:
        s = self._viable_sorted()
        return s[0] if s else None

    def niches(self) -> dict[str, Member]:
        """Best viable member per strategy label (MAP-Elites grid)."""
        grid: dict[str, Member] = {}
        for m in self._viable_sorted():   # best-first, so first seen per key wins
            grid.setdefault(m.strategy, m)
        return grid

    def salvageable(self) -> list[Member]:
        """Viable non-elite members worth rescuing: those the agent judged
        ``fixable`` (fallback: any viable non-elite)."""
        elite_ids = {m.id for m in self.elites()}
        pool = [m for m in self.members if m.viable and m.id not in elite_ids]
        fixable = [m for m in pool if m.wall == WALL_FIXABLE]
        return fixable or pool

    def _classify(self) -> None:
        for m in self.members:
            m.status = ST_LIVE if m.viable else ST_DEAD
        for m in self.elites():
            m.status = ST_ELITE

    # --- parent selection (bandit with virtual loss) ---
    def select_parent(self, operator: str, rng: random.Random,
                      in_flight: dict[int, int]) -> Member | None:
        if operator == OP_EXPLOIT:
            pool = self.elites()
            value_fn = lambda m: m.metric  # noqa: E731
        elif operator == OP_SALVAGE:
            pool = self.salvageable()
            # prefer high estimated ceiling when given, else the measured metric
            value_fn = lambda m: (m.ceiling if m.ceiling is not None else m.metric)  # noqa: E731
        else:
            return None   # explore forks the base
        if not pool:
            return None
        weights = _bandit_weights(pool, self.direction, in_flight, value_fn, self.ucb_c)
        return _weighted_pick(pool, weights, rng)


def choose_operator(rng: random.Random, weights: dict[str, float], *,
                    have_elites: bool, n_salvage: int, n_niches: int,
                    min_niches: int) -> str:
    """Pick an operator. Forces ``explore`` until a viable elite exists; zeroes
    ``salvage`` when nothing is salvageable; and doubles the ``explore`` weight
    while the niche grid is under-populated (keep breadth up early)."""
    if not have_elites:
        return OP_EXPLORE
    w = {
        OP_EXPLORE: max(0.0, weights[OP_EXPLORE]),
        OP_EXPLOIT: max(0.0, weights[OP_EXPLOIT]),
        OP_SALVAGE: max(0.0, weights[OP_SALVAGE]),
    }
    if n_salvage == 0:
        w[OP_SALVAGE] = 0.0
    if n_niches < min_niches:
        w[OP_EXPLORE] *= 2.0
    total = sum(w.values())
    if total <= 0:
        return OP_EXPLORE
    r = rng.random() * total
    acc = 0.0
    for op, weight in w.items():
        acc += weight
        if r <= acc:
            return op
    return OP_EXPLORE
