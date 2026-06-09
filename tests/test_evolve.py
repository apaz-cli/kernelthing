"""Unit tests for the pure evolutionary-search logic (kernelthing/evolve.py)."""
import random

from kernelthing import evolve
from kernelthing.evolve import Member, Population


def _viable(pop, *, metric, strategy="s", wall=evolve.WALL_UNKNOWN, ceiling=None):
    m = Member(id=pop.next_id(), operator=evolve.OP_EXPLORE, commit=f"c{metric}",
               metric=metric, correct=True, strategy=strategy, wall=wall, ceiling=ceiling)
    pop.insert(m)
    return m


# --- descriptor parsing ---

def test_parse_descriptor_full():
    text = (
        "## Strategy Descriptor\n"
        "Strategy: 128x128 Tiling + cp.async\n"
        "Wall: fixable\n"
        "Ceiling: 92.5 %cuBLAS\n"
        "Next: double-buffer the smem stage\n"
    )
    d = evolve.parse_descriptor(text)
    assert d.strategy == "128x128 tiling + cp.async"   # lowercased, whitespace-collapsed
    assert d.wall == evolve.WALL_FIXABLE
    assert d.ceiling == 92.5
    assert d.next_lever == "double-buffer the smem stage"


def test_parse_descriptor_missing_fields_defaults():
    d = evolve.parse_descriptor("no descriptor here")
    assert d.strategy == "uncategorized"
    assert d.wall == evolve.WALL_UNKNOWN
    assert d.ceiling is None
    assert d.next_lever == ""


def test_parse_descriptor_bad_ceiling_is_none():
    d = evolve.parse_descriptor("Strategy: x\nCeiling: lots\n")
    assert d.strategy == "x"
    assert d.ceiling is None


# --- viability + frontier ---

def test_member_viability_requires_correct_commit_and_metric():
    assert not Member(id=0, operator="x").viable
    assert not Member(id=0, operator="x", correct=True, commit="c").viable  # no metric
    assert not Member(id=0, operator="x", correct=True, metric=1.0).viable  # no commit
    assert Member(id=0, operator="x", correct=True, commit="c", metric=1.0).viable


def test_elites_and_best_maximize():
    pop = Population(direction="maximize", elite_k=2)
    _viable(pop, metric=10)
    top = _viable(pop, metric=30)
    _viable(pop, metric=20)
    assert pop.best() is top
    assert [m.metric for m in pop.elites()] == [30, 20]


def test_elites_minimize_direction():
    pop = Population(direction="minimize", elite_k=2)
    _viable(pop, metric=10)
    _viable(pop, metric=30)
    best = _viable(pop, metric=5)
    assert pop.best() is best
    assert [m.metric for m in pop.elites()] == [5, 10]


def test_dead_members_excluded_from_frontier():
    pop = Population()
    dead = Member(id=pop.next_id(), operator="x", correct=False)
    pop.insert(dead)
    assert pop.best() is None
    assert pop.elites() == []
    assert dead.status == evolve.ST_DEAD


def test_status_classification():
    pop = Population(elite_k=1)
    elite = _viable(pop, metric=50)
    live = _viable(pop, metric=10)
    assert elite.status == evolve.ST_ELITE
    assert live.status == evolve.ST_LIVE


# --- niches (MAP-Elites) ---

def test_niches_keep_best_per_strategy():
    pop = Population()
    _viable(pop, metric=10, strategy="a")
    best_a = _viable(pop, metric=40, strategy="a")
    b = _viable(pop, metric=20, strategy="b")
    grid = pop.niches()
    assert set(grid) == {"a", "b"}
    assert grid["a"] is best_a
    assert grid["b"] is b


# --- salvage pool ---

def test_salvageable_prefers_fixable_non_elites():
    pop = Population(elite_k=1)
    _viable(pop, metric=100, strategy="top")           # elite
    fix = _viable(pop, metric=40, wall=evolve.WALL_FIXABLE)
    _viable(pop, metric=30, wall=evolve.WALL_FUNDAMENTAL)
    pool = pop.salvageable()
    assert fix in pool
    assert all(m.wall == evolve.WALL_FIXABLE for m in pool)


def test_salvageable_falls_back_to_any_non_elite():
    pop = Population(elite_k=1)
    _viable(pop, metric=100)                            # elite
    other = _viable(pop, metric=40, wall=evolve.WALL_UNKNOWN)
    assert other in pop.salvageable()


# --- operator selection ---

def test_choose_operator_forces_explore_without_elites():
    rng = random.Random(0)
    op = evolve.choose_operator(rng, {evolve.OP_EXPLOIT: 1.0},
                                have_elites=False, n_salvage=0, n_niches=0, min_niches=4)
    assert op == evolve.OP_EXPLORE


def test_choose_operator_zeroes_salvage_when_none():
    rng = random.Random(0)
    weights = {evolve.OP_EXPLORE: 0.0, evolve.OP_EXPLOIT: 1.0, evolve.OP_SALVAGE: 5.0}
    seen = {evolve.choose_operator(
        rng, weights, have_elites=True, n_salvage=0, n_niches=10, min_niches=4)
        for _ in range(50)}
    assert evolve.OP_SALVAGE not in seen


def test_choose_operator_respects_weights():
    rng = random.Random(1)
    ops = [evolve.choose_operator(
        rng, {evolve.OP_EXPLORE: 0.0, evolve.OP_EXPLOIT: 1.0, evolve.OP_SALVAGE: 0.0},
        have_elites=True, n_salvage=1, n_niches=10, min_niches=4) for _ in range(20)]
    assert set(ops) == {evolve.OP_EXPLOIT}


# --- parent selection / bandit ---

def test_select_parent_explore_is_none():
    pop = Population()
    _viable(pop, metric=10)
    assert pop.select_parent(evolve.OP_EXPLORE, random.Random(0), {}) is None


def test_select_parent_exploit_returns_an_elite():
    pop = Population(elite_k=2)
    _viable(pop, metric=10)
    _viable(pop, metric=30)
    _viable(pop, metric=20)
    elite_ids = {m.id for m in pop.elites()}
    picks = {pop.select_parent(evolve.OP_EXPLOIT, random.Random(i), {}).id for i in range(30)}
    assert picks <= elite_ids


def test_virtual_loss_discourages_busy_arm():
    # Two equal-metric elites; the one with many in-flight children should be
    # chosen far less often (virtual loss divides its weight).
    pop = Population(elite_k=2)
    a = _viable(pop, metric=50, strategy="a")
    b = _viable(pop, metric=50, strategy="b")
    in_flight = {a.id: 8}
    counts = {a.id: 0, b.id: 0}
    rng = random.Random(0)
    for _ in range(400):
        counts[pop.select_parent(evolve.OP_EXPLOIT, rng, in_flight).id] += 1
    assert counts[b.id] > counts[a.id] * 3
