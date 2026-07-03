"""Unit tests for the pure evolutionary-search logic (kernelthing/evolve.py)."""

import random

from kernelthing import evolve
from kernelthing.evolve import Member, Population


def _viable(pop, *, metric, message="", commit="c"):
    m = Member(
        id=pop.next_id(),
        operator=evolve.OP_EXPLORE,
        commit=commit or f"c{metric}",
        commit_message=message,
        metric=metric,
        correct=True,
    )
    pop.insert(m)
    return m


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


# --- niches (commit-message derived) ---


def test_niches_keep_best_per_message():
    pop = Population()
    _viable(pop, metric=10, message="add tiling")
    best_a = _viable(pop, metric=40, message="add tiling")
    b = _viable(pop, metric=20, message="fuse layernorm")
    grid = pop.niches()
    assert set(grid) == {"add tiling", "fuse layernorm"}
    assert grid["add tiling"] is best_a
    assert grid["fuse layernorm"] is b


def test_niche_key_from_commit_message():
    pop = Population()
    m = _viable(pop, metric=10, message="  perf:  Precompute  PRNG  into shared  memory (3.95x)  ")
    assert m.niche_key() == "perf: precompute prng into shared memory (3.95x)"


# --- operator selection ---


def test_choose_operator_forces_explore_without_elites():
    rng = random.Random(0)
    op = evolve.choose_operator(
        rng, {evolve.OP_EXPLOIT: 1.0}, have_elites=False, n_niches=0, min_niches=4
    )
    assert op == evolve.OP_EXPLORE


def test_choose_operator_respects_weights():
    rng = random.Random(1)
    ops = [
        evolve.choose_operator(
            rng,
            {evolve.OP_EXPLORE: 1.0, evolve.OP_EXPLOIT: 0.0},
            have_elites=True,
            n_niches=10,
            min_niches=4,
        )
        for _ in range(20)
    ]
    assert set(ops) == {evolve.OP_EXPLORE}


def test_choose_operator_double_explore_when_underpopulated():
    rng = random.Random(0)
    # With equal weights but 0 niches (under min_niches=4), explore is doubled
    seen = set()
    for _ in range(100):
        seen.add(
            evolve.choose_operator(
                rng,
                {evolve.OP_EXPLORE: 0.5, evolve.OP_EXPLOIT: 0.5},
                have_elites=True,
                n_niches=0,
                min_niches=4,
            )
        )
    # Both operators are possible but explore should dominate (2:1 odds)
    assert "explore" in seen


# --- parent selection / bandit ---


def test_select_parent_explore_returns_viable_member():
    pop = Population()
    a = _viable(pop, metric=10)
    b = _viable(pop, metric=30)
    # explore selects from all viable members, biased toward unvisited
    p = pop.select_parent(evolve.OP_EXPLORE, random.Random(0), {})
    assert p is not None
    assert p.id in {a.id, b.id}


def test_select_parent_explore_empty_pool_returns_none():
    pop = Population()
    assert pop.select_parent(evolve.OP_EXPLORE, random.Random(0), {}) is None


def test_select_parent_explore_favors_unvisited():
    pop = Population()
    a = _viable(pop, metric=10, message="a")
    b = _viable(pop, metric=30, message="b")
    a.children = 10  # heavily visited
    b.children = 0  # untouched
    # b should be picked far more often
    rng = random.Random(42)
    counts = {a.id: 0, b.id: 0}
    for _ in range(200):
        counts[pop.select_parent(evolve.OP_EXPLORE, rng, {}).id] += 1
    assert counts[b.id] > counts[a.id]


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
    a = _viable(pop, metric=50, message="a")
    b = _viable(pop, metric=50, message="b")
    in_flight = {a.id: 8}
    counts = {a.id: 0, b.id: 0}
    rng = random.Random(0)
    for _ in range(400):
        counts[pop.select_parent(evolve.OP_EXPLOIT, rng, in_flight).id] += 1
    assert counts[b.id] > counts[a.id] * 3
