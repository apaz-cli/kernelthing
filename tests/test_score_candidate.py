"""The static cheat gate (kernelguard) must run BEFORE the expensive benchmark in
the evolutionary search, and a detected cheat must skip scoring entirely
(_guarded_score, used by every evolve worker)."""

from kernelthing import gates
from kernelthing.config import Config
from kernelthing.orchestrator import Orchestrator
from kernelthing.problem import Problem


def _orch(tmp_path):
    prob = Problem(
        name="p",
        repo_root=tmp_path,
        rel_dir=".",
        plan="plan.md",
        edit_files=["k.cu"],
        score_command="echo",
    )
    return Orchestrator(prob, Config(kernelguard=True), bus=None)


def test_cheat_disqualified_without_scoring(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    monkeypatch.setattr(gates, "kernelguard_violations", lambda *a, **k: [{"file": "k.cu"}])
    scored = []
    monkeypatch.setattr(
        orch, "_score_worktree", lambda wt, **kw: scored.append(wt) or (True, 99.0, None)
    )
    correct, metric, err = orch._guarded_score(tmp_path)
    assert correct is False
    assert metric is None
    assert err.startswith("kernelguard")
    assert scored == []


def test_clean_candidate_is_scored(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    monkeypatch.setattr(gates, "kernelguard_violations", lambda *a, **k: [])
    monkeypatch.setattr(orch, "_score_worktree", lambda wt, **kw: (True, 99.0, None))
    correct, metric, err = orch._guarded_score(tmp_path)
    assert correct is True and metric == 99.0 and err is None
