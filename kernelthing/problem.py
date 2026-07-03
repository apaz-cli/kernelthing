"""Problem manifest: how kernelthing targets an arbitrary kernel problem.

A *problem* is a directory inside a git repo containing a ``problem.json``
manifest, a plan, the editable kernel file(s), and a self-contained ``score``
command that builds + checks correctness against the problem's own baseline
(cuBLAS, torch, a CPU reference, ...) and prints a JSON line:

    {"correct": true, "metric": 88.0, "unit": "%cuBLAS", ...}

kernelthing stays language/baseline-agnostic: it only runs the score command
and reads that JSON. All paths the orchestrator uses are resolved relative to
the enclosing git repo root (so worktrees and @file mentions work).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Problem:
    name: str
    repo_root: Path  # git toplevel == orchestrator working dir
    rel_dir: str  # problem dir, relative to repo_root
    plan: str  # repo-relative path to the plan
    edit_files: list[str]  # repo-relative paths the agent may edit
    # Plain score command (cwd = <worktree>/<rel_dir>), used as the agent's
    # self-test command; the authoritative scoring always goes through pygpubench.
    # Problems may set it to a custom check or leave it empty (the loop fills in
    # ``kernelthing score .`` for the agent).
    score_command: str = ""
    metric_name: str = "metric"
    unit: str = ""
    direction: str = "maximize"  # or "minimize"
    # Expected GPU model name (matched against nvidia-smi --query-gpu=name).
    # Empty means "no restriction" — pre-existing or hand-authored problems
    # are allowed to run on any GPU. The bootstrap process fills this in
    # automatically from the GPU it runs on, and kernelthing rejects --gpu
    # indices whose model name doesn't match.
    gpu: str = ""
    # pygpubench config: submission_qualname, task_module, generator,
    # test_args, repeats, seed, timeout, landlock/mseal/allow_root. See bench.py.
    bench: dict[str, Any] = field(default_factory=dict)
    # metric derivation for pygpubench: kind + (flops|baseline_qualname).
    metric: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.repo_root / self.rel_dir


def git_toplevel(path: Path) -> Path:
    r = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"{path} is not inside a git repository")
    return Path(r.stdout.strip())


def load_problem(path: str | Path) -> Problem:
    """Load a problem from a directory or a problem.json path."""
    p = Path(path).resolve()
    manifest = p / "problem.json" if p.is_dir() else p
    if not manifest.is_file():
        raise FileNotFoundError(f"no problem.json at {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    prob_dir = manifest.parent
    repo_root = git_toplevel(prob_dir)
    rel_dir = str(prob_dir.relative_to(repo_root))

    def repo_rel(rel_to_problem: str) -> str:
        # manifest paths are relative to the problem dir; make them repo-relative
        return str(Path(rel_dir) / rel_to_problem) if rel_dir != "." else rel_to_problem

    return Problem(
        name=data["name"],
        repo_root=repo_root,
        rel_dir=rel_dir,
        plan=repo_rel(data["plan"]),
        edit_files=[repo_rel(f) for f in data["edit_files"]],
        score_command=data.get("score_command", ""),
        metric_name=data.get("metric_name", "metric"),
        unit=data.get("unit", ""),
        direction=data.get("direction", "maximize"),
        gpu=data.get("gpu", ""),
        bench=dict(data.get("bench", {})),
        metric=dict(data.get("metric", {})),
    )


def prepare_problem(problem: Problem, managed_root: Path) -> Problem:
    """Copy the problem dir into a standalone git repo at ``managed_root/<name>/``
    and return a new Problem rooted there. All worktrees branch from this repo,
    so the source repo (kernelthing itself) is never touched."""
    dest = managed_root / problem.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for item in problem.dir.iterdir():
        if item.name == "__pycache__":
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest / item.name)

    rewrite_plan_for_worktree(dest, problem)

    subprocess.run(["git", "init", "-b", "main"], cwd=dest, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "initial problem"],
        cwd=dest,
        check=True,
        capture_output=True,
    )

    return load_problem(dest / "problem.json")


def set_gpu_model(problem_dir: Path, model_name: str) -> None:
    """Write *model_name* into the problem's ``problem.json`` under the ``gpu`` key.

    The bootstrap process calls this after validation to lock the problem to the
    GPU model it was authored on. The manifest is updated in-place; a subsequent
    ``git commit`` will include it.
    """
    manifest = problem_dir / "problem.json"
    if not manifest.is_file():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("gpu") == model_name:
        return
    data["gpu"] = model_name
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rewrite_plan_for_worktree(dest: Path, problem: Problem) -> None:
    """Update the copied plan.md for standalone worktree context.

    The source plan references the kernelthing repo layout (e.g. ``kernelthing
    score problems/<name>``).  In the managed worktree the problem files live at
    the repo root, not under ``problems/<name>/``, so these references are wrong.
    Replace them with the worktree-appropriate equivalent and prepend a context
    notice so agents know where they are.
    """
    import sys

    plan_path = dest / problem.plan
    if not plan_path.is_file():
        return
    text = plan_path.read_text(encoding="utf-8")
    original = text

    source_dir = f"problems/{problem.name}"
    if source_dir in text:
        text = text.replace(source_dir, ".")
        text = text.replace("./plan.md", "plan.md")

    venv_bin = Path(sys.executable).parent
    kt = str(venv_bin / "kernelthing")
    text = text.replace("kernelthing score", f"{kt} score")

    if text != original:
        plan_path.write_text(text, encoding="utf-8")
