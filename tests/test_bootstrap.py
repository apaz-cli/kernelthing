"""Tests for the problem-bootstrap flow (kernelthing/bootstrap.py).

opencode/pygpubench/a GPU are not available in CI, so the setup agent is replaced
by a fake ``opencode_client.run`` that authors a complete problem dir, and runtime
validation fails open (``bench.available()`` -> False). This exercises the real
control flow: slug/dir derivation, the SETUP_BLOCKED and auto-without-objective
guards, manifest validation, and the git commit of the new dir.
"""

import subprocess
import textwrap

import pytest

from kernelthing import bootstrap
from kernelthing.config import MARKER_COMPLETE, MARKER_SETUP_BLOCKED, Config
from kernelthing.opencode_client import OpencodeResult

# --- pure helpers ---


def test_slugify_from_text():
    assert bootstrap.slugify("Fused FP16 softmax over [B,S,S]") == "fused-fp16-softmax-over-b"


def test_slugify_empty_falls_back_to_timestamp():
    slug = bootstrap.slugify("!!!")
    assert slug.startswith("problem-")


def test_unique_dir_suffixes(tmp_path):
    (tmp_path / "gemm").mkdir()
    assert bootstrap.unique_dir(tmp_path, "gemm").name == "gemm-2"


# --- fixtures: a managed-root dir + a fake setup agent ---


@pytest.fixture
def managed(tmp_path, monkeypatch):
    """The managed problem root bootstrap authors standalone repos under.

    Bootstrap ``git init``s the per-problem repo itself; pin a committer identity
    via env so the commit succeeds regardless of the box's global git config.
    """
    for k, v in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(k, v)
    root = tmp_path / "managed"
    root.mkdir()
    return root


def _author_complete_problem(target, *, edit_files='["kernel.py"]'):
    """Write a minimal but valid+loadable problem into ``target``.

    The kernel lives in its own file; ``submission.py`` is the fixed adapter and is
    never an edit file (the reward-hacking guard rejects that).
    """
    (target / "problem.json").write_text(
        textwrap.dedent(f"""
        {{"name": "p", "plan": "plan.md", "edit_files": {edit_files},
         "bench": {{"submission_qualname": "submission.kernel"}},
         "metric": {{"kind": "latency_us"}}}}
    """)
    )
    (target / "plan.md").write_text("# plan\n")
    (target / "kernel.py").write_text("def run(x):\n    return x\n")
    (target / "submission.py").write_text("from kernel import run as kernel\n")
    (target / "task.py").write_text(
        "def generate_test_case(*, seed):\n    return (1,), (1, 0, 0)\n"
    )


def _fake_run(text, *, author=True, **author_kw):
    """Build a stand-in for opencode_client.run that authors the problem repo.

    The problem IS the repo root, which bootstrap passes as ``working_dir``.
    """

    def run(prompt, *, working_dir, **kw):
        if author:
            _author_complete_problem(working_dir, **author_kw)
        return OpencodeResult(text=text, session_id="sess1", cost=0.0, tokens={}, exit_code=0)

    return run


@pytest.fixture
def no_runtime_validation(monkeypatch):
    # Fail open: no pygpubench installed -> skip the GPU correctness check.
    monkeypatch.setattr(bootstrap.bench, "available", lambda: False)


# --- behavior ---


def test_auto_without_objective_errors(managed):
    with pytest.raises(RuntimeError, match="needs an objective"):
        bootstrap.bootstrap_problem(None, cfg=Config(), auto=True, managed_root=managed)


def test_auto_happy_path_authors_and_commits(managed, monkeypatch, no_runtime_validation):
    monkeypatch.setattr(bootstrap.opencode_client, "run", _fake_run(f"done\n{MARKER_COMPLETE}"))
    target = bootstrap.bootstrap_problem(
        "fused softmax", cfg=Config(sandbox=False), auto=True, managed_root=managed
    )
    assert target.parent == managed
    assert (target / "problem.json").is_file()
    # The problem was committed in its standalone repo so worktrees branch from it...
    listed = subprocess.run(
        ["git", "-C", str(target), "ls-files"], capture_output=True, text=True
    ).stdout
    assert "problem.json" in listed and "kernel.py" in listed
    # ...but the noisy bootstrap artifacts are kept local, not committed.
    assert "bootstrap-opencode.log" not in listed


def test_edit_files_may_not_include_adapter(managed, monkeypatch, no_runtime_validation):
    # Reward-hacking guard: submission.py (the adapter) must not be editable.
    monkeypatch.setattr(
        bootstrap.opencode_client,
        "run",
        _fake_run(f"{MARKER_COMPLETE}", edit_files='["submission.py"]'),
    )
    with pytest.raises(RuntimeError, match="adapter/spec files"):
        bootstrap.bootstrap_problem("x", cfg=Config(sandbox=False), auto=True, managed_root=managed)


def test_setup_blocked_raises(managed, monkeypatch, no_runtime_validation):
    monkeypatch.setattr(
        bootstrap.opencode_client,
        "run",
        _fake_run(f"cannot do it\n{MARKER_SETUP_BLOCKED}", author=False),
    )
    with pytest.raises(RuntimeError, match="SETUP_BLOCKED"):
        bootstrap.bootstrap_problem(
            "vague", cfg=Config(sandbox=False), auto=True, managed_root=managed
        )


def test_auto_validation_failure_raises(managed, monkeypatch, no_runtime_validation):
    # Agent claims COMPLETE but writes nothing -> validation finds no manifest.
    monkeypatch.setattr(
        bootstrap.opencode_client, "run", _fake_run(f"{MARKER_COMPLETE}", author=False)
    )
    with pytest.raises(RuntimeError, match="validation failed"):
        bootstrap.bootstrap_problem("x", cfg=Config(sandbox=False), auto=True, managed_root=managed)
