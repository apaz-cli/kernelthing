"""Bootstrap a new problem dir from a natural-language objective.

When kernelthing is launched without an existing problem dir, an opencode agent
authors a fresh problem dir (problem.json + plan.md + baseline + submission +
task spec) inside a standalone git repo under the managed problem root (see
``Config.problem_root``) from the operator's objective, before the loop starts.

Two modes mirror the loop's interactivity flags:

* ``auto=True`` (``--auto-setup``) runs the agent headless and accepts on runtime
  validation -- requires an objective string up front, since there is no one to
  ask interactively.
* ``auto=False`` (default) drafts headlessly, then drops the operator into an
  interactive opencode session (resuming the same session) to refine/answer, with
  an ``(e)dit / (a)pprove / (q)uit`` review loop -- the same shape as the loop's
  pygpubench setup review.

The result is always a *complete* problem dir, committed so the loop's worktrees
branch from it; the orchestrator therefore does no further spec setup.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from . import bench, gates, gpupool, opencode_client, prompts
from .config import MARKER_COMPLETE, MARKER_SETUP_BLOCKED, Config
from .problem import Problem, git_toplevel, load_problem
from .state import new_timestamp


def slugify(text: str, *, max_words: int = 5) -> str:
    """Derive a filesystem-safe slug from objective text (or a timestamp)."""
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    slug = "-".join(words).strip("-")
    return slug or f"problem-{new_timestamp()}"


def unique_dir(problems_root: Path, slug: str) -> Path:
    """``problems_root/slug``, suffixed ``-2``, ``-3`` ... if it already exists."""
    candidate = problems_root / slug
    n = 2
    while candidate.exists():
        candidate = problems_root / f"{slug}-{n}"
        n += 1
    return candidate


def repo_root_for_cwd() -> Path:
    """The enclosing git toplevel of the current directory (where problems/ lives)."""
    return git_toplevel(Path.cwd())


def protected_files(problem: Problem) -> set[str]:
    """Basenames the optimizer must never edit: the adapter module that exposes
    ``submission_qualname`` plus the objective spec (``task.py`` / ``baseline.py``).

    A qualname's module is its first dotted segment (``submission.kernel`` ->
    ``submission`` -> ``submission.py``), so this catches the adapter wherever it
    is named.
    """
    bench_cfg = problem.bench or {}
    metric = problem.metric or {}
    mods = {bench_cfg.get("task_module", "task")}
    for q in (bench_cfg.get("submission_qualname"), metric.get("baseline_qualname")):
        if q:
            mods.add(q.split(".", 1)[0])
    return {f"{m}.py" for m in mods if m}


def validate_problem(target: Path, *, gpu_index: int = 0) -> tuple[bool, str | None, Problem | None]:
    """Runtime validation: a loadable manifest whose shipped submission scores correct.

    Fails open when pygpubench/torch are not installed (ok=True with a note) -- the
    same philosophy as the loop's gates -- so bootstrap can run on a box without the
    optional benchmark deps; the loop's own scoring surfaces a clear error later if
    they are genuinely needed.
    """
    if not (target / "problem.json").is_file():
        return False, "no problem.json was authored", None
    try:
        problem = load_problem(target)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        return False, f"problem.json is invalid: {e}", None
    if not (problem.repo_root / problem.plan).is_file():
        return False, f"plan not found: {problem.plan}", problem
    for f in problem.edit_files:
        if not (problem.repo_root / f).is_file():
            return False, f"edit file not found: {f}", problem
    # Reward-hacking guard: the optimizer may only edit the kernel source, never the
    # adapter (which exposes submission_qualname) or the objective spec (task.py /
    # baseline.py) -- editing those lets it bypass or fake the real computation.
    forbidden = protected_files(problem)
    clash = {Path(f).name for f in problem.edit_files} & forbidden
    if clash:
        return (
            False,
            f"edit_files must not include the adapter/spec files "
            f"{sorted(clash)} (only the kernel source is editable)",
            problem,
        )
    # GPU access is serialized by the libktgpu.so shim inside pygpubench's
    # isolated worker (see bench._gpu_env); no in-process lock needed.
    correct, _metric, err, _detail = bench.score(problem, problem.repo_root, gpu_index=gpu_index)
    if not correct:
        return False, err or "shipped submission did not score correct", problem
    return True, None, problem


def render_prompt(objective: str | None, target: Path, repo_root: Path, *, auto: bool) -> str:
    # The bootstrap prompt is mode-aware: autonomous (--auto-setup) has no operator
    # to converse with, interactive does. Render the matching directive's own
    # markers first, then inject it -- the main render is single-pass and does not
    # rescan inserted values, so {{COMPLETE}} inside the directive must already be
    # resolved.
    mode_file = "claude/bootstrap-mode-auto.md" if auto else "claude/bootstrap-mode-interactive.md"
    mode_directive = prompts.load_and_render_safe(
        mode_file, "", COMPLETE=MARKER_COMPLETE, SETUP_BLOCKED=MARKER_SETUP_BLOCKED
    )
    return prompts.load_and_render_safe(
        "claude/bootstrap-problem.md",
        "",
        OBJECTIVE=objective.strip()
        if objective and objective.strip()
        else "(none given yet -- ask the operator, who is in this session with you, now)",
        TARGET_DIR=str(target.relative_to(repo_root)),
        MODE_DIRECTIVE=mode_directive,
        COMPLETE=MARKER_COMPLETE,
        SETUP_BLOCKED=MARKER_SETUP_BLOCKED,
    )


# The fully-rendered bootstrap prompt is kept next to the problem as a convenience
# reference; ``.bootstrap-meta.json`` stores the inputs needed to re-render it, so
# the loop can refresh that snapshot from the *current* template on every run rather
# than letting a frozen copy drift from prompts/claude/bootstrap-problem.md.
PROMPT_SNAPSHOT = "bootstrap-prompt.md"
PROMPT_META = ".bootstrap-meta.json"


def write_bootstrap_prompt(
    target: Path, repo_root: Path, objective: str | None, *, auto: bool
) -> str:
    """Render the bootstrap prompt from the current template and write it (plus the
    inputs to re-render it later) into the problem dir. Returns the rendered text."""
    prompt = render_prompt(objective, target, repo_root, auto=auto)
    (target / PROMPT_SNAPSHOT).write_text(prompt, encoding="utf-8")
    (target / PROMPT_META).write_text(
        json.dumps({"objective": objective or "", "auto": bool(auto)}), encoding="utf-8"
    )
    return prompt


def refresh_bootstrap_prompt(target: Path, repo_root: Path) -> bool:
    """Re-render the ``bootstrap-prompt.md`` snapshot from the *current* template.

    Reuses the persisted ``(objective, auto)`` when present (faithful re-render);
    falls back to the neutral interactive prompt otherwise, so problems authored
    before the metadata existed still pick up template changes. Best-effort and
    idempotent: only refreshes a snapshot that already exists (never introduces one
    into a hand-authored dir), only rewrites the prompt (never the metadata), and
    swallows IO errors -- the snapshot is a convenience, never load-bearing. Returns
    True iff it rewrote the snapshot.
    """
    meta_path = target / PROMPT_META
    if not (target / PROMPT_SNAPSHOT).is_file() and not meta_path.is_file():
        return False
    objective: str | None = None
    auto = False
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            objective = (meta.get("objective") or "") or None
            auto = bool(meta.get("auto", False))
        except (ValueError, OSError):
            pass
    try:
        prompt = render_prompt(objective, target, repo_root, auto=auto)
        (target / PROMPT_SNAPSHOT).write_text(prompt, encoding="utf-8")
    except OSError:
        return False
    return True


def print_summary(target: Path, ok: bool, note: str | None) -> None:
    lines = ["", "-- Bootstrapped problem --", f"  dir: {target}"]
    manifest = target / "problem.json"
    lines.append(f"  manifest: {'present' if manifest.is_file() else 'MISSING'}")
    lines.append(f"  validation: {'VALID' if ok else 'INVALID'}" + (f" -- {note}" if note else ""))
    print("\n".join(lines), file=sys.stderr, flush=True)


def commit_problem(target: Path, repo_root: Path) -> None:
    """Commit the new problem dir so the loop's worktrees branch from it."""
    if target != repo_root:
        rel = str(target.relative_to(repo_root))
        gates.git(["add", rel], repo_root)
    else:
        gates.git(["add", "-A"], repo_root)
    if gates.git(["diff", "--cached", "--quiet"], repo_root).returncode != 0:
        gates.git(["commit", "-m", f"bootstrap: initialize problem {target.name}"], repo_root)


def interactive_bootstrap(target: Path, repo_root: Path, prompt: str, cfg: Config) -> None:
    """Open the opencode TUI seeded with the bootstrap prompt and converse to author
    the problem, then validate + ``(a)pprove / (e)dit more / (q)uit`` in a loop."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "bootstrap: no problem dir and no tty for an interactive "
            "session -- pass a problem dir, or use --auto-setup with an "
            "objective"
        )
    first = True
    while True:
        # First pass: launch the TUI with the bootstrap prompt so the agent greets the
        # operator and authors files live. Later passes: resume that session (-c) so
        # follow-up edits keep full context.
        opencode_client.run_interactive(
            working_dir=repo_root,
            model=cfg.model,
            prompt=prompt if first else None,
            continue_last=not first,
            gpu_pool=cfg.gpu_indices,
            writable=True,
            sandboxed=cfg.sandbox,
            ncu=cfg.ncu,
        )
        first = False
        ok, note, _ = validate_problem(target, gpu_index=cfg.gpu_indices[0])
        print_summary(target, ok, note)
        question = (
            "(a)pprove, (e)dit more, or (q)uit? [a/e/q] "
            if ok
            else "problem is not valid yet -- (e)dit more or (q)uit? [e/q] "
        )
        ans = input(f"\n[bootstrap] {question}").strip().lower()
        if ans in ("q", "quit"):
            raise RuntimeError("bootstrap aborted by user")
        if ok and ans in ("a", "approve", "y", "yes", ""):
            return
        # otherwise loop back into the (resumed) session for more edits


def bootstrap_problem(
    objective: str | None, *, cfg: Config, auto: bool, managed_root: Path
) -> Path:
    """Author a new problem dir and return its path. See module docstring for modes.

    The problem is authored directly inside a standalone git repo at
    ``managed_root/<slug>/`` so the loop's worktrees always branch from committed
    state; the kernelthing source repo is never touched.

    Raises ``RuntimeError`` on an unrecoverable failure (agent gave up, validation
    failed in auto mode, or the operator quit the review).
    """
    if auto and not (objective and objective.strip()):
        raise RuntimeError(
            "bootstrap: --auto-setup needs an objective "
            "(pass it as the argument or via --objective-file)"
        )

    repo_root = unique_dir(managed_root, slugify(objective or ""))
    repo_root.mkdir(parents=True, exist_ok=True)
    target = repo_root  # the problem IS the repo root

    # Init a standalone git repo so worktrees branch from committed state.
    subprocess.run(["git", "init", "-b", "main"], cwd=target, check=True, capture_output=True)

    print(
        f"[bootstrap] authoring a new problem at {target} ({'auto' if auto else 'interactive'})",
        file=sys.stderr,
    )

    # Keep the bootstrap artifacts next to the problem for debugging, but out of the
    # commit (``git add <dir>`` honours this .gitignore for untracked files).
    (target / ".gitignore").write_text(
        f"{PROMPT_SNAPSHOT}\n{PROMPT_META}\nbootstrap-opencode.log\n", encoding="utf-8"
    )

    prompt = write_bootstrap_prompt(target, repo_root, objective, auto=auto)

    if auto:
        # No operator to converse with: run the agent headless and accept on validation.
        res = opencode_client.run(
            prompt,
            working_dir=repo_root,
            model=cfg.model,
            session=None,
            timeout=cfg.opencode_timeout,
            gpu_pool=cfg.gpu_indices,
            writable=True,
            sandboxed=cfg.sandbox,
            log_path=target / "bootstrap-opencode.log",
            ncu=cfg.ncu,
        )
        if gates.has_setup_blocked(res.text):
            raise RuntimeError(
                "bootstrap: --auto-setup agent emitted SETUP_BLOCKED -- cannot "
                "author the problem from the given objective:\n" + res.text.strip()[-2000:]
            )
        ok, note, _ = validate_problem(target, gpu_index=cfg.gpu_indices[0])
        if not ok:
            raise RuntimeError(f"bootstrap: --auto-setup validation failed: {note}")
        print("[bootstrap] problem validated; --auto-setup accepting", file=sys.stderr)
    else:
        # Drop the operator straight into the opencode TUI, seeded with the bootstrap
        # prompt, to describe the objective and author the files conversationally.
        interactive_bootstrap(target, repo_root, prompt, cfg)

    # Lock this problem to the GPU model it was authored on.
    from .problem import set_gpu_model
    set_gpu_model(target, gpupool.gpu_name(cfg.gpu_indices[0]))
    print(f"[bootstrap] locked problem to GPU: {gpupool.gpu_name(cfg.gpu_indices[0])}", file=sys.stderr)

    commit_problem(target, repo_root)
    return target
