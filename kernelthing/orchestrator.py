"""The orchestration layer for the asynchronous evolutionary kernel search.

The search logic (population, operators, selection) is pure in ``evolve.py``;
this module is the side-effecting controller: it owns git worktrees, the agent
turns, the serialized GPU benchmark, and the loop budget. Problem-agnostic --
the objective fitness comes from the problem's ``score`` command (JSON {correct,
metric}); see problem.py and ``run()``.

Everything that happens is journaled to the run dir (see journal.py/state.py):
events to ``events.ndjson``, per-candidate artifacts to ``members/<id>/``. The
web UI is a pure reader of those files; live tuning (``-j``, budgets, stop)
arrives through ``control.json``, re-read at dispatch boundaries.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from . import evolve, gates, opencode_client, prompts
from .config import Config, format_duration
from .journal import MAX_PARALLELISM, Journal, LiveLock, LoopControl
from .problem import Problem
from .state import LoopDirs, State, new_timestamp, save_run

EXIT_COMPLETE = "complete"
EXIT_MAXITER = "maxiter"
EXIT_STOP = "stop"
EXIT_ERROR = "error"
EXIT_STALL = "stalled_out"  # several consecutive no-progress rounds; HEAD kept
EXIT_STOPPED = "stopped"  # user requested stop via UI


@dataclass
class RunContext:
    """Mutable state owned by a single evolutionary-search invocation.

    Separated from Orchestrator so the five closures that were tangled inside
    ``run()`` are regular methods taking this as a parameter.
    """

    dispatched: int = 0
    in_flight: dict[int, int] = field(default_factory=dict)
    futures: dict[Any, Any] = field(default_factory=dict)
    search_start: float | None = None


# --- prompts (rendered with problem fields) ---

# Methodology Analysis phase (ported from Humanize; adapted for headless opencode
# -- the Opus-subagent / AskUserQuestion / gh-issue flow is replaced by the agent
# writing the retrospective itself, since there is no interactive user here).
METHODOLOGY_PROMPT = """# Methodology Analysis (loop exit)

The optimization loop has exited.
- Exit reason: {{EXIT_REASON}} -- {{EXIT_REASON_DESCRIPTION}}
- Candidates dispatched: {{DISPATCHED}} (search budget: {{BUDGET}})
- Best result reached: {{BEST}}{{UNIT}}

Perform a retrospective on the *methodology* of this run (HOW the loop worked),
not the project itself. Read the development records in @{{LOOP_DIR}}:
- the candidate summaries and results (`members/*/summary.md`, `members/*/result.json`)
- the structured event journal (`events.ndjson`)
- the full narrative (`loop.log`)

Analyze from a pure methodology perspective. Focus areas:
- Iteration efficiency: were rounds productive, or repetitive?
- Best-of-N effectiveness: did parallel candidates explore genuinely diverse
  strategies, and did winners beat the incumbent by real margins vs. noise?
- Stagnation / plateau: where did progress slow, and why?
- Feedback quality: did reviewer feedback lead to real improvements?
- Benchmark trust: any signs of noisy or misleading measurements?
- Plan-to-execution alignment and round-count vs. progress.

Write a structured retrospective of general, transferable improvements to the
optimization methodology (describe patterns and process changes; avoid dumping
project-specific code) to:
  `{{LOOP_DIR}}/methodology-analysis-report.md`
If the methodology worked well, say so briefly. Then write a one-line completion
note to:
  `{{LOOP_DIR}}/methodology-analysis-done.md`

Do NOT edit any source files and do NOT commit -- only write those two files.
"""

# --- evolutionary-search operator prompts (Orchestrator.run) ---

EVOLVE_EXPLORE_PROMPT = """Your work is not finished. Read and execute the below with ultrathink.

## Plan
@{{PLAN}}

You are an **EXPLORE** candidate in an evolutionary kernel search: open a NEW line
of attack. The working tree is at a kernel that reaches {{PARENT_METRIC}}{{UNIT}}
(commit: {{PARENT_COMMIT_MESSAGE}}). Make ONE focused, correct improvement by
editing ONLY {{EDIT_FILES}}, using an optimization strategy DISTINCT from those
already tried:
{{KNOWN_STRATEGIES}}
Pick a genuinely different angle -- do not just retune one of the above.
"""

EVOLVE_EXPLOIT_PROMPT = """Your work is not finished. Read and execute the below with ultrathink.

## Plan
@{{PLAN}}

You are an **EXPLOIT** candidate: deepen a current best kernel. The working tree
is ALREADY at that kernel, which reaches {{PARENT_METRIC}}{{UNIT}}. The parent
commit message was:

  {{PARENT_COMMIT_MESSAGE}}

Push it further along the SAME approach by editing ONLY {{EDIT_FILES}}. Make ONE
focused, correct, measured improvement on top of it -- keep correctness.
"""

EVOLVE_DESCRIPTOR_FOOTER = """

---

## How to finish (REQUIRED)
Self-test by running the scorer:

    {{SCORE_CMD}}

It must report `"correct": true`. **Commit as soon as
you have ANY correct improvement** (`git add -A && git commit -m "..."`) -- your
last committed correct version is what gets scored, so a timeout never wastes the
run. Do NOT edit anything outside {{EDIT_FILES}}.

Write @candidate-summary.md with a brief description of the changes made -- 2-3
sentences or short bullet points, with the final measured metric ({{UNIT}}).
Keep it under 1000 characters.
"""


class Orchestrator:
    def __init__(self, problem: Problem, cfg: Config):
        self.problem = problem
        self.wd = Path(problem.repo_root).resolve()
        self.cfg = cfg
        # Multi-GPU pool: track in-flight tasks per GPU index. The first GPU in
        # the list is the "default" for operations (bootstrap, methodology) that
        # only need one device. Dispatch picks the least-busy GPU.
        self._gpu_indices = list(cfg.gpu_indices)
        self._gpu_in_flight: dict[int, int] = dict.fromkeys(self._gpu_indices, 0)
        # Run-dir plumbing, created in setup(): the journal is the event record,
        # control the live-knob channel, live_lock the liveness beacon.
        self.journal: Journal | None = None
        self.control: LoopControl | None = None
        self._live_lock: LiveLock | None = None
        self.impl_session: str | None = None
        self._best: float | None = None
        # Per-GPU pinned baseline (us). Keyed by GPU index. Each GPU gets its own
        # baseline measurement so pct_baseline/speedup ratios are valid when
        # candidates on different GPUs share one denominator.
        self._baselines: dict[int, float | None] = dict.fromkeys(self._gpu_indices)
        self._dispatched = 0
        self._logfile: Path | None = None
        self._git_lock = threading.Lock()

    # --- helpers ---
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[kernelthing] {msg}", file=sys.stderr, flush=True)
        if self._logfile is not None:
            try:
                with open(self._logfile, "a", encoding="utf-8") as f:
                    f.write(f"{stamp}  {msg}\n")
            except OSError:
                pass

    def _emit(self, type: str, **fields: Any) -> None:
        if self.journal is not None:
            self.journal.emit(type, **fields)

    def _rel(self, path: Path) -> str:
        return os.path.relpath(Path(path).resolve(), self.wd)

    def _git(self, args: list[str]) -> str:
        return gates.git(args, self.wd).stdout.strip()

    def _parallelism(self) -> int:
        return self.control.parallelism() if self.control else self.cfg.parallelism

    def _default_gpu(self) -> int:
        """First GPU in the pool (for seed, methodology, and other singleton ops)."""
        return self._gpu_indices[0]

    @contextlib.contextmanager
    def _hardware_locked(self) -> Iterator[None]:
        """Lock GPU clocks and power limit for reproducible benchmarking.

        Applied once before the seed baseline and held for the entire run;
        reset on exit even on crash/KeyboardInterrupt. A no-op when no
        hardware settings are configured.
        """
        cfg = self.cfg
        if cfg.power_limit is None and cfg.gpu_clock_lock is None and cfg.mem_clock_lock is None:
            yield
            return
        from . import gpucontrol

        hw = gpucontrol.HardwareConfig(
            power_limit_watts=cfg.power_limit,
            gpu_clock_lock=cfg.gpu_clock_lock,
            mem_clock_lock=cfg.mem_clock_lock,
            device_ids=list(self._gpu_indices),
        )
        try:
            with gpucontrol.HardwareLock(hw) as warnings:
                if warnings:
                    for w in warnings:
                        self._log(f"HW  WARNING {w}")
                else:
                    parts = []
                    if cfg.power_limit:
                        parts.append(f"power {cfg.power_limit}W")
                    if cfg.gpu_clock_lock:
                        parts.append(f"gpu clk {cfg.gpu_clock_lock[0]}-{cfg.gpu_clock_lock[1]}MHz")
                    if cfg.mem_clock_lock:
                        parts.append(f"mem clk {cfg.mem_clock_lock[0]}-{cfg.mem_clock_lock[1]}MHz")
                    self._log(f"HARDWARE locked: {', '.join(parts)}")
                yield
        finally:
            self._log("HARDWARE reset to defaults")

    def _gpu_names(self) -> dict[str, str]:
        """Product name per pool GPU for run.json (cross-run comparability)."""
        from . import gpupool

        names: dict[str, str] = {}
        for i in self._gpu_indices:
            try:
                names[str(i)] = gpupool.gpu_name(i)
            except Exception:
                names[str(i)] = f"GPU {i}"
        return names

    def _pick_gpu(self) -> int:
        """Return the GPU index with fewest in-flight tasks (round-robin on ties)."""
        return min(self._gpu_in_flight, key=lambda i: self._gpu_in_flight[i])

    def _budget_desc(self) -> str:
        """Human description of the real search budget -- the candidate/wall-clock
        bound ``want_more()`` enforces -- for agent prompts and logs."""
        parts = []
        if self.cfg.max_candidates:
            parts.append(f"{self.cfg.max_candidates} candidates")
        if self.cfg.wall_clock_s:
            parts.append(f"{format_duration(self.cfg.wall_clock_s)} wall-clock")
        if not parts:
            return "unbounded (runs until a manual stop)"
        return " or ".join(parts) + (" (whichever comes first)" if len(parts) > 1 else "")

    def _guard(
        self,
        dirs: LoopDirs,
        rnd: int,
        phase: str,
        project_root: Path | None = None,
        loop_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Context handed to the opencode PreToolUse guard (oc_guard/guard.js).

        Lets the guard reject writes/edits/reads/bash that would corrupt loop
        infrastructure (state file, plan, plan backup, prompt/summary/contract
        files, git push). ``phase`` is one of impl/review/finalize/methodology.
        """
        from .bootstrap import protected_files

        root = Path(project_root or self.wd).resolve()
        return {
            "loopDir": str((loop_dir or dirs.base).resolve()),
            "projectRoot": str(root),
            "planFile": self.problem.plan,
            "currentRound": rnd,
            "phase": phase,
            "editFiles": [str(Path(f)) for f in self.problem.edit_files],
            "editDir": str(root / self.problem.rel_dir)
            if self.problem.rel_dir not in ("", ".")
            else str(root),
            "protectedFiles": sorted(protected_files(self.problem)),
        }

    def _score_cmd_str(self) -> str:
        """Absolute-path scoring command the agent can self-test with.

        Constructs one from the venv so the agent never has to discover kernelthing
        on PATH. The problem's own ``score_command`` (if set) takes priority."""
        venv_bin = Path(sys.executable).parent
        kt = str(venv_bin / "kernelthing")
        if self.problem.score_command:
            return self.problem.score_command
        return f"{kt} score ."

    def _prompt_common(self, state: State) -> dict[str, Any]:
        """Template fields shared by every operator prompt + the descriptor footer."""

        return {
            "PLAN": state.plan_file,
            "PLAN_FILE": state.plan_file,
            "EDIT_FILES": ", ".join(self.problem.edit_files),
            "SCORE_CMD": self._score_cmd_str(),
            "UNIT": self.problem.unit or self.problem.metric_name,
        }

    @cached_property
    def _kernel_tools_block(self) -> str:
        """Assemble the optional kernel-domain tool guidance (KDA skills) appended
        to implementer prompts. Each part is included only when its flag is on;
        with both off this returns '' (no section). Absolute paths into the
        kernelthing install's ``vendor/`` are used -- the whole filesystem is
        read-only-bound in the sandbox, so they resolve from any worktree.

        Cached: every input (cfg flags, ``sys.executable``, ``REPO_ROOT``) is
        fixed for the run, so the block is built once instead of per candidate.
        """
        from .config import REPO_ROOT

        parts: list[str] = []
        pyexe = sys.executable or "python3"
        if self.cfg.sandbox:
            parts.append(
                "### Shared GPU — allocation is automatic\n\n"
                "You share a pool of GPUs with other agents and the scorer, and "
                "concurrent use of one card corrupts timings and can OOM. This is "
                "handled for you: just run your benchmark, `ncu`, or `nsys` "
                "normally — the harness transparently assigns your process a free "
                "GPU and holds it for that process's lifetime. **Do not** set "
                "`CUDA_VISIBLE_DEVICES` yourself; it is managed for you and "
                "overriding it will not give you a different card.\n\n"
                "If every GPU is busy, your process pauses at its first CUDA call "
                "until one frees — this is normal queuing, not a hang. Do not run "
                "`nvidia-smi` or `ps` to debug it; the command proceeds on its own. "
                "Use the wait time to plan or read."
            )
        if self.cfg.wiki:
            parts.append(
                prompts.load_and_render_safe(
                    "claude/kernel-tools-wiki.md",
                    "",
                    PYTHON=pyexe,
                    WIKI_DIR=str(REPO_ROOT / "vendor" / "KernelWiki"),
                )
            )
        if self.cfg.ncu:
            ncu_pp = sorted((Path("/opt/nvidia/nsight-compute")).glob("*/extras/python"))
            parts.append(
                prompts.load_and_render_safe(
                    "claude/kernel-tools-ncu.md",
                    "",
                    PYTHON=pyexe,
                    NCU_DIR=str(REPO_ROOT / "vendor" / "ncu-report-skill"),
                    NCU_BIN=shutil.which("ncu") or "/usr/local/cuda/bin/ncu",
                    NCU_PYTHONPATH=str(ncu_pp[-1]) if ncu_pp else "",
                )
            )
        parts = [p for p in parts if p.strip()]
        if not parts:
            return ""
        return "\n\n---\n\n## Kernel optimization tools (available in your sandbox)\n" + "\n".join(
            parts
        )

    # --- setup ---
    def setup(self) -> tuple[State, LoopDirs]:
        if not (self.wd / self.problem.plan).is_file():
            raise FileNotFoundError(f"plan not found: {self.wd / self.problem.plan}")
        if gates.git(["rev-parse", "--git-dir"], self.wd).returncode != 0:
            raise RuntimeError(f"{self.wd} is not a git repository")

        start_branch = self._git(["rev-parse", "--abbrev-ref", "HEAD"]) or "HEAD"
        base_commit = self._git(["rev-parse", "HEAD"])
        ts = new_timestamp()
        dirs = LoopDirs(self.wd, ts).ensure()
        self._logfile = dirs.logfile
        state = State(
            timestamp=ts,
            plan_file=self.problem.plan,
            model=self.cfg.model,
            start_branch=start_branch,
            base_branch=start_branch,
            base_commit=base_commit,
            methodology=self.cfg.methodology,
        )
        shutil.copyfile(self.wd / self.problem.plan, dirs.plan_backup)
        problem_meta = {
            "name": self.problem.name,
            "unit": self.problem.unit,
            "direction": self.problem.direction,
            "metric_name": self.problem.metric_name,
        }
        config_meta = {
            "parallelism": self.cfg.parallelism,
            "max_candidates": self.cfg.max_candidates,
            "wall_clock_s": self.cfg.wall_clock_s,
            "elite_k": self.cfg.elite_k,
            "min_niches": self.cfg.min_niches,
            "gpus": self._gpu_indices,
            "gpu_names": self._gpu_names(),
            "sandbox": self.cfg.sandbox,
            "kernelguard": self.cfg.kernelguard,
        }
        save_run(dirs, state, problem=problem_meta, config=config_meta)
        self._live_lock = LiveLock(dirs.live_lock)
        self._live_lock.acquire()
        self.journal = Journal(dirs.events_file)
        self.control = LoopControl(
            dirs.control_file,
            self.journal,
            parallelism=self.cfg.parallelism,
            elite_k=self.cfg.elite_k,
            wall_clock_s=self.cfg.wall_clock_s,
            max_candidates=self.cfg.max_candidates,
        )
        self._emit(
            "run_start",
            problem=problem_meta,
            config=config_meta,
            model=self.cfg.model,
            base_commit=base_commit,
            base_branch=start_branch,
        )
        self._log("=" * 72)
        self._log(f"SETUP  problem '{self.problem.name}'")
        self._log(f"  repo        {self.wd}")
        self._log(
            f"  metric      {self.problem.metric_name or self.problem.unit} "
            f"({self.problem.unit}, {self.problem.direction})"
        )
        self._log(f"  start       branch {start_branch} @ {base_commit[:8]}")
        self._log(
            f"  config      budget {self.cfg.max_candidates or '∞'} candidates"
            + (f"/{format_duration(self.cfg.wall_clock_s)}" if self.cfg.wall_clock_s else "")
            + f" | parallelism {self._parallelism()} | elite_k {self.cfg.elite_k} | "
            f"model {self.cfg.model}"
        )
        self._log(f"  artifacts   {dirs.base}  (full log: loop.log)")
        self._log("=" * 72)
        return state, dirs

    def _run_implementer(
        self,
        prompt: str,
        log_path: Path,
        *,
        guard: dict[str, Any] | None = None,
    ) -> opencode_client.OpencodeResult:
        res = opencode_client.run(
            prompt,
            working_dir=self.wd,
            model=self.cfg.model,
            session=self.impl_session,
            timeout=self.cfg.opencode_timeout,
            gpu_pool=self._gpu_indices,
            writable=True,
            sandboxed=self.cfg.sandbox,
            log_path=log_path,
            err_path=Path(str(log_path) + ".stderr"),
            guard=guard,
            ncu=self.cfg.ncu,
        )
        if res.session_id:
            self.impl_session = res.session_id
        return res

    def _cli_score(
        self,
        wt: Path,
        gpu_index: int,
        *,
        baseline_median: float | None = None,
        emit_baseline: bool = False,
    ) -> dict[str, Any]:
        """Score a worktree by shelling out to ``kernelthing score`` and parsing its
        JSON. Runs the *same* code path agents use, and -- crucially -- in its own
        process, so concurrent scorings never race on the shared in-process import
        state (``bench._importable`` mutates ``sys.path``/``sys.modules``/cwd).

        Returns the full verdict dict: ``{correct, metric, error, unit, bench}``
        plus ``baseline_median`` when ``emit_baseline`` is set (so the seed can
        pin it for later scores) and ``stderr_tail`` when the scorer wrote to
        stderr. ``bench`` is the raw measurement record (per-repeat timings).
        GPU exclusivity is handled inside the subprocess by the libktgpu shim.
        """
        prob_dir = wt / self.problem.rel_dir
        cmd = [
            sys.executable, "-m", "kernelthing", "score", str(prob_dir),
            "--gpu", str(gpu_index), "--override-gpu",
        ]
        if emit_baseline:
            cmd.append("--emit-baseline")
        elif baseline_median is not None:
            cmd += ["--baseline-median", repr(baseline_median)]
        try:
            r = subprocess.run(
                cmd, cwd=str(prob_dir), capture_output=True, text=True, timeout=1800
            )
        except subprocess.TimeoutExpired:
            return {"correct": False, "metric": None, "error": "score timeout"}
        stderr_tail = (r.stderr or "").strip()[-2000:]
        line = next(
            (ln.strip() for ln in reversed(r.stdout.splitlines()) if ln.strip().startswith("{")),
            "",
        )
        if not line:
            err = stderr_tail[-500:] or "score emitted no JSON"
            return {"correct": False, "metric": None, "error": err, "stderr_tail": stderr_tail}
        try:
            d: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return {
                "correct": False,
                "metric": None,
                "error": "score emitted no parseable JSON",
                "stderr_tail": stderr_tail,
            }
        if stderr_tail:
            d["stderr_tail"] = stderr_tail
        return d

    @staticmethod
    def _score_tuple(d: dict[str, Any]) -> tuple[bool, float | None, str | None, dict[str, Any]]:
        """Split a ``_cli_score`` verdict into (correct, metric, err, detail); the
        detail keeps everything else (bench timings, stderr tail) for result.json."""
        detail = {k: v for k, v in d.items() if k not in ("correct", "metric", "error", "unit")}
        return bool(d.get("correct")), d.get("metric"), d.get("error"), detail

    def _score_worktree(
        self, wt: Path, *, gpu_index: int = 0
    ) -> tuple[bool, float | None, str | None, dict[str, Any]]:
        """Score the worktree, returning (correct, metric, err, detail).

        Shells out to ``kernelthing score`` (one process per score -> no shared-state
        race), pinning this GPU's baseline denominator.
        """
        baseline = self._baselines.get(gpu_index)
        return self._score_tuple(self._cli_score(wt, gpu_index, baseline_median=baseline))

    def _guarded_score(
        self, wt: Path, *, gpu_index: int = 0
    ) -> tuple[bool, float | None, str | None, dict[str, Any]]:
        """Run the cheap static cheat gate (kernelguard) BEFORE the expensive bench:
        a detected cheat is disqualified outright and never scored. Scoring shells
        out to ``kernelthing score`` (see ``_cli_score``), which warm-builds the
        kernel off-lock itself, so the benchmark only pays runtime."""
        if self.cfg.kernelguard:
            cheats = gates.kernelguard_violations(
                self.problem.edit_files,
                wt,
                profile=self.cfg.kernelguard_profile,
                metadata={"problem_name": self.problem.name},
            )
            if cheats:
                err = "kernelguard: " + ", ".join(x["file"] for x in cheats)
                return False, None, err, {"kernelguard": cheats}
        # GPU access is serialized by the libktgpu.so shim inside pygpubench's
        # isolated worker (see bench._gpu_env) -- no in-process flock needed.
        return self._score_worktree(wt, gpu_index=gpu_index)

    # --- async evolutionary search ---
    @staticmethod
    def _fmt(x: float | None) -> str:
        return f"{x:.1f}" if isinstance(x, (int, float)) else "?"

    def _evolve_ref(self, ts: str, member_id: int) -> str:
        return f"refs/kernelthing/{ts}/mem-{member_id}"

    def _evolve_prompt(
        self, operator: str, parent: evolve.Member | None, state: State, known_strategies: str
    ) -> str:
        common = self._prompt_common(state)
        if operator == evolve.OP_EXPLORE:
            body = prompts.render(
                EVOLVE_EXPLORE_PROMPT,
                PARENT_METRIC=self._fmt(parent.metric) if parent else "?",
                PARENT_COMMIT_MESSAGE=(parent.commit_message if parent else "baseline"),
                KNOWN_STRATEGIES=known_strategies or "(none yet)",
                **common,
            )
        else:  # exploit
            assert parent is not None
            body = prompts.render(
                EVOLVE_EXPLOIT_PROMPT,
                PARENT_METRIC=self._fmt(parent.metric),
                PARENT_COMMIT_MESSAGE=parent.commit_message or "(no commit message)",
                **common,
            )
        return (
            body + prompts.render(EVOLVE_DESCRIPTOR_FOOTER, **common) + self._kernel_tools_block
        )

    def _evolve_task(
        self, task: evolve.Task, base: str, wt_root: Path, dirs: LoopDirs, gpu_index: int
    ) -> evolve.Member:
        """Worker: fork a worktree, run one agent turn, then (serialized) score it.

        Pure git plumbing is taken under ``self._git_lock``; the agent turn and the
        benchmark run outside it. GPU exclusivity is handled by the ``libktgpu.so``
        LD_PRELOAD shim: the agent's processes and the bench's isolated worker
        each flock a per-device lockfile (named in kernelthing/gpupool.py) on
        first CUDA use. Returns the scored Member; never raises.

        Everything about the attempt lands in ``members/<id>/`` -- the exact
        prompt, the live agent transcript, the summary, and (when it committed)
        the diff against its parent, which survives the run's ref cleanup.
        """
        m = evolve.Member(
            id=task.member_id, operator=task.operator, parent_id=task.parent_id, gpu=gpu_index
        )
        parent_commit = task.parent_commit or base
        wt = wt_root / f"m{task.member_id}"
        mdir = dirs.ensure_member(task.member_id)
        (mdir / "prompt.md").write_text(task.prompt, encoding="utf-8")
        try:
            with self._git_lock:
                rc = gates.git(
                    ["worktree", "add", "--detach", "--force", str(wt), parent_commit], self.wd
                ).returncode
            if rc != 0:
                m.error = "worktree add failed"
                return m
            guard = self._guard(
                dirs,
                task.member_id,
                "impl",
                project_root=wt,
                loop_dir=wt / ".humanize" / "rlcr" / "candidate",
            )
            t0 = time.time()
            res = opencode_client.run(
                task.prompt,
                working_dir=wt,
                model=self.cfg.model,
                session=None,
                timeout=self.cfg.opencode_timeout,
                gpu_pool=self._gpu_indices,
                writable=True,
                sandboxed=self.cfg.sandbox,
                log_path=dirs.member_log(task.member_id),
                err_path=dirs.member_stderr(task.member_id),
                data_dir=wt / ".humanize" / "oc-data",
                extra_writable=[self.wd / ".git"],
                guard=guard,
                ncu=self.cfg.ncu,
            )
            m.agent_s = round(time.time() - t0, 1)
            m.cost = res.cost
            m.tokens = res.tokens
            m.tool_calls = res.tool_calls
            m.agent_exit = res.exit_code
            head = gates.git(["rev-parse", "HEAD"], wt).stdout.strip()
            m.commit = head if head and head != parent_commit else None
            if m.commit:
                m.commit_message = gates.git(
                    ["log", "-1", "--format=%s", "HEAD"], wt
                ).stdout.strip()
                # Every commit of the turn, oldest first -- intermediate attempts
                # are part of the record even though only HEAD gets scored.
                m.commits = list(
                    reversed(
                        gates.git(
                            ["log", "--format=%h %s", f"{parent_commit}..HEAD"], wt
                        ).stdout.strip().splitlines()
                    )
                )
            sm = wt / "candidate-summary.md"
            m.summary_text = sm.read_text(encoding="utf-8") if sm.exists() else ""
            if m.commit is None:
                m.error = m.error or ("agent turn timed out" if res.exit_code == 124 else "no commit")
                return m
            # diff.patch is the *solution* diff only: agents are told to commit
            # with `git add -A`, so the raw parent..HEAD diff drags in profiling
            # dumps and other worktree junk. The full inventory of what else was
            # committed is kept as a name list (changed_files) in result.json.
            diff = gates.git(
                ["diff", f"{parent_commit}..HEAD", "--", *self.problem.edit_files], wt
            ).stdout
            if diff:
                (mdir / "diff.patch").write_text(diff, encoding="utf-8")
            m.changed_files = gates.git(
                ["diff", "--name-only", f"{parent_commit}..HEAD"], wt
            ).stdout.split()
            with self._git_lock:
                gates.git(["checkout", "--", *self.problem.edit_files], wt)
            t0 = time.time()
            m.correct, m.metric, m.error, m.score_detail = self._guarded_score(
                wt, gpu_index=gpu_index
            )
            m.score_s = round(time.time() - t0, 1)
        except Exception as e:
            m.error = repr(e)
        finally:
            with self._git_lock:
                gates.git(["worktree", "remove", "--force", str(wt)], self.wd)
        return m

    def _evolve_seed(self, base: str, wt_root: Path, pop: evolve.Population) -> evolve.Member:
        """Score the starting HEAD as member 0 so explore/exploit have an incumbent.

        If it scores, it is the first elite; if not, the population starts empty and
        every operator falls back to explore (forking the base) until one works.

        Baselines are measured on every GPU in the pool so each device gets its own
        pinned denominator for pct_baseline/speedup metrics. Scoring shells out to
        ``kernelthing score`` (see ``_cli_score``) once per GPU -- each in its own
        process, so nothing races on shared in-process import state.
        """
        m = evolve.Member(id=pop.next_id(), operator="seed", commit=base, commit_message="baseline")
        seed_gpu = self._default_gpu()
        m.gpu = seed_gpu
        wt = wt_root / "seed"
        with self._git_lock:
            gates.git(["worktree", "add", "--detach", "--force", str(wt), base], self.wd)
        try:
            if self.cfg.kernelguard:
                cheats = gates.kernelguard_violations(
                    self.problem.edit_files,
                    wt,
                    profile=self.cfg.kernelguard_profile,
                    metadata={"problem_name": self.problem.name},
                )
                if cheats:
                    m.error = "kernelguard: " + ", ".join(x["file"] for x in cheats)
                    m.score_detail = {"kernelguard": cheats}
                    return m

            # Score the seed and pin each GPU's baseline denominator by shelling out
            # to `kernelthing score --emit-baseline`, once per GPU. Each score is its
            # own process, so there is no shared-state race.
            for gpu in self._gpu_indices:
                correct, metric, err, detail = self._score_tuple(
                    self._cli_score(wt, gpu, emit_baseline=True)
                )
                bl = detail.get("baseline_median")
                if gpu == seed_gpu:
                    m.correct, m.metric, m.error, m.score_detail = correct, metric, err, detail
                if bl is not None:
                    self._baselines[gpu] = bl
                    self._emit("baseline_pinned", gpu=gpu, median_us=bl)
                    self._log(
                        f"GPU {gpu} baseline pinned: {bl:.1f}us"
                        + (" (= 100%, fixed for the run)" if gpu == seed_gpu else "")
                    )
                elif err:
                    self._emit("baseline_failed", gpu=gpu, error=err)
                    self._log(f"GPU {gpu} baseline pin failed ({err})")
        finally:
            with self._git_lock:
                gates.git(["worktree", "remove", "--force", str(wt)], self.wd)
        pop.insert(m)
        return m

    def _record_member(self, dirs: LoopDirs, m: evolve.Member) -> None:
        """Persist a settled member: summary + result.json + the journal event."""
        dirs.ensure_member(m.id)
        if m.summary_text:
            dirs.member_summary(m.id).write_text(m.summary_text, encoding="utf-8")
        rec = m.record()
        dirs.member_result(m.id).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        self._emit("member_result", **rec)

    # --- evolutionary-search control (extracted from run() closures) ---

    def _ev_wall_limit(self) -> int:
        """Live wall-clock budget in seconds (0 = off)."""
        return self.control.wall_clock() if self.control else self.cfg.wall_clock_s

    def _ev_target(self) -> int:
        """Live -j: how many agents to keep in flight right now. The worker pool
        is sized to MAX_PARALLELISM (threads spawn lazily), so this can be
        raised as well as lowered mid-run."""
        return max(1, min(self._parallelism(), MAX_PARALLELISM))

    def _ev_want_more(self, rc: RunContext) -> bool:
        """True while neither the candidate budget, wall-clock, nor stop flag is hit."""
        if self.control and self.control.stop_requested():
            return False
        maxc = self.control.max_candidates() if self.control else self.cfg.max_candidates
        if maxc and rc.dispatched >= maxc:
            return False
        limit = self._ev_wall_limit()
        assert rc.search_start is not None  # set before dispatch loop begins
        return not (limit and time.time() - rc.search_start >= limit)

    def _ev_dispatch(
        self,
        rc: RunContext,
        pop: evolve.Population,
        state: State,
        base: str,
        wt_root: Path,
        dirs: LoopDirs,
        rng: random.Random,
        ex: ThreadPoolExecutor,
    ) -> None:
        """Pick an operator + parent, fork a worker, and register the future."""
        cfg = self.cfg
        if self.control:
            # Live -k: resize the exploit frontier before selection sees it.
            pop.elite_k = self.control.elite_k()
        have_elites = bool(pop.elites())
        if self.control and not self.control.explore_auto():
            explore_frac = self.control.explore_bias() / 100.0
        elif cfg.max_candidates:
            progress = min(1.0, rc.dispatched / max(cfg.max_candidates, 1))
            explore_frac = 0.8 - 0.6 * progress
        else:
            explore_frac = 0.5
        op = evolve.choose_operator(
            rng,
            {evolve.OP_EXPLORE: explore_frac, evolve.OP_EXPLOIT: 1.0 - explore_frac},
            have_elites=have_elites,
            n_niches=len(pop.niches()),
            min_niches=cfg.min_niches,
        )
        parent = pop.select_parent(op, rng, rc.in_flight)
        if parent is None:
            op = evolve.OP_EXPLORE
        mid = pop.next_id()
        prompt = self._evolve_prompt(op, parent, state, ", ".join(sorted(pop.niches().keys())))
        task = evolve.Task(
            member_id=mid,
            operator=op,
            parent_id=(parent.id if parent else None),
            parent_commit=(parent.commit if parent else None),
            prompt=prompt,
        )
        gpu = self._pick_gpu()
        self._gpu_in_flight[gpu] += 1
        fut = ex.submit(self._evolve_task, task, base, wt_root, dirs, gpu)
        rc.futures[fut] = (task, gpu)
        if parent is not None:
            rc.in_flight[parent.id] = rc.in_flight.get(parent.id, 0) + 1
            parent.children += 1
        rc.dispatched += 1
        self._emit(
            "dispatch",
            member=mid,
            op=op,
            parent=(parent.id if parent else None),
            parent_metric=(parent.metric if parent else None),
            gpu=gpu,
            in_flight=len(rc.futures),
            dispatched=rc.dispatched,
            # Selection context at dispatch time -- why the search made this
            # choice is otherwise unreconstructable (RNG + live population).
            explore_frac=round(explore_frac, 3),
            niches=len(pop.niches()),
            elites=len(pop.elites()),
        )
        ptxt = f" <- mem {parent.id}" if parent else ""
        self._log(f"dispatch mem {mid}: {op}{ptxt} -> GPU {gpu}  (in-flight {len(rc.futures)})")

    def _ev_collect(
        self,
        rc: RunContext,
        pop: evolve.Population,
        state: State,
        dirs: LoopDirs,
        unit: str,
        fut: Any,
    ) -> None:
        """Absorb one completed future into the population (the run loop refills)."""
        task, gpu = rc.futures.pop(fut)
        self._gpu_in_flight[gpu] = max(0, self._gpu_in_flight.get(gpu, 1) - 1)
        if task.parent_id is not None:
            rc.in_flight[task.parent_id] = max(0, rc.in_flight.get(task.parent_id, 1) - 1)
        try:
            m = fut.result()
        except Exception as e:
            m = evolve.Member(
                id=task.member_id,
                operator=task.operator,
                parent_id=task.parent_id,
                error=repr(e),
            )
        pop.insert(m)
        if m.viable:
            assert m.commit is not None
            with self._git_lock:
                gates.git(
                    ["update-ref", self._evolve_ref(state.timestamp, m.id), m.commit],
                    self.wd,
                )
        self._record_member(dirs, m)
        best = pop.best()
        prev_best = self._best
        self._best = best.metric if best else self._best
        if best is not None and self._best != prev_best:
            self._emit("new_best", member=best.id, metric=best.metric)
        ptxt = f"<-{task.parent_id} " if task.parent_id is not None else ""
        res = (
            f"{m.metric:.1f}{unit} ✓ [{m.commit_message[:50]}]"
            if m.viable
            else f"✗ {m.error or 'no result'}"
        )
        self._log(
            f"result mem {m.id} ({m.operator} {ptxt}): {res}  · best {self._fmt(self._best)}{unit}"
        )

    # --- run: the evolutionary search loop ---

    def run(self) -> str:
        """Steady-state asynchronous evolutionary search (see kernelthing/evolve.py).

        Keeps up to ``-j``/parallelism agents editing at once (live-tunable both
        ways via the web UI, along with -k/-m/-w); all GPU work is serialized by
        the per-device flock. Dispatches explore/exploit tasks against a durable
        population until the budget is spent or a stop is requested, then
        promotes the best kernel to HEAD.
        """
        try:
            with self._hardware_locked():
                return self._run()
        finally:
            if self.journal is not None:
                self.journal.close()
            if self._live_lock is not None:
                self._live_lock.release()

    def _run(self) -> str:
        state, dirs = self.setup()
        cfg = self.cfg
        unit = self.problem.unit
        rng = random.Random(cfg.evolve_seed)
        pop = evolve.Population(direction=self.problem.direction, elite_k=cfg.elite_k)
        base = self._git(["rev-parse", "HEAD"])
        self._base_commit = base
        wt_root = cfg.problem_root / "wt" / state.timestamp / "evolve"
        shutil.rmtree(wt_root, ignore_errors=True)
        wt_root.mkdir(parents=True, exist_ok=True)

        pool_cap = max(1, cfg.parallelism)
        self._log("")
        self._log(
            f"──── EVOLVE  parallelism {pool_cap} · budget "
            f"{cfg.max_candidates or '∞'} candidates"
            + (f" / {format_duration(cfg.wall_clock_s)}" if cfg.wall_clock_s else "")
            + " ────"
        )

        rc = RunContext()
        self._emit("phase", phase="evolve")

        seed = self._evolve_seed(base, wt_root, pop)
        self._best = seed.metric if seed.viable else None
        self._record_member(dirs, seed)
        if seed.viable:
            self._emit("new_best", member=seed.id, metric=seed.metric)
        self._log(
            f"seed (HEAD {base[:8]}): "
            + (f"{seed.metric:.1f}{unit} ✓" if seed.viable else f"✗ {seed.error or 'no score'}")
        )

        rc.search_start = time.time()
        self._emit("search_start")

        # The pool is sized to the hard cap, not -j: threads spawn lazily, so the
        # live parallelism target (_ev_target) alone decides how many agents run.
        # That is what lets -j be raised mid-run, not just lowered.
        with ThreadPoolExecutor(max_workers=MAX_PARALLELISM) as ex:
            while self._ev_want_more(rc) and len(rc.futures) < self._ev_target():
                self._ev_dispatch(rc, pop, state, base, wt_root, dirs, rng, ex)
            while rc.futures:
                # Bounded wait: wake periodically to re-read control.json so a
                # raised -j refills immediately instead of at the next result.
                done, _ = wait(list(rc.futures), timeout=15, return_when=FIRST_COMPLETED)
                for fut in done:
                    self._ev_collect(rc, pop, state, dirs, unit, fut)
                while self._ev_want_more(rc) and len(rc.futures) < self._ev_target():
                    self._ev_dispatch(rc, pop, state, base, wt_root, dirs, rng, ex)

        self._dispatched = rc.dispatched
        best = pop.best()
        if best and best.commit:
            with self._git_lock:
                gates.git(["reset", "--hard", best.commit], self.wd)
            self._emit("promoted", member=best.id, commit=best.commit, metric=best.metric)
            self._log(
                f"evolve: promoted mem {best.id} @ {best.metric:.1f}{unit}"
                + (f" [{best.commit_message[:50]}]" if best.commit_message else "")
                + " to HEAD"
            )
            self._copy_best_kernel()
        else:
            self._log("evolve: no viable kernel found; HEAD unchanged")
        self._evolve_cleanup(state, pop, wt_root)

        if self.control and self.control.stop_requested():
            return self._finish(state, dirs, EXIT_STOPPED, "stopped by user via the web UI")
        if best is None:
            return self._finish(
                state,
                dirs,
                EXIT_STALL,
                f"evolutionary search ({rc.dispatched} candidates) found nothing viable",
            )
        return self._finish(
            state,
            dirs,
            EXIT_MAXITER,
            f"evolutionary search budget spent: {rc.dispatched} candidates, "
            f"best {best.metric:.1f}{unit} [{best.operator}]",
        )

    def _copy_best_kernel(self) -> Path:
        """Copy the current HEAD kernel files to the stable best-kernel path."""
        out = self.cfg.problem_root / f"{self.problem.name}-best"
        out.mkdir(parents=True, exist_ok=True)
        for f in self.problem.edit_files:
            src = self.wd / self.problem.rel_dir / f
            if src.is_file():
                shutil.copy(src, out / Path(f).name)
        self._log(f"persisted best kernel to {out}")
        return out

    def persist_current_head(self) -> None:
        """Copy whatever kernel is at HEAD to the stable best-kernel path.

        Called on KeyboardInterrupt so a killed run still saves the last promoted
        kernel, not just a clean exit. Best-effort; swallows all errors."""
        try:
            head = self._git(["rev-parse", "HEAD"])
        except Exception:
            return
        base = getattr(self, "_base_commit", "")
        if base and head == base:
            self._log("interrupted: HEAD unchanged from baseline, nothing to persist")
            return
        if not base:
            self._log("interrupted before seed scoring, nothing to persist")
            return
        self._copy_best_kernel()

    def _evolve_cleanup(self, state: State, pop: evolve.Population, wt_root: Path) -> None:
        with self._git_lock:
            for m in pop.members:
                if m.viable:
                    gates.git(
                        ["update-ref", "-d", self._evolve_ref(state.timestamp, m.id)], self.wd
                    )
            gates.git(["worktree", "prune"], self.wd)
        shutil.rmtree(wt_root, ignore_errors=True)

    # --- terminal exit + optional methodology analysis (ported from Humanize) ---
    def _finish(self, state: State, dirs: LoopDirs, reason: str, desc: str) -> str:
        if self.cfg.methodology and reason in (EXIT_COMPLETE, EXIT_MAXITER, EXIT_STALL, EXIT_STOP):
            try:
                self._methodology_phase(state, dirs, reason, desc)
            except Exception as e:  # methodology must never break the exit
                self._log(f"methodology phase error (ignored): {e!r}")
        self._log(f"loop exit: {reason} ({desc})")
        self._emit(
            "run_end", reason=reason, desc=desc, dispatched=self._dispatched, best=self._best
        )
        return reason

    def _methodology_phase(self, state: State, dirs: LoopDirs, exit_reason: str, desc: str) -> None:
        """Final retrospective on the run's methodology, written to the loop dir.

        Faithful to Humanize: analyze the round summaries/reviews from a pure
        methodology perspective and write a report + completion marker; gate on
        both existing with content (retry a few times), then exit.
        """
        done = dirs.base / "methodology-analysis-done.md"
        report = dirs.base / "methodology-analysis-report.md"
        if done.is_file() and done.read_text(encoding="utf-8").strip():
            return  # already done
        self._log("")
        self._log(f"──── METHODOLOGY ANALYSIS (exit: {exit_reason}) ────")
        self._emit("phase", phase="methodology")
        prompt = prompts.render(
            METHODOLOGY_PROMPT,
            LOOP_DIR=self._rel(dirs.base),
            EXIT_REASON=exit_reason,
            EXIT_REASON_DESCRIPTION=desc,
            DISPATCHED=self._dispatched,
            BUDGET=self._budget_desc(),
            BEST=(self._best if self._best is not None else "n/a"),
            UNIT=self.problem.unit,
        )
        log_path = dirs.base / "methodology-opencode.log"
        guard = self._guard(dirs, state.current_round, "methodology")
        for _ in range(3):
            res = self._run_implementer(prompt, log_path, guard=guard)
            self._emit(
                "methodology_turn",
                cost=res.cost,
                tokens=res.tokens,
                tool_calls=res.tool_calls,
                exit=res.exit_code,
            )
            self._log(
                f"methodology: analysis turn done (tools={res.tool_calls}, cost=${res.cost:.4f})"
            )
            if (
                report.is_file()
                and report.read_text(encoding="utf-8").strip()
                and done.is_file()
                and done.read_text(encoding="utf-8").strip()
            ):
                self._log(f"methodology: retrospective written -> {self._rel(report)}")
                return
            prompt = (
                f"The methodology analysis is incomplete. Write the retrospective to "
                f"{self._rel(report)} and a one-line completion note to {self._rel(done)}."
            )
        self._log("methodology: still incomplete after retries; continuing exit")
