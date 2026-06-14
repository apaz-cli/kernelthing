"""The orchestration layer for the asynchronous evolutionary kernel search.

The search logic (population, operators, selection) is pure in ``evolve.py``;
this module is the side-effecting controller: it owns git worktrees, the agent
turns, the serialized GPU benchmark, and the loop budget. Problem-agnostic --
the objective fitness comes from the problem's ``score`` command (JSON {correct,
metric}); see problem.py and ``run()``. ``-j``/parallelism (agents at once) and
stop flow through an optional LoopBus shared with the web UI.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import bench, evolve, gates, gpulock, opencode_client, prompts
from .bus import LoopBus
from .config import Config, format_duration
from .problem import Problem
from .state import LoopDirs, State, new_timestamp, save_state

EXIT_COMPLETE = "complete"
EXIT_MAXITER = "maxiter"
EXIT_STOP = "stop"
EXIT_ERROR = "error"
EXIT_STALL = "stalled_out"   # several consecutive no-progress rounds; HEAD kept
EXIT_STOPPED = "stopped"     # user requested stop via UI

# --- prompts (rendered with problem fields) ---

GOAL_TRACKER_SEED = """# Goal Tracker

## IMMUTABLE SECTION (set in Round 0)

### Ultimate Goal
[To be filled in Round 0 from the plan]

### Acceptance Criteria
[To be filled in Round 0 from the plan]

## MUTABLE SECTION (updated every round)

#### Active Tasks
#### Completed and Verified
#### Explicitly Deferred
#### Blocking Side Issues
#### Queued Side Issues
#### Plan Evolution Log
"""

BITLESSON_SEED = """# BitLessons

Project-specific, hard-won lessons. Each round summary records a
`## BitLesson Delta` (Action: none|add|update).
"""

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
- the round summaries (`round-*-summary.md`)
- the reviewer results (`round-*-review-result.md`)
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

    {{GPU_RUN}} {{SCORE_CMD}}

It must report `"correct": true`. **Commit as soon as
you have ANY correct improvement** (`git add -A && git commit -m "..."`) -- your
last committed correct version is what gets scored, so a timeout never wastes the
run. You may then attempt a bigger gain, keeping the first as a fallback. Do NOT
edit anything outside {{EDIT_FILES}}.

Write @candidate-summary.md with a brief description of the changes made -- 2-3
sentences or short bullet points, with the final measured metric ({{UNIT}}).
Keep it under 1000 characters.
"""


class Orchestrator:
    def __init__(self, problem: Problem, cfg: Config, bus: LoopBus | None = None):
        self.problem = problem
        self.wd = Path(problem.repo_root).resolve()
        self.cfg = cfg
        # gpu_index is the single source of truth for which device everything runs
        # on. Pin it on our own process env so the default pygpubench benchmark
        # (bench.score -> pygpubench's subprocess, which only *inherits* env) lands
        # on the same device as the agent turns and the legacy score_command, both
        # of which already force CUDA_VISIBLE_DEVICES=gpu_index on their children.
        # The loop targets exactly one device, so this is a single index.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)
        self.bus = bus
        self.impl_session: str | None = None   # reused only by the methodology turn
        self._history: list[dict] = []   # [{round: member_id, metric}] of viable members
        self._best: float | None = None
        # Baseline reference time (us), measured once at run start and pinned as the
        # fixed denominator for pct_baseline/speedup so every candidate shares one
        # baseline (see _evolve_seed / bench.measure_baseline). None = not pinned;
        # scoring then falls back to per-candidate baseline measurement.
        self._baseline_us: float | None = None
        self._dispatched = 0   # candidates dispatched this run (for the methodology phase)
        self._logfile: Path | None = None  # full persistent narrative (set in setup)
        # Serializes git plumbing that touches shared repo state (worktree
        # add/remove, update-ref, reset) across the evolve worker threads.
        self._git_lock = threading.Lock()

    # --- helpers ---
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[kernelthing] {msg}", file=sys.stderr, flush=True)
        if self.bus:
            self.bus.log(msg)
        if self._logfile is not None:
            try:
                with open(self._logfile, "a", encoding="utf-8") as f:
                    f.write(f"{stamp}  {msg}\n")
            except OSError:
                pass

    def _publish(self, **kw) -> None:
        if self.bus:
            self.bus.publish(**kw)

    def _rel(self, path: Path) -> str:
        return os.path.relpath(Path(path).resolve(), self.wd)

    def _git(self, args: list[str]) -> str:
        return gates.git(args, self.wd).stdout.strip()

    def _parallelism(self) -> int:
        return self.bus.parallelism() if self.bus else self.cfg.parallelism

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

    def _guard(self, dirs: LoopDirs, rnd: int, phase: str,
               project_root: Path | None = None, loop_dir: Path | None = None) -> dict:
        """Context handed to the opencode PreToolUse guard (oc_guard/guard.js).

        Lets the guard reject writes/edits/reads/bash that would corrupt loop
        infrastructure (state file, plan, plan backup, prompt/summary/contract
        files, git push). ``phase`` is one of impl/review/finalize/methodology.
        """
        from .bootstrap import _protected_files
        from .config import REPO_ROOT
        root = Path(project_root or self.wd).resolve()
        return {
            "loopDir": str((loop_dir or dirs.base).resolve()),
            "projectRoot": str(root),
            "planFile": self.problem.plan,
            "currentRound": rnd,
            "phase": phase,
            "gpuRun": str(REPO_ROOT / "kernelthing" / "gpu_run.sh"),
            "editFiles": [str(Path(f)) for f in self.problem.edit_files],
            "editDir": str(root / self.problem.rel_dir) if self.problem.rel_dir not in ("", ".") else str(root),
            "protectedFiles": sorted(_protected_files(self.problem)),
        }

    def _score_cmd_str(self) -> str:
        """Absolute-path scoring command the agent can self-test with.
        
        When pygpubench is on the problem's score_command is empty (the orchestrator
        uses bench.score directly), but the agent still needs a working self-test
        command. Construct one from the venv so the agent never has to discover
        kernelthing on PATH.
        """
        from .config import REPO_ROOT
        venv_bin = Path(sys.executable).parent
        kt = str(venv_bin / "kernelthing")
        if self.problem.score_command:
            return self.problem.score_command
        return f"{kt} score ."

    def _prompt_common(self, state: State) -> dict:
        """Template fields shared by every operator prompt + the descriptor footer."""
        from .config import REPO_ROOT
        return dict(
            PLAN=state.plan_file,
            PLAN_FILE=state.plan_file,
            EDIT_FILES=", ".join(self.problem.edit_files),
            SCORE_CMD=self._score_cmd_str(),
            GPU_RUN=str(REPO_ROOT / "kernelthing" / "gpu_run.sh"),
            UNIT=self.problem.unit or self.problem.metric_name,
            BITLESSON_FILE=state.bitlesson_file,
            KERNEL_TOOLS=self._kernel_tools_block(),
        )

    def _kernel_tools_block(self) -> str:
        """Assemble the optional kernel-domain tool guidance (KDA skills) appended
        to implementer prompts. Each part is included only when its flag is on;
        with both off this returns '' (no section). Absolute paths into the
        kernelthing install's ``vendor/`` are used -- the whole filesystem is
        read-only-bound in the sandbox, so they resolve from any worktree.
        """
        from .config import REPO_ROOT
        parts: list[str] = []
        pyexe = sys.executable or "python3"
        gpu_run = str(REPO_ROOT / "kernelthing" / "gpu_run.sh")
        if self.cfg.sandbox:
            parts.append(
                "### Shared GPU — serialize every GPU command\n\n"
                "You share one physical GPU with other agents and the scorer. "
                "Running on it concurrently corrupts timings and can OOM, so EVERY "
                "command that executes on the device — your compiled benchmark, "
                f"`ncu`, `nsys` — MUST be wrapped in `{gpu_run}`, which holds a "
                "per-device lock for the command's duration:\n\n"
                f"```bash\n{gpu_run} ./your_bench --args\n```\n\n"
                "When another agent holds the lock, `gpu-run` will **block silently** "
                "until released — this is normal queuing, not a hang. Do not run "
                "`nvidia-smi` or `ps` to debug it; the command will proceed on its "
                "own. Use the wait time to plan or read.\n\n"
                "CPU-only commands (building with `nvcc`, querying KernelWiki, "
                "parsing `.ncu-rep` files) do NOT need it. Profilers are enforced; "
                "wrap your benchmark binaries too.")
        if self.cfg.wiki:
            parts.append(prompts.load_and_render_safe(
                "claude/kernel-tools-wiki.md", "",
                PYTHON=pyexe,
                WIKI_DIR=str(REPO_ROOT / "vendor" / "KernelWiki")))
        if self.cfg.ncu:
            ncu_pp = sorted((Path("/opt/nvidia/nsight-compute")).glob("*/extras/python"))
            parts.append(prompts.load_and_render_safe(
                "claude/kernel-tools-ncu.md", "",
                PYTHON=pyexe,
                GPU_RUN=gpu_run,
                NCU_DIR=str(REPO_ROOT / "vendor" / "ncu-report-skill"),
                NCU_BIN=shutil.which("ncu") or "/usr/local/cuda/bin/ncu",
                NCU_PYTHONPATH=str(ncu_pp[-1]) if ncu_pp else ""))
        parts = [p for p in parts if p.strip()]
        if not parts:
            return ""
        return "\n\n---\n\n## Kernel optimization tools (available in your sandbox)\n" + "\n".join(parts)

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
        self._logfile = dirs.base / "loop.log"
        state = State(
            timestamp=ts, plan_file=self.problem.plan,
            model=self.cfg.model,
            start_branch=start_branch, base_branch=start_branch, base_commit=base_commit,
            bitlesson_required=self.cfg.bitlesson_required,
            bitlesson_file=".humanize/bitlesson.md", methodology=self.cfg.methodology,
        )
        shutil.copyfile(self.wd / self.problem.plan, dirs.plan_backup)
        dirs.goal_tracker.write_text(GOAL_TRACKER_SEED, encoding="utf-8")
        bl = self.wd / state.bitlesson_file
        bl.parent.mkdir(parents=True, exist_ok=True)
        if not bl.exists():
            bl.write_text(BITLESSON_SEED, encoding="utf-8")
        save_state(dirs, state)
        self._log("=" * 72)
        self._log(f"SETUP  problem '{self.problem.name}'")
        self._log(f"  repo        {self.wd}")
        self._log(f"  metric      {self.problem.metric_name or self.problem.unit} "
                  f"({self.problem.unit}, {self.problem.direction})")
        self._log(f"  start       branch {start_branch} @ {base_commit[:8]}")
        self._log(f"  config      budget {self.cfg.max_candidates or '∞'} candidates"
                  + (f"/{format_duration(self.cfg.wall_clock_s)}" if self.cfg.wall_clock_s else "")
                  + f" | parallelism {self._parallelism()} | elite_k {self.cfg.elite_k} | "
                  f"model {self.cfg.model}")
        self._log(f"  artifacts   {dirs.base}  (full log: loop.log)")
        self._log("=" * 72)
        self._publish(problem=self.problem.name, unit=self.problem.unit,
                      direction=self.problem.direction,
                      loop_dir=str(dirs.base), logfile="loop.log",
                      history=self._history, best=self._best, agents=[])
        return state, dirs

    def _run_implementer(self, prompt: str, log_path: Path,
                         guard: dict | None = None) -> opencode_client.OpencodeResult:
        res = opencode_client.run(
            prompt, working_dir=self.wd, model=self.cfg.model, session=self.impl_session,
            timeout=self.cfg.opencode_timeout, gpu_index=self.cfg.gpu_index,
            writable=True, sandboxed=self.cfg.sandbox, log_path=log_path, guard=guard,
            ncu=self.cfg.ncu, gpu_lock=gpulock.lock_path(self.cfg.gpu_index))
        if res.session_id:
            self.impl_session = res.session_id
        return res

    def _score_worktree(self, wt: Path) -> tuple[bool, float | None, str | None]:
        """Score the worktree, returning (correct, metric, err).

        Default (cfg.pygpubench): delegate to the in-process adversarial benchmark
        (bench.score), which owns its own repeats -- we do NOT loop it.
        cfg.pygpubench off: run the problem's score command bench_runs times;
        all_correct requires correct on every run (rejects flaky/racy
        implementations); best_metric is the best observed across runs per the
        problem's direction (robust to jitter).
        """
        if self.cfg.pygpubench:
            return bench.score(self.problem, wt, baseline_median=self._baseline_us)
        cwd = wt / self.problem.rel_dir
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.cfg.gpu_index))
        all_correct, metrics, err = True, [], None
        for _ in range(self.problem.bench_runs):
            try:
                r = subprocess.run(self.problem.score_command, shell=True, cwd=str(cwd),
                                   capture_output=True, text=True, env=env, timeout=900)
            except subprocess.TimeoutExpired:
                return False, None, "score timeout"
            correct, metric = bench.parse_score(r.stdout)
            if not correct:
                all_correct = False
            if metric is not None:
                metrics.append(metric)
            elif err is None:
                err = "score emitted no parseable JSON"
        if not metrics:
            return False, None, err or "no metric"
        best = max(metrics) if self.problem.direction == "maximize" else min(metrics)
        return all_correct, best, None

    def _guarded_score(self, wt: Path) -> tuple[bool, float | None, str | None]:
        """Run the cheap static cheat gate (kernelguard) BEFORE the expensive bench:
        a detected cheat is disqualified outright and never scored. The kernel is
        pre-compiled outside the GPU lock so the benchmark only pays runtime."""
        if self.cfg.pygpubench:
            bench.warm_build(self.problem, wt)
        if self.cfg.kernelguard:
            cheats = gates.kernelguard_violations(
                self.problem.edit_files, wt, profile=self.cfg.kernelguard_profile,
                metadata={"problem_name": self.problem.name})
            if cheats:
                return False, None, "kernelguard: " + ", ".join(x["file"] for x in cheats)
        with gpulock.gpu_lock(self.cfg.gpu_index):
            return self._score_worktree(wt)

    # --- async evolutionary search ---
    @staticmethod
    def _fmt(x: float | None) -> str:
        return f"{x:.1f}" if isinstance(x, (int, float)) else "?"

    def _evolve_ref(self, ts: str, member_id: int) -> str:
        return f"refs/kernelthing/{ts}/mem-{member_id}"

    def _evolve_prompt(self, operator: str, parent: evolve.Member | None,
                       state: State, known_strategies: str) -> str:
        common = self._prompt_common(state)
        if operator == evolve.OP_EXPLORE:
            body = prompts.render(EVOLVE_EXPLORE_PROMPT,
                                  PARENT_METRIC=self._fmt(parent.metric) if parent else "?",
                                  PARENT_COMMIT_MESSAGE=(parent.commit_message if parent
                                                         else "baseline"),
                                  KNOWN_STRATEGIES=known_strategies or "(none yet)", **common)
        else:  # exploit
            assert parent is not None
            body = prompts.render(EVOLVE_EXPLOIT_PROMPT,
                                  PARENT_METRIC=self._fmt(parent.metric),
                                  PARENT_COMMIT_MESSAGE=parent.commit_message or "(no commit message)",
                                  **common)
        return body + prompts.render(EVOLVE_DESCRIPTOR_FOOTER, **common) + common["KERNEL_TOOLS"]

    def _evolve_task(self, task: evolve.Task, base: str, wt_root: Path,
                     dirs: LoopDirs) -> evolve.Member:
        """Worker: fork a worktree, run one agent turn, then (serialized) score it.

        Pure git plumbing is taken under ``self._git_lock``; the agent turn and the
        benchmark run outside it. GPU exclusivity is handled by the per-device
        flock (kernelthing/gpulock.py): the agent holds it for its own GPU work via
        the ``gpu-run`` wrapper, and ``_guarded_score`` holds it for the bench.
        Returns the scored Member; never raises.
        """
        m = evolve.Member(id=task.member_id, operator=task.operator, parent_id=task.parent_id)
        parent_commit = task.parent_commit or base
        wt = wt_root / f"m{task.member_id}"
        try:
            with self._git_lock:
                rc = gates.git(["worktree", "add", "--detach", "--force",
                                str(wt), parent_commit], self.wd).returncode
            if rc != 0:
                m.error = "worktree add failed"
                return m
            log = dirs.base / f"mem-{task.member_id}-{task.operator}-opencode.log"
            guard = self._guard(dirs, task.member_id, "impl", project_root=wt,
                                loop_dir=wt / ".humanize" / "rlcr" / "candidate")
            opencode_client.run(
                task.prompt, working_dir=wt, model=self.cfg.model, session=None,
                timeout=self.cfg.opencode_timeout, gpu_index=self.cfg.gpu_index,
                writable=True, sandboxed=self.cfg.sandbox, log_path=log,
                data_dir=wt / ".humanize" / "oc-data", extra_writable=[self.wd / ".git"],
                guard=guard, ncu=self.cfg.ncu,
                gpu_lock=gpulock.lock_path(self.cfg.gpu_index))
            head = gates.git(["rev-parse", "HEAD"], wt).stdout.strip()
            m.commit = head if head and head != parent_commit else None
            if m.commit:
                m.commit_message = gates.git(
                    ["log", "-1", "--format=%s", "HEAD"], wt).stdout.strip()
            sm = wt / "candidate-summary.md"
            m.summary_text = sm.read_text(encoding="utf-8") if sm.exists() else ""
            if m.commit is None:
                m.error = m.error or "no commit"
                return m
            # score the COMMITTED state: restore edit files from HEAD, gate, bench
            with self._git_lock:
                gates.git(["checkout", "--", *self.problem.edit_files], wt)
            m.correct, m.metric, m.error = self._guarded_score(wt)
        except Exception as e:  # noqa: BLE001  -- a crashed worker must not kill the loop
            m.error = repr(e)
        finally:
            with self._git_lock:
                gates.git(["worktree", "remove", "--force", str(wt)], self.wd)
        return m

    def _evolve_seed(self, base: str, wt_root: Path, pop: evolve.Population) -> evolve.Member:
        """Score the starting HEAD as member 0 so explore/exploit have an incumbent.

        If it scores, it is the first elite; if not, the population starts empty and
        every operator falls back to explore (forking the base) until one works.
        """
        m = evolve.Member(id=pop.next_id(), operator="seed", commit=base,
                          commit_message="baseline")
        wt = wt_root / "seed"
        with self._git_lock:
            gates.git(["worktree", "add", "--detach", "--force", str(wt), base], self.wd)
        try:
            with gpulock.gpu_lock(self.cfg.gpu_index):
                # Pin the baseline ONCE here so every candidate (and the seed
                # itself) is scored against one fixed denominator -- the baseline
                # is then exactly 100%. Falls back to per-candidate measurement if
                # it fails or the problem has no baseline (measure_baseline -> None).
                if self.cfg.pygpubench:
                    self._baseline_us, berr = bench.measure_baseline(self.problem, wt)
                    if berr:
                        self._log(f"baseline pin failed ({berr}); "
                                  "falling back to per-candidate baseline")
                    elif self._baseline_us is not None:
                        self._log(f"baseline pinned: {self._baseline_us:.1f}us "
                                  "(= 100%, fixed for the run)")
                m.correct, m.metric, m.error = self._score_worktree(wt)
        finally:
            with self._git_lock:
                gates.git(["worktree", "remove", "--force", str(wt)], self.wd)
        pop.insert(m)
        return m

    def _evolve_members(self, pop: evolve.Population) -> list[dict]:
        """Full population snapshot for the UI -- drives the fitness chart, the
        niches, the lineage tree, and the leaderboard. Compact rows;
        the in-flight agents are published separately (they are not yet members)."""
        best = pop.best()
        best_id = best.id if best else None
        return [{
            "id": m.id, "op": m.operator, "parent": m.parent_id,
            "metric": m.metric, "correct": m.correct,
            "status": m.status, "commit": (m.commit or "")[:8],
            "message": m.commit_message, "error": m.error,
            "best": m.id == best_id,
        } for m in pop.members]

    def run(self) -> str:
        """Steady-state asynchronous evolutionary search (see kernelthing/evolve.py).

        Keeps up to ``-j``/parallelism agents editing at once (live-tunable down via
        the web UI); all GPU work (each agent's build/run/profile via ``gpu-run``
        and the authoritative benchmark) is serialized by the per-device flock
        (kernelthing/gpulock.py), so the device is never shared. Dispatches
        explore/exploit tasks against a durable population until the
        candidate / wall-clock budget is spent or a stop is requested, then
        promotes the single best-scoring kernel to HEAD.
        """
        state, dirs = self.setup()
        cfg = self.cfg
        u = self.problem.unit
        rng = random.Random(cfg.evolve_seed)
        pop = evolve.Population(direction=self.problem.direction, elite_k=cfg.elite_k)
        base = self._git(["rev-parse", "HEAD"])
        self._base_commit = base
        wt_root = cfg.problem_root / "wt" / state.timestamp / "evolve"
        shutil.rmtree(wt_root, ignore_errors=True)
        wt_root.mkdir(parents=True, exist_ok=True)
        search_start: float | None = None   # set after the seed benchmark (see below)

        def wall_limit() -> int:
            """Live wall-clock budget in seconds (0 = off). The web UI can retune it
            mid-run via the bus; headless falls back to the static config value."""
            return self.bus.wall_clock() if self.bus else cfg.wall_clock_s

        def clock(running: bool) -> dict:
            """Time payload for the UI: epoch start (so the client can tick between
            polls), elapsed-so-far, and the current (live) limit."""
            limit = wall_limit()
            elapsed = int(time.time() - search_start) if search_start is not None else 0
            return {"start": search_start, "elapsed": elapsed,
                    "limit": limit, "running": running}

        # Launch -j is the hard pool size; the live bus value (web UI slider) can
        # only throttle dispatch down within that cap, never above it.
        pool_cap = max(1, cfg.parallelism)
        target = lambda: max(1, min(self._parallelism(), pool_cap))   # noqa: E731

        self._log("")
        self._log(f"──── EVOLVE  parallelism {pool_cap} · budget "
                  f"{cfg.max_candidates or '∞'} candidates"
                  + (f" / {format_duration(cfg.wall_clock_s)}" if cfg.wall_clock_s else "") + " ────")
        in_flight: dict[int, int] = {}        # parent_id -> dispatched-not-yet-scored
        futures: dict = {}                    # future -> Task
        dispatched = 0

        def publish_state() -> None:
            """Push the full live search state to the UI: the completed population
            (chart / niches / lineage / leaderboard) and the in-flight agents, each
            tagged with its opencode log basename so the web server can tail it for
            live tool calls (the controller can't see inside a running turn)."""
            self._publish(
                mode="evolve", best=self._best, submitted=dispatched,
                members=self._evolve_members(pop), clock=clock(True),
                explore_bias=(self.bus.explore_bias() if self.bus else 50),
                explore_auto=(self.bus.explore_auto() if self.bus else False),
                agents=[{"id": t.member_id, "op": t.operator, "parent": t.parent_id,
                         "log_file": f"mem-{t.member_id}-{t.operator}-opencode.log"}
                        for t in futures.values()])

        self._publish(phase="evolve", mode="evolve", scoreboard=[], members=[], agents=[],
                      submitted=0, parallelism=pool_cap, clock=clock(False),
                      budget={"candidates": cfg.max_candidates or None,
                              "wall_clock_s": cfg.wall_clock_s or None})

        seed = self._evolve_seed(base, wt_root, pop)
        self._best = seed.metric if seed.viable else None
        self._log(f"seed (HEAD {base[:8]}): "
                  + (f"{seed.metric:.1f}{u} ✓" if seed.viable else f"✗ {seed.error or 'no score'}"))
        publish_state()

        # Start the wall-clock budget *after* the fixed-cost seed benchmark: a CUDA
        # compile + pygpubench repeats can themselves exceed a short -w, which would
        # otherwise burn the whole budget before a single candidate is dispatched.
        search_start = time.time()

        def want_more() -> bool:
            if self.bus and self.bus.stop_requested():
                return False
            maxc = self.bus.max_candidates() if self.bus else cfg.max_candidates
            if maxc and dispatched >= maxc:
                return False
            limit = wall_limit()   # live: the UI can raise/lower it mid-run
            if limit and time.time() - search_start >= limit:
                return False
            return True

        def submit_one(ex: ThreadPoolExecutor) -> None:
            nonlocal dispatched
            have_elites = bool(pop.elites())
            # Effective explore fraction: auto-schedule ramps from 80→20 over the run,
            # or manual slider value if the user moved it.
            if self.bus and not self.bus.explore_auto():
                explore_frac = self.bus.explore_bias() / 100.0
            elif cfg.max_candidates:
                progress = min(1.0, dispatched / max(cfg.max_candidates, 1))
                explore_frac = 0.8 - 0.6 * progress   # 0.8 → 0.8 → 0.2
            else:
                explore_frac = 0.5   # unbounded + no bus = equal
            op = evolve.choose_operator(
                rng, {evolve.OP_EXPLORE: explore_frac, evolve.OP_EXPLOIT: 1.0 - explore_frac},
                have_elites=have_elites,
                n_niches=len(pop.niches()), min_niches=cfg.min_niches)
            parent = pop.select_parent(op, rng, in_flight)
            if parent is None:
                op = evolve.OP_EXPLORE   # no viable members to exploit or explore from
            mid = pop.next_id()
            prompt = self._evolve_prompt(op, parent, state,
                                         ", ".join(sorted(pop.niches().keys())))
            task = evolve.Task(member_id=mid, operator=op,
                               parent_id=(parent.id if parent else None),
                               parent_commit=(parent.commit if parent else None), prompt=prompt)
            fut = ex.submit(self._evolve_task, task, base, wt_root, dirs)
            futures[fut] = task
            if parent is not None:
                in_flight[parent.id] = in_flight.get(parent.id, 0) + 1
                parent.children += 1
            dispatched += 1
            ptxt = f" <- mem {parent.id}" if parent else ""
            self._log(f"dispatch mem {mid}: {op}{ptxt}  (in-flight {len(futures)})")
            publish_state()

        with ThreadPoolExecutor(max_workers=pool_cap) as ex:
            while want_more() and len(futures) < target():
                submit_one(ex)
            while futures:
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for fut in done:
                    task = futures.pop(fut)
                    if task.parent_id is not None:
                        in_flight[task.parent_id] = max(0, in_flight.get(task.parent_id, 1) - 1)
                    try:
                        m = fut.result()
                    except Exception as e:  # noqa: BLE001
                        m = evolve.Member(id=task.member_id, operator=task.operator,
                                          parent_id=task.parent_id, error=repr(e))
                    pop.insert(m)
                    if m.viable:
                        assert m.commit is not None
                        with self._git_lock:
                            gates.git(["update-ref", self._evolve_ref(state.timestamp, m.id),
                                       m.commit], self.wd)
                        dirs.summary(m.id).write_text(m.summary_text or "(no summary)",
                                                      encoding="utf-8")
                    best = pop.best()
                    self._best = best.metric if best else self._best
                    ptxt = f"<-{task.parent_id} " if task.parent_id is not None else ""
                    res = (f"{m.metric:.1f}{u} ✓ [{m.commit_message[:50]}]" if m.viable
                           else f"✗ {m.error or 'no result'}")
                    self._log(f"result mem {m.id} ({m.operator} {ptxt}): {res}"
                              f"  · best {self._fmt(self._best)}{u}")
                    if m.viable:
                        self._history.append({"round": m.id, "metric": m.metric})
                    publish_state()
                    if want_more():
                        while len(futures) < target() and want_more():
                            submit_one(ex)

        self._dispatched = dispatched
        self._publish(clock=clock(False))   # freeze the UI timer at the final elapsed
        best = pop.best()
        if best and best.commit:
            with self._git_lock:
                gates.git(["reset", "--hard", best.commit], self.wd)
            self._log(f"evolve: promoted mem {best.id} @ {best.metric:.1f}{u}"
                      + (f" [{best.commit_message[:50]}]" if best.commit_message else "")
                      + " to HEAD")
            self._persist_best(best)
        else:
            self._log("evolve: no viable kernel found; HEAD unchanged")
        self._evolve_cleanup(state, pop, wt_root)

        if self.bus and self.bus.stop_requested():
            return self._finish(state, dirs, EXIT_STOPPED, "stopped by user via the web UI")
        if best is None:
            return self._finish(state, dirs, EXIT_STALL,
                                f"evolutionary search ({dispatched} candidates) found nothing viable")
        return self._finish(state, dirs, EXIT_MAXITER,
                            f"evolutionary search budget spent: {dispatched} candidates, "
                            f"best {best.metric:.1f}{u} [{best.operator}]")

    def _persist_best(self, best: evolve.Member) -> None:
        """Copy the best kernel to a stable path that survives managed-repo wipes."""
        out = self.cfg.problem_root / f"{self.problem.name}-best"
        out.mkdir(parents=True, exist_ok=True)
        for f in self.problem.edit_files:
            src = self.wd / self.problem.rel_dir / f
            shutil.copy(src, out / Path(f).name)
        self._log(f"persisted best kernel to {out}")

    def persist_current_head(self) -> None:
        """Copy whatever kernel is at HEAD to the stable best-kernel path.
        
        Called on KeyboardInterrupt so a killed run still saves the last promoted
        kernel, not just a clean exit."""
        try:
            head = self._git(["rev-parse", "HEAD"])
        except Exception:
            return
        if head == getattr(self, '_base_commit', ''):
            return
        out = self.cfg.problem_root / f"{self.problem.name}-best"
        out.mkdir(parents=True, exist_ok=True)
        for f in self.problem.edit_files:
            src = self.wd / self.problem.rel_dir / f
            if src.is_file():
                shutil.copy(src, out / Path(f).name)
        self._log(f"persisted best kernel to {out}")

    def _evolve_cleanup(self, state: State, pop: evolve.Population, wt_root: Path) -> None:
        with self._git_lock:
            for m in pop.members:
                if m.viable:
                    gates.git(["update-ref", "-d", self._evolve_ref(state.timestamp, m.id)], self.wd)
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
        self._publish(phase=f"done ({reason})")
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
        self._publish(phase="methodology analysis")
        prompt = prompts.render(
            METHODOLOGY_PROMPT, LOOP_DIR=self._rel(dirs.base), EXIT_REASON=exit_reason,
            EXIT_REASON_DESCRIPTION=desc, DISPATCHED=self._dispatched,
            BUDGET=self._budget_desc(),
            BEST=(self._best if self._best is not None else "n/a"), UNIT=self.problem.unit)
        log_path = dirs.base / "methodology-opencode.log"
        guard = self._guard(dirs, state.current_round, "methodology")
        for _ in range(3):
            res = self._run_implementer(prompt, log_path, guard=guard)
            self._log(f"methodology: analysis turn done (tools={res.tool_calls}, cost=${res.cost:.4f})")
            if (report.is_file() and report.read_text(encoding="utf-8").strip()
                    and done.is_file() and done.read_text(encoding="utf-8").strip()):
                self._log(f"methodology: retrospective written -> {self._rel(report)}")
                return
            prompt = (f"The methodology analysis is incomplete. Write the retrospective to "
                      f"{self._rel(report)} and a one-line completion note to {self._rel(done)}.")
        self._log("methodology: still incomplete after retries; continuing exit")
