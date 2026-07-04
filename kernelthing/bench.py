"""In-process pygpubench scorer -- the default benchmarking engine.

pygpubench (https://github.com/gpu-mode/pygpubench) runs a kernel in a supervised
subprocess (seccomp + landlock + mseal), defeats timer monkeypatching (its timing
core is compiled C++), and detects L2/replay cheats via canaries + GPU-memory
relocation. We use it as the *runtime* benchmark behind the orchestrator's score
step -- the runtime half kernelguard's static scan can't cover.

This module is the single entry point the orchestrator calls
(``bench.score(problem, worktree)``); there is no CLI and no ``python -m`` path.
pygpubench (and torch) are imported **lazily**: a missing pygpubench surfaces a
clear scoring error rather than breaking the loop's import.

The contract a pygpubench problem ships in its problem dir:
  * ``submission.py`` -- the editable kernel adapter; exposes the function named
    by ``bench.submission_qualname`` (e.g. ``submission.kernel``). This is what the
    agent edits (it is the problem's ``edit_files``).
  * ``task.py`` -- defines ``generate_test_case(*, seed, **test_args)`` returning
    ``((inputs...), (expected, atol, rtol))`` (the pygpubench test-case contract).
    NOT agent-editable -- it is the scoring objective.
  * optionally ``baseline.py`` -- a reference kernel (e.g. ``torch.matmul``) whose
    measured time is the denominator for a ``pct_baseline`` / ``speedup`` metric.

The qualname is resolved by *import* inside pygpubench's subprocess, so the
worktree's problem dir must be importable there. We maximise that by (1) inserting
it at ``sys.path[0]``, (2) exporting it on ``PYTHONPATH`` for any spawned child,
and (3) ``chdir``-ing into it for the duration. Cached ``submission``/``task``/
``baseline`` modules are purged around each call so a later worktree never reuses
an earlier one's code (same module name, different file on disk).
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gpulock
from .problem import Problem

# Protects the process-global env vars (CUDA_VISIBLE_DEVICES / LD_PRELOAD /
# KERNELTHING_GPU_POOL) while a pygpubench subprocess inherits them.
_env_lock = threading.Lock()

# Protects the in-process sys.path / sys.modules / cwd mutation inside _importable
# so concurrent scorings don't race on the shared import state.
_import_lock = threading.Lock()

# pygpubench raises ``RuntimeError('Benchmark subprocess failed with exit code N')``
# when the supervised benchmark process dies. A *negative* N is Python's convention
# for "killed by signal -N" -- the actionable signal hides inside the message.
EXIT_CODE_RE = re.compile(r"exit code (-?\d+)")

# Common ways a CUDA kernel takes down the benchmark subprocess, with the most
# likely cause for this workload (kernel editing) so the agent gets a next step.
SIGNAL_HINT = {
    signal.SIGSEGV: "illegal memory access / out-of-bounds -- check indexing, "
    "launch grid/block bounds, and shared-memory sizes",
    signal.SIGABRT: "abort() -- often a failed CUDA API check or libc++ assertion "
    "(e.g. cudaErrorIllegalAddress surfaced at sync)",
    signal.SIGKILL: "killed -- usually out-of-memory (host or GPU) or the sandbox "
    "timeout; reduce allocation or kernel runtime",
    signal.SIGBUS: "bus error -- misaligned access or bad memory mapping",
    signal.SIGFPE: "arithmetic fault -- integer divide-by-zero in index math",
    signal.SIGILL: "illegal instruction -- ABI/arch mismatch in the compiled kernel",
}


def explain_bench_error(e: BaseException) -> str:
    """Turn pygpubench's bare ``exit code -11`` into a signal-named, actionable line.

    Falls back to the plain ``repr`` when the message has no decodable exit code
    (so non-crash failures -- import errors, timeouts already labelled, etc. -- are
    untouched)."""
    m = EXIT_CODE_RE.search(str(e))
    if not m:
        return f"pygpubench score error: {e!r}"
    code = int(m.group(1))
    if code >= 0:
        return f"pygpubench score error (subprocess exit code {code}): {e!r}"
    try:
        sig = signal.Signals(-code)
    except ValueError:
        return f"pygpubench score error (killed by signal {-code}): {e!r}"
    hint = SIGNAL_HINT.get(sig, "subprocess crashed")
    return (
        f"benchmark subprocess crashed: {sig.name} (signal {-code}) -- {hint}. "
        "This is a fault inside the kernel/submission, not a kernelthing bug."
    )


# Module names we own in a problem dir; purged before/after each scoring so two
# worktrees with the same module name but different files never alias.
_PROBLEM_MODULES = ("submission", "task", "baseline")

# Benchmark defaults when the manifest's ``bench`` block omits them (the common
# case -- a problem need only name its submission/baseline). A problem may
# override any of them.
DEFAULT_REPEATS = 5  # timed measurements per scoring (median is taken)
DEFAULT_SEED = 0  # RNG seed handed to the test-case generator
DEFAULT_TIMEOUT_S = 600  # per-benchmark wall-clock cap


@dataclass
class _BenchSetup:
    """Resolved benchmark configuration extracted from a problem manifest."""

    bench_cfg: dict[str, Any]
    prob_dir: Path
    task_module: str
    generator: str
    test_args: dict[str, Any]
    repeats: int
    seed: int


def resolve_bench_config(problem: Problem, worktree: Path) -> _BenchSetup:
    """Extract benchmark configuration from a problem, with defaults."""
    bench_cfg = problem.bench or {}
    return _BenchSetup(
        bench_cfg=bench_cfg,
        prob_dir=Path(worktree) / problem.rel_dir,
        task_module=bench_cfg.get("task_module", "task"),
        generator=bench_cfg.get("generator", "generate_test_case"),
        test_args=bench_cfg.get("test_args", {}),
        repeats=int(bench_cfg.get("repeats", DEFAULT_REPEATS)),
        seed=int(bench_cfg.get("seed", DEFAULT_SEED)),
    )


def _purge_problem_modules() -> None:
    for name in list(sys.modules):
        if name in _PROBLEM_MODULES or name.split(".", 1)[0] in _PROBLEM_MODULES:
            del sys.modules[name]


@contextlib.contextmanager
def _importable(prob_dir: Path) -> Generator[None, None, None]:
    """Make ``prob_dir`` importable in-process and for spawned children."""
    prob_dir_str = str(prob_dir)
    old_cwd = os.getcwd()
    old_pp = os.environ.get("PYTHONPATH")
    with _import_lock:
        _purge_problem_modules()
        sys.path.insert(0, prob_dir_str)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [prob_dir_str, *(old_pp.split(os.pathsep) if old_pp else [])]
        )
        try:
            os.chdir(prob_dir_str)
            yield
        finally:
            os.chdir(old_cwd)
            with contextlib.suppress(ValueError):
                sys.path.remove(prob_dir_str)
            if old_pp is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pp
            _purge_problem_modules()


def do_bench(
    pygpubench: Any,
    qualname: str,
    gen: _GeneratorAdapter,
    args: dict[str, Any],
    repeats: int,
    seed: int,
    opts: dict[str, Any],
    writable_paths: list[str] | None = None,
) -> Any:
    """One isolated benchmark; returns the BenchmarkResult or raises."""
    if writable_paths is None:
        writable_paths = list(opts.get("writable_paths", ["/tmp"]))
    return pygpubench.do_bench_isolated(
        qualname,
        gen,
        dict(args),
        int(repeats),
        int(seed),
        discard=True,
        timeout=int(opts["timeout"]) if "timeout" in opts else DEFAULT_TIMEOUT_S,
        landlock=bool(opts["landlock"]) if "landlock" in opts else True,
        mseal=bool(opts["mseal"]) if "mseal" in opts else True,
        allow_root=bool(opts["allow_root"]) if "allow_root" in opts else False,
        writable_paths=writable_paths,
    )


# ---------------------------------------------------------------------------
# pygpubench protocol adaptation
# ---------------------------------------------------------------------------
# pygpubench requires the kernel to mutate its *first* argument in-place (the
# output buffer) and ignores the return value.  Problem authors write natural
# signatures:  ``def kernel(A, B): return C`` and generators return
# ``((A, B), (expected, ...))``.  We transparently adapt both sides.
#
# Generator wrapper: prepends a pre-allocated output buffer matching the
# expected tensor so pygpubench's validation can compare against it.
#
# Kernel shim: a tiny module written into the problem directory that imports
# the real kernel, calls it with the trailing args, and copies the return
# value into the output buffer.  The shim qualifies as the pygpubench
# submission qualname and is cleaned up after scoring.


class _GeneratorAdapter:
    """Adapts a natural generator to pygpubench's in-place protocol.

    The instance is picklable because it references *gen* by its importable
    qualified name (e.g. ``task.generate_test_case``).
    """

    def __init__(self, gen: Any) -> None:
        self._gen = gen

    def __call__(self, **kwargs: Any) -> Any:
        inputs, expected = self._gen(**kwargs)
        output = expected[0].new_empty(expected[0].size())
        return (output, *inputs), expected


def _shim_qualname(qualname: str, prob_dir: Path) -> tuple[str, Path]:
    """Write a pygpubench-in-place shim for *qualname* into *prob_dir*.

    Returns ``(shim_qualname, shim_path)``.  The caller must unlink
    *shim_path* (and any ``.pyc`` sibling) after scoring.
    """
    module_name, __, func_name = qualname.rpartition(".")
    safe = qualname.replace(".", "_")
    shim_mod = f"_kt_shim_{safe}"
    shim_path = prob_dir / f"{shim_mod}.py"
    shim_path.write_text(f"""\
import torch
import {module_name} as _real
_orig = _real.{func_name}
def {func_name}(output, *args, **kwargs):
    result = _orig(*args, **kwargs)
    if result is not None and result is not output and hasattr(output, 'copy_'):
        output.copy_(result)
    return output
""")
    return f"{shim_mod}.{func_name}", shim_path


def _cleanup_shim(shim_path: Path) -> None:
    """Remove the shim file and any cached bytecode."""
    for p in (shim_path, shim_path.with_suffix(".pyc")):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


# Minimum completed iterations for a usable timing. pygpubench truncates the
# repeat count for a long-running kernel and flags the run ``full=False`` (so
# ``result.success`` is False), but the iterations that *did* complete are still
# valid timings -- we take their median rather than discarding the score, else any
# kernel slower than pygpubench's per-benchmark wall-time budget is permanently
# unscoreable (baseline can't pin, every candidate reads ?%baseline). Two is the
# floor ``basic_stats`` needs (its variance divides by ``runs - 1``).
MIN_VALID_REPEATS = 2


def median_us(pygpubench: Any, result: Any) -> float | None:
    """Median of the completed timed iterations, or ``None`` if unusable.

    A result is usable when no iteration reported a correctness/execution error
    (``result.errors is None``) and at least ``MIN_VALID_REPEATS`` iterations
    produced a positive time. This deliberately accepts a *truncated* run
    (``full=False`` / ``success=False``) whose only "fault" is that the kernel was
    too slow to finish all repeats within pygpubench's budget -- those completed
    iterations are real measurements. ``basic_stats`` already ignores the ``-1``
    sentinels of un-run iterations. Returns ``None`` only on a genuine failure: an
    iteration errored (``errors is not None``) or too few timings completed.
    """
    if result.errors is not None:
        return None
    valid = [t for t in result.time_us if t > 0]
    if len(valid) < MIN_VALID_REPEATS:
        return None
    return float(pygpubench.basic_stats(result.time_us).median)


def describe_failure(result: Any) -> str:
    """Human-readable reason a benchmark result is unusable, for error messages.

    Distinguishes the cases ``errors=None`` alone can't: a real correctness/exec
    error vs. a kernel that merely ran too few iterations (too slow) vs. one that
    produced no timing at all. Replaces the old bare ``errors={result.errors}``,
    which printed a useless ``errors=None`` for the common too-slow case.
    """
    if result.errors is not None:
        return f"{result.errors} iteration(s) reported a correctness/execution error"
    valid = [t for t in result.time_us if t > 0]
    if not valid:
        return "no timed iteration completed (kernel crashed or was killed before timing)"
    slowest_ms = max(valid) / 1000.0
    return (
        f"only {len(valid)}/{len(result.time_us)} iterations completed before "
        f"pygpubench's per-benchmark budget (kernel too slow: ~{slowest_ms:.0f}ms/iter); "
        f"need at least {MIN_VALID_REPEATS} for a stable measurement"
    )


@contextlib.contextmanager
def _gpu_env(gpu_pool: list[int]) -> Generator[None, None, None]:
    """Arrange serialized GPU access for pygpubench's isolated subprocess.

    The kernel never runs in this process -- ``pygpubench.do_bench_isolated``
    forks a sandboxed worker to run it. We hand that worker the ``libktgpu.so``
    LD_PRELOAD shim together with ``gpu_pool``: on its first CUDA call the shim
    probes the pool for a free card (non-blocking), blocking on the first only if
    every card is busy, flocks it, and holds it for the worker's (short) lifetime.
    This is the same per-UUID flock agents take through the shim, so all GPU work
    serializes on one inode -- with no in-process lock to deadlock against.

    An outer shim's pool wins (an agent running ``kernelthing score`` -- let its
    full pool probe). ``CUDA_VISIBLE_DEVICES`` is blanked (fail-closed); the shim
    pins the locked card's UUID. If the shim isn't built, fall back to pinning the
    first pool index directly.
    """
    keys = ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "KERNELTHING_GPU_POOL")
    saved = {k: os.environ.get(k) for k in keys}
    with _env_lock:
        gpulock.apply_shim_env(os.environ, gpu_pool, overwrite_pool=False)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def derive_metric(
    pygpubench: Any,
    problem: Problem,
    gen: _GeneratorAdapter,
    args: dict[str, Any],
    repeats: int,
    seed: int,
    opts: dict[str, Any],
    cand_median: float,
    *,
    baseline_shim_q: str = "",
    pinned_baseline: float | None = None,
    writable_paths: list[str] | None = None,
) -> tuple[float | None, str | None]:
    """Turn the candidate's median time (us) into the problem's metric.

    kinds:
      * ``latency_us``        -> the median time itself (direction: minimize)
      * ``tflops``            -> ``flops / (median_us * 1e6)`` (needs ``flops``)
      * ``pct_baseline``      -> ``baseline_median / cand_median * 100`` -- e.g.
        %cuBLAS when ``baseline_qualname`` benchmarks torch.matmul (equal FLOPs,
        so the time ratio is the throughput ratio)
      * ``speedup``           -> ``baseline_median / cand_median``

    ``pinned_baseline`` (us): when given, used as the fixed denominator for
    ``pct_baseline`` / ``speedup`` instead of re-benchmarking the baseline here.
    The orchestrator measures the baseline once per run (see ``measure_baseline``)
    and pins it so every candidate shares one denominator -- the baseline is then
    definitionally 100% and identical kernels score identically rather than
    drifting with per-scoring GPU noise.
    """
    metric = problem.metric or {}
    kind = metric.get("kind", "latency_us")
    if kind == "latency_us":
        return cand_median, None
    if kind == "tflops":
        flops = metric.get("flops", None)
        if not flops:
            return None, "metric.kind=tflops requires metric.flops"
        return float(flops) / (cand_median * 1e6), None
    if kind in ("pct_baseline", "speedup"):
        base_median: float | None
        if pinned_baseline is not None:
            base_median = pinned_baseline
        else:
            base_q = baseline_shim_q or metric.get("baseline_qualname", "")
            if not base_q:
                return None, f"metric.kind={kind} requires metric.baseline_qualname"
            base_res = do_bench(
                pygpubench, base_q, gen, args, repeats, seed, opts, writable_paths=writable_paths
            )
            base_median = median_us(pygpubench, base_res)
            if base_median is None:
                return None, f"baseline '{base_q}' failed: {describe_failure(base_res)}"
        ratio = base_median / cand_median
        return (ratio * 100.0 if kind == "pct_baseline" else ratio), None
    return None, f"unknown metric.kind '{kind}'"


def measure_baseline(
    problem: Problem, worktree: Path, *, gpu_index: int = 0, gpu_pool: list[int] | None = None
) -> tuple[float | None, str | None]:
    """Benchmark the baseline reference kernel once, returning ``(median_us, err)``."""
    pool = gpu_pool if gpu_pool is not None else [gpu_index]
    try:
        with _gpu_env(pool):
            return _measure_baseline_impl(problem, worktree)
    except Exception as e:
        return None, explain_bench_error(e)


def _measure_baseline_impl(
    problem: Problem, worktree: Path
) -> tuple[float | None, str | None]:
    """Body of ``measure_baseline`` — runs inside ``_gpu_env``."""
    try:
        import pygpubench
    except Exception as e:
        return None, f"pygpubench not installed: {e!r}"

    base_q = (problem.metric or {}).get("baseline_qualname", "")
    if not base_q:
        return None, None
    s = resolve_bench_config(problem, worktree)
    if not s.prob_dir.is_dir():
        return None, f"problem dir not found in worktree: {s.prob_dir}"

    try:
        with _importable(s.prob_dir):
            task_mod = importlib.import_module(s.task_module)
            gen = _GeneratorAdapter(getattr(task_mod, s.generator))
            base_shim_q, base_shim_path = _shim_qualname(base_q, s.prob_dir)
            try:
                res = do_bench(
                    pygpubench,
                    base_shim_q,
                    gen,
                    s.test_args,
                    s.repeats,
                    s.seed,
                    s.bench_cfg,
                    writable_paths=[str(s.prob_dir)],
                )
                median = median_us(pygpubench, res)
                if median is None:
                    return None, f"baseline '{base_q}' failed: {describe_failure(res)}"
                return median, None
            finally:
                _cleanup_shim(base_shim_path)
    except Exception as e:
        return None, explain_bench_error(e)


def _sweep_stale_batons(build_root: Path) -> None:
    """Delete orphaned torch ``file_baton`` lock files under *build_root*.

    ``torch.utils.cpp_extension`` guards each build directory with a ``FileBaton``
    (``torch/utils/file_baton.py``): ``try_acquire`` creates ``<dir>/lock`` with
    ``O_EXCL`` and only ``release`` removes it. A compiler killed mid-build (a
    timeout, OOM, or crash) never releases, so the lock file is orphaned -- and
    every later import that would compile that extension spins **forever** in
    ``FileBaton.wait()`` (it has no timeout and no liveness check).

    Call this only while holding ``_compile_lock`` for the same tree: that flock
    guarantees no live kernelthing compiler is using these directories, so any
    ``lock`` present is provably stale and safe to remove.
    """
    with contextlib.suppress(OSError):
        for lock in build_root.rglob("lock"):
            with contextlib.suppress(OSError):
                lock.unlink()


@contextlib.contextmanager
def _compile_lock(prob_dir: Path) -> Generator[None, None, None]:
    """Crash-safe exclusive lock over *prob_dir*'s build tree.

    Uses ``fcntl.flock`` -- released automatically by the kernel when the fd is
    closed or the holding process dies -- so a compiler killed mid-build can never
    wedge the next build, the crash-safety torch's ``O_EXCL`` file_baton lacks.
    This is what lets ``_sweep_stale_batons`` delete a stale baton without racing a
    live compiler: while we hold this flock, no other kernelthing build is running
    in the tree. Best-effort -- if the lock can't be taken we proceed unlocked
    rather than block scoring.
    """
    build_root = prob_dir / "build"
    with contextlib.suppress(OSError):
        build_root.mkdir(parents=True, exist_ok=True)
    lf = None
    try:
        with contextlib.suppress(OSError):
            lf = open(build_root / ".kt-compile.lock", "w")  # noqa: SIM115 (held across yield for the flock)
            fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        if lf is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()


def warm_build(problem: Problem, worktree: Path, *, timeout: int | None = None, arch: str = "") -> None:
    """Compile the worktree's kernels in a subprocess outside the GPU lock, so the
    real benchmark only pays runtime, not compile time.

    Runs ``import task, submission`` inside the problem dir to force
    ``torch.utils.cpp_extension`` to build and cache both the reference kernel
    (``task`` imports ``baseline``, which the sandboxed worker also compiles when
    it generates test cases) and the submission kernel. The build is pure host-side
    ``nvcc``/``cicc`` and must stay *off* the GPU lock; the real pygpubench worker
    then reuses the cached ``.so``s and pays only runtime. ``task`` is imported
    first so the fixed reference still warms even when a broken submission fails to
    compile.

    Crash-safe compile coordination (the ``file_baton`` fix): the build is wrapped
    in ``_compile_lock`` (an ``fcntl.flock`` released on process death) and any
    orphaned torch baton lock is swept first (``_sweep_stale_batons``). Together
    these replace torch's non-crash-safe ``O_EXCL`` file_baton -- a compiler killed
    mid-build can no longer wedge later builds in ``FileBaton.wait()``. Because this
    completes the build off-lock, the GPU-locked worker finds an up-to-date ``.so``
    and never touches the baton at all, so its own timeout-kill can't orphan one.

    The env is set so the build cannot touch the GPU or its lock:
      * ``TORCH_CUDA_ARCH_LIST`` (``arch``, e.g. ``"8.9"``) makes ``cpp_extension``
        pick gencode flags from the string instead of calling
        ``get_device_capability()`` — so the compile makes no CUDA call at all
        (inherited from the environment when ``arch`` is empty).
      * ``LD_PRELOAD`` / ``KERNELTHING_GPU_POOL`` are stripped so the libktgpu
        shim isn't even loaded here — the build physically cannot flock a card
        even if some import path did touch CUDA.
      * ``CUDA_VISIBLE_DEVICES=""`` — no device visible.

    ``timeout`` is ``None`` by default: the compile runs to completion off-lock (a
    bounded stall on the caller, which has *not* taken the GPU lock yet). The old
    fixed cap silently ``SIGKILL``ed slow ``nvcc`` builds mid-compile, orphaning a
    baton and forcing the *next* build to recompile under the GPU lock -- the exact
    stall this function exists to prevent. Best-effort otherwise: on any failure the
    caller proceeds and the real benchmark just compiles in-line.
    """
    prob_dir = Path(worktree) / problem.rel_dir
    if not prob_dir.is_dir():
        return
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env.pop("KERNELTHING_GPU_POOL", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    if arch:
        env["TORCH_CUDA_ARCH_LIST"] = arch
    with _compile_lock(prob_dir):
        _sweep_stale_batons(prob_dir / "build")
        with contextlib.suppress(Exception):
            subprocess.run(
                [sys.executable, "-c", "import task, submission"],
                cwd=str(prob_dir),
                capture_output=True,
                timeout=timeout,
                env=env,
            )


def score(
    problem: Problem,
    worktree: Path,
    *,
    baseline_median: float | None = None,
    gpu_index: int = 0,
    gpu_pool: list[int] | None = None,
) -> tuple[bool, float | None, str | None]:
    """Benchmark the worktree's submission through pygpubench.

    Returns ``(correct, metric, err)`` -- the same tuple the legacy
    ``_score_worktree`` returns, so the orchestrator treats both scoring paths
    identically. ``correct`` is pygpubench's ``result.success`` (every repeat
    passed correctness within tolerance). Repeats are pygpubench's job (its
    ``repeats`` arg), so the orchestrator does not loop this call.

    ``baseline_median`` (us): a pinned denominator for ``pct_baseline`` /
    ``speedup`` metrics. When given, the baseline reference kernel is NOT
    re-benchmarked here -- the run measures it once (``measure_baseline``) and
    reuses it so every candidate is scored against one fixed baseline.

    ``gpu_pool`` (or a single ``gpu_index``): candidate cards handed to the shim,
    which probes them for a free one and flocks it for the worker's lifetime.
    """
    pool = gpu_pool if gpu_pool is not None else [gpu_index]
    try:
        with _gpu_env(pool):
            return _score_impl(problem, worktree, baseline_median)
    except Exception as e:
        return False, None, explain_bench_error(e)


def _score_impl(
    problem: Problem, worktree: Path, baseline_median: float | None
) -> tuple[bool, float | None, str | None]:
    """Body of ``score`` — runs inside ``_gpu_env`` so torch sees the GPU at import time."""
    try:
        import pygpubench
    except Exception as e:
        return (
            False,
            None,
            (f"pygpubench not installed (pip install 'kernelthing[pygpubench]'): {e!r}"),
        )

    qualname = (problem.bench or {}).get("submission_qualname")
    if not qualname:
        return False, None, "problem.bench.submission_qualname is required"
    s = resolve_bench_config(problem, worktree)
    if not s.prob_dir.is_dir():
        return False, None, f"problem dir not found in worktree: {s.prob_dir}"

    try:
        with _importable(s.prob_dir):
            task_mod = importlib.import_module(s.task_module)
            gen = _GeneratorAdapter(getattr(task_mod, s.generator))

            sub_shim_q, sub_shim_path = _shim_qualname(qualname, s.prob_dir)
            base_shim_q = ""
            base_shim_path = None
            base_q = (problem.metric or {}).get("baseline_qualname", "")
            if base_q and baseline_median is None:
                base_shim_q, base_shim_path = _shim_qualname(base_q, s.prob_dir)

            try:
                writable = [str(s.prob_dir)]
                cand_res = do_bench(
                    pygpubench,
                    sub_shim_q,
                    gen,
                    s.test_args,
                    s.repeats,
                    s.seed,
                    s.bench_cfg,
                    writable_paths=writable,
                )
                cand_median = median_us(pygpubench, cand_res)
                if cand_median is None:
                    return False, None, f"submission failed: {describe_failure(cand_res)}"
                metric, err = derive_metric(
                    pygpubench,
                    problem,
                    gen,
                    s.test_args,
                    s.repeats,
                    s.seed,
                    s.bench_cfg,
                    cand_median,
                    baseline_shim_q=base_shim_q,
                    pinned_baseline=baseline_median,
                    writable_paths=writable,
                )
                if err:
                    return True, None, err
                return True, metric, None
            finally:
                _cleanup_shim(sub_shim_path)
                if base_shim_path:
                    _cleanup_shim(base_shim_path)
    except Exception as e:
        return False, None, explain_bench_error(e)
