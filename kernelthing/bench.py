"""In-process pygpubench scorer -- the default benchmarking engine.

pygpubench (https://github.com/gpu-mode/pygpubench) runs a kernel in a supervised
subprocess (seccomp + landlock + mseal), defeats timer monkeypatching (its timing
core is compiled C++), and detects L2/replay cheats via canaries + GPU-memory
relocation. We use it as the *runtime* benchmark behind the orchestrator's score
step -- the runtime half kernelguard's static scan can't cover.

This module is the single entry point the orchestrator calls
(``bench.score(problem, worktree)``); there is no CLI and no ``python -m`` path.
pygpubench (and torch) are imported **lazily**: with pygpubench scoring turned off
(``cfg.pygpubench=False``) the loop falls back to the plain ``score_command`` and
needs neither, and a missing pygpubench degrades to a clear scoring error rather
than breaking the loop's import.

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
import importlib
import json
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

from .problem import Problem

# Protects the process-global CUDA_VISIBLE_DEVICES env var while a pygpubench
# subprocess inherits it; the actual GPU work is already serialized per-device by
# the per-GPU flock (gpulock.py), so this lock is held only for microseconds.
_env_lock = threading.Lock()

JSON_OBJ = re.compile(r"\{.*\}")

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


def parse_score(stdout: str) -> tuple[bool, float | None]:
    """Return (correct, metric) from the last JSON line emitted by a score command.

    Used only by the ``cfg.pygpubench=False`` fallback (the plain ``score_command``
    path). Tolerates extra non-JSON output (build logs etc.): scans bottom-up for
    the last line that parses as a JSON object containing ``correct`` and ``metric``.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            m = JSON_OBJ.search(line)
            if not m:
                continue
            line = m.group(0)
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "correct" in obj and "metric" in obj:
            metric = obj["metric"]
            try:
                metric = float(metric)
            except (TypeError, ValueError):
                metric = None
            return bool(obj["correct"]), metric
    return False, None


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


def available() -> bool:
    """True if pygpubench can be imported (it and torch are optional deps)."""
    try:
        import pygpubench  # noqa: F401

        return True
    except Exception:
        return False


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
        return (output, *tuple(inputs)), expected


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
    output.copy_(result)
    return output
""")
    return f"{shim_mod}.{func_name}", shim_path


def _cleanup_shim(shim_path: Path) -> None:
    """Remove the shim file and any cached bytecode."""
    for p in (shim_path, shim_path.with_suffix(".pyc")):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


def median_us(pygpubench: Any, result: Any) -> float | None:
    if not result.success:
        return None
    return float(pygpubench.basic_stats(result.time_us).median)


@contextlib.contextmanager
def _gpu_env(gpu_index: int) -> Generator[None, None, None]:
    """Temporarily set ``CUDA_VISIBLE_DEVICES`` for a pygpubench subprocess to inherit.

    The per-GPU flock serialises actual device work; this lock only serialises the
    env-var mutation (microseconds), so two GPUs can bench concurrently.
    """
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    with _env_lock:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old


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
                return None, f"baseline '{base_q}' failed: errors={base_res.errors}"
        ratio = base_median / cand_median
        return (ratio * 100.0 if kind == "pct_baseline" else ratio), None
    return None, f"unknown metric.kind '{kind}'"


def measure_baseline(
    problem: Problem, worktree: Path, *, gpu_index: int = 0
) -> tuple[float | None, str | None]:
    """Benchmark the baseline reference kernel once, returning ``(median_us, err)``."""
    try:
        with _gpu_env(gpu_index):
            return _measure_baseline_impl(problem, worktree, gpu_index)
    except Exception as e:
        return None, _explain_bench_error(e)


def _measure_baseline_impl(
    problem: Problem, worktree: Path, gpu_index: int
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
                    return None, f"baseline '{base_q}' failed: errors={res.errors}"
                return median, None
            finally:
                _cleanup_shim(base_shim_path)
    except Exception as e:
        return None, explain_bench_error(e)


def warm_build(problem: Problem, worktree: Path, *, timeout: int = 60) -> None:
    """Trigger compilation of the worktree's kernel in a subprocess outside the GPU
    lock, so the real benchmark only pays runtime, not compile time.

    Runs ``python -c 'import submission'`` inside the problem dir with
    ``CUDA_VISIBLE_DEVICES=""`` — enough to force ``torch.utils.cpp_extension.load``
    to compile and cache the ``.so``, but the import itself will fail when the
    kernel launch is actually attempted (no GPU visible). That is fine: the build
    artifacts survive and the real pygpubench benchmark reuses them.

    Best-effort: if the subprocess fails or times out the caller proceeds anyway
    and the real benchmark just compiles in-line.
    """
    prob_dir = Path(worktree) / problem.rel_dir
    if not prob_dir.is_dir():
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            [sys.executable, "-c", "import submission"],
            cwd=str(prob_dir),
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )


def score(
    problem: Problem, worktree: Path, *, baseline_median: float | None = None, gpu_index: int = 0
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

    ``gpu_index``: set ``CUDA_VISIBLE_DEVICES`` for the pygpubench subprocess.
    """
    try:
        with _gpu_env(gpu_index):
            return _score_impl(problem, worktree, baseline_median, gpu_index)
    except Exception as e:
        return False, None, _explain_bench_error(e)


def _score_impl(
    problem: Problem, worktree: Path, baseline_median: float | None, gpu_index: int
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
                    return False, None, f"submission failed correctness: errors={cand_res.errors}"
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
