"""Tests for the pygpubench scorer (kernelthing/bench.py) and the setup gate.

pygpubench (+ torch + a GPU) is not available in CI, so the scorer is exercised
against an injected *fake* pygpubench module. The fake faithfully reproduces the
parts of the contract bench.py depends on: it resolves the submission/baseline by
qualified name (which also tests bench.py's import-path machinery), calls the
test generator, checks the output against the expected value, and returns canned
timings. This lets us test the real control flow + metric derivation with no GPU.
"""

import sys
import textwrap
import types
from typing import ClassVar

import pytest

from kernelthing import bench, gates
from kernelthing.problem import Problem

# --- setup-blocked marker (used by the bootstrap auto path) ---


def test_has_setup_blocked():
    assert gates.has_setup_blocked("explanation...\nSETUP_BLOCKED")
    assert not gates.has_setup_blocked("SETUP_BLOCKED\ntrailing text")


# --- parse_score (the cfg.pygpubench=False fallback that reads score_command JSON) ---


def test_parse_score_basic():
    assert bench.parse_score('{"correct": true, "metric": 88.0, "unit": "%cuBLAS"}') == (True, 88.0)
    assert bench.parse_score('{"correct": false, "metric": 0}') == (False, 0.0)


def test_parse_score_last_json_amid_logs():
    out = 'make: building\nptxas info...\n{"correct": true, "metric": 91.5}\n'
    assert bench.parse_score(out) == (True, 91.5)


def test_parse_score_none_when_absent():
    assert bench.parse_score("no json here\nbuild failed") == (False, None)


def test_parse_score_int_metric():
    assert bench.parse_score('{"correct": true, "metric": 7}') == (True, 7.0)


# --- fake pygpubench harness for the scorer ---


class _Stats:
    def __init__(self, median):
        self.median = median


class _Result:
    def __init__(self, success, time_us, errors=None):
        self.success = success
        self.time_us = time_us
        self.errors = errors


class _FakePygpubench(types.ModuleType):
    """Resolves the qualname, runs the generator+kernel, checks correctness."""

    # median time (us) returned per qualname; lets us predict pct_baseline.
    TIMES: ClassVar[dict[str, float]] = {"submission.kernel": 10.0, "baseline.matmul": 20.0}

    def __init__(self):
        super().__init__("pygpubench")

    def do_bench_isolated(self, qualname, gen, args, repeats, seed, **kw):
        # bench.py wraps the natural generator in _GeneratorAdapter (which prepends
        # a torch output buffer) and the submission/baseline in a _shim_qualname
        # module (which copies the kernel's return into that buffer). So *gen* is
        # the adapter and *qualname* is the shim's qualname -- mirror that contract.
        import importlib

        import torch

        mod_name, attr = qualname.rsplit(".", 1)
        # Real pygpubench imports in a fresh subprocess; in-process we must (a) drop the
        # FileFinder's stale dir listing so the just-written shim is visible, and (b)
        # evict any shim module cached from a prior test so it re-executes against the
        # current problem dir (bench._purge_problem_modules only covers task/submission/
        # baseline, not the _kt_shim_* wrappers).
        sys.modules.pop(mod_name, None)
        importlib.invalidate_caches()
        fn = getattr(importlib.import_module(mod_name), attr)
        call_args, expected = gen(seed=seed, **args)  # ((output, *inputs), (exp, atol, rtol))
        out = fn(*call_args)
        ok = bool(torch.allclose(out, expected[0]))
        # the shim module name embeds the original qualname, so route the time by it
        key = "submission.kernel" if "submission" in qualname else "baseline.matmul"
        return _Result(ok, [self.TIMES[key]] * repeats, errors=None if ok else 1)

    def basic_stats(self, time_us):
        s = sorted(time_us)
        return _Stats(s[len(s) // 2])


@pytest.fixture
def fake_pygpubench(monkeypatch):
    pytest.importorskip("torch")  # the scorer's adapter/shim require torch
    fake = _FakePygpubench()
    monkeypatch.setitem(sys.modules, "pygpubench", fake)
    return fake


def _write_problem_dir(root, kernel_body="return x * 2"):
    d = root / "prob"
    d.mkdir()
    (d / "submission.py").write_text(f"def kernel(x):\n    {kernel_body}\n")
    (d / "task.py").write_text(
        textwrap.dedent("""
        import torch
        def generate_test_case(*, seed, n):
            x = torch.arange(n, dtype=torch.float32)
            return (x,), (x * 2, 0, 0)
    """)
    )
    (d / "baseline.py").write_text("def matmul(x):\n    return x * 2\n")
    return d


def _problem(root):
    return Problem(
        name="p",
        repo_root=root,
        rel_dir="prob",
        plan="plan.md",
        edit_files=["prob/submission.py"],
        score_command="",
        direction="maximize",
        bench_runs=1,
        bench={
            "submission_qualname": "submission.kernel",
            "task_module": "task",
            "test_args": {"n": 3},
            "repeats": 4,
            "seed": 7,
        },
        metric={"kind": "pct_baseline", "baseline_qualname": "baseline.matmul"},
    )


def test_score_correct_kernel_pct_baseline(tmp_path, fake_pygpubench):
    _write_problem_dir(tmp_path)
    correct, metric, err = bench.score(_problem(tmp_path), tmp_path)
    assert correct and err is None
    # baseline 20us / candidate 10us * 100 == 200 %baseline
    assert metric == pytest.approx(200.0)


def test_score_wrong_kernel_is_incorrect(tmp_path, fake_pygpubench):
    _write_problem_dir(tmp_path, kernel_body="return x * 3")  # wrong output
    correct, metric, err = bench.score(_problem(tmp_path), tmp_path)
    assert correct is False and metric is None and err


def test_score_latency_metric(tmp_path, fake_pygpubench):
    _write_problem_dir(tmp_path)
    prob = _problem(tmp_path)
    prob.metric = {"kind": "latency_us"}
    correct, metric, _err = bench.score(prob, tmp_path)
    assert correct and metric == pytest.approx(10.0)


def test_score_missing_pygpubench(tmp_path, monkeypatch):
    # No fake injected and not installed -> a clear error, not an exception.
    monkeypatch.setitem(sys.modules, "pygpubench", None)
    _write_problem_dir(tmp_path)
    correct, _metric, err = bench.score(_problem(tmp_path), tmp_path)
    assert correct is False and "pygpubench" in err
