"""Command-line entry point: ``kernelthing [<problem-dir> | <objective>]``.

A *problem* is a directory containing a ``problem.json`` manifest (see
problem.py). Pass an existing problem dir to optimize it directly. Omit it (or
pass a natural-language objective instead) and kernelthing first *bootstraps* a
new problem dir with an opencode agent (see bootstrap.py): interactively by
default, or non-interactively with ``--auto-setup`` (which then needs an
objective). Either way it auto-detects the enclosing git repo, runs the loop, and
serves a web UI for watching progress and live-tuning N / the turn cap / stop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import bootstrap
from .config import Config
from .orchestrator import Orchestrator
from .problem import Problem, load_problem, prepare_problem


def _check_gpu_model(problem: Problem, gpus: list[int]) -> str | None:
    """Return an error message if any GPU doesn't match the problem's required model.

    When ``problem.gpu`` is empty (pre-existing or hand-authored problems) this is
    a no-op — no restriction.
    """
    if not problem.gpu:
        return None
    from . import gpupool

    bad: list[str] = []
    for idx in gpus:
        name = gpupool.gpu_name(idx)
        if name != problem.gpu:
            bad.append(f"  GPU {idx}: {name}")
    if not bad:
        return None
    return (
        f"\nProblem '{problem.name}' requires GPU model: {problem.gpu}\n"
        "but the following GPU(s) don't match:\n" + "\n".join(bad) + "\n\n"
        "Use --gpu to pick matching GPUs, or remove the 'gpu' field from\n"
        "the problem's problem.json if this restriction is wrong.\n"
    )

BANNER_ART = r"""
 __                                          ___     __      __
/\ \                                        /\_ \   /\ \__  /\ \        __
\ \ \/'\       __    _ __     ___       __  \//\ \  \ \ ,_\ \ \ \___   /\_\     ___       __
 \ \ , <     /'__`\ /\`'__\ /' _ `\   /'__`\  \ \ \  \ \ \/  \ \  _ `\ \/\ \  /' _ `\   /'_ `\
  \ \ \\`\  /\  __/ \ \ \/  /\ \/\ \ /\  __/   \_\ \_ \ \ \_  \ \ \ \ \ \ \ \ /\ \/\ \ /\ \L\ \
   \ \_\ \_\\ \____\ \ \_\  \ \_\ \_\\ \____\  /\____\ \ \__\  \ \_\ \_\ \ \_\\ \_\ \_\\ \____ \
    \/_/\/_/ \/____/  \/_/   \/_/\/_/ \/____/  \/____/  \/__/   \/_/\/_/  \/_/ \/_/\/_/ \/___L\ \
        Evolutionary Autoresearch Optimization of GPU Kernels.                            /\____/
                                                                                          \/___/
"""

# bright green for the figlet, bright blue for the tagline prose.
GREEN, BLUE, RESET = "\033[92m", "\033[94m", "\033[0m"
TAGLINE = "Evolutionary Autoresearch Optimization of GPU Kernels."


def colorize_banner(art: str) -> str:
    """Bright-green the art; recolor the tagline prose bright blue (its line also
    carries art glyphs on the right, so the two halves are colored separately)."""
    lines = []
    for line in art.split("\n"):
        if TAGLINE in line:
            end = line.index(TAGLINE) + len(TAGLINE)
            line = f"{BLUE}{line[:end]}{GREEN}{line[end:]}"
        lines.append(line)
    return f"{GREEN}{chr(10).join(lines)}{RESET}"


# Color only when writing to a terminal; piped/redirected --help stays plain.
BANNER = colorize_banner(BANNER_ART) if sys.stdout.isatty() else BANNER_ART


def duration(text: str) -> int:
    """argparse type for ``-w/--wall-clock``: parse a duration into whole seconds.

    Accepts a number with an optional ``s/m/h/d/w`` suffix (``10m``, ``2h``,
    ``1d``, ``90s``); a bare number is seconds. ``0`` means 'off'. This exists
    because bare-seconds was a footgun -- ``-w 10`` reads as 10 *seconds*, not the
    10 minutes one might expect."""
    from .config import parse_duration

    try:
        return parse_duration(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"invalid duration '{text}'; use e.g. 90s, 10m, 2h, 1d, 1w (a bare number is seconds)"
        ) from err


def default_gpu() -> list[int]:
    """Seed ``--gpu`` from ``CUDA_VISIBLE_DEVICES``.

    A bare ``CUDA_VISIBLE_DEVICES=1 kernelthing ...`` is the natural way to pick a
    GPU, but the env var alone is ignored: kernelthing *overrides*
    CUDA_VISIBLE_DEVICES on every subprocess from the configured GPU indices. So
    honour it here as the default for ``--gpu`` (an explicit ``--gpu`` still wins).
    Accepts a comma-separated list too (``CUDA_VISIBLE_DEVICES=0,1``).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [0]
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    return indices or [0]


def read_objective(args: argparse.Namespace) -> str | None:
    """The bootstrap objective: ``--objective-file`` wins, else a non-path positional."""
    if args.objective_file:
        return Path(args.objective_file).read_text(encoding="utf-8")
    return (
        str(args.problem) if args.problem is not None else None
    )  # a positional that wasn't a problem dir is objective text


def resolve_problem(args: argparse.Namespace, cfg: Config) -> Problem:
    """Load an existing problem dir, or bootstrap a new one from an objective."""
    src = args.problem
    if src and not args.objective_file:
        p = Path(src)
        if (p.is_dir() and (p / "problem.json").is_file()) or (
            p.is_file() and p.name == "problem.json"
        ):
            problem = load_problem(p)
            # Keep the problem's bootstrap-prompt.md snapshot current with the live
            # template before the run copies the dir (the copy inherits the refresh).
            bootstrap.refresh_bootstrap_prompt(problem.dir, problem.repo_root)
            return prepare_problem(problem, cfg.problem_root)
        if p.exists():
            raise RuntimeError(
                f"{src} exists but is not a problem dir (no problem.json); "
                "pass a problem dir, or an objective to bootstrap from"
            )
    # Bootstrap mode: build a new problem dir inside a managed repo.
    target = bootstrap.bootstrap_problem(
        read_objective(args), cfg=cfg, auto=args.auto_setup, managed_root=cfg.problem_root
    )
    return load_problem(target)


def run_loop(args: argparse.Namespace) -> int:
    gpus: list[int] = args.gpu if args.gpu else default_gpu()
    cfg = Config(
        model=args.model,
        opencode_timeout=args.timeout,
        gpu_indices=gpus,
        methodology=args.methodology,
        sandbox=not args.no_sandbox,
        parallelism=args.parallelism,
        kernelguard=not args.no_kernelguard,
        ncu=not args.no_ncu,
        wiki=not args.no_wiki,
        auto_setup=args.auto_setup,
        max_candidates=args.max_candidates,
        wall_clock_s=args.wall_clock,
        elite_k=args.elite_k,
        min_niches=args.min_niches,
        problem_root=args.problem_root,
    )

    from . import gpupool

    gpupool.warm_cache(gpus)

    if not args.override_gpu:
        arch_warning = gpupool.check_architecture_mismatch(gpus)
        if arch_warning:
            print(arch_warning, file=sys.stderr)
            ans = input("[kernelthing] Proceed anyway? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                return 1

    try:
        problem = resolve_problem(args, cfg)
    except (FileNotFoundError, RuntimeError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not args.override_gpu:
        gpu_err = _check_gpu_model(problem, gpus)
        if gpu_err:
            print(gpu_err, file=sys.stderr)
            return 2

    if not args.no_web:
        from . import webui

        try:
            _httpd, port = webui.start_server(cfg.problem_root, port=args.web_port)
            print(f"[kernelthing] web UI:   http://127.0.0.1:{port}", file=sys.stderr)
        except OSError as e:
            print(f"[kernelthing] web UI disabled ({e}); continuing headless", file=sys.stderr)
    else:
        print("[kernelthing] running headless (--no-web)", file=sys.stderr)

    print(f"[kernelthing] problem:   {problem.name}", file=sys.stderr)
    print(f"[kernelthing] artifacts: {problem.repo_root}", file=sys.stderr)
    from .config import format_duration

    wall_str = format_duration(args.wall_clock) if args.wall_clock else "none"
    budget_str = f"{args.max_candidates} candidates" if args.max_candidates else "unbounded"
    gpu_str = f"{len(gpus)} GPU{'s' if len(gpus) > 1 else ''} ({', '.join(map(str, gpus))})"
    print(
        f"[kernelthing] agents:    {args.parallelism}  wall: {wall_str}  budget: {budget_str}",
        file=sys.stderr,
    )
    print(f"[kernelthing] gpus:      {gpu_str}", file=sys.stderr)

    orch = Orchestrator(problem, cfg)
    try:
        exit_reason = orch.run()
    except KeyboardInterrupt:
        print("\n[kernelthing] interrupted", file=sys.stderr)
        orch.persist_current_head()
        return 130
    print(f"[kernelthing] loop finished: {exit_reason}", file=sys.stderr)
    # complete / stalled_out / stopped / maxiter all leave a correct HEAD.
    return 0 if exit_reason in ("complete", "stalled_out", "stopped", "maxiter") else 1


def score_command(argv: list[str]) -> int:
    """``kernelthing score [<dir>]``: run the authoritative scorer on a problem dir
    and print its JSON verdict.

    This is the *same* ``bench.score`` call that gates bootstrap
    (``validate_problem``) and every loop round -- so a green here means the
    problem (or a kernel edit) scores correct for real.

    ``--gpu N`` (repeatable) picks the GPU; without it, any free GPU of the same
    model as the default device (``CUDA_VISIBLE_DEVICES``/GPU 0) is used, falling
    back to a blocking lock on GPU 0 if all are busy.
    """
    from . import bench, gpupool

    p = argparse.ArgumentParser(
        prog="kernelthing score",
        description="Score a problem dir with the authoritative pygpubench benchmark "
        "and print {correct, metric, unit}. Same code path the loop scores "
        "with -- use it to check a freshly authored problem or a kernel edit.",
    )
    p.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="problem dir containing problem.json (default: current dir)",
    )
    p.add_argument(
        "--gpu",
        type=int,
        action="append",
        help="CUDA device index to score on (may be repeated). Without it the "
        "scorer picks any free GPU of the same model as the default device.",
    )
    p.add_argument(
        "--override-gpu",
        action="store_true",
        help=argparse.SUPPRESS,
        default=False,
    )
    # Orchestrator-internal (used by Orchestrator._cli_score); hidden from agents,
    # who only ever run bare `kernelthing score .`.
    p.add_argument("--baseline-median", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--emit-baseline", action="store_true", default=False, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    try:
        problem = load_problem(Path(args.dir))
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not args.override_gpu and args.gpu:
        gpu_err = _check_gpu_model(problem, args.gpu)
        if gpu_err:
            print(gpu_err, file=sys.stderr)
            return 2

    if args.gpu:
        pool = gpupool.candidate_gpus(preferred=args.gpu)
    elif problem.gpu:
        pool = gpupool.candidate_gpus(model=problem.gpu)
    else:
        default_index = default_gpu()[0]
        pool = gpupool.candidate_gpus(
            model=gpupool.gpu_name(default_index),
            arch=gpupool.gpu_architecture(default_index),
        )
    # Compile the kernel off-lock first (pure host-side nvcc, no GPU): the shimmed
    # pygpubench worker then reuses the cached .so instead of compiling while
    # holding the card, so the lock covers only the actual run.
    bench.warm_build(problem, problem.repo_root, arch=gpupool.torch_arch_list(pool))

    # Baseline denominator for pct_baseline/speedup: pin the one passed in, or (with
    # --emit-baseline) measure it once here and report it so the caller can pin it
    # on subsequent scores. Both run sequentially in this single process, so there
    # is no shared-state race -- unlike scoring concurrently in threads.
    result: dict[str, Any] = {"unit": problem.unit}
    baseline_median = args.baseline_median
    if args.emit_baseline and baseline_median is None:
        baseline_median, berr = bench.measure_baseline(problem, problem.repo_root, gpu_pool=pool)
        result["baseline_median"] = baseline_median
        if berr:
            print(json.dumps({**result, "correct": False, "metric": None, "error": berr}))
            return 1
    # GPU access is serialized inside bench.score: it hands pygpubench's isolated
    # worker the libktgpu.so LD_PRELOAD shim together with this candidate pool. The
    # shim probes the pool for a free card and flocks it for the worker's lifetime.
    # No in-process flock here -- taking one would deadlock against the shim's.
    correct, metric, err, detail = bench.score(
        problem, problem.repo_root, gpu_pool=pool, baseline_median=baseline_median
    )
    result.update({"correct": correct, "metric": metric, "error": err, "bench": detail})
    print(json.dumps(result))
    return 0 if correct else 1


def web_command(argv: list[str]) -> int:
    """``kernelthing web``: serve the web UI standalone over a directory of runs.

    Discovers every run (live or finished) under ``--root`` -- each run dir is
    self-describing on disk (run.json / events.ndjson / members/), so no run
    process needs to be alive. Use it to inspect old runs, or as a single UI
    over several concurrent ``kernelthing --no-web`` runs.
    """
    from . import webui

    p = argparse.ArgumentParser(
        prog="kernelthing web",
        description="Serve the kernelthing web UI over a directory of runs "
        "(live and finished). Runs live in <root>/<problem>/.humanize/rlcr/.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache" / "kernelthing",
        help="directory to discover runs under (default: %(default)s)",
    )
    p.add_argument("--port", type=int, default=8765, help="port (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    args = p.parse_args(argv)

    httpd = webui.make_server(args.root, port=args.port, host=args.host)
    print(
        f"[kernelthing] web UI: http://{args.host}:{httpd.server_address[1]}  "
        f"(runs from {args.root})",
        file=sys.stderr,
    )
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "score":
        return score_command(argv[1:])
    if argv and argv[0] == "web":
        return web_command(argv[1:])

    parser = argparse.ArgumentParser(
        prog="kernelthing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=BANNER,
        epilog="subcommands:\n"
        "  score [<dir>]   score a problem dir with the authoritative benchmark and\n"
        "                  print {correct, metric, unit} -- the same scoring the loop\n"
        "                  uses; run `kernelthing score --help` for details\n"
        "  web             serve the web UI standalone over all runs (live and\n"
        "                  finished) under --root; run `kernelthing web --help`",
    )
    parser.add_argument(
        "problem",
        nargs="?",
        help="path to a problem dir (containing problem.json) or the manifest. "
        "Omit it (or pass a natural-language objective instead) to bootstrap "
        "a new problem dir first; without --auto-setup this is interactive.",
    )

    search = parser.add_argument_group(
        "search budget & shape",
        "how long the evolutionary search runs and how wide it goes; "
        "-j/-k/-m/-w are all live-tunable from the web UI while the run is going",
    )
    search.add_argument(
        "-j",
        "--parallelism",
        type=int,
        default=4,
        help="max agents editing at once (live-tunable up and down in the web "
        "UI; the GPU benchmark stays serial). Default 4.",
    )
    search.add_argument(
        "-k",
        "--elite-k",
        type=int,
        default=4,
        help="size of the top-K frontier (the exploit pool); live-tunable "
        "in the web UI. Default 4.",
    )
    search.add_argument(
        "--min-niches",
        type=int,
        default=4,
        help="below this many strategy niches, bias toward explore "
        "to fill out the diversity grid. Default 4.",
    )
    search.add_argument(
        "-m",
        "--max-candidates",
        type=int,
        default=24,
        help="budget: total candidates to dispatch, then stop. 0 = run until "
        "--wall-clock / web-UI stop. Live-tunable in the web UI. Default 24.",
    )
    search.add_argument(
        "-w",
        "--wall-clock",
        type=duration,
        default=0,
        metavar="DUR",
        help="budget: wall-clock duration, then stop. Accepts an s/m/h/d/w "
        "suffix (e.g. 10m, 2h, 1d); a bare number is seconds. 0 = off. "
        "Live-tunable in the web UI. Default 0.",
    )

    models = parser.add_argument_group("model")
    models.add_argument(
        "--model",
        default="deepseek/deepseek-v4-pro",
        help="opencode model that edits kernels and authors problems (default: %(default)s)",
    )
    models.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="per-opencode-turn timeout in seconds (default: %(default)s)",
    )

    boot = parser.add_argument_group("bootstrap (authoring a new problem)")
    boot.add_argument(
        "--auto-setup",
        action="store_true",
        help="build the new problem dir non-interactively and auto-accept on "
        "validation pass (no interactive review). Requires an objective "
        "(positional arg or --objective-file).",
    )
    boot.add_argument(
        "--objective-file",
        metavar="PATH",
        help="read the objective description from this file instead of the "
        "positional argument (useful for long descriptions)",
    )

    tools = parser.add_argument_group("agent tools & sandboxing")
    tools.add_argument("--no-sandbox", action="store_true", help="disable bwrap (debug only)")
    tools.add_argument(
        "--no-kernelguard",
        action="store_true",
        help="disable kernelguard benchmark-cheat detection (rollback/disqualify)",
    )
    tools.add_argument(
        "--no-ncu",
        action="store_true",
        help="don't offer the agent the Nsight Compute (ncu) profiling skill, "
        "and don't bind the GPU perf-counter nodes in the sandbox",
    )
    tools.add_argument(
        "--no-wiki",
        action="store_true",
        help="don't offer the agent the KernelWiki kernel-optimization knowledge base",
    )

    web = parser.add_argument_group("web UI")
    web.add_argument(
        "--web-port", type=int, default=8765, help="web UI port (default: %(default)s)"
    )
    web.add_argument("--no-web", action="store_true", help="run headless (no web UI)")

    runtime = parser.add_argument_group("runtime & output")
    runtime.add_argument(
        "--gpu",
        type=int,
        action="append",
        help="CUDA device index to pin (may be given multiple times, e.g. "
        "--gpu 0 --gpu 1). Defaults to $CUDA_VISIBLE_DEVICES "
        "when set, else [0].",
    )
    runtime.add_argument(
        "--problem-root",
        type=Path,
        default=Path.home() / ".cache" / "kernelthing",
        help="managed problem repo root (worktrees branch from copies here)",
    )
    runtime.add_argument(
        "--methodology",
        action="store_true",
        help="at loop exit, run a retrospective that writes a sanitized "
        "methodology report (methodology-analysis-report.md) to the loop dir",
    )
    runtime.add_argument(
        "--override-gpu",
        action="store_true",
        help=argparse.SUPPRESS,
        default=False,
    )

    args = parser.parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
