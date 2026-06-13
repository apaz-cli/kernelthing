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

from . import bootstrap
from .bus import LoopBus
from .config import Config
from .orchestrator import Orchestrator
from .problem import load_problem, prepare_problem


_BANNER_ART = r"""
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
_GREEN, _BLUE, _RESET = "\033[92m", "\033[94m", "\033[0m"
_TAGLINE = "Evolutionary Autoresearch Optimization of GPU Kernels."


def _colorize_banner(art: str) -> str:
    """Bright-green the art; recolor the tagline prose bright blue (its line also
    carries art glyphs on the right, so the two halves are colored separately)."""
    lines = []
    for line in art.split("\n"):
        if _TAGLINE in line:
            end = line.index(_TAGLINE) + len(_TAGLINE)
            line = f"{_BLUE}{line[:end]}{_GREEN}{line[end:]}"
        lines.append(line)
    return f"{_GREEN}{chr(10).join(lines)}{_RESET}"


# Color only when writing to a terminal; piped/redirected --help stays plain.
BANNER = _colorize_banner(_BANNER_ART) if sys.stdout.isatty() else _BANNER_ART


def _duration(text: str) -> int:
    """argparse type for ``-w/--wall-clock``: parse a duration into whole seconds.

    Accepts a number with an optional ``s/m/h/d/w`` suffix (``10m``, ``2h``,
    ``1d``, ``90s``); a bare number is seconds. ``0`` means 'off'. This exists
    because bare-seconds was a footgun -- ``-w 10`` reads as 10 *seconds*, not the
    10 minutes one might expect."""
    from .config import format_duration, parse_duration

    try:
        return parse_duration(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid duration '{text}'; use e.g. 90s, 10m, 2h, 1d, 1w "
            "(a bare number is seconds)")


def _default_gpu() -> int:
    """Seed ``--gpu`` from ``CUDA_VISIBLE_DEVICES`` when it names a single device.

    A bare ``CUDA_VISIBLE_DEVICES=1 kernelthing ...`` is the natural way to pick a
    GPU, but the env var alone is ignored: kernelthing *overrides*
    CUDA_VISIBLE_DEVICES on every subprocess from ``gpu_index`` (see
    opencode_client.run and the scoring subprocess in orchestrator). So honour it
    here as the default for ``--gpu`` (an explicit ``--gpu`` still wins). Only a
    single integer is meaningful -- the loop pins exactly one device -- so a list
    ("0,1") or a UUID falls back to 0.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return int(raw) if raw.isdigit() else 0


def _read_objective(args: argparse.Namespace) -> str | None:
    """The bootstrap objective: ``--objective-file`` wins, else a non-path positional."""
    if args.objective_file:
        return Path(args.objective_file).read_text(encoding="utf-8")
    return args.problem  # a positional that wasn't a problem dir is objective text


def _resolve_problem(args: argparse.Namespace, cfg: Config):
    """Load an existing problem dir, or bootstrap a new one from an objective."""
    src = args.problem
    if src and not args.objective_file:
        p = Path(src)
        if (p.is_dir() and (p / "problem.json").is_file()) or \
                (p.is_file() and p.name == "problem.json"):
            problem = load_problem(p)
            # Keep the problem's bootstrap-prompt.md snapshot current with the live
            # template before the run copies the dir (the copy inherits the refresh).
            bootstrap.refresh_bootstrap_prompt(problem.dir, problem.repo_root)
            return prepare_problem(problem, cfg.problem_root)
        if p.exists():
            raise RuntimeError(f"{src} exists but is not a problem dir (no problem.json); "
                               "pass a problem dir, or an objective to bootstrap from")
    # Bootstrap mode: build a new problem dir inside a managed repo.
    target = bootstrap.bootstrap_problem(
        _read_objective(args), cfg=cfg, auto=args.auto_setup,
        managed_root=cfg.problem_root)
    return load_problem(target)


def _run(args: argparse.Namespace) -> int:
    cfg = Config(
        model=args.model,
        opencode_timeout=args.timeout,
        gpu_index=args.gpu,
        methodology=args.methodology,
        sandbox=not args.no_sandbox,
        parallelism=args.parallelism,
        kernelguard=not args.no_kernelguard,
        ncu=not args.no_ncu,
        wiki=not args.no_wiki,
        pygpubench=not args.no_pygpubench,
        auto_setup=args.auto_setup,
        max_candidates=args.max_candidates,
        wall_clock_s=args.wall_clock,
        elite_k=args.elite_k,
        problem_root=args.problem_root,
    )

    try:
        problem = _resolve_problem(args, cfg)
    except (FileNotFoundError, RuntimeError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    bus = None
    if not args.no_web:
        from . import webui
        bus = LoopBus(args.parallelism, args.wall_clock)
        try:
            _httpd, port = webui.start_server(bus, port=args.web_port)
            print(f"[kernelthing] web UI:   http://127.0.0.1:{port}", file=sys.stderr)
        except OSError as e:
            print(f"[kernelthing] web UI disabled ({e}); continuing headless", file=sys.stderr)
            bus = None
    else:
        print(f"[kernelthing] running headless (--no-web)", file=sys.stderr)

    print(f"[kernelthing] problem:   {problem.name}", file=sys.stderr)
    print(f"[kernelthing] artifacts: {problem.repo_root}", file=sys.stderr)
    from .config import format_duration
    wall_str = format_duration(args.wall_clock) if args.wall_clock else "none"
    budget_str = f"{args.max_candidates} candidates" if args.max_candidates else "unbounded"
    print(f"[kernelthing] agents:    {args.parallelism}  wall: {wall_str}  budget: {budget_str}", file=sys.stderr)

    orch = Orchestrator(problem, cfg, bus)
    try:
        exit_reason = orch.run()
    except KeyboardInterrupt:
        print("\n[kernelthing] interrupted", file=sys.stderr)
        return 130
    print(f"[kernelthing] loop finished: {exit_reason}", file=sys.stderr)
    # complete / stalled_out / stopped / maxiter all leave a correct HEAD.
    return 0 if exit_reason in ("complete", "stalled_out", "stopped", "maxiter") else 1


def _score_cmd(argv: list[str]) -> int:
    """``kernelthing score [<dir>]``: run the authoritative scorer on a problem dir
    and print its JSON verdict.

    This is the *same* ``bench.score`` call that gates bootstrap (``_validate``) and
    every loop round -- so a green here means the problem (or a kernel edit) scores
    correct for real, with no bespoke self-test harness to drift out of sync.
    """
    from . import bench

    p = argparse.ArgumentParser(
        prog="kernelthing score",
        description="Score a problem dir with the authoritative pygpubench benchmark "
                    "and print {correct, metric, unit}. Same code path the loop scores "
                    "with -- use it to check a freshly authored problem or a kernel edit.")
    p.add_argument("dir", nargs="?", default=".",
                   help="problem dir containing problem.json (default: current dir)")
    args = p.parse_args(argv)

    try:
        problem = load_problem(Path(args.dir))
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    correct, metric, err = bench.score(problem, problem.repo_root)
    print(json.dumps({"correct": correct, "metric": metric,
                      "unit": problem.unit, "error": err}))
    return 0 if correct else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "score":
        return _score_cmd(argv[1:])

    parser = argparse.ArgumentParser(
        prog="kernelthing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=BANNER,
        epilog="subcommands:\n"
               "  score [<dir>]   score a problem dir with the authoritative benchmark and\n"
               "                  print {correct, metric, unit} -- the same scoring the loop\n"
               "                  uses; run `kernelthing score --help` for details")
    parser.add_argument("problem", nargs="?",
                        help="path to a problem dir (containing problem.json) or the manifest. "
                             "Omit it (or pass a natural-language objective instead) to bootstrap "
                             "a new problem dir first; without --auto-setup this is interactive.")

    search = parser.add_argument_group(
        "search budget & shape", "how long the evolutionary search runs and how wide it goes")
    search.add_argument("-j", "--parallelism", type=int, default=4,
                        help="max agents editing at once (live-tunable down in the web UI; "
                             "the GPU benchmark stays serial). Default 4.")
    search.add_argument("-k", "--elite-k", type=int, default=4,
                        help="size of the top-K frontier (the exploit pool). Default 4.")
    search.add_argument("-m", "--max-candidates", type=int, default=24,
                        help="budget: total candidates to dispatch, then stop. "
                             "0 = run until --wall-clock / web-UI stop. Default 24.")
    search.add_argument("-w", "--wall-clock", type=_duration, default=0, metavar="DUR",
                        help="budget: wall-clock duration, then stop. Accepts an s/m/h/d/w "
                             "suffix (e.g. 10m, 2h, 1d); a bare number is seconds. 0 = off. "
                             "Default 0.")

    models = parser.add_argument_group("model")
    models.add_argument("--model", default="deepseek/deepseek-v4-pro",
                        help="opencode model that edits kernels and authors problems "
                             "(default: %(default)s)")
    models.add_argument("--timeout", type=int, default=5400,
                        help="per-opencode-turn timeout in seconds (default: %(default)s)")

    boot = parser.add_argument_group("bootstrap (authoring a new problem)")
    boot.add_argument("--auto-setup", action="store_true",
                      help="build the new problem dir non-interactively and auto-accept on "
                           "validation pass (no interactive review). Requires an objective "
                           "(positional arg or --objective-file).")
    boot.add_argument("--objective-file", metavar="PATH",
                      help="read the objective description from this file instead of the "
                           "positional argument (useful for long descriptions)")

    tools = parser.add_argument_group("agent tools & sandboxing")
    tools.add_argument("--no-sandbox", action="store_true", help="disable bwrap (debug only)")
    tools.add_argument("--no-kernelguard", action="store_true",
                       help="disable kernelguard benchmark-cheat detection (rollback/disqualify)")
    tools.add_argument("--no-ncu", action="store_true",
                       help="don't offer the agent the Nsight Compute (ncu) profiling skill, "
                            "and don't bind the GPU perf-counter nodes in the sandbox")
    tools.add_argument("--no-wiki", action="store_true",
                       help="don't offer the agent the KernelWiki kernel-optimization knowledge base")
    tools.add_argument("--no-pygpubench", action="store_true",
                       help="score with the problem's plain score_command instead of the "
                            "pygpubench adversarial sandboxed benchmark (debug / no-torch boxes)")

    web = parser.add_argument_group("web UI")
    web.add_argument("--web-port", type=int, default=8765, help="web UI port (default: %(default)s)")
    web.add_argument("--no-web", action="store_true", help="run headless (no web UI)")

    runtime = parser.add_argument_group("runtime & output")
    runtime.add_argument("--gpu", type=int, default=_default_gpu(),
                         help="CUDA device index to pin (defaults to $CUDA_VISIBLE_DEVICES "
                              "when it is a single index, else 0)")
    runtime.add_argument("--problem-root", type=Path,
                         default=Path.home() / ".cache" / "kernelthing",
                         help="managed problem repo root (worktrees branch from copies here)")
    runtime.add_argument("--methodology", action="store_true",
                         help="at loop exit, run a retrospective that writes a sanitized "
                              "methodology report (methodology-analysis-report.md) to the loop dir")

    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
