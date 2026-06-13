# kernelthing

A kernel optimization autoresearch loop built on top of [`opencode`](https://github.com/sst/opencode) and DeepSeek V4.

Iteratively improves a GPU kernel, scoring every attempt against a known-correct
baseline and keeping only genuine improvements. Originally a DSV4/Opencode
reimplementation of [Humanize](https://github.com/PolyArch/humanize), but now
is rather different and has many more features.

## Setup

```bash
python -m pip install -e .
git submodule update --init # fetch vendored KernelWiki + ncu-report-skill

# external prerequisites
opencode --version     # CLI agent on PATH + DEEPSEEK_API_KEY
bwrap --version        # bubblewrap sandbox
nvcc --version         # CUDA compiler (for problems/gemm)
# As well as whatever else your kernels might need.
```

## Quick start

```bash
kernelthing                                                  # Jump into an opencode session to define a new problem interactively
kernelthing -j 8                                             # Launch with 8 concurrent agents working on improving kernels
kernelthing problems/gemm -j 8                               # opens web UI at localhost:8765
kernelthing problems/gemm --max-candidates 50                # headless, set budgets
```

Running without `--no-web` will open a WebUI at [http://localhost:8765](http://localhost:8765) from which to view progress and tweak parameters.


## Targeting your own problem

A **problem** is a directory inside a git repo with a `problem.json` manifest:

```json
{
  "name": "my-kernel",
  "plan": "plan.md",
  "edit_files": ["kernels/mykernel.cu"],
  "score_command": "bash score.sh",
  "metric_name": "pct_baseline", "unit": "%baseline",
  "direction": "maximize", "bench_runs": 3
}
```

You can write it manually, or have the agent write it on launch. Having the agent write it is recommended.

### Score scripts must benchmark honestly

A single timed run is not a measurement. The bundled `problems/gemm` harness shows
the bar: **multi-seed correctness**, **steady-state warmup**, an **L2-cache flush
between timed iterations**, and **median of many iters with min/max spread**.
kernelthing additionally re-runs your score command `bench_runs` times and keeps only
the best of all-correct runs. Hold your own problems to the same standard.

## Web UI

`http://127.0.0.1:<port>` (stdlib-only, zero deps). Live search view:

- **Best vs. submitted** — best-so-far staircase chart with each attempt a dot
  colored by operator (explore/exploit/salvage), failures marked.
- **Agents (live)** — cards showing operator, parent, tool-call count, cost, and
  the latest tool call + reasoning line from the streaming opencode log. Click for
  full transcript.
- **Niches** (MAP-Elites: best per strategy), **Lineage** (parent→child mutation
  tree), **Leaderboard** with self-triage (fixable/fundamental + estimated ceiling).

Disable with `--no-web`.

## Search strategy: 

The search strategy is the biggest difference between kernelthing and humanize. Humanize takes an RLCR approach.
We take an async evolutionary population approach. This allows us to separate parallel tasks like code editing from
tasks like benchmarking, and allows us to maintain idea diversity.

- a **controller** owns the population/archive and a task queue;
- a pool of **mutation workers** (many, concurrent) each take a parent + operator,
  edit in an isolated worktree, and submit the result;
- **all GPU access is serialized** by a per-device lock — not just the
  authoritative benchmark, but every agent's own build/run/profile too.

Results flow back continuously, so the GPU is always fed and agents always working.

**Three operators**, with compute budget split across them:

| Operator | Purpose | How |
|---|---|---|
| **Explore** (breadth) | new lineages | start from a seed, take a strategy *not yet in the archive* |
| **Exploit** (depth) | deepen winners | refine a top scoring commit |
| **Salvage / crossover** | rescue near-misses | fix the *named* reason a stalled candidate underperformed, or merge two ideas |

**Diversity via MAP-Elites** niches keyed on agent-reported strategy descriptors
(tiling, vectorization, tensor-core use, …): the best kernel *per niche* is kept,
and exploration targets empty niches so the search can't collapse onto one lineage.

**Prune vs. salvage uses agent judgment, backstopped by the metric.** Each candidate
self-triages — `fundamental` vs. `fixable` plus an estimated ceiling. That judgment
only *allocates compute*; the measured score still decides what is elite.

**Honesty**: the measured benchmark is the only thing that decides what is elite or
gets promoted. The run stops on a global budget (wall-clock / candidate count), not a
round count. `-j` sets max concurrent agents; all GPU work stays serialized.

**One GPU at a time.** A per-device `flock` (`kernelthing/gpulock.py`) keyed on the
physical GPU **UUID** (not the CUDA index, which is relative to each process's
`CUDA_VISIBLE_DEVICES`) gates every GPU command — the authoritative benchmark and
each agent's own runs/`ncu` profiles, which go through a `gpu-run` wrapper bound
into the sandbox. So nothing ever contends on the device, even across separate
kernelthing processes targeting the same card. Multi-GPU is one process per device
(`--gpu N` / `CUDA_VISIBLE_DEVICES=N`); each acquires only its own card's lock.

## Sandboxing

Every edit-capable agent runs under **bubblewrap**: filesystem read-only except the
candidate's worktree, opencode's own state, and `/tmp`; GPU device nodes bound through;
GPU pinned via `CUDA_VISIBLE_DEVICES`. Network stays up (the model API needs it);
the filesystem is the confinement boundary. opencode's `--dangerously-skip-permissions`
is only safe because of this.

## Kernel tooling (KernelWiki + ncu profiling)

Two vendored [KDA](https://github.com/mit-han-lab/kernel-design-agents) skills
(git submodules under `vendor/`), injected into the agent's prompt:

- **KernelWiki** (`vendor/KernelWiki`) — Blackwell/Hopper kernel-optimization
  knowledge base (read-only query).
- **ncu-report-skill** (`vendor/ncu-report-skill`) — Nsight Compute profiling
  workflow.

Disable with `--no-wiki` / `--no-ncu`.

## GPU profiling permission (for ncu)

NVIDIA drivers restrict performance counters to admins by default. The agent runs
non-root under bubblewrap (which sets `no_new_privs`), so profiling fails with
`ERR_NVGPUCTRPERM` until you allow non-root access:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo reboot
```

Verify: `cat /proc/driver/nvidia/params | grep Profiling` → `RmProfilingAdminOnly: 0`.
Run with `--no-ncu` to skip profiling entirely.
