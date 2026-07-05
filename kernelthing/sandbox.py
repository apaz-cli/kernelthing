"""Bubblewrap sandbox for edit-capable (and read-only) agent processes.

Basic, not bulletproof: the property that matters is that an edit-capable
opencode run cannot write anywhere except the project worktree, opencode's own
state dir, and /tmp. The whole filesystem is mounted read-only and specific
paths are re-bound writable on top. Network is left shared (the DeepSeek API
needs it); the filesystem is the confinement boundary. The GPU device nodes are
bound through so kernels can compile and run.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from . import gpupool

# GPU control nodes shared by every device -- always bound when any GPU is used.
NVIDIA_CTRL_NODES = [
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
]


def _nvidia_nodes(indices: Sequence[int]) -> list[str]:
    """Per-device nodes for the pool (``/dev/nvidia<i>``) plus the shared control
    nodes. With no indices given, fall back to the first two cards so callers that
    don't specify a pool (e.g. tests) still get a working GPU binding."""
    idx = list(indices) if indices else [0, 1]
    return [f"/dev/nvidia{i}" for i in idx] + NVIDIA_CTRL_NODES

# Extra device nodes Nsight Compute (ncu) needs to read GPU performance
# counters. Only bound when profiling is enabled -- a non-ncu run needs no
# access to them.
NVIDIA_CAPS_NODES = [
    "/dev/nvidia-caps/nvidia-cap1",
    "/dev/nvidia-caps/nvidia-cap2",
]

# opencode loads skills from these directories. Mask them entirely so no
# user-level skills (flywheel, prose-restructurer, etc.) load into the agent's
# system prompt. The loop provides its own kernel-domain tooling via prompt
# injection; user skills are pure noise.
SKILL_HOMES = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".agents" / "skills",
]


def available() -> bool:
    return shutil.which("bwrap") is not None


def wrap(
    inner_argv: list[str],
    *,
    project_dir: Path,
    writable: bool,
    writable_extra: list[Path] | tuple[Path, ...] = (),
    enabled: bool = True,
    ncu: bool = True,
    gpu_indices: Sequence[int] = (),
) -> list[str]:
    """Return ``inner_argv`` wrapped in a bwrap invocation (or unchanged if disabled).

    ``writable``: project_dir is bound read-write (implementer) vs read-only
    (reviewer). ``writable_extra`` paths are always bound read-write at their
    real locations -- used for opencode's own session/cache state so the
    implementer's ``-s`` session persists across rounds. ``ncu``: also bind the
    GPU performance-counter capability nodes so Nsight Compute can profile.
    ``gpu_indices``: the GPU pool. Their per-device nodes are bound so kernels can
    run, and each device's flock lockfile (``gpupool.lock_path``) is bound
    read-write at its real path so the ``libktgpu.so`` shim inside the sandbox
    shares the same inode -- and therefore the same flock -- as the orchestrator's
    benchmark (the locks live under /tmp, which the sandbox otherwise masks with a
    fresh tmpfs; the explicit binds reach through it).
    """
    if not enabled:
        return inner_argv
    project_dir = Path(project_dir).resolve()

    args: list[str] = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",  # everything readable, nothing writable...
        "--proc",
        "/proc",
        "--dev",
        "/dev",  # fresh devtmpfs (null/zero/random/...)
        "--tmpfs",
        "/tmp",
    ]
    # ...then re-bind the few writable paths on top.
    if writable:
        args += ["--bind", str(project_dir), str(project_dir)]
    else:
        args += ["--ro-bind", str(project_dir), str(project_dir)]
    for extra in writable_extra:
        extra = Path(extra)
        extra.mkdir(parents=True, exist_ok=True)
        args += ["--bind", str(extra), str(extra)]

    # GPU device nodes (only those present on the host).
    nodes = _nvidia_nodes(gpu_indices) + (NVIDIA_CAPS_NODES if ncu else [])
    for node in nodes:
        if Path(node).exists():
            args += ["--dev-bind", node, node]

    # GPU mutex lockfiles (one per pool device): bound after --tmpfs /tmp so they
    # reach through the mask (same inode as the host => the shim's flock
    # coordinates with the scorer across the sandbox boundary).
    for idx in gpu_indices:
        lock = gpupool.lock_path(idx)
        if Path(lock).exists():
            args += ["--bind", str(lock), str(lock)]

    # Mask skill directories so opencode loads no user-level skills.
    for home in SKILL_HOMES:
        if home.exists():
            args += ["--tmpfs", str(home)]

    # Run inside the project worktree.
    args += ["--chdir", str(project_dir)]
    args += inner_argv
    return args
