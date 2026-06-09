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
from pathlib import Path

NVIDIA_NODES = [
    "/dev/nvidia0",
    "/dev/nvidia1",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
]

# Extra device nodes Nsight Compute (ncu) needs to read GPU performance
# counters. Only bound when profiling is enabled -- a non-ncu run needs no
# access to them.
NVIDIA_CAPS_NODES = [
    "/dev/nvidia-caps/nvidia-cap1",
    "/dev/nvidia-caps/nvidia-cap2",
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
) -> list[str]:
    """Return ``inner_argv`` wrapped in a bwrap invocation (or unchanged if disabled).

    ``writable``: project_dir is bound read-write (implementer) vs read-only
    (reviewer). ``writable_extra`` paths are always bound read-write at their
    real locations -- used for opencode's own session/cache state so the
    implementer's ``-s`` session persists across rounds. ``ncu``: also bind the
    GPU performance-counter capability nodes so Nsight Compute can profile.
    """
    if not enabled:
        return inner_argv
    project_dir = Path(project_dir).resolve()

    args: list[str] = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--ro-bind", "/", "/",        # everything readable, nothing writable...
        "--proc", "/proc",
        "--dev", "/dev",              # fresh devtmpfs (null/zero/random/...)
        "--tmpfs", "/tmp",
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
    nodes = NVIDIA_NODES + (NVIDIA_CAPS_NODES if ncu else [])
    for node in nodes:
        if Path(node).exists():
            args += ["--dev-bind", node, node]

    # Run inside the project worktree.
    args += ["--chdir", str(project_dir)]
    args += inner_argv
    return args
