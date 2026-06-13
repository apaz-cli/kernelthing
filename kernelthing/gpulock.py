"""Cross-process GPU mutex keyed on the *physical* device UUID.

The GPU is a shared, serially-used resource: the authoritative benchmark and the
agents' own build/run/profile work must never hit the same device at once
(concurrent runs corrupt timing and can OOM). A ``threading.Semaphore`` can't
coordinate this -- the agents are separate ``opencode`` subprocesses, and several
kernelthing instances may target one box -- so the lock is an OS-level ``flock``
on a file shared by everyone using that device.

The key is the device's persistent UUID (``nvidia-smi --query-gpu=uuid``), not the
CUDA index: the index is relative to each process's ``CUDA_VISIBLE_DEVICES``
masking/ordering, so index 0 in one process can be a different card than index 0
in another. The UUID is invariant, so two processes that name the same GPU by
different indices still share one lock.

``flock`` releases automatically when the holding fd is closed (including on
process death), so a crashed agent or a SIGKILLed benchmark child can never wedge
the device. The lockfile lives in the system temp dir; the orchestrator binds it
into each agent's bubblewrap sandbox at the same path (see ``sandbox.wrap``) so
the inode -- and therefore the lock -- is shared across the sandbox boundary.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_UUID_CACHE: dict[int, str] = {}


def gpu_uuid(index: int) -> str:
    """Physical UUID of CUDA device *index*, or a stable ``index-N`` fallback.

    Resolved once per index via ``nvidia-smi`` and cached. Any failure (no
    nvidia-smi, query error) degrades to ``index-<n>`` -- the lock still works
    for matching indices on one host, it just loses the cross-ordering
    invariance the UUID gives.
    """
    if index in _UUID_CACHE:
        return _UUID_CACHE[index]
    uuid = f"index-{index}"
    smi = shutil.which("nvidia-smi")
    if smi:
        with contextlib.suppress(Exception):
            out = subprocess.run(
                [smi, f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                val = out.stdout.strip().splitlines()
                if val and val[0].strip():
                    uuid = val[0].strip()
    _UUID_CACHE[index] = uuid
    return uuid


def _slug(uuid: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", uuid) or "unknown"


def lock_path(index: int) -> Path:
    """Path to the lockfile for the physical GPU behind CUDA *index*.

    Lives in the system temp dir, named by device UUID so every process on the
    box targeting this card -- agents and benchmarks, across kernelthing runs --
    opens the same file. The empty file is created on first request (bwrap needs
    the bind source to exist before it can mount it into a sandbox).
    """
    p = Path(tempfile.gettempdir()) / f"kt-gpu-{_slug(gpu_uuid(index))}.lock"
    with contextlib.suppress(OSError):
        p.touch(exist_ok=True)
    return p


@contextlib.contextmanager
def gpu_lock(index: int):
    """Hold an exclusive flock on GPU *index* for the duration of the block.

    Blocking: waits until no other process (an agent's build/run/profile or
    another benchmark) holds the device. Released on exit and on process death.
    """
    path = lock_path(index)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
