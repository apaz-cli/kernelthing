"""The GPU pool: device discovery, lockfile naming, and the shim's environment.

kernelthing gives each GPU-using process exclusive use of one card. The
mechanism that enforces this -- flocking a per-card lockfile on the process's
first CUDA call -- lives entirely in the LD_PRELOAD shim (``native/ktgpu.c``,
whose header tells that story). Nothing in this module takes a GPU lock; it is
everything the orchestrator does *around* the shim, before any shimmed process
exists:

* **Identify cards** (``gpu_uuid``, ``gpu_name``, ``gpu_architecture``,
  ``discover_gpus``, ``warm_cache``): cached ``nvidia-smi`` queries.

* **Name each card's lockfile** (``lock_path``): a per-UUID file in the system
  temp dir, created empty here so bubblewrap can bind it into each sandbox at
  the same path (see ``sandbox.wrap``). Every process on the box that targets
  a card therefore flocks the same inode -- across sandboxes and across
  kernelthing instances.

* **Build the shim's environment** (``gpu_pool_spec``, ``apply_shim_env``):
  serialize the pool as ``UUID=lockpath;...``, inject ``LD_PRELOAD``, blank
  ``CUDA_VISIBLE_DEVICES`` (fail-closed), and set ``TORCH_CUDA_ARCH_LIST`` so
  kernel builds never touch CUDA and can run outside the lock.

* **Choose the pool and warn about it** (``candidate_gpus``,
  ``check_architecture_mismatch``).

Everything is keyed on the device's *physical* UUID (``nvidia-smi
--query-gpu=uuid``), never the CUDA index: the index is relative to each
process's ``CUDA_VISIBLE_DEVICES`` masking/ordering, so index 0 in one process
can be a different card than index 0 in another. The UUID is invariant, so two
processes that name the same GPU by different indices still share one lock.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

_UUID_CACHE: dict[int, str] = {}
_ARCH_CACHE: dict[int, str] = {}
_NAME_CACHE: dict[int, str] = {}

# libktgpu.so LD_PRELOAD shim, compiled next to this package by setup.py's build
# hook. Injected into scorer/agent processes so the first CUDA call flocks a free
# card from the pool; see native/ktgpu.c and gpu_pool_spec / apply_shim_env.
SHIM_PATH = Path(__file__).resolve().parent / "libktgpu.so"


def _smi_query(index: int, field: str) -> str | None:
    """First non-empty line of ``nvidia-smi --id=<index> --query-gpu=<field>``.

    Returns ``None`` on any failure (no nvidia-smi, query error, empty output).
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    with contextlib.suppress(Exception):
        out = subprocess.run(
            [smi, f"--id={index}", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            if lines and lines[0].strip():
                return lines[0].strip()
    return None


def _parse_arch(cc: str) -> str:
    """Turn a ``compute_cap`` string (e.g. ``"9.0"``) into an SM arch (``"sm_90"``)."""
    major, minor = cc.split(".") if "." in cc else (cc, "0")
    return f"sm_{major}{minor}"


def gpu_uuid(index: int) -> str:
    """Physical UUID of CUDA device *index*, or a stable ``index-N`` fallback.

    Resolved once per index via ``nvidia-smi`` and cached. Any failure (no
    nvidia-smi, query error) degrades to ``index-<n>`` -- the lock still works
    for matching indices on one host, it just loses the cross-ordering
    invariance the UUID gives.
    """
    if index in _UUID_CACHE:
        return _UUID_CACHE[index]
    uuid = _smi_query(index, "uuid") or f"index-{index}"
    _UUID_CACHE[index] = uuid
    return uuid


def slug(uuid: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", uuid) or "unknown"


def lock_path(index: int) -> Path:
    """Path to the lockfile for the physical GPU behind CUDA *index*.

    Lives in the system temp dir, named by device UUID so every process on the
    box targeting this card -- agents and benchmarks, across kernelthing runs --
    opens the same file. The empty file is created on first request (bwrap needs
    the bind source to exist before it can mount it into a sandbox).
    """
    p = Path(tempfile.gettempdir()) / f"kt-gpu-{slug(gpu_uuid(index))}.lock"
    with contextlib.suppress(OSError):
        p.touch(exist_ok=True)
    return p


def gpu_architecture(index: int) -> str:
    """SM compute capability string for CUDA device *index* (e.g. ``"sm_90"``, ``"sm_120"``).

    Queried once per index via ``nvidia-smi`` and cached. Any failure degrades to
    ``"unknown"``; the caller can still proceed (the user has been warned).
    """
    if index in _ARCH_CACHE:
        return _ARCH_CACHE[index]
    cc = _smi_query(index, "compute_cap")
    arch = _parse_arch(cc) if cc else "unknown"
    _ARCH_CACHE[index] = arch
    return arch


def gpu_name(index: int) -> str:
    """Human-readable GPU product name for CUDA device *index*. Best-effort."""
    if index in _NAME_CACHE:
        return _NAME_CACHE[index]
    name = _smi_query(index, "name") or f"GPU {index}"
    _NAME_CACHE[index] = name
    return name


def check_architecture_mismatch(indices: list[int]) -> str | None:
    """If the given GPU indices have mixed SM architectures, return a warning
    message suitable for display. Returns ``None`` when homogeneous or information
    is unavailable.

    Mixed architectures mean the same compiled kernel (PTX/SASS) may not be valid
    on all devices and absolute timing comparisons across GPUs are meaningless.
    """
    if len(indices) <= 1:
        return None
    arches: dict[str, list[int]] = {}
    for i in indices:
        a = gpu_architecture(i)
        arches.setdefault(a, []).append(i)
    if len(arches) <= 1:
        return None
    lines = [
        "",
        "=" * 72,
        "WARNING: GPU architecture mismatch detected!",
        "Different GPUs may not run the same compiled kernel correctly, and",
        "absolute performance comparisons across architectures are not valid.",
        "",
    ]
    for arch, idxs in sorted(arches.items()):
        names = [f"  GPU {j} ({gpu_name(j)}, {arch})" for j in idxs]
        lines.extend(names)
    lines.append("")
    lines.append("Proceed only if you understand these implications.")
    lines.append("=" * 72)
    return "\n".join(lines)


def discover_gpus() -> list[dict[str, object]]:
    """Return every visible GPU as ``[{index, name, arch}, ...]``.

    If ``nvidia-smi`` is unavailable or errors, returns an empty list.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    with contextlib.suppress(Exception):
        out = subprocess.run(
            [smi, "--query-gpu=index,name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            result: list[dict[str, object]] = []
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",", 2)]
                if len(parts) < 2:
                    continue
                idx = int(parts[0].strip())
                name = parts[1].strip() if len(parts) > 1 else f"GPU {idx}"
                cc = parts[2].strip() if len(parts) > 2 else "0.0"
                result.append({"index": idx, "name": name, "arch": _parse_arch(cc)})
            return result
    return []


def warm_cache(indices: list[int] | None = None) -> bool:
    """Populate UUID, architecture, and name caches for every visible GPU in one
    ``nvidia-smi`` call.  When *indices* is given only those indices are resolved
    (still in one batch call) and unrecognised indices get a best-effort
    ``_smi_query``.  Returns ``True`` when ``nvidia-smi`` succeeded (caches are
    populated from a live query); ``False`` means fallback values are in play.

    Call this once at startup before any per-GPU queries to avoid paying the
    ``nvidia-smi`` startup penalty N times.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    fields = "index,name,compute_cap,uuid"
    with contextlib.suppress(Exception):
        out = subprocess.run(
            [smi, f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",", 3)]
                if len(parts) < 2:
                    continue
                idx = int(parts[0].strip())
                if len(parts) > 1 and parts[1]:
                    _NAME_CACHE[idx] = parts[1]
                if len(parts) > 2 and parts[2]:
                    _ARCH_CACHE[idx] = _parse_arch(parts[2])
                if len(parts) > 3 and parts[3]:
                    _UUID_CACHE[idx] = parts[3]
            return True
    return False


def gpu_pool_spec(indices: list[int]) -> str:
    """Serialize a GPU pool as ``UUID=lockpath;UUID2=lockpath2;...`` for the shim.

    Consumed by the ``libktgpu.so`` LD_PRELOAD shim (``KERNELTHING_GPU_POOL``): on
    first CUDA use a shimmed process flocks the first free lockfile and pins
    ``CUDA_VISIBLE_DEVICES`` to its UUID. The lockfile paths are exactly those
    ``lock_path`` creates, so every shimmed process -- agents and the benchmark's
    isolated worker alike -- serializes on the same inode. The UUID form is what
    ``CUDA_VISIBLE_DEVICES`` accepts unambiguously regardless of device
    enumeration order.
    """
    return ";".join(f"{gpu_uuid(i)}={lock_path(i)}" for i in indices)


def torch_arch_list(indices: list[int]) -> str:
    """``TORCH_CUDA_ARCH_LIST`` value covering the given GPUs (e.g. ``"8.9"`` for an
    RTX 4070, ``"8.9;12.0"`` for a mixed pool). Empty if no arch is determinable.

    Setting this lets ``torch.utils.cpp_extension`` pick its ``-gencode`` flags
    without calling ``torch.cuda.get_device_capability()`` -- so compilation never
    creates a CUDA context, never trips the libktgpu shim into taking the GPU
    lock, and therefore runs *outside* the lock (concurrently across agents). The
    lock is then held only for the actual benchmark run, not the build.
    """
    seen: list[str] = []
    for i in indices:
        arch = gpu_architecture(i)  # e.g. "sm_89", "sm_120"
        digits = arch[3:] if arch.startswith("sm_") else ""
        if not digits.isdigit() or len(digits) < 2:
            continue
        cc = f"{digits[:-1]}.{digits[-1]}"  # sm_89 -> 8.9 ; sm_120 -> 12.0
        if cc not in seen:
            seen.append(cc)
    return ";".join(seen)


def apply_shim_env(env: MutableMapping[str, str], indices: list[int], *,
                   overwrite_pool: bool = True) -> None:
    """Route CUDA in *env* through the ``libktgpu.so`` shim for the given pool.

    Mutates *env* (an ``os.environ``-like mapping) in place: blanks
    ``CUDA_VISIBLE_DEVICES`` (fail-closed -- a process that evades the shim sees
    no GPU, not an unlocked one), sets ``KERNELTHING_GPU_POOL`` from *indices*,
    prepends the shim to ``LD_PRELOAD`` (deduped), and sets
    ``TORCH_CUDA_ARCH_LIST`` (if unset) so kernel compilation doesn't create a
    CUDA context and can run outside the GPU lock. When *overwrite_pool* is
    ``False`` an already-set pool is left alone (an outer shim's full pool wins).

    Raises ``FileNotFoundError`` if ``libktgpu.so`` is not built -- there is no
    safe fallback without the shim.
    """
    if not SHIM_PATH.exists():
        raise FileNotFoundError(
            f"libktgpu.so not found at {SHIM_PATH}; rebuild with `python setup.py build`"
        )
    env["CUDA_VISIBLE_DEVICES"] = ""
    if overwrite_pool or not env.get("KERNELTHING_GPU_POOL"):
        spec = gpu_pool_spec(indices)
        if spec:
            env["KERNELTHING_GPU_POOL"] = spec
    if not env.get("TORCH_CUDA_ARCH_LIST"):
        arch = torch_arch_list(indices)
        if arch:
            env["TORCH_CUDA_ARCH_LIST"] = arch
    prev = env.get("LD_PRELOAD") or ""
    if str(SHIM_PATH) not in prev.split(":"):
        env["LD_PRELOAD"] = f"{SHIM_PATH}:{prev}" if prev else str(SHIM_PATH)


def candidate_gpus(preferred: list[int] | None = None,
                   *, model: str | None = None,
                   arch: str | None = None) -> list[int]:
    """Return the pool of GPU indices to hand the shim: *preferred* if given, else
    every visible GPU matching *model*/*arch* (else all visible GPUs, else ``[0]``).

    No card is selected or locked here -- this only decides *which* cards are
    candidates. The ``libktgpu.so`` shim owns all locking: given this pool it
    probes for a free card (non-blocking), and blocks on the first only if every
    one is busy. Used by ``kernelthing score`` to build the pool it hands the shim.
    """
    if preferred:
        return list(preferred)
    gpus = discover_gpus()
    matched = [
        int(g["index"])  # type: ignore[call-overload]
        for g in gpus
        if (model is None or g["name"] == model) and (arch is None or g["arch"] == arch)
    ]
    if matched:
        return matched
    if gpus:
        return [int(g["index"]) for g in gpus]  # type: ignore[call-overload]
    return [0]
