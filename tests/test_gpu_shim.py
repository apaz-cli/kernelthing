"""Tests for the GPU-lock LD_PRELOAD shim and its wiring.

The shim (kernelthing/native/ktgpu.c -> libktgpu.so) claims a free GPU from the
pool on first CUDA use. Its device-selection/flock core is exercised here without
a real GPU: we drive ``ktgpu_acquire`` directly against temp lockfiles in a fresh
subprocess (the shim acquires once per process), and assert it picks a free card
and pins CUDA_VISIBLE_DEVICES -- including overriding an inherited value.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernelthing import gpulock, opencode_client

REPO = Path(__file__).resolve().parent.parent
SHIM_SRC = REPO / "kernelthing" / "native" / "ktgpu.c"


@pytest.fixture(scope="module")
def shim_so(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Prefer the packaged .so; otherwise compile it (skip if no C compiler)."""
    packaged = REPO / "kernelthing" / "libktgpu.so"
    if packaged.exists():
        return str(packaged)
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler and no prebuilt libktgpu.so")
    out = tmp_path_factory.mktemp("shim") / "libktgpu.so"
    subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", str(SHIM_SRC), "-o", str(out), "-ldl", "-lpthread"],
        check=True,
    )
    return str(out)


# Runs inside a fresh interpreter: load the shim, acquire, report chosen CVD.
_DRIVER = """
import ctypes, os, sys
libc = ctypes.CDLL(None)
libc.getenv.restype = ctypes.c_char_p
if len(sys.argv) > 3:
    libc.setenv(b"CUDA_VISIBLE_DEVICES", sys.argv[3].encode(), 1)
lib = ctypes.CDLL(sys.argv[1])
cvd_after_ctor = libc.getenv(b"CUDA_VISIBLE_DEVICES")
rc = lib.ktgpu_acquire()
cvd = libc.getenv(b"CUDA_VISIBLE_DEVICES")
import json
print(json.dumps({
    "rc": rc,
    "ctor": None if cvd_after_ctor is None else cvd_after_ctor.decode(),
    "cvd": None if cvd is None else cvd.decode(),
}))
"""


def _drive(shim: str, pool: str, preset_cvd: str | None = None) -> dict:
    argv = [sys.executable, "-c", _DRIVER, shim]
    if preset_cvd is not None:
        argv += ["", preset_cvd]
    env = dict(os.environ, KERNELTHING_GPU_POOL=pool)
    env.pop("LD_PRELOAD", None)
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_shim_picks_free_card(shim_so: str, tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    a.touch()
    b.touch()
    # Hold A busy from this process so the shim must fall through to B.
    held = os.open(str(a), os.O_RDWR)
    import fcntl

    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        out = _drive(shim_so, f"GPU-AAA={a};GPU-BBB={b}")
    finally:
        os.close(held)
    assert out["rc"] == 0
    assert out["cvd"] == "GPU-BBB", out


def test_shim_overrides_inherited_cvd(shim_so: str, tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    a.touch()
    # Launched with a bogus CUDA_VISIBLE_DEVICES=7 (an agent trying to dodge the
    # lock): the constructor must blank it, then acquire pins the locked card.
    out = _drive(shim_so, f"GPU-AAA={a}", preset_cvd="7")
    assert out["ctor"] == "", "constructor should blank inherited CVD"
    assert out["cvd"] == "GPU-AAA", out


def test_shim_inherits_held_card_without_relocking(shim_so: str, tmp_path: Path) -> None:
    # Simulate an ancestor that already locked GPU-AAA and exported it via
    # KERNELTHING_GPU_HELD (inherited through exec). The shim must adopt that card
    # WITHOUT taking a second flock -- even though the card's lockfile is busy.
    # If re-entrancy were broken it would block here and the test would time out
    # (the parent/child self-deadlock on a single-card pool).
    import fcntl

    a = tmp_path / "a.lock"
    a.touch()
    held = os.open(str(a), os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        argv = [sys.executable, "-c", _DRIVER, shim_so]
        env = dict(
            os.environ,
            KERNELTHING_GPU_POOL=f"GPU-AAA={a}",
            KERNELTHING_GPU_HELD="GPU-AAA",
        )
        env.pop("LD_PRELOAD", None)
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        out = json.loads(proc.stdout.strip())
    finally:
        os.close(held)
    assert out["rc"] == 0
    assert out["ctor"] == "GPU-AAA", "constructor should pin the inherited card"
    assert out["cvd"] == "GPU-AAA", out


def test_shim_inert_without_pool(shim_so: str) -> None:
    # No pool configured -> shim is a no-op and leaves CVD untouched.
    argv = [sys.executable, "-c", _DRIVER, shim_so, "", "5"]
    env = dict(os.environ)
    env.pop("KERNELTHING_GPU_POOL", None)
    env.pop("LD_PRELOAD", None)
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip())
    assert out["rc"] == -1
    assert out["cvd"] == "5", out


def test_gpu_pool_spec_format() -> None:
    spec = gpulock.gpu_pool_spec([0])
    assert "=" in spec
    uuid, _, lock = spec.partition("=")
    assert uuid == gpulock.gpu_uuid(0)
    assert lock == str(gpulock.lock_path(0))


def test_build_env_injects_shim_and_pool() -> None:
    if not gpulock.SHIM_PATH.exists():
        pytest.skip("shim not built")
    env, _ = opencode_client.build_opencode_env(gpu_pool=[0])
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["KERNELTHING_GPU_POOL"] == gpulock.gpu_pool_spec([0])
    assert "KERNELTHING_GPU_INDEX" not in env
    assert "KERNELTHING_GPU_LOCK" not in env
    assert env["LD_PRELOAD"].split(":")[0] == str(gpulock.SHIM_PATH)


def _cuda_build_toolchain() -> bool:
    """True if we can actually compile a CUDA extension (torch + nvcc present)."""
    if importlib.util.find_spec("torch") is None:
        return False
    if shutil.which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    return (Path(cuda_home) / "bin" / "nvcc").exists()


# Runs in a fresh interpreter under the shim: compile a trivial CUDA extension,
# then report whether a CUDA context was created and what CVD the shim left.
_ARCH_BUILD_DRIVER = r"""
import json, os, tempfile, torch
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name="kt_archtest",
    cpp_sources="int answer();",
    cuda_sources="#include <torch/extension.h>\n__global__ void noop(){}\nint answer(){return 42;}\n",
    functions=["answer"],
    verbose=False,
    build_directory=tempfile.mkdtemp(),
)
print(json.dumps({
    "answer": mod.answer(),
    "cuda_initialized": bool(torch.cuda.is_initialized()),
    "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
}))
"""


@pytest.mark.skipif(not _cuda_build_toolchain(), reason="needs torch + nvcc to compile")
def test_arch_list_build_creates_no_context_and_shim_stays_idle(
    shim_so: str, tmp_path: Path
) -> None:
    """Compiling a CUDA extension with ``TORCH_CUDA_ARCH_LIST`` set must not create
    a CUDA context -- so under the libktgpu shim the build claims no card. That is
    exactly what lets builds run outside the GPU lock and overlap across agents.

    We use the shim as the detector: run the build under LD_PRELOAD with a pool
    pointing at a temp lockfile. If the build had made any CUDA call the shim would
    have flocked the card and pinned CUDA_VISIBLE_DEVICES to its UUID; a build that
    touches no CUDA leaves CVD blanked (the constructor's fail-closed default).
    """
    lock = tmp_path / "fake.lock"
    lock.touch()
    prev_preload = os.environ.get("LD_PRELOAD", "")
    env = dict(
        os.environ,
        LD_PRELOAD=f"{shim_so}:{prev_preload}" if prev_preload else shim_so,
        KERNELTHING_GPU_POOL=f"GPU-FAKE={lock}",
        TORCH_CUDA_ARCH_LIST="8.9",
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)  # let the shim constructor blank it
    proc = subprocess.run(
        [sys.executable, "-c", _ARCH_BUILD_DRIVER],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["answer"] == 42
    assert out["cuda_initialized"] is False, "arch-list build must not initialize CUDA"
    assert out["cvd"] == "", f"shim should not have claimed a card; CVD={out['cvd']!r}"
