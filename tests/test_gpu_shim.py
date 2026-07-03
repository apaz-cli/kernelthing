"""Tests for the GPU-lock LD_PRELOAD shim and its wiring.

The shim (kernelthing/native/ktgpu.c -> libktgpu.so) claims a free GPU from the
pool on first CUDA use. Its device-selection/flock core is exercised here without
a real GPU: we drive ``ktgpu_acquire`` directly against temp lockfiles in a fresh
subprocess (the shim acquires once per process), and assert it picks a free card
and pins CUDA_VISIBLE_DEVICES -- including overriding an inherited value.
"""

from __future__ import annotations

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
    env, _ = opencode_client.build_opencode_env(gpu_pool=[0])
    # Fail-closed backstop stays.
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    # Pool is passed to the shim; the old per-agent pinning vars are gone.
    assert env["KERNELTHING_GPU_POOL"] == gpulock.gpu_pool_spec([0])
    assert "KERNELTHING_GPU_INDEX" not in env
    assert "KERNELTHING_GPU_LOCK" not in env
    if opencode_client.GPU_SHIM.exists():
        assert env["LD_PRELOAD"].split(":")[0] == str(opencode_client.GPU_SHIM)
