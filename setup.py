"""Build hook that compiles the GPU-lock LD_PRELOAD shim (kernelthing/native/ktgpu.c)
into ``kernelthing/libktgpu.so`` as part of the normal build.

Everything else about the package is declared in pyproject.toml; setuptools only
needs this file for the native step. The shim needs no CUDA headers -- it resolves
the real CUDA symbols at runtime via dlsym -- so a plain C compiler is the only
build-time requirement.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).parent.resolve()
SHIM_SRC = HERE / "kernelthing" / "native" / "ktgpu.c"
SHIM_NAME = "libktgpu.so"


def _compile_shim(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / SHIM_NAME
    cc = os.environ.get("CC", "cc")
    cmd = [
        cc, "-shared", "-fPIC", "-O2", "-Wall",
        str(SHIM_SRC), "-o", str(out), "-ldl", "-lpthread",
    ]
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            f"kernelthing: failed to build the GPU-lock shim ({SHIM_NAME}).\n"
            f"A C compiler (cc/gcc) is required. Command was:\n  {' '.join(cmd)}\n"
            f"Underlying error: {e}"
        ) from e


class BuildPyWithShim(build_py):
    def run(self) -> None:
        super().run()
        # Installed/wheel build: place the .so alongside the copied package.
        self._compile_into(Path(self.build_lib) / "kernelthing")
        # Editable/in-tree: keep the source checkout runnable too.
        self._compile_into(HERE / "kernelthing")

    def _compile_into(self, pkg_dir: Path) -> None:
        if pkg_dir.exists():
            _compile_shim(pkg_dir)


setup(cmdclass={"build_py": BuildPyWithShim})
