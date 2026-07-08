"""NVIDIA GPU hardware control for reproducible benchmarking.

Locks power limits and clock frequencies so every benchmark run sees the same
hardware state. Uses ``nvidia-smi`` subprocess calls (requires root for power
limits and clock locking on most systems).

The ``HardwareLock`` context manager applies a requested config and resets on
exit (even on exception — reset runs in ``__exit__``). Everything is
best-effort: failures produce warnings and the run proceeds on default
hardware state.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_SMI_TIMEOUT = 15


@dataclass
class HardwareConfig:
    """Requested hardware state to apply before benchmarking.

    All fields are optional — ``None`` means "don't change this setting".
    Clock locks are ``(min_mhz, max_mhz)`` pairs.
    """

    power_limit_watts: int | None = None
    gpu_clock_lock: tuple[int, int] | None = None
    mem_clock_lock: tuple[int, int] | None = None
    device_ids: list[int] = field(default_factory=lambda: [0])


def parse_mhz_pair(spec: str | None) -> tuple[int, int] | None:
    """Parse a CLI-style ``"MIN,MAX"`` MHz string into a clock-lock pair.

    ``None``, empty, or a value without a comma-separated pair is silently
    ignored (returns ``None``).
    """
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 2:
        return None
    return int(parts[0]), int(parts[1])


def _smi(*args: str) -> subprocess.CompletedProcess[str]:
    """Run nvidia-smi, or return a returncode=-1 sentinel when it's unavailable."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return subprocess.CompletedProcess([], -1, stdout="", stderr="nvidia-smi not found")
    return subprocess.run([smi, *args], capture_output=True, text=True, timeout=_SMI_TIMEOUT)


def query_power_limit(device_id: int) -> int | None:
    """Current power limit in watts, or ``None`` if unreadable."""
    r = _smi("-i", str(device_id), "--query-gpu=power.limit", "--format=csv,noheader,nounits")
    if r.returncode != 0:
        return None
    try:
        return int(float(r.stdout.strip()))
    except ValueError:  # "[N/A]"
        return None


def set_power_limit(device_id: int, watts: int) -> bool:
    """Set the GPU's power limit in watts. Requires root."""
    r = _smi("-i", str(device_id), "-pl", str(watts))
    if r.returncode != 0:
        logger.warning("Failed to set power limit on GPU %d: %s", device_id, r.stderr.strip())
        return False
    logger.info("GPU %d power limit set to %d W", device_id, watts)
    return True


def lock_gpu_clocks(device_id: int, min_mhz: int, max_mhz: int) -> bool:
    """Lock GPU core clock to a fixed range ``[min_mhz, max_mhz]``.

    GPU must be idle (no compute/graphics processes active). Requires root.
    """
    r = _smi("-i", str(device_id), "-lgc", f"{min_mhz},{max_mhz}")
    if r.returncode != 0:
        logger.warning(
            "Failed to lock GPU clocks on GPU %d (%d-%d MHz): %s",
            device_id, min_mhz, max_mhz, r.stderr.strip(),
        )
        return False
    logger.info("GPU %d clocks locked to %d-%d MHz", device_id, min_mhz, max_mhz)
    return True


def lock_mem_clocks(device_id: int, min_mhz: int, max_mhz: int) -> bool:
    """Lock GPU memory clock to a fixed range ``[min_mhz, max_mhz]``.

    GPU must be idle. Requires root.
    """
    r = _smi("-i", str(device_id), "-lmc", f"{min_mhz},{max_mhz}")
    if r.returncode != 0:
        logger.warning(
            "Failed to lock memory clocks on GPU %d (%d-%d MHz): %s",
            device_id, min_mhz, max_mhz, r.stderr.strip(),
        )
        return False
    logger.info("GPU %d memory clocks locked to %d-%d MHz", device_id, min_mhz, max_mhz)
    return True


def reset_clocks(device_id: int) -> bool:
    """Reset GPU and memory clocks to default (variable) behaviour."""
    ok = True
    for flag, name in [("-rgc", "graphics"), ("-rmc", "memory")]:
        r = _smi("-i", str(device_id), flag)
        if r.returncode != 0:
            logger.warning(
                "Failed to reset %s clocks on GPU %d: %s",
                name, device_id, r.stderr.strip(),
            )
            ok = False
    if ok:
        logger.info("GPU %d clocks reset to defaults", device_id)
    return ok


def apply_config(cfg: HardwareConfig) -> list[str]:
    """Apply a ``HardwareConfig`` to all specified GPUs.

    Returns a list of warning/error messages for anything that failed.
    Attempts every setting on every GPU; one failure does not skip the rest.
    """
    warnings: list[str] = []
    for dev in cfg.device_ids:
        if cfg.power_limit_watts is not None and not set_power_limit(dev, cfg.power_limit_watts):
            warnings.append(f"GPU {dev}: failed to set power limit")
        if cfg.gpu_clock_lock and not lock_gpu_clocks(dev, *cfg.gpu_clock_lock):
            warnings.append(f"GPU {dev}: failed to lock GPU clocks")
        if cfg.mem_clock_lock and not lock_mem_clocks(dev, *cfg.mem_clock_lock):
            warnings.append(f"GPU {dev}: failed to lock memory clocks")
    return warnings


class HardwareLock:
    """Context manager that applies GPU hardware state and resets on exit.

    Usage::

        cfg = HardwareConfig(power_limit_watts=300, gpu_clock_lock=(1500, 1500))
        with HardwareLock(cfg) as warnings:
            if warnings:
                print("Some settings failed:", warnings)
            # Benchmarks here run under locked hardware.
        # Clocks and power limit restored.

    ``__enter__`` snapshots each device's current power limit (nvidia-smi
    reports the overridden value as the current limit, so the pre-change value
    must be captured up front). ``__exit__`` resets only what the config
    touched: locked clocks back to variable behaviour, power limits back to
    the snapshot.
    """

    def __init__(self, cfg: HardwareConfig):
        self._cfg = cfg
        self._orig_power_limits: dict[int, int] = {}
        self.warnings: list[str] = []

    def __enter__(self) -> list[str]:
        cfg = self._cfg
        for dev in cfg.device_ids:
            # Persistence mode so the settings survive driver unload.
            _smi("-i", str(dev), "-pm", "1")
            if cfg.power_limit_watts is not None:
                orig = query_power_limit(dev)
                if orig is not None:
                    self._orig_power_limits[dev] = orig
        self.warnings = apply_config(cfg)
        return self.warnings

    def __exit__(self, *exc: Any) -> None:
        cfg = self._cfg
        for dev in cfg.device_ids:
            with contextlib.suppress(Exception):
                if cfg.gpu_clock_lock or cfg.mem_clock_lock:
                    reset_clocks(dev)
                orig = self._orig_power_limits.get(dev)
                if orig is not None:
                    set_power_limit(dev, orig)
        self._orig_power_limits.clear()
