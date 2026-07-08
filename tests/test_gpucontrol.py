"""Tests for GPU hardware control (kernelthing/gpucontrol.py)."""

from __future__ import annotations

import pytest

from kernelthing import gpucontrol, gpupool

# One nvidia-smi probe for the whole session; every hardware test shares it.
_HAS_GPU = bool(gpupool.discover_gpus())
needs_gpu = pytest.mark.skipif(not _HAS_GPU, reason="needs nvidia-smi + GPU")


def test_parse_mhz_pair() -> None:
    assert gpucontrol.parse_mhz_pair(None) is None
    assert gpucontrol.parse_mhz_pair("") is None
    # Malformed (no comma pair) is silently ignored.
    assert gpucontrol.parse_mhz_pair("1500") is None
    assert gpucontrol.parse_mhz_pair("1500,1600") == (1500, 1600)
    assert gpucontrol.parse_mhz_pair(" 7000 , 7000 ") == (7000, 7000)


def test_hardware_lock_noop_without_devices() -> None:
    with gpucontrol.HardwareLock(gpucontrol.HardwareConfig(device_ids=[])) as warnings:
        assert warnings == []


@needs_gpu
def test_query_power_limit_reads() -> None:
    watts = gpucontrol.query_power_limit(0)
    # A GPU with power management off reports [N/A] -> None; otherwise sane.
    assert watts is None or watts > 0


@needs_gpu
def test_hardware_lock_applies_and_resets_power_limit() -> None:
    """Apply a power limit, verify it took, and verify reset restores original."""
    dev = 0
    before = gpucontrol.query_power_limit(dev)

    # Pick a limit that differs from current (if we can read it)
    target = 200
    if before is not None and before <= 200:
        target = before + 50

    cfg = gpucontrol.HardwareConfig(power_limit_watts=target, device_ids=[dev])
    with gpucontrol.HardwareLock(cfg) as warnings:
        if warnings:
            pytest.skip(f"power limit not supported: {warnings}")
        after = gpucontrol.query_power_limit(dev)
        assert after == target, f"expected {target}W, got {after}W"

    if before is not None:
        assert gpucontrol.query_power_limit(dev) == before
