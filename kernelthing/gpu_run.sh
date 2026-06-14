#!/usr/bin/env bash
# kernelthing GPU mutex wrapper.
#
# Run ANY command that touches the GPU through this -- your compiled benchmark,
# ncu, nsys -- so it holds the per-device lock for its whole duration. That keeps
# the agents' own GPU work from overlapping with each other or with the
# authoritative benchmark (concurrent runs corrupt timing and can OOM). The
# lockfile is keyed on the physical device UUID and bound into the sandbox by the
# orchestrator; see kernelthing/gpulock.py.
#
# The agent sandbox runs with CUDA_VISIBLE_DEVICES="" by default so bare python
# commands cannot touch the GPU. This wrapper sets CUDA_VISIBLE_DEVICES to the
# configured GPU index only while holding the per-device lock, so the GPU is
# accessible only through serialized access.
#
#   gpu-run ./your_bench --args
#   gpu-run ncu --set full -o profile/run1 ./your_bench
#
# If no lock is configured (sandbox disabled / standalone use) the command runs
# unwrapped, so this is always safe to prefix.
set -euo pipefail
lock="${KERNELTHING_GPU_LOCK:-}"
gpu="${KERNELTHING_GPU_INDEX:-0}"
timeout="${KERNELTHING_GPU_TIMEOUT:-120}"
# Pre-compile the kernel outside the lock so the real command only pays runtime.
# Best-effort: if there is no submission.py in cwd this fails silently.
python3 -c "import submission" 2>/dev/null || true
if [ -z "$lock" ]; then
  exec env CUDA_VISIBLE_DEVICES="$gpu" "$@"
fi
exec timeout "$timeout" flock "$lock" env CUDA_VISIBLE_DEVICES="$gpu" "$@"
