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
#   gpu-run ./your_bench --args
#   gpu-run ncu --set full -o profile/run1 ./your_bench
#
# If no lock is configured (sandbox disabled / standalone use) the command runs
# unwrapped, so this is always safe to prefix.
set -euo pipefail
lock="${KERNELTHING_GPU_LOCK:-}"
if [ -z "$lock" ]; then
  exec "$@"
fi
exec flock "$lock" "$@"
