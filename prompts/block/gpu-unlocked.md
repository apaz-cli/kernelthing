# GPU command must hold the shared-GPU lock

You tried to run a GPU profiler (`ncu`/`nsys`) directly. You share one physical
GPU with other agents and the authoritative scorer — running on it concurrently
corrupts every timing measurement and can OOM the device.

Wrap any command that executes on the GPU in the `gpu-run` lock wrapper, which
holds a per-device lock for the command's whole duration:

```bash
{{GPU_RUN}} ncu --set full -o profile/run1 --force-overwrite ./your_bench <args>
{{GPU_RUN}} ./your_bench <args>
```

CPU-only steps (building with `nvcc`, parsing `.ncu-rep` reports, querying
KernelWiki) do not need the wrapper.
