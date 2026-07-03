# GPU allocation is managed for you

You referenced part of the GPU-allocation machinery (`CUDA_VISIBLE_DEVICES`,
`LD_PRELOAD`, `KERNELTHING_*`, or the GPU-lock shim). This is not something you
configure or inspect.

A free GPU is assigned to each of your processes automatically and held for that
process's lifetime, so kernels compile, run, and profile without any wrapper or
environment setup. Setting or unsetting these variables will not give you a
different card — just run your benchmark, `ncu`, or `nsys` normally.

If every GPU is busy your process pauses at its first CUDA call until one frees;
that is normal queuing, not a hang.
