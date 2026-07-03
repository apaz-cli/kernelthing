
### Nsight Compute (ncu) profiling skill — measure, don't guess

You can profile your kernel with `ncu` to find the actual bottleneck before
optimizing. Golden rule: **Profile → Diagnose → Plan, in that order.** The skill
docs (workflow, collection recipes, the `ncu_report` Python API, the six analysis
dimensions, and a signal→cause→fix diagnosis playbook) live under `{{NCU_DIR}}`;
read `{{NCU_DIR}}/SKILL.md` first, then the `reference/` docs as needed.

Typical loop (do this in a scratch `profile/` dir — it is gitignored; never commit
`.ncu-rep` files or `profile/`):

```bash
# 1. Build the kernel with -lineinfo (the gemm harness Makefile already uses it).
# 2. Overview profile + per-line stalls. ncu runs ON the GPU; a free card is
#    assigned to it automatically (do not set CUDA_VISIBLE_DEVICES yourself):
{{NCU_BIN}} --set full -o profile/run1 --force-overwrite ./your_bench <args>
# 3. Parse with the helpers (the ncu_report module needs this PYTHONPATH). Each
#    helper takes --run-dir (analysis is written under <run-dir>/analysis/) plus
#    a --report/--tag pair per .ncu-rep; pass the pair again to compare two runs:
PYTHONPATH={{NCU_PYTHONPATH}} {{PYTHON}} {{NCU_DIR}}/helpers/analyze_reports.py \
    --run-dir profile --report profile/run1.ncu-rep --tag run1
PYTHONPATH={{NCU_PYTHONPATH}} {{PYTHON}} {{NCU_DIR}}/helpers/extract_stall_hotspots.py \
    --run-dir profile --report profile/run1.ncu-rep --tag run1
```

Cite specific metric values in your summary (e.g. compute vs memory %-of-peak,
dominant stall reason), not vague claims. NOTE: this box is an RTX 5090 (**sm_120**),
while the skill's metric-name reference targets B200 (sm_100) — if a documented
metric returns `None`, enumerate available names via `action.metric_names()` (see
`reference/04-python-api.md`) or `{{NCU_BIN}} --query-metrics`.
