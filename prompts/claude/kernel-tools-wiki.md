
### KernelWiki — Blackwell/Hopper kernel-optimization knowledge base

A cross-referenced wiki of GPU kernel optimization (2179 merged PRs from CUTLASS,
vLLM, SGLang, FlashInfer, PyTorch, DeepGEMM + 48 synthesis pages). Consult it
*before* guessing at a technique — cite the page/PR you took an idea from in your
summary. Use it for tensor-core GEMM/attention/MoE patterns, tcgen05/TMEM/TMA,
warp specialization, FP8/FP4 block scaling, CuTe-DSL/PTX/Triton on Blackwell.

Query it (runs read-only, from any directory):

```bash
# natural-language search
{{PYTHON}} {{WIKI_DIR}}/scripts/query.py "how to overlap TMA loads with tcgen05 mma" --limit 5
# filtered search
{{PYTHON}} {{WIKI_DIR}}/scripts/query.py --tag gemm --architecture sm100 --limit 10
# fetch a specific page (id or path), optionally with its sources
{{PYTHON}} {{WIKI_DIR}}/scripts/get_page.py kernel-flash-attention-4 --follow-sources
# regex search across wiki + PR bodies
{{PYTHON}} {{WIKI_DIR}}/scripts/grep_wiki.py "tcgen05\.fence"
```

Start broad with `references/primer.md` under `{{WIKI_DIR}}` if unsure what to ask.
