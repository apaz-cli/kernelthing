Your work is not finished. Read and execute the below with ultrathink.

# Setup phase: define the scoring objective

Before any optimization rounds can run, the **scoring objective** must be defined
as a pygpubench task spec. You are deriving it now from the plan and the existing
program. This is a one-time setup turn; you are NOT optimizing yet.

## Plan
@{{PLAN}}

## What you must produce
pygpubench scores a kernel by importing it **by qualified name** in an isolated,
sandboxed subprocess and timing it against a reference test case. You must author
these spec file(s) in this problem directory so that is possible:

{{SPEC_FILES}}

Concretely:
1. **`task.py`** — define `generate_test_case(*, seed, **test_args)` returning a
   tuple `((inputs...), (expected, atol, rtol))`:
   - the first element is the tuple of arguments the kernel is called with;
   - `expected` is the correct output tensor (compute it with a trustworthy
     reference — e.g. a torch op);
   - `atol`/`rtol` are the absolute/relative tolerances for correctness.
   It will be called with these test args: `{{TEST_ARGS}}`.
2. The submission entry point **`{{SUBMISSION_QUALNAME}}`** must exist and accept
   exactly the inputs `generate_test_case` returns. The editable kernel adapter is
   @{{SUBMISSION_FILE}} — make sure its public function matches.
3. If the metric needs a baseline (e.g. %-of-cuBLAS), author **`baseline.py`** with
   a reference kernel of the same signature (it is timed for the denominator).

Replace every `[To be ...]` placeholder. Derive shapes/dtypes/tolerances from the
plan and the program; do not guess silently — if the plan is ambiguous, choose the
most defensible interpretation and state it in your summary.

## Self-test
Confirm the objective actually runs and the shipped submission is correct:
`{{SELFTEST_CMD}}`
It must report the submission as correct.

## Finishing
- If you succeeded, end your message with a short "where things are" summary
  (which file defines the reference, the input shapes, the tolerances, the
  submission binding) and then the single word `{{COMPLETE}}` on its own line.
- If you genuinely cannot determine the objective (missing reference, unknowable
  shapes/tolerances), explain precisely what is missing and end with the single
  word `{{SETUP_BLOCKED}}` on its own line. Do not fabricate a plausible-but-wrong
  objective — a wrong objective silently corrupts every later round.

Do not edit anything except the spec file(s) and, if needed, the submission
adapter. Do not start optimizing.
