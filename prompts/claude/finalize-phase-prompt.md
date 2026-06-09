# Finalize Phase

the reviewer's diff review has passed. The implementation is complete and all acceptance criteria have been met.

You are now in the **Finalize Phase**. This is your opportunity to simplify and refactor the code before final completion.

## Your Task

Review and simplify the recently changed code yourself, using your normal editing tools. Cleanup only -- do not change behavior.

## Constraints

These constraints are **non-negotiable**:

1. **Must NOT change existing functionality** - All features must work exactly as before
2. **Must NOT fail existing tests** - Run tests to verify nothing is broken
3. **Must NOT introduce new bugs** - Be careful with refactoring
4. **Only perform functionality-equivalent changes** - Simplification and cleanup only

## Focus Areas

The code simplification agent should focus on:
- Code that was recently added or modified
- Focus more on changes between branch from `{{BASE_BRANCH}}` to `{{START_BRANCH}}`
- Removing unnecessary complexity
- Improving readability and maintainability
- Consolidating duplicate code
- Simplifying control flow where possible
- Removing dead code or unused variables

## Reference Files

- Original plan: @{{PLAN_FILE}}
- Goal tracker: @{{GOAL_TRACKER_FILE}}

## Before Exiting

1. Complete all `[mainline]` and `[blocking]` todos (mark each `completed` with the `todowrite` tool; an untagged todo counts as blocking)
2. `[queued]` todos may remain only if they are documented as non-blocking follow-up work
3. Commit your changes with a descriptive message
4. Write your finalize summary to: **{{FINALIZE_SUMMARY_FILE}}**

Your summary should include:
- What simplifications were made
- Files modified during the Finalize Phase
- Confirmation that tests still pass
- Any notes about the refactoring decisions
