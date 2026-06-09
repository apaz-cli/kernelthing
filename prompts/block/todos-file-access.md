# Todos File Access Blocked

Do NOT create or access `round-*-todos.md` files.

**Use opencode's built-in todo tool instead (`todowrite` / `todoread`).**

The todo tool tracks task state in the session itself (with `[mainline]` /
`[blocking]` / `[queued]` lane tags), which the loop reads to verify all
blocking work is finished before the round can complete.
