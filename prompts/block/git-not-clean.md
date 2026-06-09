# Git Not Clean

You are trying to stop, but you have **{{GIT_ISSUES}}**.
{{SPECIAL_NOTES}}
**Required Actions**:
0. If the code simplification plugin is installed, use it to review and simplify your code before committing. Invoke via: code simplification, code simplification, or code simplification
1. Review untracked files - add build artifacts to `.gitignore`
2. Stage only real changes with specific paths: `git add <files>`
3. Commit with a descriptive message following project conventions

**Important Rules**:
- Do NOT use `git add -A`, `git add --all`, or `git add .` during an active RLCR loop
- Never stage `.humanize/` or legacy `.humanize-*` loop artifacts
- Commit message must follow project conventions
- AI tools (implementer, reviewer, etc.) must NOT have authorship in commits
- Do NOT include `Co-Authored-By: implementer` or similar AI attribution

After committing all changes, you may attempt to exit again.
