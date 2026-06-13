// kernelthing loop guard -- pure decision logic (the port of Humanize's
// hooks/loop-*-validator.sh predicates). This module has NO opencode coupling
// and registers NO plugin hooks, so it is safe to unit-test directly (see
// tests/test_oc_guard.py). The opencode plugin wrapper lives in guard.js.
//
// IMPORTANT: opencode treats every exported function in a loaded plugin file as
// a plugin factory. That is why the pure logic (which exports decide/GuardBlock)
// lives here, separate from guard.js -- opencode only loads guard.js.
//
// Block messages reuse the existing prompts/block/*.md templates verbatim
// (rendered with {{VAR}} substitution), so the wording stays in one place.

import fs from "node:fs";
import path from "node:path";

// --- path helpers -----------------------------------------------------------

const lower = (s) => String(s || "").toLowerCase();
const baseName = (p) => path.basename(String(p || ""));

function absPath(cfg, p) {
  const s = String(p || "");
  return path.normalize(path.isAbsolute(s) ? s : path.resolve(cfg.projectRoot, s));
}

function inLoop(cfg, abs) {
  const dir = path.normalize(cfg.loopDir);
  return abs === dir || abs.startsWith(dir + path.sep);
}

function underProject(cfg, abs) {
  const root = path.normalize(cfg.projectRoot);
  return abs === root || abs.startsWith(root + path.sep);
}

const RE_TODOS = /round-\d+-todos\.md$/;
const RE_PROMPT = /round-\d+-prompt\.md$/;
const RE_REVIEW_PROMPT = /round-\d+-review-prompt\.md$/;
const RE_SUMMARY = /round-\d+-summary\.md$/;
const RE_CONTRACT = /round-\d+-contract\.md$/;
const RE_ROUNDFILE = /round-(\d+)-(summary|prompt|contract)\.md$/;

function roundOf(nameLower) {
  const m = nameLower.match(/round-(\d+)-(summary|prompt|contract)\.md$/);
  return m ? Number(m[1]) : null;
}

// --- block-message rendering ------------------------------------------------

function render(cfg, name, vars, fallback) {
  let text = fallback || "";
  try {
    if (cfg.blockDir) {
      const p = path.join(cfg.blockDir, name + ".md");
      if (fs.existsSync(p)) text = fs.readFileSync(p, "utf8");
    }
  } catch {
    /* fall through to fallback */
  }
  for (const [k, v] of Object.entries(vars || {})) {
    text = text.split("{{" + k + "}}").join(String(v));
  }
  return text;
}

// Sentinel so internal errors (which fail open) are distinguishable from an
// intentional block (which must propagate to opencode as a tool rejection).
export class GuardBlock extends Error {}

function block(cfg, name, vars, fallback) {
  return new GuardBlock(render(cfg, name, vars, fallback));
}

// --- goal-tracker immutable-section preservation ----------------------------

function immutableSection(text) {
  const lines = String(text || "").split("\n");
  let capture = false;
  const out = [];
  for (const line of lines) {
    if (/^## IMMUTABLE SECTION\s*$/.test(line)) { capture = true; continue; }
    if (capture && (/^## MUTABLE SECTION\s*$/.test(line) || /^---\s*$/.test(line))) break;
    if (capture) out.push(line);
  }
  return out.join("\n").trim();
}

// Returns true if the proposed new content keeps the immutable section intact
// (or if the on-disk tracker has no immutable section -- legacy/None case).
function immutablePreserved(trackerPath, newContent) {
  let current = "";
  try { current = fs.readFileSync(trackerPath, "utf8"); } catch { return true; }
  const cur = immutableSection(current);
  if (!cur) return true;
  return cur === immutableSection(newContent);
}

function applyEdit(filePath, oldString, newString, replaceAll) {
  let content = "";
  try { content = fs.readFileSync(filePath, "utf8"); } catch { return null; }
  if (oldString == null) return content;
  if (replaceAll) return content.split(oldString).join(newString ?? "");
  const i = content.indexOf(oldString);
  return i < 0 ? content : content.slice(0, i) + (newString ?? "") + content.slice(i + oldString.length);
}

// --- bash command analysis (port of command_modifies_file) ------------------

function commandModifiesFile(commandLower, filePattern) {
  const fp = filePattern; // already a regex fragment matching the filename
  const patterns = [
    `>\\s*[^\\s]*${fp}`,
    `>>\\s*[^\\s]*${fp}`,
    `tee\\s+(-a\\s+)?[^\\s]*${fp}`,
    `sed\\s+-i[^|]*${fp}`,
    `awk\\s+-i\\s+inplace[^|]*${fp}`,
    `perl\\s+-[^\\s]*i[^|]*${fp}`,
    `(mv|cp)\\s+[^\\s]+\\s+[^\\s]*${fp}`,
    `rm\\s+(-[rfv]+\\s+)?[^\\s]*${fp}`,
    `dd\\s+.*of=[^\\s]*${fp}`,
    `truncate\\s+[^|]*${fp}`,
    `printf\\s+.*>\\s*[^\\s]*${fp}`,
    `exec\\s+[0-9]*>\\s*[^\\s]*${fp}`,
  ];
  return patterns.some((p) => new RegExp(p).test(commandLower));
}

function gitAddsHumanize(commandLower) {
  if (!/\bgit\s+add\b/.test(commandLower)) return false;
  if (/\bgit\s+add\b[^&|;]*\.humanize/.test(commandLower)) return true;
  if (/\bgit\s+add\b\s+(-a|--all)\b/.test(commandLower)) return true;
  if (/\bgit\s+add\b\s+(-f\s+)?\.(\s|$)/.test(commandLower)) return true;
  return false;
}

// --- per-tool decision logic ------------------------------------------------

function checkWriteLike(cfg, filePath, resultingContent) {
  if (!filePath) return null;
  const abs = absPath(cfg, filePath);
  const nameLower = lower(baseName(abs));

  // Edit-file enforcement: the agent may only write to the problem's designated
  // edit_files (e.g. kernel.cu).  Other problem assets (task.py, submission.py,
  // baseline.py, kernel_ref.cu, etc.) are protected.
  if (cfg.editFiles && cfg.editFiles.length > 0 && cfg.editDir) {
    const editDir = path.normalize(cfg.editDir) + path.sep;
    const inEditDir = abs === path.normalize(cfg.editDir) || abs.startsWith(editDir);
    if (inEditDir) {
      const isEditFile = cfg.editFiles.some(f => abs === absPath(cfg, f));
      const isProtected = (cfg.protectedFiles || []).some(f => lower(f) === nameLower);
      const rel = path.relative(cfg.editDir, abs);
      const hasBuildPrefix = rel.split(path.sep).some(seg => seg.startsWith("_"));
      const isBuildArtifact = hasBuildPrefix || nameLower === ".gitignore" || nameLower === "bootstrap-prompt.md" || nameLower.startsWith(".humanize");
      if (!isEditFile && !isBuildArtifact) {
        return block(cfg, "edit-file-protected",
          { EDIT_FILES: cfg.editFiles.join(", ") },
          "# Edit Blocked\n\nOnly the designated edit files may be modified in this problem directory:\n\n  " + cfg.editFiles.join("\n  "));
      }
      if (isProtected) {
        return block(cfg, "edit-file-protected",
          { EDIT_FILES: cfg.editFiles.join(", ") },
          "# Edit Blocked\n\nOnly the designated edit files may be modified in this problem directory:\n\n  " + cfg.editFiles.join("\n  "));
      }
    }
  }

  // Methodology phase: only the two sanitized artifacts may be written.
  if (cfg.phase === "methodology") {
    if (inLoop(cfg, abs) &&
        (nameLower === "methodology-analysis-report.md" || nameLower === "methodology-analysis-done.md")) {
      return null;
    }
    return block(cfg, "methodology-analysis-state-file-modification", {},
      "# Write Blocked During Methodology Analysis\n\nOnly methodology-analysis-report.md and methodology-analysis-done.md may be written during this phase.");
  }

  // Todos files: task state lives in opencode's todo tool, not a markdown file.
  if (RE_TODOS.test(nameLower)) {
    return block(cfg, "todos-file-access", {},
      "Do not create round-*-todos.md files; use the todowrite tool instead.");
  }

  // Prompt files: instructions FROM the reviewer; the implementer cannot edit them.
  if (RE_PROMPT.test(nameLower) || RE_REVIEW_PROMPT.test(nameLower)) {
    return block(cfg, "prompt-file-write", {}, "You cannot write to round-*-prompt.md files.");
  }

  // Loop state files.
  if (nameLower === "finalize-state.json" && inLoop(cfg, abs)) {
    return block(cfg, "finalize-state-file-modification", {},
      "You cannot modify finalize-state.json; it is managed by the loop.");
  }
  if (nameLower === "state.json" && inLoop(cfg, abs)) {
    return block(cfg, "state-file-modification", {}, "You cannot modify the loop state file.");
  }

  // Plan backup inside the loop dir.
  if (nameLower === "plan.md" && inLoop(cfg, abs)) {
    return block(cfg, "plan-backup-protected", {}, "The plan.md backup in the loop directory cannot be modified.");
  }

  // The real plan file (read-only for the duration of the run).
  if (cfg.planFile && abs === absPath(cfg, cfg.planFile)) {
    return block(cfg, "plan-file-modified",
      { PLAN_FILE: cfg.planFile, BACKUP_PATH: path.join(cfg.loopDir, "plan.md") },
      "Modifying the plan file is forbidden during an active session.");
  }

  // Goal tracker: only the active loop's, and (after round 0) only its mutable section.
  if (nameLower === "goal-tracker.md") {
    const correct = path.join(cfg.loopDir, "goal-tracker.md");
    if (abs !== correct) {
      return block(cfg, "goal-tracker-modification",
        { CURRENT_ROUND: cfg.currentRound, CORRECT_PATH: correct },
        "Write the goal tracker to the active loop directory only.");
    }
    if (cfg.currentRound > 0 && resultingContent != null && !immutablePreserved(correct, resultingContent)) {
      return block(cfg, "goal-tracker-modification",
        { CURRENT_ROUND: cfg.currentRound, CORRECT_PATH: correct },
        "After round 0 you may change only the MUTABLE SECTION of the goal tracker.");
    }
    return null;
  }

  // Summary / contract files.
  const isSummary = RE_SUMMARY.test(nameLower);
  const isContract = RE_CONTRACT.test(nameLower);
  // No active round contract exists during the finalize phase.
  if (cfg.phase === "finalize" && isContract) {
    return block(cfg, "finalize-contract-access", { ACTION: "write to" },
      "There is no active round contract during the finalize phase.");
  }
  if (nameLower === "finalize-summary.md") {
    return abs === path.join(cfg.loopDir, "finalize-summary.md")
      ? null
      : block(cfg, "wrong-summary-location",
          { CORRECT_PATH: path.join(cfg.loopDir, "finalize-summary.md") },
          "Write the finalize summary to the active loop directory.");
  }
  if (isSummary || isContract) {
    const type = isContract ? "contract" : "summary";
    const correctRound = path.join(cfg.loopDir, `round-${cfg.currentRound}-${type}.md`);
    if (!inLoop(cfg, abs)) {
      return block(cfg, isContract ? "wrong-contract-location" : "wrong-summary-location",
        { CORRECT_PATH: correctRound }, `Write the ${type} into the active loop directory: ${correctRound}`);
    }
    const r = roundOf(nameLower);
    if (r != null && r !== cfg.currentRound) {
      return block(cfg, "wrong-round-number",
        { ACTION: "write to", CLAUDE_ROUND: r, FILE_TYPE: type, CURRENT_ROUND: cfg.currentRound, CORRECT_PATH: correctRound },
        `Current round is ${cfg.currentRound}; write to ${correctRound}`);
    }
    const correct = path.join(cfg.loopDir, baseName(abs));
    if (abs !== correct) {
      return block(cfg, "wrong-directory-path",
        { ACTION: "write to", FILE_PATH: filePath, CORRECT_PATH: correct }, `Correct path: ${correct}`);
    }
  }
  return null;
}

function checkRead(cfg, filePath) {
  if (!filePath) return null;
  const abs = absPath(cfg, filePath);
  const nameLower = lower(baseName(abs));

  // Methodology phase: read only sanitized artifacts; no project files.
  if (cfg.phase === "methodology") {
    if (inLoop(cfg, abs)) {
      const ok = ["methodology-analysis-report.md", "methodology-analysis-done.md", "state.json"];
      if (ok.includes(nameLower) || RE_SUMMARY.test(nameLower) || /round-\d+-review-result\.md$/.test(nameLower) || nameLower === "loop.log") {
        return null; // the retrospective is allowed to read the development records
      }
      return block(cfg, null, {},
        "# Read Blocked During Methodology Analysis\n\nOnly methodology and development-record artifacts may be read during this phase.");
    }
    if (underProject(cfg, abs)) {
      return block(cfg, null, {},
        "# Read Blocked During Methodology Analysis\n\nReading project files is not allowed during the methodology phase.");
    }
    return null;
  }

  // No active round contract exists during the finalize phase.
  if (cfg.phase === "finalize" && RE_CONTRACT.test(nameLower)) {
    return block(cfg, "finalize-contract-access", { ACTION: "read" },
      "There is no active round contract during the finalize phase.");
  }

  // Goal tracker must be the active loop's.
  if (nameLower === "goal-tracker.md") {
    const correct = path.join(cfg.loopDir, "goal-tracker.md");
    if (abs !== correct) {
      return block(cfg, "wrong-file-location",
        { FILE_PATH: filePath, ACTIVE_LOOP_DIR: cfg.loopDir, CURRENT_ROUND: cfg.currentRound, CORRECT_PATH: correct },
        `Read the active loop goal tracker: ${correct}`);
    }
    return null;
  }

  // Round files (summary/prompt/contract) must come from the active loop dir.
  if (RE_ROUNDFILE.test(nameLower) && !inLoop(cfg, abs)) {
    return block(cfg, "wrong-file-location",
      { FILE_PATH: filePath, ACTIVE_LOOP_DIR: cfg.loopDir, CURRENT_ROUND: cfg.currentRound },
      `Loop files live in ${cfg.loopDir}/`);
  }
  // Wrong-round reads within the loop dir are blocked only OUTSIDE the review
  // phase: review prompts legitimately @-reference prior rounds' summaries, so
  // the reviewer must keep history access; the implementer must use the current round.
  if (cfg.phase !== "review" && RE_ROUNDFILE.test(nameLower) && inLoop(cfg, abs)) {
    const r = roundOf(nameLower);
    const type = nameLower.match(RE_ROUNDFILE)[2];
    if (r != null && r !== cfg.currentRound) {
      return block(cfg, "wrong-round-file",
        { CLAUDE_ROUND: r, FILE_TYPE: type, CURRENT_ROUND: cfg.currentRound,
          ACTIVE_LOOP_DIR: cfg.loopDir, FILE_PATH: filePath },
        `Current round is ${cfg.currentRound}; read round-${cfg.currentRound}-${type}.md`);
    }
  }
  return null;
}

function checkBash(cfg, command) {
  if (!command) return null;
  const c = lower(command);

  if (/\bgit\s+push\b/.test(c)) {
    return block(cfg, "git-push", {}, "Pushing to a remote is blocked during the loop.");
  }
  if (gitAddsHumanize(c)) {
    return block(cfg, "git-add-humanize", {}, "Do not git-add the .humanize loop-state directory; stage specific files instead.");
  }
  if (commandModifiesFile(c, "state\\.json")) {
    return block(cfg, "state-file-modification", {}, "Do not modify the loop state file via bash.");
  }
  if (commandModifiesFile(c, "round-\\d+-todos\\.md")) {
    return block(cfg, "todos-file-access", {},
      "Do not create round-*-todos.md files via bash; use the todowrite tool instead.");
  }
  if (commandModifiesFile(c, "round-\\d+-summary\\.md")) {
    return block(cfg, "summary-bash-write",
      { CORRECT_PATH: path.join(cfg.loopDir, `round-${cfg.currentRound}-summary.md`) },
      "Use the write/edit tool for summary files, not bash redirection.");
  }
  if (commandModifiesFile(c, "round-\\d+-contract\\.md")) {
    return block(cfg, "round-contract-bash-write",
      { CORRECT_PATH: path.join(cfg.loopDir, `round-${cfg.currentRound}-contract.md`) },
      "Use the write/edit tool for round contracts, not bash redirection.");
  }
  if (commandModifiesFile(c, "goal-tracker\\.md")) {
    return block(cfg, "goal-tracker-bash-write",
      { CORRECT_PATH: path.join(cfg.loopDir, "goal-tracker.md") },
      "Use the write/edit tool for goal-tracker.md, not bash redirection.");
  }
  // plan.md backup: only the copy living in the loop dir is protected.
  if (/\.humanize\b/.test(c) && commandModifiesFile(c, "plan\\.md")) {
    return block(cfg, "plan-backup-protected", {}, "The plan.md backup in the loop directory cannot be modified.");
  }
  // Shared-GPU serialization: GPU profilers must hold the per-device lock (run via
  // the gpu-run wrapper / flock) so two agents never profile the same card at once.
  // We can only *reliably* recognize the named GPU tools from a command string --
  // matched as a bare or path-prefixed binary, so "ncu-report-skill" (the CPU-only
  // report parser) is NOT caught -- and the prompt instructs gpu-run for plain
  // benchmark binaries too. Commands already wrapped in flock/gpu-run are allowed.
  const GPU_TOOL = /(?:^|[\s;&|(>])(?:[^\s;&|()]*\/)?(ncu|nv-nsight-cu-cli|nsys)(?:\s|$)/;
  if (GPU_TOOL.test(c) && !/\b(flock|gpu[_-]run)\b/.test(c)) {
    const gr = cfg.gpuRun || "gpu-run";
    return block(cfg, "gpu-unlocked", { GPU_RUN: gr },
      "Wrap GPU commands in the shared-GPU lock so only one agent uses the device " +
      `at a time, e.g.  ${gr} ncu --set full ...`);
  }
  return null;
}

// Pure decision function (exported for tests). Returns a GuardBlock to reject,
// or null to allow. Never throws for ordinary inputs.
export function decide(cfg, tool, args) {
  if (!cfg) return null;
  args = args || {};
  switch (tool) {
    case "write":
      return checkWriteLike(cfg, args.filePath, args.content);
    case "edit": {
      let resulting = null;
      if (lower(baseName(args.filePath || "")) === "goal-tracker.md") {
        resulting = applyEdit(absPath(cfg, args.filePath), args.oldString, args.newString, !!args.replaceAll);
      }
      return checkWriteLike(cfg, args.filePath, resulting);
    }
    case "patch":
      return checkWriteLike(cfg, args.filePath, null);
    case "read":
      return checkRead(cfg, args.filePath);
    case "bash":
      return checkBash(cfg, args.command);
    default:
      return null;
  }
}
