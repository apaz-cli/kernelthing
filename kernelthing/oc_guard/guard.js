// opencode plugin: kernelthing loop guard (PreToolUse enforcement).
//
// kernelthing's port of Humanize's hooks/loop-*-validator.sh layer. Humanize
// enforces loop-integrity conditions via Claude Code / Codex PreToolUse hooks
// that reject a tool call before it runs; kernelthing drives opencode headless,
// so we reimplement the same interception via opencode's `tool.execute.before`
// hook -- throwing aborts the tool and feeds the message back to the model
// (verified against opencode 1.16).
//
// Loop context (active loop dir, current round, real plan file, phase) arrives
// from the orchestrator through the KERNELTHING_GUARD env var (JSON); we do NOT
// scan the filesystem to discover the loop the way the shell hooks did, because
// kernelthing already knows it. If the env var is absent or malformed, the guard
// is a no-op (fail open) so it can never wedge a run.
//
// NOTE: opencode treats every exported function in this file as a plugin
// factory, so this module exports exactly ONE thing. The decision logic and its
// own exports live in guard_core.js, which opencode never loads directly.

import { decide, GuardBlock } from "./guard_core.js";

function loadConfig() {
  const raw = process.env.KERNELTHING_GUARD;
  if (!raw) return null;
  try {
    const c = JSON.parse(raw);
    if (!c || !c.loopDir || !c.projectRoot) return null;
    c.currentRound = Number(c.currentRound ?? 0);
    c.phase = c.phase || "impl";
    return c;
  } catch {
    return null;
  }
}

export const KernelthingGuard = async () => {
  const cfg = loadConfig();
  return {
    "tool.execute.before": async (input, output) => {
      if (!cfg) return; // fail open: no context => no enforcement
      let decision = null;
      try {
        decision = decide(cfg, input.tool, output && output.args);
      } catch {
        return; // internal guard error => fail open, never wedge the run
      }
      if (decision instanceof GuardBlock) {
        throw new Error(decision.message);
      }
    },
  };
};
