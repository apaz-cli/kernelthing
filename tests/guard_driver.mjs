// Test driver: reads a JSON array of {cfg, tool, args} cases on stdin and
// writes a JSON array of {blocked, message} by calling the guard's decide().
import { decide } from "../kernelthing/oc_guard/guard_core.js";

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const cases = JSON.parse(input);
  const out = cases.map((c) => {
    const d = decide(c.cfg, c.tool, c.args);
    return { blocked: !!d, message: d ? d.message : null };
  });
  process.stdout.write(JSON.stringify(out));
});
