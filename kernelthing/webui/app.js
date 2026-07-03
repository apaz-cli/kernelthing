let UNIT = "", DIR = "maximize", MODE = "";
let selFile = null;   // transcript currently shown (auto-refreshed)
const $ = id => document.getElementById(id);
const fmt = v => v == null ? '-' : (Math.round(v * 10) / 10) + UNIT;
const opClass = op => 'op op-' + (['explore', 'exploit', 'seed'].includes(op) ? op : 'seed');
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));

// ---- wall-clock timer (server gives epoch start + live limit; client ticks) ----
let CLOCK = {start: null, elapsed: 0, limit: 0, running: false};

const hum = s => {
  s = Math.max(0, Math.round(s));
  const d = (s / 86400 | 0); s %= 86400;
  const h = (s / 3600 | 0); s %= 3600;
  const m = (s / 60 | 0), ss = s % 60;
  return d ? (h ? `${d}d${h}h` : `${d}d`) : h ? (m ? `${h}h${m}m` : `${h}h`) : m ? (ss ? `${m}m${ss}s` : `${m}m`) : ss + 's';
};

const fmtDur = s => {
  s = Math.max(0, Math.round(s));
  for (const [u, z] of [['w', 604800], ['d', 86400], ['h', 3600], ['m', 60]])
    if (s >= z && s % z === 0) return (s / z) + u;
  return s + 's';
};

function renderClock() {
  const ck = CLOCK;
  let el = ck.elapsed || 0;
  if (ck.running && ck.start) el = Math.floor(Date.now() / 1000 - ck.start);
  const over = ck.limit && el >= ck.limit;
  $('clock').innerHTML = '⏱ <b' + (over ? ' style=color:var(--bad)' : '') + '>' + hum(el) + '</b> / ' + (ck.limit ? hum(ck.limit) : '∞');
  if (document.activeElement.id !== 'W') $('W').value = ck.limit ? fmtDur(ck.limit) : '0';
}

function setW() { ctl({wall_clock: $('W').value}); }
function setM() { ctl({max_candidates: +$('M').value}); }

const memLog = m => `mem-${m.id}-${m.op}-opencode.log`;
const better = (a, b) => a != null && b != null && (DIR === 'maximize' ? a > b : a < b);

async function poll() {
  try {
    const s = await (await fetch('/api/status')).json();
    UNIT = s.unit || ""; DIR = s.direction || "maximize"; MODE = s.mode || "";
    $('problem').innerHTML = '<b>' + esc(s.problem || '-') + '</b>' + (UNIT ? ' · ' + UNIT : '');
    $('phase').textContent = 'phase ' + (s.phase || '-');
    $('best').innerHTML = 'best <b>' + fmt(s.best) + '</b>';
    const cap = (s.budget && s.budget.candidates) ? ('/' + s.budget.candidates) : '';
    $('kernels').innerHTML = '<b>' + (s.submitted ?? 0) + cap + '</b> kernels';
    const agents = s.agents || [];
    $('agentsN').innerHTML = '<b>' + agents.length + '</b> agents';
    const c = s.control || {};
    $('ctl').textContent = c.stop ? 'stop requested — finishing in-flight work…' : '';
    CLOCK = Object.assign({}, CLOCK, s.clock || {});
    if (c.wall_clock_s != null) CLOCK.limit = c.wall_clock_s;
    renderClock();
    if (document.activeElement.id !== 'M') {
      const mc = c.max_candidates != null ? c.max_candidates : 0;
      $('M').value = mc || '';
    }
    // explore bias slider — read live value from server if not currently focused
    if (document.activeElement.id !== 'E') {
      const eb = c.explore_bias != null ? c.explore_bias : 50;
      $('E').value = eb; $('Eb').textContent = eb;
      if (c.explore_auto) { $('Eauto').classList.add('active'); $('E').disabled = true; }
      else { $('Eauto').classList.remove('active'); $('E').disabled = false; }
    }

    if (MODE === 'evolve') {
      $('sbCard').classList.add('hidden');
      $('agentsCard').classList.remove('hidden');
      $('lineageCard').classList.remove('hidden');
      $('lbCard').classList.remove('hidden');
      const mem = s.members || [];
      drawChart(mem); drawAgents(agents); drawLineage(mem); drawLeaderboard(mem);
    } else {
      $('sbCard').classList.remove('hidden');
      $('agentsCard').classList.add('hidden');
      $('lineageCard').classList.add('hidden');
      $('lbCard').classList.add('hidden');
      drawScoreboard(s.scoreboard || []);
      drawChart((s.history || []).map(h => ({id: h.round, metric: h.metric, correct: true, op: 'seed'})));
    }
    await refreshLog(); await refreshTx();
  } catch(e) { /* transient; next tick retries */ }
  finally { setTimeout(poll, 1500); }
}

/* ---------- fitness chart: best-so-far staircase + every attempt ---------- */
function drawChart(mem) {
  const svg = $('chart'), H = 260, pl = 46, pr = 12, pt = 12, pb = 22;
  if (!svg) return;
  const W = Math.max(320, Math.round(svg.clientWidth || svg.getBoundingClientRect().width || 1000));
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const pts = mem.filter(m => m.metric != null);
  if (!pts.length) {
    svg.innerHTML = '<text x=12 y=24 fill=#6b7785 font-size=13>no kernels scored yet</text>';
    $('chartSub').textContent = '';
    return;
  }
  const xs = pts.map(m => m.id);
  let ys = pts.filter(m => m.correct).map(m => m.metric);
  if (!ys.length) ys = pts.map(m => m.metric);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = ys.length ? Math.min(...ys) : 0, y1 = ys.length ? Math.max(...ys) : 100;
  if (y1 === y0) { y1 += 1; y0 -= 1; }
  const padY = (y1 - y0) * 0.08; y0 -= padY; y1 += padY;
  const X = v => pl + (W - pl - pr) * (v - x0) / (x1 - x0 || 1);
  const Y = v => H - pb - (H - pt - pb) * (v - y0) / ((y1 - y0) || 1);
  let g = '';
  // grid + y labels
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + (y1 - y0) * i / 4, yy = Y(yv);
    g += `<line x1=${pl} y1=${yy} x2=${W - pr} y2=${yy} stroke=#21262d />` +
         `<text x=${pl - 6} y=${yy + 4} fill=#6b7785 font-size=11 text-anchor=end>${yv.toFixed(0)}</text>`;
  }
  // best-so-far staircase (correct points only)
  const sorted = pts.slice().sort((a, b) => a.id - b.id);
  let run = null, step = [];
  for (const m of sorted) {
    if (m.correct && (run == null || better(m.metric, run))) {
      if (run != null) step.push([X(m.id), Y(run)]);
      run = m.metric;
      step.push([X(m.id), Y(run)]);
    }
  }
  if (run != null) step.push([X(x1), Y(run)]);
  if (step.length) g += `<polyline fill=none stroke=#7ee787 stroke-width=2 points="${step.map(p => p[0] + ',' + p[1]).join(' ')}" />`;
  // every attempt as a dot
  const col = {explore: '#79c0ff', exploit: '#7ee787', seed: '#9aa5b1'};
  for (const m of pts) {
    const x = X(m.id), y = Y(m.metric);
    if (!m.correct) {
      g += `<path d="M${x - 3} ${y - 3}L${x + 3} ${y + 3}M${x - 3} ${y + 3}L${x + 3} ${y - 3}" stroke=#f85149 stroke-width=1.5 />`;
      continue;
    }
    if (m.best) {
      g += `<path d="M${x} ${y - 6}L${x + 6} ${y}L${x} ${y + 6}L${x - 6} ${y}Z" fill=#d2a8ff stroke=#fff stroke-width=1 />`;
    } else {
      g += `<circle cx=${x} cy=${y} r=3 fill=${col[m.op] || '#9aa5b1'} />`;
    }
  }
  g += `<text x=${(pl + W - pr) / 2} y=${H - 4} fill=#6b7785 font-size=11 text-anchor=middle>kernels submitted →</text>`;
  svg.innerHTML = g;
  const ok = pts.filter(m => m.correct).length;
  $('chartSub').textContent = `${pts.length} scored · ${ok} correct · ${pts.length - ok} failed`;
}

/* ---------- live agents (each enriched server-side from its NDJSON log) ---------- */
function drawAgents(agents) {
  const d = $('agents');
  if (!agents.length) { d.innerHTML = '<div class=muted>no agents in flight</div>'; return; }
  d.innerHTML = agents.map(a => {
    const f = memLog(a); const par = a.parent != null ? `← mem ${a.parent}` : 'fork base';
    const cost = a.cost ? (' · $' + a.cost.toFixed(3)) : '';
    const tool = a.last_tool ? `<div class="ln tool">▶ ${esc(a.last_tool)}</div>` : '';
    const text = a.last_text ? `<div class=ln>💬 ${esc(a.last_text)}</div>` : '';
    return `<div class="agent${selFile === f ? ' sel' : ''}" onclick="loadTx('${f}',this)">
      <div class=top><span class="${opClass(a.op)}">${a.op}</span>
        <span class=id>mem ${a.id}</span><span class=meta>${par}</span></div>
      <div class=meta>⚙ ${a.tools || 0} tools${cost}</div>${tool}${text}</div>`;
  }).join('');
}

/* ---------- lineage tree from parent links ---------- */
function drawLineage(mem) {
  const by = {}, kids = {};
  for (const m of mem) { by[m.id] = m; (kids[m.parent] = kids[m.parent] || []).push(m); }
  const col = {explore: '#79c0ff', exploit: '#7ee787', seed: '#9aa5b1'};
  const roots = mem.filter(m => m.parent == null || !(m.parent in by)).sort((a, b) => a.id - b.id);
  let out = '';
  const walk = (m, depth) => {
    const c = m.correct ? (col[m.op] || '#9aa5b1') : '#f85149';
    const val = m.metric != null ? fmt(m.metric) : (m.error ? '✗' : '…');
    out += `<div class=node style="padding-left:${depth * 14}px;cursor:pointer" onclick="loadTx('${memLog(m)}',null)">
      <span class=dot style=background:${c}></span><span class="${opClass(m.op)}">${m.op}</span>
      <span>mem ${m.id}</span><span class="${m.correct ? 'good' : 'muted'}">${val}</span>
      ${m.best ? '<span style=color:var(--elite)>◆</span>' : ''}</div>`;
    (kids[m.id] || []).sort((a, b) => a.id - b.id).forEach(k => walk(k, depth + 1));
  };
  roots.forEach(r => walk(r, 0));
  $('lineage').innerHTML = out || '<div class=muted>no lineage yet</div>';
}

/* ---------- leaderboard ---------- */
function drawLeaderboard(mem) {
  const v = mem.filter(m => m.correct && m.metric != null).sort((a, b) => better(a.metric, b.metric) ? -1 : 1).slice(0, 10);
  if (!v.length) { $('lb').innerHTML = '<tr><td colspan=7 class=muted>no viable kernels yet</td></tr>'; return; }
  $('lb').innerHTML = v.map((m, i) => {
    return `<tr onclick="loadTx('${memLog(m)}',null)" style=cursor:pointer>
      <td>${m.best ? '◆' : (i + 1)}</td><td>mem ${m.id}</td><td class="num good">${fmt(m.metric)}</td>
      <td><span class="${opClass(m.op)}">${m.op}</span></td><td>${esc(m.message || '')}</td>
      <td class=muted>${m.parent != null ? 'mem ' + m.parent : '—'}</td><td class=muted>${esc(m.commit)}</td></tr>`;
  }).join('');
}

/* ---------- legacy scoreboard (tournament / sequential) ---------- */
function drawScoreboard(raw) {
  const rows = raw.slice().sort((a, b) => (b.metric ?? -1e9) - (a.metric ?? -1e9));
  const done = rows.filter(r => r.correct && r.metric != null && !r.running).map(r => r.metric);
  const bestM = done.length ? Math.max(...done) : null;
  $('sb').innerHTML = rows.length ? rows.map(r => {
    const isHead = r.index === -1;
    const cls = isHead ? 'muted' : (r.running ? '' : (r.correct && r.metric === bestM ? 'good' : (r.correct ? '' : 'bad')));
    const st = r.running ? '<span style=color:#e3b341>running…</span>' : (r.correct ? '✓' : '✗');
    const f = `mem-${r.index}-${r.op || 'impl'}-opencode.log`;
    return `<tr onclick="loadTx('${f}',null)" style=cursor:pointer><td class="${cls}">${isHead ? 'HEAD' : ('#' + r.index)}</td>
      <td>${st}</td><td class="num ${cls}">${r.running ? '-' : fmt(r.metric)}</td>
      <td class=muted>${esc(r.commit || '')}</td><td class=bad>${r.running ? '' : esc(r.error || '')}</td></tr>`;
  }).join('')
  : '<tr><td colspan=5 class=muted>no candidates</td></tr>';
}

/* ---------- transcript + controller log ---------- */
function loadTx(f, el) {
  selFile = f;
  $('txWho').textContent = f;
  for (const a of document.querySelectorAll('.agent')) a.classList.remove('sel');
  if (el) el.classList.add('sel');
  refreshTx(true);
}

async function refreshTx(jump) {
  if (!selFile) return;
  try {
    const t = await (await fetch('/api/candlog?file=' + encodeURIComponent(selFile))).text();
    const p = $('tx');
    const atBottom = p.scrollHeight - p.scrollTop - p.clientHeight < 60;
    if (p.textContent !== t) { p.textContent = t; if (jump || atBottom) p.scrollTop = p.scrollHeight; }
  } catch(e) {}
}

async function refreshLog() {
  try {
    const lt = await (await fetch('/api/log')).text();
    const lg = $('log');
    const atBottom = lg.scrollHeight - lg.scrollTop - lg.clientHeight < 60;
    if (lg.textContent !== lt) { lg.textContent = lt; if (atBottom) lg.scrollTop = lg.scrollHeight; }
  } catch(e) {}
}

async function ctl(b) {
  await fetch('/api/control', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(b)});
}

function stop() {
  if (confirm('Stop the search? (in-flight work finishes, best kernel is kept)')) ctl({stop: true});
}

function setE() {
  ctl({explore_bias: +$('E').value});
  $('Eauto').classList.remove('active');
  $('E').disabled = false;
  $('Eb').textContent = $('E').value;
}

function autoE() { ctl({explore_auto: true}); }

$('E').addEventListener('input', () => { $('Eb').textContent = $('E').value; });
$('E').addEventListener('change', setE);
setInterval(renderClock, 1000);
poll();
