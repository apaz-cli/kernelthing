// kernelthing web UI: a pure reader of run journals.
//
// The server exposes run dirs; each run's UI state is a fold over its
// events.ndjson (fetched incrementally by byte offset). The same reducer
// renders a finished run (one big fetch, no polling) and a live one (poll,
// apply, redraw). The only write path is POST /api/control on a live run.

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
const opClass = op => 'op op-' + (['explore', 'exploit', 'seed'].includes(op) ? op : 'seed');

let RUN = null;        // selected run id
let LIVE = false;      // is the selected run live
let OFFSET = 0;        // journal byte offset consumed so far
let S = null;          // reduced state (fold over events)
let AGENTS = {};       // live enrichment per in-flight member (from /api/agents)
let selMember = null;  // member shown in the detail pane
let selTab = 'transcript';
let pollTimer = null;

const fmt = v => v == null ? '-' : (Math.round(v * 10) / 10) + (S ? S.unit : '');
const better = (a, b) => a != null && b != null && (S.direction === 'maximize' ? a > b : a < b);

/* ---------- the reducer: UI state = fold(applyEvent, events) ---------- */
function initState() {
  return {
    problem: '', unit: '', direction: 'maximize', model: '',
    control: {parallelism: 0, elite_k: 4, wall_clock_s: 0, max_candidates: 0,
              explore_bias: 50, explore_auto: true, stop: false},
    phase: '—', baselines: {},
    members: new Map(), inflight: new Map(),
    best: null, dispatched: 0, cost: 0,
    startT: null, searchStart: null, end: null, lastT: null,
  };
}

function applyEvent(s, e) {
  switch (e.type) {
    case 'run_start':
      s.problem = e.problem.name; s.unit = e.problem.unit || '';
      s.direction = e.problem.direction || 'maximize'; s.model = e.model;
      Object.assign(s.control, {
        parallelism: e.config.parallelism, elite_k: e.config.elite_k,
        wall_clock_s: e.config.wall_clock_s, max_candidates: e.config.max_candidates,
      });
      s.startT = e.t; s.phase = 'starting';
      break;
    case 'phase': s.phase = e.phase; break;
    case 'search_start': s.searchStart = e.t; break;
    case 'baseline_pinned': s.baselines[e.gpu] = e.median_us; break;
    case 'control_changed': Object.assign(s.control, e.changes); break;
    case 'dispatch':
      s.inflight.set(e.member, {id: e.member, op: e.op, parent: e.parent, gpu: e.gpu, t: e.t});
      s.dispatched = e.dispatched;
      break;
    case 'member_result':
      s.inflight.delete(e.member);
      s.members.set(e.member, {
        id: e.member, op: e.op, parent: e.parent, metric: e.metric, correct: e.correct,
        message: e.message || '', commit: (e.commit || '').slice(0, 8), error: e.error,
        cost: e.cost || 0, gpu: e.gpu, agent_s: e.agent_s, score_s: e.score_s,
      });
      s.cost += e.cost || 0;
      break;
    case 'new_best': s.best = e.metric; break;
    case 'run_end': s.end = e; s.phase = 'done (' + e.reason + ')'; break;
  }
  s.lastT = e.t;
}

/* ---------- run selection ---------- */
async function loadRuns(autoselect) {
  let runs = [];
  try { runs = await (await fetch('/api/runs')).json(); } catch (e) { return; }
  const sel = $('runSel'), cur = sel.value;
  sel.innerHTML = runs.map(r => {
    const name = (r.run.problem && r.run.problem.name) || r.id;
    const label = `${r.live ? '● ' : ''}${name} · ${r.run.timestamp}`;
    return `<option value="${esc(r.id)}">${esc(label)}</option>`;
  }).join('') || '<option value="">(no runs found)</option>';
  const ids = runs.map(r => r.id);
  if (cur && ids.includes(cur)) sel.value = cur;
  if (autoselect && runs.length) {
    const live = runs.find(r => r.live);
    selectRun(live ? live.id : runs[0].id);
    sel.value = RUN;
  }
  // liveness of the selected run can flip without new events
  const mine = runs.find(r => r.id === RUN);
  if (mine) LIVE = mine.live;
}

function selectRun(id) {
  if (!id || id === RUN) return;
  RUN = id; OFFSET = 0; S = initState(); AGENTS = {}; selMember = null; LIVE = false;
  dirty.clear();
  $('tx').textContent = '—'; delete $('tx').dataset.raw;
  $('txWho').textContent = 'select an agent or member';
  clearTimeout(pollTimer);
  poll();
}

const api = path => path + (path.includes('?') ? '&' : '?') + 'run=' + encodeURIComponent(RUN);

/* ---------- polling ---------- */
async function poll() {
  if (!RUN) return;
  try {
    const r = await (await fetch(api(`/api/events?offset=${OFFSET}`))).json();
    LIVE = r.live;
    for (const e of r.events) applyEvent(S, e);
    OFFSET = r.offset;
    if (LIVE && S.inflight.size) {
      const ids = [...S.inflight.keys()].join(',');
      AGENTS = await (await fetch(api(`/api/agents?ids=${ids}`))).json();
    } else if (!LIVE) {
      AGENTS = {};
    }
    render();
    await refreshLog(); await refreshTx();
  } catch (e) { /* transient; next tick retries */ }
  finally { pollTimer = setTimeout(poll, LIVE ? 1500 : 8000); }
}

/* ---------- render ---------- */
function render() {
  if (!S) return;
  $('problem').innerHTML = '<b>' + esc(S.problem || '-') + '</b>' + (S.unit ? ' · ' + S.unit : '');
  $('phase').textContent = 'phase ' + (S.phase || '-');
  $('best').innerHTML = 'best <b>' + fmt(S.best) + '</b>';
  const cap = S.control.max_candidates ? ('/' + S.control.max_candidates) : '';
  $('kernels').innerHTML = '<b>' + S.dispatched + cap + '</b> kernels';
  $('agentsN').innerHTML = '<b>' + S.inflight.size +
    (LIVE ? '/' + S.control.parallelism : '') + '</b> agents';
  $('cost').innerHTML = '$<b>' + S.cost.toFixed(2) + '</b>';
  $('ctl').textContent = LIVE && S.control.stop ? 'stop requested — finishing in-flight work…' : '';
  $('liveCtls').classList.toggle('hidden', !LIVE);
  renderClock();
  if (LIVE) {
    syncKnobs();
    if (document.activeElement.id !== 'E') {
      $('E').value = S.control.explore_bias; $('Eb').textContent = S.control.explore_bias;
      $('Eauto').classList.toggle('active', S.control.explore_auto);
      $('E').disabled = S.control.explore_auto;
    }
  }
  const mem = [...S.members.values()].sort((a, b) => a.id - b.id);
  drawChart(mem); drawAgents(); drawLineage(mem); drawLeaderboard(mem);
}

/* ---------- wall clock ---------- */
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
  if (!S) return;
  let el = 0;
  if (S.searchStart != null) {
    const end = S.end ? S.end.t : (LIVE ? Date.now() / 1000 : S.lastT);
    el = Math.max(0, Math.floor((end || S.searchStart) - S.searchStart));
  }
  const limit = S.control.wall_clock_s || 0;
  const over = limit && el >= limit;
  $('clock').innerHTML = '⏱ <b' + (over ? ' style=color:var(--bad)' : '') + '>' + hum(el) + '</b> / ' + (limit ? hum(limit) : '∞');
}

/* ---------- fitness chart: best-so-far staircase + every attempt ---------- */
function bestId(mem) {
  let b = null;
  for (const m of mem) if (m.correct && m.metric != null && (!b || better(m.metric, b.metric))) b = m;
  return b ? b.id : null;
}

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
  const bid = bestId(mem);
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
  let run = null, step = [];
  for (const m of pts) {
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
    if (m.id === bid) {
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

/* ---------- live agents (enriched from their NDJSON transcripts) ---------- */
function drawAgents() {
  const d = $('agents');
  const inflight = [...S.inflight.values()].sort((a, b) => a.id - b.id);
  if (!inflight.length) {
    d.innerHTML = `<div class=muted>${LIVE ? 'no agents in flight' : 'run is not live'}</div>`;
    return;
  }
  d.innerHTML = inflight.map(a => {
    const en = AGENTS[a.id] || {};
    const par = a.parent != null ? `← mem ${a.parent}` : 'fork base';
    const cost = en.cost ? (' · $' + en.cost.toFixed(3)) : '';
    const tool = en.last_tool ? `<div class="ln tool">▶ ${esc(en.last_tool)}</div>` : '';
    const text = en.last_text ? `<div class=ln>💬 ${esc(en.last_text)}</div>` : '';
    return `<div class="agent${selMember === a.id ? ' sel' : ''}" onclick="loadMember(${a.id})">
      <div class=top><span class="${opClass(a.op)}">${a.op}</span>
        <span class=id>mem ${a.id}</span><span class=meta>${par} · GPU ${a.gpu}</span></div>
      <div class=meta>⚙ ${en.tools || 0} tools${cost}</div>${tool}${text}</div>`;
  }).join('');
}

/* ---------- lineage tree from parent links ---------- */
function drawLineage(mem) {
  const by = {}, kids = {};
  for (const m of mem) { by[m.id] = m; (kids[m.parent] = kids[m.parent] || []).push(m); }
  const col = {explore: '#79c0ff', exploit: '#7ee787', seed: '#9aa5b1'};
  const bid = bestId(mem);
  const roots = mem.filter(m => m.parent == null || !(m.parent in by)).sort((a, b) => a.id - b.id);
  let out = '';
  const walk = (m, depth) => {
    const c = m.correct ? (col[m.op] || '#9aa5b1') : '#f85149';
    const val = m.metric != null ? fmt(m.metric) : (m.error ? '✗' : '…');
    out += `<div class=node style="padding-left:${depth * 14}px;cursor:pointer" onclick="loadMember(${m.id})">
      <span class=dot style=background:${c}></span><span class="${opClass(m.op)}">${m.op}</span>
      <span>mem ${m.id}</span><span class="${m.correct ? 'good' : 'muted'}">${val}</span>
      ${m.id === bid ? '<span style=color:var(--elite)>◆</span>' : ''}</div>`;
    (kids[m.id] || []).sort((a, b) => a.id - b.id).forEach(k => walk(k, depth + 1));
  };
  roots.forEach(r => walk(r, 0));
  $('lineage').innerHTML = out || '<div class=muted>no lineage yet</div>';
}

/* ---------- leaderboard ---------- */
function drawLeaderboard(mem) {
  const bid = bestId(mem);
  const v = mem.filter(m => m.correct && m.metric != null)
    .sort((a, b) => better(a.metric, b.metric) ? -1 : 1).slice(0, 10);
  if (!v.length) { $('lb').innerHTML = '<tr><td colspan=8 class=muted>no viable kernels yet</td></tr>'; return; }
  $('lb').innerHTML = v.map((m, i) => {
    return `<tr onclick="loadMember(${m.id})" style=cursor:pointer>
      <td>${m.id === bid ? '◆' : (i + 1)}</td><td>mem ${m.id}</td><td class="num good">${fmt(m.metric)}</td>
      <td><span class="${opClass(m.op)}">${m.op}</span></td><td>${esc(m.message)}</td>
      <td class=muted>${m.parent != null ? 'mem ' + m.parent : '—'}</td><td class=muted>${esc(m.commit)}</td>
      <td class="num muted">${m.cost ? m.cost.toFixed(2) : '-'}</td></tr>`;
  }).join('');
}

/* ---------- member detail pane (transcript / prompt / diff / summary) ---------- */
function loadMember(id) {
  selMember = id;
  $('txWho').textContent = 'mem ' + id;
  for (const b of document.querySelectorAll('#txTabs .mini'))
    b.classList.toggle('active', b.dataset.tab === selTab);
  refreshTx(true);
  render();
}

function setTab(tab) {
  selTab = tab;
  for (const b of document.querySelectorAll('#txTabs .mini'))
    b.classList.toggle('active', b.dataset.tab === tab);
  refreshTx(true);
}

// Unified-diff coloring: one span per line, keyed on the line prefix. File
// headers before hunk markers so '+++'/'---' aren't mistaken for add/delete.
function renderDiff(t) {
  return t.split('\n').map(ln => {
    const e = esc(ln);
    if (/^(diff |index |new file|deleted file|\+\+\+|---)/.test(ln)) return `<span class=d-meta>${e}</span>`;
    if (ln.startsWith('@@')) return `<span class=d-hunk>${e}</span>`;
    if (ln.startsWith('+')) return `<span class=d-add>${e}</span>`;
    if (ln.startsWith('-')) return `<span class=d-del>${e}</span>`;
    return e;
  }).join('\n');
}

// The whole agent context, opencode-style: assistant prose inline, thinking and
// tool input/output as <details> collapsed by default. The item list is
// append-only while the agent runs, so indices are stable and open state
// survives a live re-render (restored by data-i).
function renderTranscript(items) {
  return items.map((it, i) => {
    if (it.kind === 'text') return `<div class=say>${esc(it.text)}</div>`;
    if (it.kind === 'think')
      return `<details class=think data-i=${i}><summary>thinking</summary><div class=say>${esc(it.text)}</div></details>`;
    if (it.kind === 'tool') {
      const st = it.status && it.status !== 'completed' ? ` · ${esc(it.status)}` : '';
      const body = (it.input ? `<pre class=io>${esc(it.input)}</pre>` : '') +
        (it.output ? `<pre class=io>${esc(it.output)}</pre>` : '<div class="say muted">(no output)</div>');
      return `<details class=toolcall data-i=${i}><summary>${esc(it.line)}${st}</summary>${body}</details>`;
    }
    return '';
  }).join('');
}

async function refreshTx(jump) {
  if (selMember == null) return;
  try {
    const t = await (await fetch(api(`/api/member?id=${selMember}&file=${selTab}`))).text();
    const p = $('tx');
    const atBottom = p.scrollHeight - p.scrollTop - p.clientHeight < 60;
    if (p.dataset.raw !== t) {
      p.dataset.raw = t;
      if (selTab === 'diff') p.innerHTML = renderDiff(t);
      else if (selTab === 'transcript') {
        const open = new Set([...p.querySelectorAll('details[open]')].map(d => d.dataset.i));
        let items = [];
        try { items = JSON.parse(t); } catch (e) {}
        p.innerHTML = renderTranscript(items) || '(no transcript yet)';
        for (const d of p.querySelectorAll('details')) if (open.has(d.dataset.i)) d.open = true;
      } else p.textContent = t;
      if (jump || atBottom) p.scrollTop = p.scrollHeight;
    }
    if (jump && selTab !== 'transcript') p.scrollTop = 0;
  } catch (e) {}
}

async function refreshLog() {
  try {
    const lt = await (await fetch(api('/api/log'))).text();
    const lg = $('log');
    const atBottom = lg.scrollHeight - lg.scrollTop - lg.clientHeight < 60;
    if (lg.textContent !== lt) { lg.textContent = lt; if (atBottom) lg.scrollTop = lg.scrollHeight; }
  } catch (e) {}
}

/* ---------- live controls ----------
 * Knob protocol (-j -k -m -w): editing marks the input dirty (amber) and the
 * poll loop stops syncing it, so a refresh can never clobber what you typed.
 * Enter or the apply button POSTs every dirty knob at once; Escape reverts one.
 * The POST response carries the authoritative (clamped) control.json, which is
 * folded into S immediately -- the loop picks it up at its next dispatch
 * boundary and journals a control_changed event when it does. */

const KNOBS = {
  J: {show: c => c.parallelism || '', body: v => ({parallelism: +v || 1})},
  K: {show: c => c.elite_k || '', body: v => ({elite_k: +v || 1})},
  M: {show: c => c.max_candidates || '0', body: v => ({max_candidates: +v || 0})},
  W: {show: c => c.wall_clock_s ? fmtDur(c.wall_clock_s) : '0',
      body: v => ({wall_clock: v})},  // human form ("10m"); the server parses it
};
const dirty = new Set();

function syncKnobs() {
  for (const id of Object.keys(KNOBS)) {
    const el = $(id);
    if (!dirty.has(id) && document.activeElement !== el) el.value = KNOBS[id].show(S.control);
  }
  $('applyBtn').classList.toggle('dirty', dirty.size > 0);
}

async function applyKnobs() {
  const body = {};
  for (const id of dirty) Object.assign(body, KNOBS[id].body($(id).value.trim()));
  if (!Object.keys(body).length) return;
  for (const id of dirty) $(id).classList.remove('dirty');
  dirty.clear();
  await ctl(body);
}

function revertKnob(id) {
  dirty.delete(id);
  $(id).classList.remove('dirty');
  syncKnobs();
}

for (const id of Object.keys(KNOBS)) {
  $(id).addEventListener('input', () => { dirty.add(id); $(id).classList.add('dirty'); $('applyBtn').classList.add('dirty'); });
  $(id).addEventListener('keydown', e => {
    if (e.key === 'Enter') { applyKnobs(); $(id).blur(); }
    else if (e.key === 'Escape') { revertKnob(id); $(id).blur(); }
  });
}

async function ctl(b) {
  try {
    const r = await fetch(api('/api/control'), {
      method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(b),
    });
    const d = await r.json();
    // The response is the authoritative clamped control.json: fold it in now so
    // the UI shows the real committed values instead of snapping back to stale
    // state until the loop's next control_changed event.
    if (r.ok && d && d.control) { Object.assign(S.control, d.control); render(); }
  } catch (e) { /* transient; state re-syncs on the next poll */ }
}

function stop() {
  if (confirm('Stop the search? (in-flight work finishes, best kernel is kept)')) ctl({stop: true});
}

function setE() {
  ctl({explore_bias: +$('E').value, explore_auto: false});
  $('Eauto').classList.remove('active');
  $('E').disabled = false;
  $('Eb').textContent = $('E').value;
}

function autoE() { ctl({explore_auto: true}); }

$('E').addEventListener('input', () => { $('Eb').textContent = $('E').value; });
$('E').addEventListener('change', setE);
setInterval(renderClock, 1000);
setInterval(() => loadRuns(false), 5000);
loadRuns(true);
