"""Embedded stdlib web UI for watching/controlling a running loop.

Zero dependencies: a ThreadingHTTPServer in a daemon thread serves a single HTML
page that polls ``/api/status`` and renders the live evolutionary search -- a
fitness chart (best vs. kernels submitted), the in-flight agents (each streaming
its exact tool calls), the MAP-Elites niches, the lineage tree, and a
leaderboard. Shares a :class:`LoopBus` with the loop.

The agents stream for free: every opencode turn writes its NDJSON event log to a
per-agent file *as it arrives*. The controller can't see inside a running turn
(it blocks on the subprocess), but this server can tail the file -- so
``/api/status`` enriches each in-flight agent with the live tool/text summary of
its log, and ``/api/candlog`` returns the full readable transcript on demand.

Endpoints:
  * ``GET  /``            -- the page
  * ``GET  /api/status``  -- the snapshot, agents enriched with live log tails
  * ``GET  /api/log``     -- the controller narrative (loop.log)
  * ``GET  /api/candlog?file=`` -- one agent's full transcript (basename-restricted)
  * ``POST /api/control`` -- stop / set parallelism / set turn cap
"""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bus import LoopBus

PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>kernelthing</title><style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#30363d;--line2:#21262d;--ink:#d6deeb;--dim:#9aa5b1;--mut:#6b7785;
       --explore:#79c0ff;--exploit:#7ee787;--seed:#9aa5b1;--bad:#f85149;--elite:#d2a8ff}
 *{box-sizing:border-box}
 body{font:14px/1.45 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
 header{padding:9px 16px;background:var(--panel);border-bottom:1px solid var(--line);
        display:flex;gap:8px 16px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:5}
 h1{font-size:15px;margin:0;color:var(--exploit);letter-spacing:.02em}
 .pill{background:var(--line2);border:1px solid var(--line);border-radius:12px;padding:2px 10px;white-space:nowrap}
 .pill b{color:#fff}
 .grow{flex:1}
 main{padding:14px;display:flex;flex-direction:column;gap:14px}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
 .card h2{font-size:12px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;
          display:flex;justify-content:space-between;align-items:baseline}
 .card h2 .sub{text-transform:none;letter-spacing:0;color:var(--mut);font-weight:400}
 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line2);white-space:nowrap}
 th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 td.num{text-align:right}
 .good{color:var(--exploit)}.bad{color:var(--bad)}.muted{color:var(--mut)}
 input{width:64px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:4px}
 button{background:#238636;color:#fff;border:0;border-radius:4px;padding:6px 12px;cursor:pointer;font:inherit}
 button.stop{background:#da3633}
  button.mini{background:var(--line2);border:1px solid var(--line);color:var(--ink);padding:3px 8px;font-size:12px}
  button.mini.active{background:var(--exploit);color:#0e1116;font-weight:600}
 pre{background:var(--bg);border:1px solid var(--line2);border-radius:6px;padding:8px;max-height:340px;
     overflow:auto;font-size:12px;margin:0;white-space:pre-wrap;word-break:break-word}
  svg{display:block;width:100%}
 /* operator badge */
 .op{display:inline-block;border-radius:4px;padding:0 6px;font-size:11px;font-weight:600;color:#0e1116}
  .op-explore{background:var(--explore)}.op-exploit{background:var(--exploit)}
  .op-seed{background:var(--seed)}
 /* agent cards */
 #agents{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
 .agent{border:1px solid var(--line);border-radius:7px;padding:9px;background:var(--bg);cursor:pointer;
        position:relative;overflow:hidden}
 .agent.sel{border-color:var(--explore)}
 .agent .top{display:flex;align-items:center;gap:8px;margin-bottom:5px}
 .agent .id{font-weight:600}
 .agent .meta{color:var(--mut);font-size:12px}
 .agent .ln{font:12px ui-monospace,Menlo,monospace;color:var(--dim);overflow:hidden;text-overflow:ellipsis;
            white-space:nowrap;margin-top:2px}
 .agent .ln.tool{color:var(--exploit)}
 /* niche bars */
 .niche{margin-bottom:7px}
 .niche .lbl{display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px}
 .niche .lbl .s{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%}
 .track{background:var(--line2);border-radius:3px;height:8px;overflow:hidden}
 .fill{height:100%;background:linear-gradient(90deg,#1f6feb,var(--exploit))}
 /* lineage */
 #lineageCard{display:flex;flex-direction:column}
 #lineage{font:12px ui-monospace,Menlo,monospace;flex:1;min-height:0;overflow:auto}
 .node{display:flex;align-items:center;gap:6px;padding:1px 0}
 .node .dot{width:8px;height:8px;border-radius:50%;flex:none}
 .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:8px}
 .legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle}
 .hidden{display:none}
</style></head><body>
<header>
 <h1>kernelthing</h1>
 <span class=pill id=problem>—</span>
 <span class=pill id=phase>phase —</span>
 <span class=pill id=best>best —</span>
 <span class=pill id=kernels>0 kernels</span>
 <span class=pill id=agentsN>0 agents</span>
 <span class=pill id=clock>⏱ —</span>
 <span class=grow></span>
  <span id=ctl class=muted style=font-size:12px></span>
  <span id=exploreCtl>explore
    <input id=E type=range min=0 max=100 value=50 style=width:80px;vertical-align:middle
           title="explore/exploit bias — auto-schedule adjusts this over time">
    exploit <b id=Eb>50</b>
    <button class=mini id=Eauto onclick=autoE() title="re-enable auto-schedule">auto</button>
  </span>
  <span id=clockCtl>limit <input id=W type=text title="wall-clock limit, e.g. 10m, 2h, 1d — 0 = off">
    <button class=mini onclick=setW()>set</button></span>
  <span id=candCtl>cands <input id=M type=number min=0 value=0 style=width:50px>
    <button class=mini onclick=setM()>set</button></span>
  <button class=stop onclick=stop()>Stop</button>
</header>
<main>
 <div class=card>
   <h2>Best vs. kernels submitted <span class=sub id=chartSub></span></h2>
   <svg id=chart height=260></svg>
   <div class=legend>
      <span><i style=background:var(--explore)></i>explore</span>
      <span><i style=background:var(--exploit)></i>exploit</span>
      <span><i style=background:var(--seed)></i>seed</span>
     <span><i style=background:var(--bad)></i>incorrect</span>
     <span>◆ best&nbsp;&nbsp;━ best-so-far</span>
   </div>
 </div>

 <div class=card id=agentsCard>
   <h2>Agents — live <span class=sub>what each agent is doing right now · click to stream its transcript</span></h2>
   <div id=agents></div>
 </div>

 <div class=row>
   <div class=card id=nichesCard><h2>Niches <span class=sub>best kernel per commit message</span></h2>
     <div id=niches></div></div>
   <div class=card id=lineageCard><h2>Lineage <span class=sub>parent → child mutations</span></h2>
     <div id=lineage></div></div>
 </div>

 <div class=card id=lbCard><h2>Leaderboard</h2>
    <table><thead><tr><th>#</th><th>mem</th><th class=num>metric</th><th>op</th><th>message</th>
      <th>parent</th><th>commit</th></tr></thead><tbody id=lb></tbody></table></div>

 <div class=card id=sbCard class=hidden><h2>Scoreboard</h2>
   <table><thead><tr><th>cand</th><th>status</th><th class=num>metric</th><th>commit</th><th>note</th></tr></thead>
   <tbody id=sb></tbody></table>
   <div id=candbtns style=margin-top:8px></div></div>

 <div class=card><h2>Transcript <span class=sub id=txWho>select an agent</span></h2><pre id=tx>—</pre></div>
 <div class=card><h2>Controller log</h2><pre id=log>—</pre></div>
</main>
<script>
let UNIT="",DIR="maximize",MODE="";
let selFile=null;               // transcript currently shown (auto-refreshed)
const $=id=>document.getElementById(id);
const fmt=v=>v==null?'-':(Math.round(v*10)/10)+UNIT;
const opClass=op=>'op op-'+(['explore','exploit','seed'].includes(op)?op:'seed');
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// ---- wall-clock timer (server gives epoch start + live limit; client ticks) ----
let CLOCK={start:null,elapsed:0,limit:0,running:false};
const hum=s=>{s=Math.max(0,Math.round(s));const d=(s/86400|0);s%=86400;const h=(s/3600|0);s%=3600;const m=(s/60|0),ss=s%60;
 return d?(h?`${d}d${h}h`:`${d}d`):h?(m?`${h}h${m}m`:`${h}h`):m?(ss?`${m}m${ss}s`:`${m}m`):ss+'s';};
const fmtDur=s=>{s=Math.max(0,Math.round(s));for(const[u,z] of [['w',604800],['d',86400],['h',3600],['m',60]])if(s>=z&&s%z===0)return (s/z)+u;return s+'s';};
function renderClock(){const ck=CLOCK;let el=ck.elapsed||0;if(ck.running&&ck.start)el=Math.floor(Date.now()/1000-ck.start);
 const over=ck.limit&&el>=ck.limit;
 $('clock').innerHTML='⏱ <b'+(over?' style=color:var(--bad)':'')+'>'+hum(el)+'</b> / '+(ck.limit?hum(ck.limit):'∞');
 if(document.activeElement.id!=='W')$('W').value=ck.limit?fmtDur(ck.limit):'0';}
function setW(){ctl({wall_clock:$('W').value});}
function setM(){ctl({max_candidates:+$('M').value});}
const memLog=m=>`mem-${m.id}-${m.op}-opencode.log`;
const better=(a,b)=>DIR==='maximize'?a>b:a<b;   // a strictly better than b

async function poll(){
 try{
  const s=await (await fetch('/api/status')).json();
  UNIT=s.unit||"";DIR=s.direction||"maximize";MODE=s.mode||"";
  $('problem').innerHTML='<b>'+esc(s.problem||'-')+'</b>'+(UNIT?' · '+UNIT:'');
  $('phase').textContent='phase '+(s.phase||'-');
  $('best').innerHTML='best <b>'+fmt(s.best)+'</b>';
  const cap=(s.budget&&s.budget.candidates)?('/'+s.budget.candidates):'';
  $('kernels').innerHTML='<b>'+(s.submitted??0)+cap+'</b> kernels';
  const agents=s.agents||[];
  $('agentsN').innerHTML='<b>'+agents.length+'</b> agents';
  const c=s.control||{};
  $('ctl').textContent=c.stop?'stop requested — finishing in-flight work…':'';
  CLOCK=Object.assign({},CLOCK,s.clock||{});
  if(c.wall_clock_s!=null)CLOCK.limit=c.wall_clock_s;   // live bus value wins
  renderClock();
  if(document.activeElement.id!=='M'){
    const mc=c.max_candidates!=null?c.max_candidates:0;
    $('M').value=mc||'';
  }
  // explore bias slider — read live value from server if not currently focused
  if(document.activeElement.id!=='E'){
    const eb=c.explore_bias!=null?c.explore_bias:50;
    $('E').value=eb;$('Eb').textContent=eb;
    if(c.explore_auto){$('Eauto').classList.add('active');$('E').disabled=true;}
    else{$('Eauto').classList.remove('active');$('E').disabled=false;}
  }

   if(MODE==='evolve'){
    $('sbCard').classList.add('hidden');
    $('agentsCard').classList.remove('hidden');
   $('nichesCard').classList.remove('hidden');$('lineageCard').classList.remove('hidden');$('lbCard').classList.remove('hidden');
   const mem=s.members||[];
   drawChart(mem);drawAgents(agents);drawNiches(mem);drawLineage(mem);drawLeaderboard(mem);
   }else{
    // legacy tournament / sequential: scoreboard + history chart
    $('sbCard').classList.remove('hidden');
    $('agentsCard').classList.add('hidden');
   $('nichesCard').classList.add('hidden');$('lineageCard').classList.add('hidden');$('lbCard').classList.add('hidden');
   if(document.activeElement.id!=='N')$('N').value=c.parallelism??'';
   drawScoreboard(s.scoreboard||[]);
   drawChart((s.history||[]).map(h=>({id:h.round,metric:h.metric,correct:true,op:'seed'})));
  }
  await refreshLog();await refreshTx();
 }catch(e){/* transient; next tick retries */}
 finally{ setTimeout(poll,1500); }
}

/* ---------- fitness chart: best-so-far staircase + every attempt ---------- */
function drawChart(mem){
 const svg=$('chart'),H=260,pl=46,pr=12,pt=12,pb=22;
 if(!svg)return;
 // render at real pixel width (1 unit = 1px) so markers/strokes don't warp
 const W=Math.max(320,Math.round(svg.clientWidth||svg.getBoundingClientRect().width||1000));
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const pts=mem.filter(m=>m.metric!=null);
 if(!pts.length){svg.innerHTML='<text x=12 y=24 fill=#6b7785 font-size=13>no kernels scored yet</text>';
  $('chartSub').textContent='';return;}
 const xs=pts.map(m=>m.id);
 let ys=pts.filter(m=>m.correct).map(m=>m.metric);
 if(!ys.length)ys=pts.map(m=>m.metric);
 const x0=Math.min(...xs),x1=Math.max(...xs,x0+1);
 let y0=ys.length?Math.min(...ys):0,y1=ys.length?Math.max(...ys):100;if(y1===y0){y1+=1;y0-=1;}
 const padY=(y1-y0)*0.08;y0-=padY;y1+=padY;
 const X=v=>pl+(W-pl-pr)*(v-x0)/(x1-x0||1);
 const Y=v=>H-pb-(H-pt-pb)*(v-y0)/((y1-y0)||1);
 let g='';
 // grid + y labels
 for(let i=0;i<=4;i++){const yv=y0+(y1-y0)*i/4,yy=Y(yv);
  g+=`<line x1=${pl} y1=${yy} x2=${W-pr} y2=${yy} stroke=#21262d />`+
     `<text x=${pl-6} y=${yy+4} fill=#6b7785 font-size=11 text-anchor=end>${yv.toFixed(0)}</text>`;}
 // best-so-far staircase (correct points only)
 const sorted=pts.slice().sort((a,b)=>a.id-b.id);let run=null,step=[];
 for(const m of sorted){if(m.correct&&(run==null||better(m.metric,run))){
   if(run!=null)step.push([X(m.id),Y(run)]);run=m.metric;step.push([X(m.id),Y(run)]);}}
 if(run!=null)step.push([X(x1),Y(run)]);
 if(step.length)g+=`<polyline fill=none stroke=#7ee787 stroke-width=2 points="${step.map(p=>p[0]+','+p[1]).join(' ')}" />`;
 // every attempt as a dot
  const col={explore:'#79c0ff',exploit:'#7ee787',seed:'#9aa5b1'};
  for(const m of pts){const x=X(m.id),y=Y(m.metric);
   if(!m.correct){g+=`<path d="M${x-3} ${y-3}L${x+3} ${y+3}M${x-3} ${y+3}L${x+3} ${y-3}" stroke=#f85149 stroke-width=1.5 />`;continue;}
  if(m.best){g+=`<path d="M${x} ${y-6}L${x+6} ${y}L${x} ${y+6}L${x-6} ${y}Z" fill=#d2a8ff stroke=#fff stroke-width=1 />`;}
  else{g+=`<circle cx=${x} cy=${y} r=3 fill=${col[m.op]||'#9aa5b1'} />`;}}
 g+=`<text x=${(pl+W-pr)/2} y=${H-4} fill=#6b7785 font-size=11 text-anchor=middle>kernels submitted →</text>`;
 svg.innerHTML=g;
 const ok=pts.filter(m=>m.correct).length;
 $('chartSub').textContent=`${pts.length} scored · ${ok} correct · ${pts.length-ok} failed`;
}

/* ---------- live agents (each enriched server-side from its NDJSON log) ---------- */
function drawAgents(agents){
 const d=$('agents');
 if(!agents.length){d.innerHTML='<div class=muted>no agents in flight</div>';return;}
 d.innerHTML=agents.map(a=>{
  const f=memLog(a);const par=a.parent!=null?`← mem ${a.parent}`:'fork base';
  const cost=a.cost?(' · $'+a.cost.toFixed(3)):'';
  const tool=a.last_tool?`<div class="ln tool">▶ ${esc(a.last_tool)}</div>`:'';
  const text=a.last_text?`<div class=ln>💬 ${esc(a.last_text)}</div>`:'';
  return `<div class="agent${selFile===f?' sel':''}" onclick="loadTx('${f}',this)">
    <div class=top><span class="${opClass(a.op)}">${a.op}</span>
      <span class=id>mem ${a.id}</span><span class=meta>${par}</span></div>
    <div class=meta>⚙ ${a.tools||0} tools${cost}</div>${tool}${text}</div>`;}).join('');
}

/* ---------- Niches: best viable member per commit message ---------- */
function drawNiches(mem){
 const grid={};
 for(const m of mem){if(!m.correct||m.metric==null)continue;
  const k=m.message||'uncategorized';
  const g=grid[k];if(!g||better(m.metric,g.metric)){grid[k]={...m,count:(g?g.count:0)+1};}
  else{g.count++;}}
 const list=Object.values(grid).sort((a,b)=>better(a.metric,b.metric)?-1:1);
 if(!list.length){$('niches').innerHTML='<div class=muted>no niches yet</div>';return;}
 const ms=list.map(n=>n.metric);const lo=Math.min(...ms),hi=Math.max(...ms);
 $('niches').innerHTML=list.map(n=>{
  const w=hi===lo?100:Math.max(6,(DIR==='maximize'?(n.metric-lo)/(hi-lo):(hi-n.metric)/(hi-lo))*100);
  const lbl=n.message||'uncategorized';
  return `<div class=niche><div class=lbl><span class=s title="${esc(lbl)}">${esc(lbl)}</span>
    <span class=good>${fmt(n.metric)} <span class=muted>×${n.count}</span></span></div>
    <div class=track><div class=fill style=width:${w}%></div></div></div>`;}).join('');
}

/* ---------- lineage tree from parent links ---------- */
function drawLineage(mem){
 const by={},kids={};for(const m of mem){by[m.id]=m;(kids[m.parent]=kids[m.parent]||[]).push(m);}
  const col={explore:'#79c0ff',exploit:'#7ee787',seed:'#9aa5b1'};
  const roots=mem.filter(m=>m.parent==null||!(m.parent in by)).sort((a,b)=>a.id-b.id);
 let out='';
 const walk=(m,depth)=>{
  const c=m.correct?(col[m.op]||'#9aa5b1'):'#f85149';
  const val=m.metric!=null?fmt(m.metric):(m.error?'✗':'…');
  out+=`<div class=node style="padding-left:${depth*14}px;cursor:pointer" onclick="loadTx('${memLog(m)}',null)">
    <span class=dot style=background:${c}></span><span class=${opClass(m.op)}>${m.op}</span>
    <span>mem ${m.id}</span><span class=${m.correct?'good':'muted'}>${val}</span>
    ${m.best?'<span style=color:var(--elite)>◆</span>':''}</div>`;
  (kids[m.id]||[]).sort((a,b)=>a.id-b.id).forEach(k=>walk(k,depth+1));};
 roots.forEach(r=>walk(r,0));
 $('lineage').innerHTML=out||'<div class=muted>no lineage yet</div>';
}

/* ---------- leaderboard ---------- */
function drawLeaderboard(mem){
 const v=mem.filter(m=>m.correct&&m.metric!=null).sort((a,b)=>better(a.metric,b.metric)?-1:1).slice(0,10);
 if(!v.length){$('lb').innerHTML='<tr><td colspan=7 class=muted>no viable kernels yet</td></tr>';return;}
  $('lb').innerHTML=v.map((m,i)=>{
   return `<tr onclick="loadTx('${memLog(m)}',null)" style=cursor:pointer>
     <td>${m.best?'◆':(i+1)}</td><td>mem ${m.id}</td><td class="num good">${fmt(m.metric)}</td>
     <td><span class=${opClass(m.op)}>${m.op}</span></td><td>${esc(m.message||'')}</td>
     <td class=muted>${m.parent!=null?'mem '+m.parent:'—'}</td><td class=muted>${esc(m.commit)}</td></tr>`;}).join('');
}

/* ---------- legacy scoreboard (tournament / sequential) ---------- */
function drawScoreboard(raw){
 const rows=raw.slice().sort((a,b)=>(b.metric??-1e9)-(a.metric??-1e9));
 const done=rows.filter(r=>r.correct&&r.metric!=null&&!r.running).map(r=>r.metric);
 const bestM=done.length?Math.max(...done):null;
 $('sb').innerHTML=rows.length?rows.map(r=>{
  const isHead=r.index===-1;const cls=isHead?'muted':(r.running?'':(r.correct&&r.metric===bestM?'good':(r.correct?'':'bad')));
  const st=r.running?'<span style=color:#e3b341>running…</span>':(r.correct?'✓':'✗');
  const f=`mem-${r.index}-${r.op||'impl'}-opencode.log`;
  return `<tr onclick="loadTx('${f}',null)" style=cursor:pointer><td class=${cls}>${isHead?'HEAD':('#'+r.index)}</td>
    <td>${st}</td><td class="num ${cls}">${r.running?'-':fmt(r.metric)}</td>
    <td class=muted>${esc(r.commit||'')}</td><td class=bad>${r.running?'':esc(r.error||'')}</td></tr>`;}).join('')
  :'<tr><td colspan=5 class=muted>no candidates</td></tr>';
}

/* ---------- transcript + controller log ---------- */
function loadTx(f,el){selFile=f;$('txWho').textContent=f;
 for(const a of document.querySelectorAll('.agent'))a.classList.remove('sel');
 if(el)el.classList.add('sel');refreshTx(true);}
async function refreshTx(jump){
 if(!selFile)return;
 try{const t=await (await fetch('/api/candlog?file='+encodeURIComponent(selFile))).text();
  const p=$('tx');const atBottom=p.scrollHeight-p.scrollTop-p.clientHeight<60;
  if(p.textContent!==t){p.textContent=t;if(jump||atBottom)p.scrollTop=p.scrollHeight;}}catch(e){}
}
async function refreshLog(){
 try{const lt=await (await fetch('/api/log')).text();const lg=$('log');
  const atBottom=lg.scrollHeight-lg.scrollTop-lg.clientHeight<60;
  if(lg.textContent!==lt){lg.textContent=lt;if(atBottom)lg.scrollTop=lg.scrollHeight;}}catch(e){}
}

async function ctl(b){await fetch('/api/control',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});}
function setN(){ctl({parallelism:+$('N').value})}
function stop(){if(confirm('Stop the search? (in-flight work finishes, best kernel is kept)'))ctl({stop:true})}
function setE(){ctl({explore_bias:+$('E').value});$('Eauto').classList.remove('active');$('E').disabled=false;$('Eb').textContent=$('E').value;}
function autoE(){ctl({explore_auto:true})}
$('E').addEventListener('input',()=>{$('Eb').textContent=$('E').value;});
$('E').addEventListener('change',setE);
setInterval(renderClock,1000);   // tick the elapsed timer between polls
poll();
</script></body></html>"""


def tail_text(path: Path, nbytes: int = 262144) -> str:
    """Read the last ``nbytes`` of a file as text, dropping a partial first line."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            raw = f.read()
    except OSError:
        return ""
    txt = raw.decode("utf-8", errors="replace")
    if size > nbytes and "\n" in txt:
        txt = txt.split("\n", 1)[1]
    return txt


def tool_line(part: dict[str, Any]) -> str:
    """One readable line for a tool event: name + its salient argument.

    opencode nests the call args under ``part.state.input`` (e.g. a read tool is
    ``{"tool":"read","state":{"input":{"filePath":...}}}``); only some shapes put
    them at ``part.input``. Check both, else the line is just the bare tool name."""
    name = part.get("tool") or part.get("name", "tool")
    state = part["state"] if isinstance(part.get("state"), dict) else {}
    inp = (
        state["input"]
        if isinstance(state.get("input"), dict)
        else (part["input"] if isinstance(part.get("input"), dict) else {})
    )
    arg = ""
    for key in (
        "command",
        "filePath",
        "file_path",
        "path",
        "pattern",
        "url",
        "query",
        "description",
        "prompt",
    ):
        if inp.get(key):
            arg = inp[key]
            break
    return " ".join((str(name) + " " + str(arg)).split())


def is_tool(d: dict[str, Any], part: dict[str, Any]) -> bool:
    return d["type"] in ("tool", "tool_use") or (
        "type" in part and part["type"] in ("tool", "tool-invocation")
    )


def summarize_agent_log(path: Path) -> dict[str, Any]:
    """Live summary of an agent's NDJSON log for its card: tool count, latest
    tool call, latest reasoning line, and accumulated cost."""
    out: dict[str, Any] = {"tools": 0, "cost": 0.0, "last_tool": "", "last_text": ""}
    if not path.is_file():
        return out
    for line in tail_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = d["part"] if "part" in d and isinstance(d["part"], dict) else {}
        if d["type"] == "text" and "text" in part and part["text"]:
            out["last_text"] = " ".join(part["text"].split())[:160]
        elif is_tool(d, part):
            out["tools"] += 1
            line_txt = tool_line(part)
            if line_txt:
                out["last_tool"] = line_txt[:160]
        elif d["type"] == "step_finish":
            if part.get("cost"):
                out["cost"] = part["cost"]
    out["cost"] = round(float(out["cost"] or 0.0), 4)
    return out


def candlog_text(path: Path, lines: int = 400) -> str:
    """Full readable transcript of an agent's NDJSON log (text + tool lines)."""
    if not path.is_file():
        return "(no log yet)"
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines()[-4000:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        part = d["part"] if "part" in d and isinstance(d["part"], dict) else {}
        if d["type"] == "text" and "text" in part and part["text"]:
            out.append("· " + " ".join(part["text"].split()))
        elif is_tool(d, part):
            out.append("$ " + (tool_line(part) or "tool"))
    return "\n".join(out[-lines:]) or "(no text yet)"


def make_handler(bus: LoopBus) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # silence
            pass

        def _send(self, code: int, body: str | bytes, ctype: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/status":
                snap = bus.snapshot()
                loop_dir = Path(bus.loop_dir())
                # Enrich each in-flight agent with the live tail of its NDJSON log
                # (the loop can't observe a running turn; this file can).
                if str(loop_dir):
                    for a in snap.get("agents", []):
                        lf = a["log_file"]
                        if lf:
                            a.update(summarize_agent_log(loop_dir / Path(lf).name))
                self._send(200, json.dumps(snap))
            elif u.path == "/api/log":
                loop_dir = Path(bus.loop_dir())
                lf = loop_dir / "loop.log"
                txt = (
                    lf.read_text(encoding="utf-8", errors="replace")
                    if lf.is_file()
                    else "(no log yet)"
                )
                self._send(200, txt, "text/plain; charset=utf-8")
            elif u.path == "/api/candlog":
                q = parse_qs(u.query)
                loop_dir = Path(bus.loop_dir())
                name = (q.get("file") or [""])[0]
                # restrict to the loop dir, basename only (no traversal)
                target = loop_dir / Path(name).name if str(loop_dir) and name else None
                self._send(
                    200,
                    candlog_text(target) if target else "(no file)",
                    "text/plain; charset=utf-8",
                )
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/control":
                self._send(404, "not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                body = {}
            if "parallelism" in body:
                bus.set_parallelism(body["parallelism"])
            if "wall_clock" in body:
                from .config import parse_duration

                with contextlib.suppress(ValueError, TypeError):
                    bus.set_wall_clock(parse_duration(body["wall_clock"]))
            if "explore_bias" in body:
                bus.set_explore_bias(body["explore_bias"])
            if "explore_auto" in body:
                bus.set_explore_auto()
            if "max_candidates" in body:
                bus.set_max_candidates(body["max_candidates"])
            if body.get("stop"):
                bus.request_stop()
            self._send(200, json.dumps({"ok": True}))

    return Handler


def start_server(
    bus: LoopBus, port: int = 8765, host: str = "127.0.0.1"
) -> tuple[ThreadingHTTPServer, int]:
    """Start the web UI in a daemon thread; returns (httpd, actual_port)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(bus))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]
