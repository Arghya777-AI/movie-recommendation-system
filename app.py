"""
CineAI — Movie Recommendation System
All recommendations pre-computed and served client-side.
Live suggestions + instant cards with zero server round-trips.
"""

import os, json
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI · Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide all Streamlit chrome — full-page component takes over
st.markdown("""
<style>
#MainMenu, header, footer,
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }
.stApp { background: #0a0c12 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load pre-computed data (instant — no ML at runtime) ───────────────────────
@st.cache_data
def load_precomputed():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'precomputed.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['movies'], data['recs'], data['top']

with st.spinner("🎬 Loading CineAI…"):
    movies_data, recs_data, top_picks = load_precomputed()

# Serialise — escape </script> so it cannot break the HTML embedding
def safe_json(obj):
    return json.dumps(obj, ensure_ascii=False).replace('</', '<\\/')

movies_json = safe_json(movies_data)
recs_json   = safe_json(recs_data)
top_json    = safe_json(top_picks)
titles_json = safe_json(sorted(movies_data.keys()))


# ── Full-page HTML/JS app ──────────────────────────────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap" rel="stylesheet">
<style>
/* ── Base ── */
.wrap{max-width:620px;margin:0 auto;padding:0 24px}

/* ── Hero ── */
.hero{text-align:center;margin-bottom:40px}
.hero h1{
  font-size:3.2rem;font-weight:900;letter-spacing:-2px;line-height:1;
  background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:8px;
}
.hero p{font-size:.88rem;color:#253045;margin-bottom:3px}
.hero small{font-size:.75rem;color:#182334}
.hero small b{color:#6d63d4;font-weight:600}

/* ── Search ── */
.search-box{position:relative;margin-bottom:0}
.s-icon{
  position:absolute;left:16px;top:50%;transform:translateY(-50%);
  width:18px;height:18px;opacity:.3;pointer-events:none;
  transition:opacity .2s;
}
.search-box:focus-within .s-icon{opacity:.7}
#inp{
  width:100%;background:#0C101C;
  border:1px solid #1C2540;border-radius:12px;
  color:#E8EDF5;font-family:'Inter',sans-serif;font-size:.95rem;font-weight:400;
  padding:15px 16px 15px 48px;outline:none;
  transition:border-color .15s,box-shadow .15s;caret-color:#818cf8;
}
#inp:focus{border-color:#4f4aaf;box-shadow:0 0 0 3px rgba(99,92,175,.13)}
#inp::placeholder{color:#1C2A40;font-weight:400}

/* ── Dropdown — absolute, results dim behind it ── */
#sugg{
  display:none;position:absolute;top:calc(100% + 5px);left:0;right:0;
  background:#090C17;border:1px solid #1C2540;border-radius:12px;
  overflow-y:auto;max-height:420px;
  box-shadow:0 20px 60px rgba(0,0,0,.75);z-index:50;
}
#sugg::-webkit-scrollbar{width:4px}
#sugg::-webkit-scrollbar-track{background:transparent}
#sugg::-webkit-scrollbar-thumb{background:#1C2540;border-radius:4px}
.si{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 18px;cursor:pointer;border-bottom:1px solid #0F1525;
  transition:background .1s;
}
.si:last-child{border-bottom:none}
.si:hover,.si.hi{background:#0E1220}
.si-name{font-size:.88rem;font-weight:500;color:#8D9BB5}
.si:hover .si-name,.si.hi .si-name{color:#E8EDF5}
.si-meta{font-size:.72rem;color:#1C2A40;display:flex;gap:10px;align-items:center}
.si-meta .rt{color:#B45309}

/* ── Results section (dims when suggestions open) ── */
#results-section{transition:opacity .15s}
#results-section.dim{opacity:.25;pointer-events:none}

/* ── Section row ── */
.section-row{
  display:flex;align-items:center;gap:10px;
  margin:28px 0 14px;
}
.section-row span{
  font-size:.68rem;font-weight:700;letter-spacing:1.6px;
  text-transform:uppercase;color:#1E2D3D;flex:1;
}
.ctl{
  background:transparent;border:1px solid #1C2540;border-radius:8px;
  color:#2A3A52;font-family:'Inter',sans-serif;font-size:.72rem;font-weight:500;
  padding:5px 10px;outline:none;cursor:pointer;-webkit-appearance:none;
  transition:border-color .15s,color .15s;
}
.ctl:hover,.ctl:focus{border-color:#3a3480;color:#818cf8}

/* ── Grid ── */
.grid{
  display:grid;grid-template-columns:1fr 1fr;gap:10px;
}
@media(max-width:500px){.grid{grid-template-columns:1fr}}

/* ── Card ── */
.card{
  background:#0C101C;border:1px solid #161E30;border-radius:12px;
  padding:16px 16px;position:relative;overflow:hidden;
  transition:border-color .15s,transform .15s;cursor:default;
}
.card:hover{border-color:#2E2A68;transform:translateY(-1px)}
.card::after{
  content:'';position:absolute;inset:0;border-radius:12px;
  background:linear-gradient(135deg,rgba(99,92,200,.04),transparent);
  opacity:0;transition:opacity .15s;
}
.card:hover::after{opacity:1}
.card.sel{
  grid-column:1/-1;border-color:#2E2A68;
  box-shadow:0 0 0 1px rgba(99,92,200,.2);
}
.rnk{
  position:absolute;top:12px;right:13px;
  font-size:.58rem;font-weight:800;color:#141D2E;letter-spacing:1.2px;
}
.ttl{
  font-size:.95rem;font-weight:600;color:#D6DCE8;
  margin-bottom:5px;padding-right:32px;line-height:1.4;letter-spacing:-.1px;
}
.mtch{
  display:inline-block;background:linear-gradient(90deg,#4f4aaf,#7c3aed);
  color:#fff;border-radius:4px;padding:1px 6px;
  font-size:.6rem;font-weight:700;margin-left:6px;vertical-align:middle;letter-spacing:.3px;
}
.meta{
  display:flex;align-items:center;flex-wrap:wrap;gap:3px;
  font-size:.7rem;color:#1E2D3D;margin-bottom:8px;
}
.meta .s{color:#B45309;font-weight:600;font-size:.73rem}
.meta .d{color:#111825;margin:0 3px}
.tgs{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.tg{
  background:rgba(79,74,175,.09);border:1px solid rgba(79,74,175,.18);
  color:#5a56a0;border-radius:4px;
  padding:2px 8px;font-size:.62rem;font-weight:500;letter-spacing:.2px;
}
.ov{
  font-size:.73rem;color:#1E2D3D;line-height:1.55;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}
.grid-div{grid-column:1/-1;height:1px;background:#131C2C;margin:6px 0 2px}
.grid-lbl{
  grid-column:1/-1;font-size:.67rem;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;color:#1A2535;margin-bottom:2px;
}

/* ── Footer ── */
footer{text-align:center;margin-top:52px;color:#0F1825;font-size:.67rem;line-height:2;letter-spacing:.3px}
</style>
</head>
<body>

<div class="wrap">

  <!-- Hero -->
  <div class="hero">
    <div style="font-size:2rem;margin-bottom:10px">🎬</div>
    <h1>CineAI</h1>
    <p>Discover your next favourite movie</p>
    <small>A project by <b>Megha Saha</b></small>
  </div>

  <!-- Search -->
  <div class="search-box" id="sb">
    <svg class="s-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    <input id="inp" type="text" placeholder="Type a movie title…" autocomplete="off" spellcheck="false">
    <div id="sugg"></div>
  </div>

  <!-- Results (dims when dropdown open) -->
  <div id="results-section">
    <div class="section-row">
      <span id="lbl"></span>
      <select id="nRes" class="ctl">
        <option value="6">6</option>
        <option value="9" selected>9</option>
        <option value="12">12</option>
      </select>
      <select id="sortBy" class="ctl">
        <option value="match">Best match</option>
        <option value="rating">Top rated</option>
        <option value="year">Newest</option>
      </select>
    </div>
    <div id="res" class="grid"></div>
  </div>

  <footer>
    CineAI &nbsp;·&nbsp; TMDB 5000 &nbsp;·&nbsp; Python &amp; Streamlit<br>
    A project by Megha Saha
  </footer>

</div>

<script>
const MOVIES=__MOVIES__,RECS=__RECS__,TOP=__TOP__,TITLES=__TITLES__;
const inp=document.getElementById('inp');
const sugg=document.getElementById('sugg');
const res=document.getElementById('res');
const lbl=document.getElementById('lbl');
const nRes=document.getElementById('nRes');
const srt=document.getElementById('sortBy');
const sec=document.getElementById('results-section');
let hi=-1;

// ── Suggestions ───────────────────────────────────────────────────────────────
function openSugg(q){
  const ql=q.toLowerCase();
  const s=TITLES.filter(t=>t.toLowerCase().startsWith(ql)).slice(0,25);
  const c=TITLES.filter(t=>t.toLowerCase().includes(ql)&&!t.toLowerCase().startsWith(ql)).slice(0,25);
  const all=[...new Set([...s,...c])].slice(0,50);
  if(!all.length){closeSugg();return;}
  sugg.innerHTML=all.map((t,i)=>{
    const m=MOVIES[t]||{};
    return `<div class="si" data-t="${t.replace(/"/g,'&quot;')}" data-i="${i}">
      <span class="si-name">${t}</span>
      <span class="si-meta">
        ${m.y?`<span>${m.y}</span>`:''}
        ${m.r?`<span class="rt">★ ${m.r}</span>`:''}
      </span>
    </div>`;
  }).join('');
  sugg.style.display='block'; sec.classList.add('dim'); hi=-1;
  sugg.querySelectorAll('.si').forEach(el=>
    el.addEventListener('mousedown',e=>{e.preventDefault();pick(el.dataset.t);})
  );
  rz();
}
function closeSugg(){sugg.style.display='none';sec.classList.remove('dim');rz();}
function pick(t){inp.value=t;closeSugg();showRecs(t);}

// ── Cards ─────────────────────────────────────────────────────────────────────
const STR=r=>{const f=Math.round(r/2);return'★'.repeat(f)+'☆'.repeat(5-f)};
function card(title,rank,sim,sel){
  const m=MOVIES[title];if(!m)return'';
  const sr=m.r?`<span class="s">${STR(m.r)} ${m.r}</span>`:'';
  const yr=m.y?`<span class="d">·</span><span>${m.y}</span>`:'';
  const rt=m.rt?`<span class="d">·</span><span>${m.rt}m</span>`:'';
  const bd=m.b?`<span class="d">·</span><span>${m.b}</span>`:'';
  const bx=m.v?`<span class="d">·</span><span>BO ${m.v}</span>`:'';
  const tg=(m.g||[]).map(g=>`<span class="tg">${g}</span>`).join('');
  const pm=sim!=null?`<span class="mtch">${sim}%</span>`:'';
  const rk=rank!=null?`<div class="rnk">#${rank}</div>`:'';
  const ov=m.o?`<div class="ov">${m.o}</div>`:'';
  return`<div class="card${sel?' sel':''}">
    ${rk}<div class="ttl">${title}${pm}</div>
    <div class="meta">${sr}${yr}${rt}${bd}${bx}</div>
    <div class="tgs">${tg}</div>${ov}
  </div>`;
}

// ── Recommendations ───────────────────────────────────────────────────────────
function sorted(list){
  const n=parseInt(nRes.value),by=srt.value,a=list.slice();
  if(by==='rating')a.sort((x,y)=>((MOVIES[y.t]||{}).r||0)-((MOVIES[x.t]||{}).r||0));
  else if(by==='year')a.sort((x,y)=>((MOVIES[y.t]||{}).y||0)-((MOVIES[x.t]||{}).y||0));
  return a.slice(0,n);
}
function showRecs(title){
  const list=RECS[title];
  if(!list){
    const q=title.toLowerCase(),n=parseInt(nRes.value);
    const ms=TITLES.filter(t=>t.toLowerCase().includes(q)).slice(0,n);
    lbl.textContent='RESULTS';
    res.innerHTML=ms.map((t,i)=>card(t,i+1,null,false)).join('');
    rz();return;
  }
  lbl.textContent='YOU SELECTED';
  res.innerHTML=card(title,null,null,true)
    +`<div class="grid-div"></div><div class="grid-lbl">Similar films</div>`
    +sorted(list).map((r,i)=>card(r.t,i+1,r.s,false)).join('');
  rz();
}
function showTop(){
  lbl.textContent='TOP PICKS';
  res.innerHTML=TOP.slice(0,parseInt(nRes.value)).map((t,i)=>card(t,i+1,null,false)).join('');
  rz();
}

// ── Input ─────────────────────────────────────────────────────────────────────
inp.addEventListener('input',function(){
  const v=this.value.trim();
  if(!v){closeSugg();showTop();return;}
  openSugg(v);
  if(RECS[v]){showRecs(v);return;}
  const ms=TITLES.filter(t=>t.toLowerCase().includes(v.toLowerCase())).slice(0,parseInt(nRes.value));
  if(ms.length){lbl.textContent='RESULTS';res.innerHTML=ms.map(t=>card(t,null,null,false)).join('');rz();}
});
inp.addEventListener('keydown',function(e){
  const it=sugg.querySelectorAll('.si');
  if(e.key==='ArrowDown'){e.preventDefault();hi=Math.min(hi+1,it.length-1);it.forEach((el,i)=>el.classList.toggle('hi',i===hi));}
  else if(e.key==='ArrowUp'){e.preventDefault();hi=Math.max(hi-1,-1);it.forEach((el,i)=>el.classList.toggle('hi',i===hi));}
  else if(e.key==='Enter'){hi>=0&&it[hi]?pick(it[hi].dataset.t):(this.value.trim()&&(closeSugg(),showRecs(this.value.trim())));}
  else if(e.key==='Escape')closeSugg();
});
document.addEventListener('click',e=>{if(!e.target.closest('#sb'))closeSugg();});
[nRes,srt].forEach(el=>el.addEventListener('change',()=>{const v=inp.value.trim();v?showRecs(v):showTop();rz();}));

// ── Auto resize iframe to exact content height ────────────────────────────────
function rz(){
  const h=document.documentElement.scrollHeight;
  window.parent.postMessage({type:'streamlit:setFrameHeight',height:h},'*');
}
// Observe ANY size change and auto-send resize
new ResizeObserver(rz).observe(document.body);

showTop();rz();
</script>
</body>
</html>
"""

HTML = (HTML_TEMPLATE
        .replace('__MOVIES__', movies_json)
        .replace('__RECS__',   recs_json)
        .replace('__TOP__',    top_json)
        .replace('__TITLES__', titles_json))

components.html(HTML, height=820, scrolling=False)
