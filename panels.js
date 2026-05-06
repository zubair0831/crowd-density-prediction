'use strict';
// ════════════════════════════════════════════════════════════════════════════
// panels.js  v3.2 — Sidebar panel update functions
//
// Optimisations vs v3.1:
//   • DOM elements cached via _el() — no repeated getElementById in hot paths
//   • NN histogram and walkable-mask canvas guard skipped frames
//   • CUSUM detail panel added (cusumPanel)
//   • Trajectory stats panel added (trajStats)
//   • Zone filter options built once with DocumentFragment (not string concat)
// ════════════════════════════════════════════════════════════════════════════

/* ── Tiny cached DOM accessor ─────────────────────────────────────────────── */
const _cache = {};
const _el = id => (_cache[id] ||= document.getElementById(id));

/* ── Behavior banner ────────────────────────────────────────────────────────  */
const BEH_CONFIG = {
  NORMAL:    { cls: 'normal',     icon: '✅', label: 'NORMAL',     col: 'var(--green)' },
  PRE_SURGE: { cls: 'pre_surge',  icon: '⚠️', label: 'PRE-SURGE',  col: 'var(--amber)' },
  SURGE:     { cls: 'surge',      icon: '🚨', label: 'SURGE',      col: 'var(--red)'   },
  DISPERSING:{ cls: 'dispersing', icon: '🔵', label: 'DISPERSING', col: 'var(--blue)'  },
};

function updateBehaviorBanner(beh) {
  if (!beh) return;
  const cfg = BEH_CONFIG[beh.state] || BEH_CONFIG.NORMAL;
  _el('behBanner').className    = 'beh-banner ' + cfg.cls;
  _el('behIcon').textContent    = cfg.icon;
  _el('behState').textContent   = cfg.label;
  _el('behConf').textContent    = Math.round((beh.confidence || 0) * 100) + '%';
  const s = beh.state, t = beh.count_trend?.toFixed(1) || 0, c = beh.compression_zones;
  _el('behDetail').textContent  =
    s === 'SURGE'      ? `Compression: ${c} zones · Density ratio: ${Math.round(beh.density_ratio*100)}%`
    : s === 'PRE_SURGE'  ? `Rising: +${t}/frame · ${c} comp. zones`
    : s === 'DISPERSING' ? `Trend: ${t}/frame · Dissipating`
    : `Flow nominal · Trend: ${t}/frame`;
}

/* ── Fruin LOS ──────────────────────────────────────────────────────────────  */
const LOS_CONFIG = {
  A: { color: '#4ade80', bg: 'rgba(74,222,128,.12)',  label: 'Free Movement',      desc: 'Safe — no restrictions'     },
  B: { color: '#86efac', bg: 'rgba(134,239,172,.1)',  label: 'Slight Restriction', desc: 'Comfortable movement'       },
  C: { color: '#fde68a', bg: 'rgba(253,230,138,.1)',  label: 'Acceptable',         desc: 'Minor flow disruption'      },
  D: { color: '#fbbf24', bg: 'rgba(251,191,36,.12)',  label: 'Uncomfortable',      desc: 'Significant restriction'    },
  E: { color: '#f97316', bg: 'rgba(249,115,22,.12)',  label: 'Very Restricted',    desc: 'Near capacity — monitor'    },
  F: { color: '#f87171', bg: 'rgba(248,113,113,.14)', label: 'Crush Risk',         desc: 'EVACUATE / Reduce capacity' },
};

function updateLOS(los) {
  const row = _el('losRow');
  if (!los?.los) { row.style.display = 'none'; return; }
  row.style.display = 'flex';
  const cfg   = LOS_CONFIG[los.los] || LOS_CONFIG.A;
  const badge = _el('losBadge');
  badge.textContent      = los.los;
  badge.style.background = cfg.bg;
  badge.style.color      = cfg.color;
  _el('losLabel').textContent = cfg.label;
  _el('losLabel').style.color = cfg.color;
  _el('losDesc').textContent  = cfg.desc;
  _el('losDensity').textContent = los.density_sqm?.toFixed(2) + ' p/m²';
}

/* ── Adaptive threshold card ────────────────────────────────────────────────  */
function updateThreshCard(thresh, mode, mean, std, samples, signals) {
  if (thresh == null) return;
  _el('tcV').textContent = thresh;
  const badge = _el('tcBadge');
  if (mode === 'dynamic_v2') {
    badge.textContent = 'Multi-Signal · ' + samples + 'f';
    badge.className   = 'tc-badge b-v2';
    _el('tcSub').textContent =
      `μ=${mean}  σ=${std}  z=${currentZ}σ` +
      (signals?.area_cap ? `  |  Area cap: ${signals.area_cap}` : '') +
      `  |  prox·${signals?.prox_factor?.toFixed(2)}  zone·${signals?.zone_factor?.toFixed(2)}`;
  } else if (mode === 'dynamic') {
    badge.textContent = 'Dynamic · ' + samples + 'f';
    badge.className   = 'tc-badge b-dy';
    _el('tcSub').textContent = `μ=${mean}  σ=${std}  EMA + ${currentZ}σ`;
  } else {
    badge.textContent = 'Warm-up · ' + (samples||0) + 'f';
    badge.className   = 'tc-badge b-wu';
    _el('tcSub').textContent = 'Building baseline — activates after 5 frames';
  }
  _el('barLbl').textContent = 'threshold: ' + thresh;

  if (signals && mode === 'dynamic_v2') {
    _el('signalStack').style.display = 'block';
    const maxSig = Math.max(signals.baseline||0, signals.area_cap||0, thresh, 1);
    const set = (barId, valId, val, maxV) => {
      const b = _el(barId), v = _el(valId);
      if (b) b.style.width   = Math.min((val/maxV)*100, 100) + '%';
      if (v) v.textContent   = val != null ? Math.round(val) : '—';
    };
    set('sig-baseline', 'sigv-baseline', signals.baseline,                     maxSig);
    set('sig-area',     'sigv-area',     signals.area_cap,                     maxSig);
    set('sig-prox',     'sigv-prox',     Math.round((signals.proximity_score||0)*100), 100);
    set('sig-zone',     'sigv-zone',     Math.round((signals.zone_cv||0)*100),          100);
    set('sig-final',    'sigv-final',    thresh,                                maxSig);
  }
}

/* ── Distance / proximity panel ─────────────────────────────────────────────  */
function updateDistancePanel(dist) {
  if (!dist) return;
  const ps  = dist.proximity_score || 0;
  const ban = _el('proxBanner'), lbl = _el('proxLabel');
  const det = _el('proxDetail'), sc  = _el('proxScore');
  sc.textContent = Math.round(ps * 100) + '%';
  if (ps >= 0.65) {
    ban.className = 'prox-banner danger';
    lbl.textContent = 'DANGER: HIGH PROXIMITY'; lbl.style.color = 'var(--red)';
    det.textContent = `${dist.violations} of ${dist.count} persons too close`;
    sc.style.color  = 'var(--red)';
  } else if (ps >= 0.30) {
    ban.className = 'prox-banner moderate';
    lbl.textContent = 'MODERATE CROWDING'; lbl.style.color = 'var(--amber)';
    det.textContent = `${dist.violations} proximity violations detected`;
    sc.style.color  = 'var(--amber)';
  } else {
    ban.className = 'prox-banner safe';
    lbl.textContent = 'SAFE SPACING'; lbl.style.color = 'var(--green)';
    det.textContent = 'All persons within safe distance';
    sc.style.color  = 'var(--green)';
  }
  _el('dViolations').textContent = dist.violations || 0;
  _el('dVioRatio').textContent   = Math.round((dist.proximity_violation_ratio||0)*100) + '%';
  _el('dMedianNN').textContent   = dist.median_nn_dist != null ? Math.round(dist.median_nn_dist)+(dist.space==='world_m'?'m':'px') : '—';
  _el('dSafeDist').textContent   = dist.min_safe_px   != null ? Math.round(dist.min_safe_px)  +(dist.space==='world_m'?'m':'px') : '—';
  // Show "world_m" badge when calibrated
  const spaceBadge = _el('distSpaceBadge');
  if (spaceBadge) { spaceBadge.textContent = dist.space==='world_m'?'📐 calibrated':'px'; spaceBadge.style.color = dist.space==='world_m'?'var(--green)':'var(--text3)'; }
}

/* ── Walkable area card ──────────────────────────────────────────────────────  */
const TIER_CONFIG = {
  'segformer+yolo': { cls: 'tier-segformer-yolo', label: '✦ SegFormer + YOLO (best)' },
  'segformer':      { cls: 'tier-segformer',       label: '✦ SegFormer only'          },
  'yolo':           { cls: 'tier-yolo',            label: '◈ YOLO only'               },
  'heuristic':      { cls: 'tier-heuristic',       label: '◇ Heuristic fallback'      },
};

function updateWalkableCard(meta) {
  if (!meta) return;
  const wpct = meta.walkable_pct || 0, opct = 100 - wpct;
  _el('walkableBar')?.style.setProperty?.('width', wpct + '%');
  if (_el('walkableBar')) _el('walkableBar').style.width = wpct + '%';
  if (_el('walkablePct')) _el('walkablePct').textContent = wpct + '%';
  if (_el('obstacleBar')) _el('obstacleBar').style.width = opct.toFixed(1) + '%';
  if (_el('obstaclePct')) _el('obstaclePct').textContent = opct.toFixed(1) + '%';
  if (_el('walkableInfo')) _el('walkableInfo').textContent =
    `Walkable: ${wpct}% · Tier: ${meta.tier_used||'unknown'} · Obstacles: ${meta.obstacle_count||0}`;

  const tierEl = _el('tierBadge');
  if (tierEl) {
    const cfg = TIER_CONFIG[meta.tier_used||'heuristic'] || TIER_CONFIG.heuristic;
    tierEl.className  = 'tier-badge ' + cfg.cls;
    tierEl.textContent = cfg.label;
  }

  // SegFormer class breakdown
  const segList = _el('segClassList');
  if (segList) {
    const classes = meta.segment_classes || [];
    if (!classes.length) {
      segList.innerHTML = '<div style="color:var(--text3);font-size:11px;">SegFormer not active — no semantic data</div>';
    } else {
      segList.innerHTML = [...classes]
        .sort((a,b)=>b.pct-a.pct).slice(0,12)
        .map(cls => {
          const w = cls.is_walkable;
          return `<div class="seg-class-item">
            <span class="${w?'seg-walkable':'seg-nonwalkable'}" style="font-size:9px;flex-shrink:0;margin-right:4px;">${w?'▣':'✕'}</span>
            <span class="seg-class-name">${cls.class_name}</span>
            <div class="seg-class-bar-wrap"><div class="seg-class-bar" style="width:${Math.min(cls.pct,100)}%;background:${w?'var(--teal)':'var(--red)'}"></div></div>
            <span class="seg-class-pct">${cls.pct}%</span>
          </div>`;
        }).join('');
    }
  }

  // YOLO obstacle list
  const ol  = _el('obstacleList');
  const obs = meta.obstacles_detected || [];
  if (!obs.length) {
    ol.innerHTML = '<div style="color:var(--text3);font-size:11px;">No YOLO obstacles detected</div>';
  } else {
    const catColors = { vehicle:'var(--orange)', nature:'var(--green)', furniture:'var(--purple)', structure:'var(--red)' };
    const catIcons  = { vehicle:'🚗', nature:'🌿', furniture:'🪑', structure:'🏗' };
    ol.innerHTML = obs.slice(0,10).map(o => {
      const col = catColors[o.category||'structure']||'var(--red)';
      const ico = catIcons[o.category||'structure']||'⬛';
      return `<div class="obstacle-item">
        <span style="display:flex;align-items:center;gap:5px;">
          <span style="font-size:12px;">${ico}</span>
          <span class="obs-class" style="color:${col};font-weight:600;">${o.class}</span>
        </span>
        <div style="display:flex;gap:5px;align-items:center;">
          <span style="font-family:var(--mono);font-size:9px;color:var(--text3)">${o.area_pct}%</span>
          <span class="obs-badge" style="background:${col}22;color:${col};border:1px solid ${col}44;">${Math.round(o.conf*100)}%</span>
        </div>
      </div>`;
    }).join('');
  }
}

/* ── NN distance histogram ──────────────────────────────────────────────────  */
function renderNNHistogram() {
  if (!currentDistMeta) return;
  const dists = currentDistMeta.nn_distances || [];
  if (!dists.length) return;
  const canvas = _el('nnHistCanvas');
  const W = canvas.offsetWidth || 300, H = 80;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const minD   = Math.min(...dists), maxD = Math.max(...dists)*1.1||100;
  const safePx = currentDistMeta.min_safe_px || 50;
  const BINS   = 20, binW = (maxD-minD)/BINS;
  const counts = new Array(BINS).fill(0);
  dists.forEach(d => { counts[Math.min(Math.floor((d-minD)/Math.max(binW,0.001)),BINS-1)]++; });
  const maxC = Math.max(...counts,1);
  const pad  = {l:20,r:8,t:8,b:20}, cH=H-pad.t-pad.b, cw=W/BINS;

  counts.forEach((c,i) => {
    const x=pad.l+i*cw, bh=(c/maxC)*cH, y=pad.t+cH-bh;
    const bc = minD+(i+0.5)*binW;
    ctx.fillStyle = bc<safePx?'rgba(248,113,113,.7)':'rgba(34,211,238,.5)';
    ctx.fillRect(x,y,cw-1,bh);
  });

  const pct=(safePx-minD)/Math.max(maxD-minD,1);
  const lx=pad.l+pct*(W-pad.l-pad.r);
  if (lx>pad.l && lx<W-pad.r) {
    ctx.strokeStyle='rgba(251,191,36,.8)'; ctx.lineWidth=1.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(lx,pad.t); ctx.lineTo(lx,H-pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font='9px "JetBrains Mono",monospace'; ctx.fillStyle='rgba(251,191,36,.8)';
    ctx.fillText('safe',lx+3,pad.t+9);
  }
  ctx.font='8px "JetBrains Mono",monospace'; ctx.fillStyle='rgba(125,143,168,.5)'; ctx.textAlign='center';
  ctx.fillText(Math.round(minD)+'px',pad.l+cw,H-4);
  ctx.fillText(Math.round(maxD)+'px',W-pad.r,H-4);
  ctx.textAlign='left';
}

/* ── Walkable mask grid canvas ──────────────────────────────────────────────  */
function renderWalkableMaskCanvas() {
  if (!currentWalkableMask) return;
  const canvas = _el('walkableMaskCanvas');
  const W=GRID_COLS*6, H=GRID_ROWS*6;
  canvas.width=W; canvas.height=H;
  const ctx=canvas.getContext('2d');
  for (let r=0;r<GRID_ROWS;r++) for (let c=0;c<GRID_COLS;c++) {
    const v=currentWalkableMask[r]?.[c]??0;
    ctx.fillStyle=v<0.1?'rgba(248,113,113,.3)':`rgb(${Math.round(v*34)},${Math.round(v*211)},${Math.round(v*238)})`;
    ctx.fillRect(c*6,r*6,6,6);
  }
}

/* ── Zone map ───────────────────────────────────────────────────────────────  */
function getDensityLevel(risk) {
  return risk>=0.75?'critical':risk>=0.50?'high':risk>=0.25?'medium':'low';
}

function setZoneMode(mode,btn) {
  zoneDisplayMode=mode;
  document.querySelectorAll('.zmc-btn').forEach(b=>b.classList.remove('act'));
  btn.classList.add('act');
  renderZoneMap();
}

function renderZoneMap() {
  const canvas=_el('zoneMapCanvas');
  const rect=canvas.getBoundingClientRect();
  const W=Math.floor(rect.width)||canvas.offsetWidth||300;
  const H=Math.floor(rect.height)||canvas.offsetHeight||225;
  if (W<10||H<10){requestAnimationFrame(renderZoneMap);return;}
  canvas.width=W; canvas.height=H;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#121a2a'; ctx.fillRect(0,0,W,H);
  const cw=W/GRID_COLS, ch=H/GRID_ROWS, t=Date.now();

  if (!currentZoneData?.length) {
    ctx.strokeStyle='rgba(255,255,255,.05)'; ctx.lineWidth=0.5;
    for (let c=0;c<=GRID_COLS;c++){ctx.beginPath();ctx.moveTo(c*cw,0);ctx.lineTo(c*cw,H);ctx.stroke();}
    for (let r=0;r<=GRID_ROWS;r++){ctx.beginPath();ctx.moveTo(0,r*ch);ctx.lineTo(W,r*ch);ctx.stroke();}
    ctx.fillStyle='rgba(125,143,168,.4)'; ctx.font='10px "JetBrains Mono",monospace';
    ctx.textAlign='center'; ctx.fillText('Awaiting data…',W/2,H/2); ctx.textAlign='left';
    return;
  }

  currentZoneData.forEach(z => {
    const x=z.col*cw, y=z.row*ch, r=z.risk;
    if (!z.is_walkable) {
      ctx.fillStyle='rgba(248,113,113,.07)'; ctx.fillRect(x,y,cw,ch);
      ctx.strokeStyle='rgba(248,113,113,.15)'; ctx.lineWidth=0.7;
      for (let d=0;d<cw+ch;d+=7){ctx.beginPath();ctx.moveTo(x+d,y);ctx.lineTo(x,y+d);ctx.stroke();}
      ctx.textAlign='center';
      ctx.font=`500 ${Math.max(6,Math.min(8,cw*0.21))}px "JetBrains Mono",monospace`;
      ctx.fillStyle='rgba(248,113,113,.4)'; ctx.fillText('N/W',x+cw/2,y+ch/2+3);
      ctx.textAlign='left'; return;
    }

    let fR=0,fG=200,fB=0,alpha=0.15+r*0.70;
    if (zoneDisplayMode==='risk'){
      fR=Math.round(r<0.5?r*2*255:255); fG=Math.round(r<0.5?255:(1-r)*2*255); fB=0;
    } else if (zoneDisplayMode==='density'){
      const dn=z.density_norm||0;
      fR=Math.round(dn<0.5?dn*2*160:160+dn*95); fG=Math.round(dn<0.5?200:(1-dn)*200);
      fB=Math.round((1-dn)*238); alpha=0.12+dn*0.75;
    } else if (zoneDisplayMode==='walkable'){
      const wf=z.walkable_frac||0;
      fR=Math.round((1-wf)*238); fG=Math.round(wf*200); fB=Math.round(wf*238);
      alpha=0.3+Math.abs(wf-0.5)*0.5;
    } else if (zoneDisplayMode==='flow'){
      const mag=Math.min(z.magnitude||0,8)/8, div=z.divergence||0;
      fR=div<-0.05?255:50; fG=div>0.05?200:50; fB=Math.round(mag*238);
      alpha=0.15+mag*0.65;
    }

    ctx.fillStyle=`rgba(${fR},${fG},${fB},${alpha})`; ctx.fillRect(x,y,cw,ch);

    if (r>0.68&&zoneDisplayMode==='risk'){
      const p=0.55+Math.sin(t/300)*0.45;
      ctx.strokeStyle=`rgba(248,113,113,${p})`; ctx.lineWidth=1.5;
      ctx.strokeRect(x+0.75,y+0.75,cw-1.5,ch-1.5);
    }
    if (selectedZone&&selectedZone.row===z.row&&selectedZone.col===z.col){
      ctx.strokeStyle='rgba(34,211,238,0.9)'; ctx.lineWidth=2;
      ctx.strokeRect(x+1,y+1,cw-2,ch-2);
    }

    const fs=Math.max(8,Math.min(13,cw*0.32));
    ctx.font=`700 ${fs}px "Epilogue",sans-serif`; ctx.textAlign='center';
    const lx=x+cw/2, ly=y+ch/2+(cw>30?2:4);
    ctx.fillStyle='rgba(0,0,0,.6)'; ctx.fillText(z.count,lx+1,ly+1);
    ctx.fillStyle='#fff';           ctx.fillText(z.count,lx,ly);
    if (cw>28&&ch>24){
      const dlblColors={low:'rgba(74,222,128,.7)',medium:'rgba(251,191,36,.7)',high:'rgba(251,146,60,.75)',critical:'rgba(248,113,113,.8)'};
      ctx.font=`500 ${Math.max(6,Math.min(8,cw*0.21))}px "JetBrains Mono",monospace`;
      ctx.fillStyle=dlblColors[getDensityLevel(r)];
      ctx.fillText(getDensityLevel(r).toUpperCase(),lx,ly+fs*0.95);
    }
    ctx.textAlign='left';
    if (z.los&&z.los!=='?'&&cw>28){
      const losCols={A:'#4ade80',B:'#86efac',C:'#fde68a',D:'#fbbf24',E:'#f97316',F:'#f87171'};
      ctx.font=`700 ${Math.max(6,Math.min(9,cw*0.25))}px "JetBrains Mono",monospace`;
      ctx.fillStyle=losCols[z.los]||'#aaa'; ctx.fillText(z.los,x+2,y+10);
    }
  });

  ctx.strokeStyle='rgba(255,255,255,.06)'; ctx.lineWidth=0.5;
  for (let c=0;c<=GRID_COLS;c++){ctx.beginPath();ctx.moveTo(c*cw,0);ctx.lineTo(c*cw,H);ctx.stroke();}
  for (let r=0;r<=GRID_ROWS;r++){ctx.beginPath();ctx.moveTo(0,r*ch);ctx.lineTo(W,r*ch);ctx.stroke();}
  ctx.font=`${Math.max(7,Math.min(9,cw*0.22))}px "JetBrains Mono",monospace`;
  ctx.fillStyle='rgba(255,255,255,.18)'; ctx.textAlign='center';
  for (let c=0;c<GRID_COLS;c++) ctx.fillText(c+1,c*cw+cw/2,9);
  ctx.textAlign='left';
  for (let r=0;r<GRID_ROWS;r++) ctx.fillText(r+1,2,r*ch+ch/2+4);

  if (currentZoneData&&isPlaying) requestAnimationFrame(renderZoneMap);
}

function updateZoneStats(zones,agg) {
  if (!zones?.length) return;
  _el('zHot').textContent      = zones.filter(z=>z.risk>0.68).length;
  _el('zPeak').textContent     = Math.max(...zones.map(z=>z.count));
  _el('zCrush').textContent    = zones.filter(z=>z.risk>0.85).length;
  _el('zWalkable').textContent = agg?.walkable_zone_count||zones.filter(z=>z.is_walkable).length;

  const sel = _el('zoneFilter');
  if (sel&&sel.options.length<2) {
    const frag = document.createDocumentFragment();
    for (let r=0;r<GRID_ROWS;r++) for (let c=0;c<GRID_COLS;c++) {
      const o=document.createElement('option');
      o.value=`${r}-${c}`; o.textContent=`Zone R${r+1}·C${c+1}`; frag.appendChild(o);
    }
    sel.appendChild(frag);
  }
}

function zoneHover(e) {
  if (!currentZoneData) return;
  const canvas=_el('zoneMapCanvas'), rect=canvas.getBoundingClientRect();
  showZoneDetail(
    Math.floor((e.clientY-rect.top) /rect.height*GRID_ROWS),
    Math.floor((e.clientX-rect.left)/rect.width *GRID_COLS)
  );
}

function zoneClick(e) {
  if (!currentZoneData) return;
  const canvas=_el('zoneMapCanvas'), rect=canvas.getBoundingClientRect();
  const col=Math.floor((e.clientX-rect.left)/rect.width *GRID_COLS);
  const row=Math.floor((e.clientY-rect.top) /rect.height*GRID_ROWS);
  selectedZone=(selectedZone&&selectedZone.row===row&&selectedZone.col===col)?null:{row,col};
  showZoneDetail(row,col,true); renderZoneMap();
}

function showZoneDetail(row,col,pinned=false) {
  if (!currentZoneData) return;
  const z=currentZoneData.find(z=>z.row===row&&z.col===col);
  if (!z) return;
  const el = _el('zoneDetail');
  if (!z.is_walkable) {
    el.innerHTML=`<div class="zd-title">ZONE R${row+1}·C${col+1}</div>
      <span class="density-badge density-nonwalkable">NON-WALKABLE</span>
      <div style="color:var(--text2);font-size:12px;margin-top:8px;line-height:1.6;">Covered by obstacles. Excluded from density calculations.</div>`;
    return;
  }
  const dlvl=getDensityLevel(z.risk);
  el.innerHTML=`<div class="zd-title">ZONE R${row+1}·C${col+1} ${pinned?'<span style="color:var(--cyan)">(pinned)</span>':''}</div>
    <div style="margin-bottom:8px;display:flex;gap:6px;align-items:center;">
      <span class="density-badge density-${dlvl}">${dlvl}</span>
      ${z.los?`<span style="font-family:var(--mono);font-size:11px;color:var(--text2)">Fruin LOS-${z.los}</span>`:''}
    </div>
    <div class="zone-grid-info">
      <div class="zgi">Count: <span>${z.count}</span></div>
      <div class="zgi">Risk: <span>${(z.risk*100).toFixed(0)}%</span></div>
      <div class="zgi">Walkable: <span>${(z.walkable_frac*100).toFixed(0)}%</span></div>
      ${z.density_sqm!=null
        ?`<div class="zgi">Density: <span>${z.density_sqm} p/m²</span></div>`
        :`<div class="zgi">Density: <span>uncalibrated</span></div>`}
      <div class="zgi">Flow Δx: <span>${z.dx.toFixed(2)}</span></div>
      <div class="zgi">Flow Δy: <span>${z.dy.toFixed(2)}</span></div>
      <div class="zgi" style="grid-column:1/-1">Divergence:
        <span style="color:${z.divergence<-0.05?'var(--red)':'var(--green)'}">
          ${z.divergence.toFixed(3)} ${z.divergence<-0.05?'⚠ compression':''}
        </span>
      </div>
    </div>`;
}

function clearZoneHover() {
  if (selectedZone) return;
  _el('zoneDetail').innerHTML='<div class="zd-title">ZONE DETAIL — hover or click a cell</div>'
    +'<div style="color:var(--text2);font-size:12.5px;line-height:1.6;">Zones clipped to walkable area. Non-walkable cells shown with hatching.</div>';
}

/* ── History chart ──────────────────────────────────────────────────────────  */
function setTimeRange(range,btn) {
  histTimeRangeValue=range;
  document.querySelectorAll('.hc-pill').forEach(b=>b.classList.remove('act'));
  btn.classList.add('act');
  updateHistoryView();
}

function updateHistoryView() {
  if (!frameData.length) return;
  const fps=parseFloat(_el('fpsIn').value)||2;
  let data=[...frameData];
  if (histTimeRangeValue!=='all'){
    const secs=parseInt(histTimeRangeValue);
    const lastT=data[data.length-1]?.frame/fps||0;
    data=data.filter(fd=>fd.frame/fps>=lastT-secs);
  }
  const counts=data.map(f=>f.count);
  const peak=Math.max(...counts,0);
  const avg=counts.length?Math.round(counts.reduce((a,b)=>a+b,0)/counts.length):0;
  const breaches=data.filter(f=>f.count>(f.dynamic_threshold||Infinity)).length;
  const dur=data.length?fmt(data[data.length-1].frame/fps-data[0].frame/fps):'0:00';
  _el('hsPeak').textContent    =peak;
  _el('hsAvg').textContent     =avg;
  _el('hsBreaches').textContent=breaches;
  _el('hsDur').textContent     =dur;
  _el('sMax').textContent=Math.max(...frameData.map(f=>f.count),0);
  _el('sAvg').textContent=Math.round(frameData.reduce((s,f)=>s+f.count,0)/Math.max(frameData.length,1));
  drawHistChart(data,fps);
  renderEventsList(data,fps);
}

function drawHistChart(data,fps) {
  const canvas=_el('histChartCanvas');
  const W=canvas.offsetWidth||300, H=160;
  canvas.width=W; canvas.height=H;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,W,H);
  if (data.length<2){
    ctx.fillStyle='rgba(125,143,168,.3)'; ctx.font='11px "JetBrains Mono",monospace';
    ctx.textAlign='center'; ctx.fillText('Awaiting data…',W/2,H/2); ctx.textAlign='left'; return;
  }
  const counts=data.map(f=>f.count), threshs=data.map(f=>f.dynamic_threshold||0);
  const areaCaps=data.map(f=>f.threshold_signals?.area_cap||null);
  const maxV=Math.max(...counts,...threshs,...areaCaps.filter(Boolean),1);
  const pad={l:28,r:12,t:14,b:24}, cW=W-pad.l-pad.r, cH=H-pad.t-pad.b, n=data.length;
  const xOf=i=>pad.l+(i/(n-1))*cW, yOf=v=>pad.t+cH-(v/maxV)*cH;

  ctx.strokeStyle='rgba(255,255,255,.05)'; ctx.lineWidth=1;
  ctx.font='9px "JetBrains Mono",monospace'; ctx.fillStyle='rgba(125,143,168,.5)'; ctx.textAlign='right';
  for (let i=0;i<=4;i++){const v=Math.round(maxV/4*i),y=yOf(v);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(v,pad.l-3,y+3);}
  ctx.textAlign='center'; ctx.fillStyle='rgba(125,143,168,.4)';
  const xStep=Math.max(1,Math.floor(n/5));
  for (let i=0;i<n;i+=xStep) ctx.fillText(fmt(data[i].frame/fps),xOf(i),H-pad.b+13);
  ctx.textAlign='left';

  // Fill area
  const grad=ctx.createLinearGradient(0,pad.t,0,pad.t+cH);
  grad.addColorStop(0,'rgba(34,211,238,.25)'); grad.addColorStop(1,'rgba(34,211,238,.02)');
  ctx.beginPath(); ctx.moveTo(xOf(0),yOf(counts[0]));
  for (let i=1;i<n;i++) ctx.lineTo(xOf(i),yOf(counts[i]));
  ctx.lineTo(xOf(n-1),pad.t+cH); ctx.lineTo(xOf(0),pad.t+cH); ctx.closePath();
  ctx.fillStyle=grad; ctx.fill();

  // Area cap line
  if (areaCaps.some(v=>v!=null)){
    ctx.strokeStyle='rgba(45,212,191,.45)'; ctx.lineWidth=1; ctx.setLineDash([3,4]);
    ctx.beginPath(); let first=true;
    for (let i=0;i<n;i++){const ac=areaCaps[i];if(ac==null)continue;first?(ctx.moveTo(xOf(i),yOf(ac)),first=false):ctx.lineTo(xOf(i),yOf(ac));}
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Threshold line
  ctx.strokeStyle='rgba(251,191,36,.4)'; ctx.lineWidth=1; ctx.setLineDash([5,4]);
  ctx.beginPath(); for (let i=0;i<n;i++){const y=yOf(threshs[i]||0);i===0?ctx.moveTo(xOf(0),y):ctx.lineTo(xOf(i),y);}
  ctx.stroke(); ctx.setLineDash([]);

  // Count line
  const lg=ctx.createLinearGradient(pad.l,0,W-pad.r,0);
  lg.addColorStop(0,'rgba(34,211,238,.95)'); lg.addColorStop(1,'rgba(232,121,160,.95)');
  ctx.strokeStyle=lg; ctx.lineWidth=2;
  ctx.beginPath(); for (let i=0;i<n;i++) i===0?ctx.moveTo(xOf(i),yOf(counts[i])):ctx.lineTo(xOf(i),yOf(counts[i])); ctx.stroke();

  // Peak/trough markers
  const pkI=counts.indexOf(Math.max(...counts)), trI=counts.indexOf(Math.min(...counts));
  [pkI,trI].forEach(idx=>{
    if (idx<0) return;
    const x=xOf(idx),y=yOf(counts[idx]);
    ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2);
    ctx.fillStyle=idx===pkI?'var(--pink)':'var(--green)'; ctx.fill();
    const lbl=String(counts[idx]);
    ctx.font='bold 10px "JetBrains Mono",monospace'; ctx.fillStyle='#fff';
    const lx=clamp(x-ctx.measureText(lbl).width/2,pad.l,W-pad.r-20);
    ctx.fillText(lbl,lx,y-8<pad.t+10?y+16:y-8);
  });

  // Alert dots
  data.forEach((fd,i)=>{
    if (fd.count>(fd.dynamic_threshold||Infinity)){
      ctx.beginPath(); ctx.arc(xOf(i),yOf(fd.count),4,0,Math.PI*2);
      ctx.fillStyle='rgba(248,113,113,.85)'; ctx.fill();
    }
  });

  // CUSUM alert ticks (v3.2 new)
  data.forEach((fd,i)=>{
    if (fd.cusum?.alert){
      const x=xOf(i), y=yOf(counts[i]);
      ctx.strokeStyle=fd.cusum.direction==='surge'?'rgba(248,113,113,.9)':'rgba(96,165,250,.9)';
      ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+cH); ctx.stroke();
    }
  });
}

function renderEventsList(data,fps) {
  const el=_el('eventsList');
  const events=[];
  if (data.length){
    const pk=data.reduce((b,f)=>f.count>b.count?f:b,data[0]);
    events.push({label:'Peak Count',count:pk.count,ts:fmt(pk.frame/fps),state:'PEAK',col:'var(--pink)',frame:pk.frame});
  }
  data.filter(f=>f.count>(f.dynamic_threshold||Infinity))
    .forEach(fd=>events.push({label:'Threshold Breach',count:fd.count,ts:fmt(fd.frame/fps),state:'BREACH',col:'var(--red)',frame:fd.frame}));
  data.filter(f=>f.behavior?.state==='SURGE')
    .forEach(fd=>events.push({label:'Surge Event',count:fd.count,ts:fmt(fd.frame/fps),state:'SURGE',col:'var(--red)',frame:fd.frame}));
  data.filter(f=>f.cusum?.alert)
    .forEach(fd=>events.push({label:'CUSUM: '+fd.cusum.direction,count:fd.count,ts:fmt(fd.frame/fps),state:'CUSUM',col:fd.cusum.direction==='surge'?'var(--red)':'var(--blue)',frame:fd.frame}));
  notableEvents.forEach(ev=>events.push({label:'Announcement: '+ev.level,count:'—',ts:ev.time,state:ev.level.toUpperCase(),col:ev.level==='critical'?'var(--red)':ev.level==='high'?'var(--orange)':'var(--amber)',frame:-1}));
  if (!events.length){el.innerHTML='<div class="hist-empty">No notable events recorded</div>';return;}
  events.sort((a,b)=>b.frame-a.frame);
  el.innerHTML=events.slice(0,30).map(ev=>`
    <div class="event-item" onclick="${ev.frame>=0?`seekToFrame(${ev.frame})`:''}" title="Click to seek">
      <div class="ev-dot" style="background:${ev.col}"></div>
      <div class="ev-body"><div class="ev-title">${ev.label}</div><div class="ev-meta">${ev.ts}${ev.state?'  ·  '+ev.state:''}</div></div>
      <div class="ev-count" style="color:${ev.col}">${ev.count}</div>
    </div>`).join('');
}

function exportCSV() {
  if (!frameData.length){alert('No data to export.');return;}
  const fps=parseFloat(_el('fpsIn').value)||2;
  const rows=[['Frame','Time','Count','Threshold','Mode','Behavior','Density Ratio','Proximity Score','Violations','Walkable%','Obstacles','Tier','CUSUM Alert','CUSUM Direction']];
  frameData.forEach(fd=>rows.push([
    fd.frame, fmt(fd.frame/fps), fd.count, fd.dynamic_threshold||'',
    fd.threshold_mode||'', fd.behavior?.state||'', fd.behavior?.density_ratio||'',
    fd.distance?.proximity_score||'', fd.distance?.violations||'',
    fd.walkable?.walkable_pct||'', fd.walkable?.obstacle_count||'',
    fd.walkable?.tier_used||'',
    fd.cusum?.alert?'1':'0', fd.cusum?.direction||'',
  ]));
  const csv=rows.map(r=>r.join(',')).join('\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='crowd_data_v3_'+Date.now()+'.csv'; a.click();
  addLog('CSV exported: '+frameData.length+' rows');
}