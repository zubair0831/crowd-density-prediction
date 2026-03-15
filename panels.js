'use strict';
// ════════════════════════════════════════════════════════════════════════════
// panels.js — Sidebar panel update functions
// ════════════════════════════════════════════════════════════════════════════

// ── Behavior banner ──────────────────────────────────────────────────────────
const BEH_CONFIG = {
  NORMAL:    { cls: 'normal',     icon: '✅', label: 'NORMAL',     col: 'var(--green)' },
  PRE_SURGE: { cls: 'pre_surge',  icon: '⚠️', label: 'PRE-SURGE',  col: 'var(--amber)' },
  SURGE:     { cls: 'surge',      icon: '🚨', label: 'SURGE',      col: 'var(--red)'   },
  DISPERSING:{ cls: 'dispersing', icon: '🔵', label: 'DISPERSING', col: 'var(--blue)'  },
};

function updateBehaviorBanner(beh) {
  if (!beh) return;
  const cfg = BEH_CONFIG[beh.state] || BEH_CONFIG.NORMAL;
  document.getElementById('behBanner').className = 'beh-banner ' + cfg.cls;
  document.getElementById('behIcon').textContent  = cfg.icon;
  document.getElementById('behState').textContent = cfg.label;
  document.getElementById('behConf').textContent  = Math.round((beh.confidence || 0) * 100) + '%';
  const detail =
    beh.state === 'SURGE'      ? `Compression: ${beh.compression_zones} zones · Density: ${Math.round(beh.density_ratio * 100)}%`
    : beh.state === 'PRE_SURGE'  ? `Rising: +${beh.count_trend?.toFixed(1)}/frame · ${beh.compression_zones} comp. zones`
    : beh.state === 'DISPERSING' ? `Trend: ${beh.count_trend?.toFixed(1)}/frame · Dissipating`
    : `Flow nominal · Trend: ${beh.count_trend?.toFixed(1) || 0}/frame`;
  document.getElementById('behDetail').textContent = detail;
}

// ── Fruin LOS ────────────────────────────────────────────────────────────────
const LOS_CONFIG = {
  A: { color: '#4ade80', bg: 'rgba(74,222,128,.12)',   label: 'Free Movement',      desc: 'Safe — no restrictions'     },
  B: { color: '#86efac', bg: 'rgba(134,239,172,.1)',   label: 'Slight Restriction', desc: 'Comfortable movement'       },
  C: { color: '#fde68a', bg: 'rgba(253,230,138,.1)',   label: 'Acceptable',         desc: 'Minor flow disruption'      },
  D: { color: '#fbbf24', bg: 'rgba(251,191,36,.12)',   label: 'Uncomfortable',      desc: 'Significant restriction'    },
  E: { color: '#f97316', bg: 'rgba(249,115,22,.12)',   label: 'Very Restricted',    desc: 'Near capacity — monitor'    },
  F: { color: '#f87171', bg: 'rgba(248,113,113,.14)',  label: 'Crush Risk',         desc: 'EVACUATE / Reduce capacity' },
};

function updateLOS(los) {
  const row = document.getElementById('losRow');
  if (!los || !los.los) { row.style.display = 'none'; return; }
  row.style.display = 'flex';
  const cfg = LOS_CONFIG[los.los] || LOS_CONFIG.A;
  const badge = document.getElementById('losBadge');
  badge.textContent = los.los;
  badge.style.background = cfg.bg;
  badge.style.color = cfg.color;
  document.getElementById('losLabel').textContent = cfg.label;
  document.getElementById('losLabel').style.color = cfg.color;
  document.getElementById('losDesc').textContent  = cfg.desc;
  document.getElementById('losDensity').textContent = los.density_sqm?.toFixed(2) + ' p/m²';
}

// ── Adaptive threshold card (multi-signal) ────────────────────────────────────
function updateThreshCard(thresh, mode, mean, std, samples, signals) {
  if (thresh == null) return;
  document.getElementById('tcV').textContent = thresh;
  const badge = document.getElementById('tcBadge');
  if (mode === 'dynamic_v2') {
    badge.textContent = 'Multi-Signal · ' + samples + 'f';
    badge.className = 'tc-badge b-v2';
    const hasArea = signals?.area_cap;
    document.getElementById('tcSub').textContent =
      `μ=${mean}  σ=${std}  z=${currentZ}σ` +
      (hasArea ? `  |  Area cap: ${signals.area_cap}` : '') +
      `  |  prox·${signals?.prox_factor?.toFixed(2)}  zone·${signals?.zone_factor?.toFixed(2)}`;
  } else if (mode === 'dynamic') {
    badge.textContent = 'Dynamic · ' + samples + 'f';
    badge.className = 'tc-badge b-dy';
    document.getElementById('tcSub').textContent = 'μ=' + mean + '  σ=' + std + '  EMA + ' + currentZ + 'σ';
  } else {
    badge.textContent = 'Warm-up · ' + samples + 'f';
    badge.className = 'tc-badge b-wu';
    document.getElementById('tcSub').textContent = 'Building baseline — activates after 5 frames';
  }
  document.getElementById('barLbl').textContent = 'threshold: ' + thresh;

  if (signals && mode === 'dynamic_v2') {
    document.getElementById('signalStack').style.display = 'block';
    const maxSig = Math.max(signals.baseline || 0, signals.area_cap || 0, thresh, 1);
    function setSig(id, valId, val, maxV) {
      const el = document.getElementById(id);
      const vEl = document.getElementById(valId);
      if (el)  el.style.width = Math.min((val / maxV) * 100, 100) + '%';
      if (vEl) vEl.textContent = val != null ? Math.round(val) : '—';
    }
    setSig('sig-baseline', 'sigv-baseline', signals.baseline, maxSig);
    setSig('sig-area',     'sigv-area',     signals.area_cap, maxSig);
    setSig('sig-prox',     'sigv-prox',     Math.round((signals.proximity_score || 0) * 100), 100);
    setSig('sig-zone',     'sigv-zone',     Math.round((signals.zone_cv || 0) * 100), 100);
    setSig('sig-final',    'sigv-final',    thresh, maxSig);
  }
}

// ── Distance / proximity panel ────────────────────────────────────────────────
function updateDistancePanel(dist) {
  if (!dist) return;
  const ps     = dist.proximity_score || 0;
  const banner = document.getElementById('proxBanner');
  const label  = document.getElementById('proxLabel');
  const detail = document.getElementById('proxDetail');
  const scoreEl= document.getElementById('proxScore');
  scoreEl.textContent = Math.round(ps * 100) + '%';
  if (ps >= 0.65) {
    banner.className = 'prox-banner danger';
    label.textContent = 'DANGER: HIGH PROXIMITY'; label.style.color = 'var(--red)';
    detail.textContent = `${dist.violations} of ${dist.count} persons too close`;
    scoreEl.style.color = 'var(--red)';
  } else if (ps >= 0.30) {
    banner.className = 'prox-banner moderate';
    label.textContent = 'MODERATE CROWDING'; label.style.color = 'var(--amber)';
    detail.textContent = `${dist.violations} proximity violations detected`;
    scoreEl.style.color = 'var(--amber)';
  } else {
    banner.className = 'prox-banner safe';
    label.textContent = 'SAFE SPACING'; label.style.color = 'var(--green)';
    detail.textContent = 'All persons within safe distance';
    scoreEl.style.color = 'var(--green)';
  }
  document.getElementById('dViolations').textContent = dist.violations || 0;
  document.getElementById('dVioRatio').textContent   = Math.round((dist.proximity_violation_ratio || 0) * 100) + '%';
  document.getElementById('dMedianNN').textContent   = dist.median_nn_dist != null ? Math.round(dist.median_nn_dist) + 'px' : '—';
  document.getElementById('dSafeDist').textContent   = dist.min_safe_px   != null ? Math.round(dist.min_safe_px)   + 'px' : '—';
}

// ── Walkable area card (v3.1 — includes tier badge + SegFormer class list) ────
function updateWalkableCard(meta) {
  if (!meta) return;
  const wpct = meta.walkable_pct || 0;
  const opct = 100 - wpct;
  document.getElementById('walkableBar').style.width   = wpct + '%';
  document.getElementById('walkablePct').textContent   = wpct + '%';
  document.getElementById('obstacleBar').style.width   = opct + '%';
  document.getElementById('obstaclePct').textContent   = opct.toFixed(1) + '%';
  document.getElementById('walkableInfo').textContent  =
    `Walkable: ${wpct}% · Tier: ${meta.tier_used || 'unknown'} · Obstacles: ${meta.obstacle_count || 0}`;

  // Tier badge
  const tierEl = document.getElementById('tierBadge');
  if (tierEl) {
    const tier = meta.tier_used || 'heuristic';
    const tierConfig = {
      'segformer+yolo': { cls: 'tier-segformer-yolo', label: '✦ SegFormer + YOLO  (best)' },
      'segformer':      { cls: 'tier-segformer',       label: '✦ SegFormer only' },
      'yolo':           { cls: 'tier-yolo',            label: '◈ YOLO only' },
      'heuristic':      { cls: 'tier-heuristic',       label: '◇ Heuristic fallback' },
    };
    const cfg = tierConfig[tier] || tierConfig['heuristic'];
    tierEl.className = 'tier-badge ' + cfg.cls;
    tierEl.textContent = cfg.label;
  }

  // SegFormer semantic class breakdown
  const segList = document.getElementById('segClassList');
  if (segList) {
    const classes = meta.segment_classes || [];
    if (!classes.length) {
      segList.innerHTML = '<div style="color:var(--text3);font-size:11px;">SegFormer not active — no semantic data</div>';
    } else {
      segList.innerHTML = [...classes].sort((a, b) => b.pct - a.pct).slice(0, 12).map(cls => {
        const isW = cls.is_walkable;
        return `<div class="seg-class-item">
          <span class="${isW ? 'seg-walkable' : 'seg-nonwalkable'}" style="font-size:9px;flex-shrink:0;margin-right:4px;">${isW ? '▣' : '✕'}</span>
          <span class="seg-class-name">${cls.class_name}</span>
          <div class="seg-class-bar-wrap"><div class="seg-class-bar" style="width:${Math.min(cls.pct, 100)}%;background:${isW ? 'var(--teal)' : 'var(--red)'}"></div></div>
          <span class="seg-class-pct">${cls.pct}%</span>
        </div>`;
      }).join('');
    }
  }

  // YOLO obstacle list
  const ol  = document.getElementById('obstacleList');
  const obs = meta.obstacles_detected || [];
  if (!obs.length) {
    ol.innerHTML = '<div style="color:var(--text3);font-size:11px;">No YOLO obstacles detected</div>';
  } else {
    const catColors = { vehicle: 'var(--orange)', nature: 'var(--green)', furniture: 'var(--purple)', structure: 'var(--red)' };
    const catIcons  = { vehicle: '🚗', nature: '🌿', furniture: '🪑', structure: '🏗' };
    ol.innerHTML = obs.slice(0, 10).map(o => {
      const col  = catColors[o.category || 'structure'] || 'var(--red)';
      const icon = catIcons[o.category  || 'structure'] || '⬛';
      return `<div class="obstacle-item">
        <span style="display:flex;align-items:center;gap:5px;">
          <span style="font-size:12px;">${icon}</span>
          <span class="obs-class" style="color:${col};font-weight:600;">${o.class}</span>
        </span>
        <div style="display:flex;gap:5px;align-items:center;">
          <span style="font-family:var(--mono);font-size:9px;color:var(--text3)">${o.area_pct}%</span>
          <span class="obs-badge" style="background:${col}22;color:${col};border:1px solid ${col}44;">${Math.round(o.conf * 100)}%</span>
        </div>
      </div>`;
    }).join('');
  }
}

// ── NN distance histogram ─────────────────────────────────────────────────────
function renderNNHistogram() {
  if (!currentDistMeta) return;
  const dists = currentDistMeta.nn_distances || [];
  if (!dists.length) return;
  const canvas = document.getElementById('nnHistCanvas');
  const W = canvas.offsetWidth || 300, H = 80;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const minD   = Math.min(...dists);
  const maxD   = Math.max(...dists) * 1.1 || 100;
  const safePx = currentDistMeta.min_safe_px || 50;
  const BINS   = 20;
  const binW   = (maxD - minD) / BINS;
  const counts = new Array(BINS).fill(0);
  dists.forEach(d => { const bi = Math.min(Math.floor((d - minD) / Math.max(binW, 0.001)), BINS - 1); counts[bi]++; });
  const maxC = Math.max(...counts, 1);
  const pad  = { l: 20, r: 8, t: 8, b: 20 };
  const cH   = H - pad.t - pad.b;
  const cw   = W / BINS;

  counts.forEach((c, i) => {
    const x  = pad.l + i * cw;
    const bh = (c / maxC) * cH;
    const y  = pad.t + cH - bh;
    const binCenter = minD + (i + 0.5) * binW;
    ctx.fillStyle = binCenter < safePx ? 'rgba(248,113,113,.7)' : 'rgba(34,211,238,.5)';
    ctx.fillRect(x, y, cw - 1, bh);
  });

  const safePct = (safePx - minD) / Math.max(maxD - minD, 1);
  const lineX   = pad.l + safePct * (W - pad.l - pad.r);
  if (lineX > pad.l && lineX < W - pad.r) {
    ctx.strokeStyle = 'rgba(251,191,36,.8)'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(lineX, pad.t); ctx.lineTo(lineX, H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = '9px "JetBrains Mono",monospace'; ctx.fillStyle = 'rgba(251,191,36,.8)';
    ctx.fillText('safe', lineX + 3, pad.t + 9);
  }

  ctx.font = '8px "JetBrains Mono",monospace'; ctx.fillStyle = 'rgba(125,143,168,.5)'; ctx.textAlign = 'center';
  ctx.fillText(Math.round(minD) + 'px', pad.l + cw, H - 4);
  ctx.fillText(Math.round(maxD) + 'px', W - pad.r,  H - 4);
  ctx.textAlign = 'left';
}

// ── Walkable mask grid canvas ─────────────────────────────────────────────────
function renderWalkableMaskCanvas() {
  if (!currentWalkableMask) return;
  const canvas = document.getElementById('walkableMaskCanvas');
  const W = GRID_COLS * 6, H = GRID_ROWS * 6;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const v = currentWalkableMask[r]?.[c] ?? 0;
      ctx.fillStyle = v < 0.1
        ? 'rgba(248,113,113,.3)'
        : `rgb(${Math.round(v * 34)},${Math.round(v * 211)},${Math.round(v * 238)})`;
      ctx.fillRect(c * 6, r * 6, 6, 6);
    }
  }
}

// ── Zone map ──────────────────────────────────────────────────────────────────
function getDensityLevel(risk) {
  if (risk >= 0.75) return 'critical';
  if (risk >= 0.50) return 'high';
  if (risk >= 0.25) return 'medium';
  return 'low';
}

function setZoneMode(mode, btn) {
  zoneDisplayMode = mode;
  document.querySelectorAll('.zmc-btn').forEach(b => b.classList.remove('act'));
  btn.classList.add('act');
  renderZoneMap();
}

function renderZoneMap() {
  const canvas = document.getElementById('zoneMapCanvas');
  const rect = canvas.getBoundingClientRect();
  const W = Math.floor(rect.width) || canvas.offsetWidth || 300;
  const H = Math.floor(rect.height) || canvas.offsetHeight || 225;
  if (W < 10 || H < 10) { requestAnimationFrame(renderZoneMap); return; }
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#121a2a'; ctx.fillRect(0, 0, W, H);
  const cw = W / GRID_COLS, ch = H / GRID_ROWS;
  const t  = Date.now();

  if (!currentZoneData || !currentZoneData.length) {
    ctx.strokeStyle = 'rgba(255,255,255,.05)'; ctx.lineWidth = 0.5;
    for (let c = 0; c <= GRID_COLS; c++) { ctx.beginPath(); ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, H); ctx.stroke(); }
    for (let r = 0; r <= GRID_ROWS; r++) { ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(W, r * ch); ctx.stroke(); }
    ctx.fillStyle = 'rgba(125,143,168,.4)'; ctx.font = '10px "JetBrains Mono",monospace';
    ctx.textAlign = 'center'; ctx.fillText('Awaiting data…', W / 2, H / 2); ctx.textAlign = 'left';
    return;
  }

  currentZoneData.forEach(z => {
    const x = z.col * cw, y = z.row * ch, r = z.risk;
    if (!z.is_walkable) {
      ctx.fillStyle = 'rgba(248,113,113,.07)'; ctx.fillRect(x, y, cw, ch);
      ctx.strokeStyle = 'rgba(248,113,113,.15)'; ctx.lineWidth = 0.7;
      for (let d = 0; d < cw + ch; d += 7) { ctx.beginPath(); ctx.moveTo(x + d, y); ctx.lineTo(x, y + d); ctx.stroke(); }
      ctx.textAlign = 'center';
      ctx.font = `500 ${Math.max(6, Math.min(8, cw * 0.21))}px "JetBrains Mono",monospace`;
      ctx.fillStyle = 'rgba(248,113,113,.4)'; ctx.fillText('N/W', x + cw / 2, y + ch / 2 + 3);
      ctx.textAlign = 'left'; return;
    }

    let fillR = 0, fillG = 200, fillB = 0, alpha = 0.15 + r * 0.70;
    if (zoneDisplayMode === 'risk') {
      fillR = Math.round(r < 0.5 ? r * 2 * 255 : 255);
      fillG = Math.round(r < 0.5 ? 255 : (1 - r) * 2 * 255);
      fillB = 0;
    } else if (zoneDisplayMode === 'density') {
      const dn = z.density_norm || 0;
      fillR = Math.round(dn < 0.5 ? dn * 2 * 160 : 160 + dn * 95);
      fillG = Math.round(dn < 0.5 ? 200 : (1 - dn) * 200);
      fillB = Math.round((1 - dn) * 238); alpha = 0.12 + dn * 0.75;
    } else if (zoneDisplayMode === 'walkable') {
      const wf = z.walkable_frac || 0;
      fillR = Math.round((1 - wf) * 238); fillG = Math.round(wf * 200); fillB = Math.round(wf * 238);
      alpha = 0.3 + Math.abs(wf - 0.5) * 0.5;
    } else if (zoneDisplayMode === 'flow') {
      const mag = Math.min(z.magnitude || 0, 8) / 8, div = z.divergence || 0;
      fillR = div < -0.05 ? 255 : 50; fillG = div > 0.05 ? 200 : 50; fillB = Math.round(mag * 238);
      alpha = 0.15 + mag * 0.65;
    }

    ctx.fillStyle = `rgba(${fillR},${fillG},${fillB},${alpha})`; ctx.fillRect(x, y, cw, ch);

    if (r > 0.68 && zoneDisplayMode === 'risk') {
      const pulse = 0.55 + Math.sin(t / 300) * 0.45;
      ctx.strokeStyle = `rgba(248,113,113,${pulse})`; ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.75, y + 0.75, cw - 1.5, ch - 1.5);
    }
    if (selectedZone && selectedZone.row === z.row && selectedZone.col === z.col) {
      ctx.strokeStyle = 'rgba(34,211,238,0.9)'; ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, cw - 2, ch - 2);
    }

    const fontSize = Math.max(8, Math.min(13, cw * 0.32));
    ctx.font = `700 ${fontSize}px "Epilogue",sans-serif`; ctx.textAlign = 'center';
    const lx = x + cw / 2, ly = y + ch / 2 + (cw > 30 ? 2 : 4);
    ctx.fillStyle = 'rgba(0,0,0,.6)'; ctx.fillText(z.count, lx + 1, ly + 1);
    ctx.fillStyle = '#fff'; ctx.fillText(z.count, lx, ly);
    if (cw > 28 && ch > 24) {
      const dlblColors = { low: 'rgba(74,222,128,.7)', medium: 'rgba(251,191,36,.7)', high: 'rgba(251,146,60,.75)', critical: 'rgba(248,113,113,.8)' };
      ctx.font = `500 ${Math.max(6, Math.min(8, cw * 0.21))}px "JetBrains Mono",monospace`;
      ctx.fillStyle = dlblColors[getDensityLevel(r)];
      ctx.fillText(getDensityLevel(r).toUpperCase(), lx, ly + fontSize * 0.95);
    }
    ctx.textAlign = 'left';
    if (z.los && z.los !== '?' && cw > 28) {
      const losCols = { A: '#4ade80', B: '#86efac', C: '#fde68a', D: '#fbbf24', E: '#f97316', F: '#f87171' };
      ctx.font = `700 ${Math.max(6, Math.min(9, cw * 0.25))}px "JetBrains Mono",monospace`;
      ctx.fillStyle = losCols[z.los] || '#aaa'; ctx.fillText(z.los, x + 2, y + 10);
    }
  });

  ctx.strokeStyle = 'rgba(255,255,255,.06)'; ctx.lineWidth = 0.5;
  for (let c = 0; c <= GRID_COLS; c++) { ctx.beginPath(); ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, H); ctx.stroke(); }
  for (let r = 0; r <= GRID_ROWS; r++) { ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(W, r * ch); ctx.stroke(); }
  ctx.font = `${Math.max(7, Math.min(9, cw * 0.22))}px "JetBrains Mono",monospace`;
  ctx.fillStyle = 'rgba(255,255,255,.18)'; ctx.textAlign = 'center';
  for (let c = 0; c < GRID_COLS; c++) ctx.fillText(c + 1, c * cw + cw / 2, 9);
  ctx.textAlign = 'left';
  for (let r = 0; r < GRID_ROWS; r++) ctx.fillText(r + 1, 2, r * ch + ch / 2 + 4);

  if (currentZoneData && isPlaying) requestAnimationFrame(renderZoneMap);
}

function updateZoneStats(zones, agg) {
  if (!zones || !zones.length) return;
  document.getElementById('zHot').textContent      = zones.filter(z => z.risk > 0.68).length;
  document.getElementById('zPeak').textContent     = Math.max(...zones.map(z => z.count));
  document.getElementById('zCrush').textContent    = zones.filter(z => z.risk > 0.85).length;
  document.getElementById('zWalkable').textContent = agg?.walkable_zone_count || zones.filter(z => z.is_walkable).length;
  const sel = document.getElementById('zoneFilter');
  if (sel.options.length < 2) {
    for (let r = 0; r < GRID_ROWS; r++)
      for (let c = 0; c < GRID_COLS; c++) {
        const o = document.createElement('option');
        o.value = `${r}-${c}`; o.textContent = `Zone R${r + 1}·C${c + 1}`;
        sel.appendChild(o);
      }
  }
}

function zoneHover(e) {
  if (!currentZoneData) return;
  const canvas = document.getElementById('zoneMapCanvas'), rect = canvas.getBoundingClientRect();
  showZoneDetail(
    Math.floor((e.clientY - rect.top)  / rect.height * GRID_ROWS),
    Math.floor((e.clientX - rect.left) / rect.width  * GRID_COLS)
  );
}

function zoneClick(e) {
  if (!currentZoneData) return;
  const canvas = document.getElementById('zoneMapCanvas'), rect = canvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / rect.width  * GRID_COLS);
  const row = Math.floor((e.clientY - rect.top)  / rect.height * GRID_ROWS);
  selectedZone = (selectedZone && selectedZone.row === row && selectedZone.col === col) ? null : { row, col };
  showZoneDetail(row, col, true); renderZoneMap();
}

function showZoneDetail(row, col, pinned = false) {
  if (!currentZoneData) return;
  const z = currentZoneData.find(z => z.row === row && z.col === col);
  if (!z) return;
  if (!z.is_walkable) {
    document.getElementById('zoneDetail').innerHTML =
      `<div class="zd-title">ZONE R${row + 1}·C${col + 1}</div>
       <span class="density-badge density-nonwalkable">NON-WALKABLE</span>
       <div style="color:var(--text2);font-size:12px;margin-top:8px;line-height:1.6;">Covered by obstacles. Excluded from density and threshold calculations.</div>`;
    return;
  }
  const dlvl = getDensityLevel(z.risk);
  document.getElementById('zoneDetail').innerHTML =
    `<div class="zd-title">ZONE R${row + 1}·C${col + 1} ${pinned ? '<span style="color:var(--cyan)">(pinned)</span>' : ''}</div>
     <div style="margin-bottom:8px;display:flex;gap:6px;align-items:center;">
       <span class="density-badge density-${dlvl}">${dlvl}</span>
       ${z.los ? `<span style="font-family:var(--mono);font-size:11px;color:var(--text2)">Fruin LOS-${z.los}</span>` : ''}
     </div>
     <div class="zone-grid-info">
       <div class="zgi">Count: <span>${z.count}</span></div>
       <div class="zgi">Risk: <span>${(z.risk * 100).toFixed(0)}%</span></div>
       <div class="zgi">Walkable: <span>${(z.walkable_frac * 100).toFixed(0)}%</span></div>
       ${z.density_sqm != null
         ? `<div class="zgi">Density: <span>${z.density_sqm} p/m²</span></div>`
         : `<div class="zgi">Density: <span>uncalibrated</span></div>`}
       <div class="zgi">Flow Δx: <span>${z.dx.toFixed(2)}</span></div>
       <div class="zgi">Flow Δy: <span>${z.dy.toFixed(2)}</span></div>
       <div class="zgi" style="grid-column:1/-1">Divergence:
         <span style="color:${z.divergence < -0.05 ? 'var(--red)' : 'var(--green)'}">
           ${z.divergence.toFixed(3)} ${z.divergence < -0.05 ? '⚠ compression' : ''}
         </span>
       </div>
     </div>`;
}

function clearZoneHover() {
  if (selectedZone) return;
  document.getElementById('zoneDetail').innerHTML =
    '<div class="zd-title">ZONE DETAIL — hover or click a cell</div>' +
    '<div style="color:var(--text2);font-size:12.5px;line-height:1.6;">Zones clipped to walkable area. Non-walkable cells shown with hatching.</div>';
}

// ── History chart ─────────────────────────────────────────────────────────────
function setTimeRange(range, btn) {
  histTimeRangeValue = range;
  document.querySelectorAll('.hc-pill').forEach(b => b.classList.remove('act'));
  btn.classList.add('act');
  updateHistoryView();
}

function updateHistoryView() {
  if (!frameData.length) return;
  const fps = parseFloat(document.getElementById('fpsIn').value) || 2;
  let data = [...frameData];
  if (histTimeRangeValue !== 'all') {
    const secs = parseInt(histTimeRangeValue);
    const lastT = data[data.length - 1]?.frame / fps || 0;
    data = data.filter(fd => fd.frame / fps >= lastT - secs);
  }
  const counts   = data.map(f => f.count);
  const peak     = Math.max(...counts, 0);
  const avg      = counts.length ? Math.round(counts.reduce((a, b) => a + b, 0) / counts.length) : 0;
  const breaches = data.filter(f => f.count > (f.dynamic_threshold || Infinity)).length;
  const dur      = data.length ? fmt(data[data.length - 1].frame / fps - data[0].frame / fps) : '0:00';
  document.getElementById('hsPeak').textContent    = peak;
  document.getElementById('hsAvg').textContent     = avg;
  document.getElementById('hsBreaches').textContent= breaches;
  document.getElementById('hsDur').textContent     = dur;
  document.getElementById('sMax').textContent = Math.max(...frameData.map(f => f.count), 0);
  document.getElementById('sAvg').textContent = Math.round(frameData.reduce((s, f) => s + f.count, 0) / Math.max(frameData.length, 1));
  drawHistChart(data, fps);
  renderEventsList(data, fps);
}

function drawHistChart(data, fps) {
  const canvas = document.getElementById('histChartCanvas');
  const W = canvas.offsetWidth || 300, H = 160;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (data.length < 2) {
    ctx.fillStyle = 'rgba(125,143,168,.3)'; ctx.font = '11px "JetBrains Mono",monospace';
    ctx.textAlign = 'center'; ctx.fillText('Awaiting data…', W / 2, H / 2); ctx.textAlign = 'left'; return;
  }
  const counts   = data.map(f => f.count);
  const threshs  = data.map(f => f.dynamic_threshold || 0);
  const areaCaps = data.map(f => f.threshold_signals?.area_cap || null);
  const maxV = Math.max(...counts, ...threshs, ...areaCaps.filter(Boolean), 1);
  const pad  = { l: 28, r: 12, t: 14, b: 24 };
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b, n = data.length;
  const xOf = i => pad.l + (i / (n - 1)) * cW;
  const yOf = v => pad.t + cH - (v / maxV) * cH;

  ctx.strokeStyle = 'rgba(255,255,255,.05)'; ctx.lineWidth = 1;
  ctx.font = '9px "JetBrains Mono",monospace'; ctx.fillStyle = 'rgba(125,143,168,.5)'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = Math.round((maxV / 4) * i), y = yOf(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v, pad.l - 3, y + 3);
  }
  ctx.textAlign = 'center'; ctx.fillStyle = 'rgba(125,143,168,.4)';
  const xStep = Math.max(1, Math.floor(n / 5));
  for (let i = 0; i < n; i += xStep) ctx.fillText(fmt(data[i].frame / fps), xOf(i), H - pad.b + 13);
  ctx.textAlign = 'left';

  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + cH);
  grad.addColorStop(0, 'rgba(34,211,238,.25)'); grad.addColorStop(1, 'rgba(34,211,238,.02)');
  ctx.beginPath(); ctx.moveTo(xOf(0), yOf(counts[0]));
  for (let i = 1; i < n; i++) ctx.lineTo(xOf(i), yOf(counts[i]));
  ctx.lineTo(xOf(n - 1), pad.t + cH); ctx.lineTo(xOf(0), pad.t + cH); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  if (areaCaps.some(v => v != null)) {
    ctx.strokeStyle = 'rgba(45,212,191,.45)'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); let first = true;
    for (let i = 0; i < n; i++) { const ac = areaCaps[i]; if (ac == null) continue; first ? (ctx.moveTo(xOf(i), yOf(ac)), first = false) : ctx.lineTo(xOf(i), yOf(ac)); }
    ctx.stroke(); ctx.setLineDash([]);
  }

  ctx.strokeStyle = 'rgba(251,191,36,.4)'; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
  ctx.beginPath(); for (let i = 0; i < n; i++) { const y = yOf(threshs[i] || 0); i === 0 ? ctx.moveTo(xOf(0), y) : ctx.lineTo(xOf(i), y); }
  ctx.stroke(); ctx.setLineDash([]);

  const lineGrad = ctx.createLinearGradient(pad.l, 0, W - pad.r, 0);
  lineGrad.addColorStop(0, 'rgba(34,211,238,.95)'); lineGrad.addColorStop(1, 'rgba(232,121,160,.95)');
  ctx.strokeStyle = lineGrad; ctx.lineWidth = 2;
  ctx.beginPath(); for (let i = 0; i < n; i++) i === 0 ? ctx.moveTo(xOf(i), yOf(counts[i])) : ctx.lineTo(xOf(i), yOf(counts[i])); ctx.stroke();

  const peakIdx   = counts.indexOf(Math.max(...counts));
  const troughIdx = counts.indexOf(Math.min(...counts));
  [peakIdx, troughIdx].forEach(idx => {
    if (idx < 0) return;
    const x = xOf(idx), y = yOf(counts[idx]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = idx === peakIdx ? 'var(--pink)' : 'var(--green)'; ctx.fill();
    const lbl = String(counts[idx]);
    ctx.font = 'bold 10px "JetBrains Mono",monospace'; ctx.fillStyle = '#fff';
    const lx = clamp(x - ctx.measureText(lbl).width / 2, pad.l, W - pad.r - 20);
    const ly = y - 8; ctx.fillText(lbl, lx, ly < pad.t + 10 ? y + 16 : ly);
  });

  data.forEach((fd, i) => {
    if (fd.count > (fd.dynamic_threshold || Infinity)) {
      ctx.beginPath(); ctx.arc(xOf(i), yOf(fd.count), 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(248,113,113,.85)'; ctx.fill();
    }
  });
}

function renderEventsList(data, fps) {
  const el = document.getElementById('eventsList');
  const events = [];
  if (data.length) {
    const peakFd = data.reduce((b, f) => f.count > b.count ? f : b, data[0]);
    events.push({ label: 'Peak Count', count: peakFd.count, ts: fmt(peakFd.frame / fps), state: 'PEAK', col: 'var(--pink)', frame: peakFd.frame });
  }
  data.filter(f => f.count > (f.dynamic_threshold || Infinity)).forEach(fd =>
    events.push({ label: 'Threshold Breach', count: fd.count, ts: fmt(fd.frame / fps), state: 'BREACH', col: 'var(--red)', frame: fd.frame })
  );
  data.filter(f => f.behavior?.state === 'SURGE').forEach(fd =>
    events.push({ label: 'Surge Event', count: fd.count, ts: fmt(fd.frame / fps), state: 'SURGE', col: 'var(--red)', frame: fd.frame })
  );
  notableEvents.forEach(ev =>
    events.push({ label: 'Announcement: ' + ev.level, count: '—', ts: ev.time, state: ev.level.toUpperCase(),
      col: ev.level === 'critical' ? 'var(--red)' : ev.level === 'high' ? 'var(--orange)' : 'var(--amber)', frame: -1 })
  );
  if (!events.length) { el.innerHTML = '<div class="hist-empty">No notable events recorded</div>'; return; }
  events.sort((a, b) => b.frame - a.frame);
  el.innerHTML = events.slice(0, 30).map(ev =>
    `<div class="event-item" onclick="${ev.frame >= 0 ? `seekToFrame(${ev.frame})` : ''}" title="Click to seek">
      <div class="ev-dot" style="background:${ev.col}"></div>
      <div class="ev-body"><div class="ev-title">${ev.label}</div><div class="ev-meta">${ev.ts}${ev.state ? '  ·  ' + ev.state : ''}</div></div>
      <div class="ev-count" style="color:${ev.col}">${ev.count}</div>
    </div>`
  ).join('');
}

function exportCSV() {
  if (!frameData.length) { alert('No data to export.'); return; }
  const fps  = parseFloat(document.getElementById('fpsIn').value) || 2;
  const rows = [['Frame','Time','Count','Threshold','Mode','Behavior','Proximity Score','Violations','Walkable%','Obstacles','Tier']];
  frameData.forEach(fd => rows.push([
    fd.frame, fmt(fd.frame / fps), fd.count, fd.dynamic_threshold || '',
    fd.threshold_mode || '', fd.behavior?.state || '',
    fd.distance?.proximity_score || '', fd.distance?.violations || '',
    fd.walkable?.walkable_pct || '', fd.walkable?.obstacle_count || '',
    fd.walkable?.tier_used || '',
  ]));
  const csv  = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a    = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'crowd_data_v3_' + Date.now() + '.csv'; a.click();
  addLog('CSV exported: ' + frameData.length + ' rows');
}