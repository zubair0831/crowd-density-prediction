'use strict';
// ════════════════════════════════════════════════════════════════════════════
// app.js — Core application: state, video controls, processing pipeline
// ════════════════════════════════════════════════════════════════════════════

/* ══════════ CONFIG */
const API            = 'http://localhost:8000';
const THUMB_INTERVAL = 1, THUMB_W = 160, THUMB_H = 90;
const GRID_COLS = 8, GRID_ROWS = 6;

/* ══════════ STATE */
const vid = document.getElementById('video');
let file = null, ws = null, frameData = [], isPlaying = false;
let animId = null, isScrubbing = false;
let currentConf = 0.5, currentZ = 2.0, smoothWin = 7;
let thumbCache = [], thumbReady = false;
let lastFrameShown = null;
let calibMode = false, calibPts = [], pixelsPerMeter = null;
let currentZoneData = null, currentWalkableMask = null;
let currentDistMeta = null, currentWalkableMeta = null;
let histTimeRangeValue = 'all', selectedZone = null;
let notableEvents = [];
let zoneDisplayMode = 'risk';
const ALERT_HISTORY = [];

/* ══════════ REVIEW MODE */
let reviewMode = false;

/* ══════════ UTILS */
const fmt   = s  => { const m = Math.floor(s / 60), ss = Math.floor(s % 60); return m + ':' + String(ss).padStart(2, '0'); };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ── NULL-SAFE addLog ──────────────────────────────────────────────────────
// enterReviewMode() replaces procWrap.innerHTML, removing #log from the DOM.
// ws.onclose fires right after and used to call addLog → null.prepend() → crash.
// Now: always wrapped in try/catch, bails silently when #log is gone.
function addLog(m) {
  try {
    const l = document.getElementById('log');
    if (!l) return;
    const d = document.createElement('div');
    d.textContent = new Date().toLocaleTimeString() + '  ' + m;
    l.prepend(d);
  } catch (_) {}
}

function updConf(v)    { currentConf = +v; document.getElementById('confV').textContent = (+v).toFixed(2); }
function updZ(v)       { currentZ    = +v; document.getElementById('zV').textContent    = (+v).toFixed(2) + 'σ'; }
function updSmooth(v)  { smoothWin   = +v; document.getElementById('smV').textContent   = v; }
function hideAlert(id) { document.getElementById(id).classList.remove('show'); }

/* ══════════ VIDEO VISIBILITY ══════════
   showVideo / hideVideo are the ONLY two places that may change videoArea.
   hideVideo is called ONLY from resetAll(). Everything else calls showVideo(). */
function showVideo() {
  document.getElementById('videoArea').classList.add('show');
  document.getElementById('uploadBox').style.display = 'none';
}
function hideVideo() {
  document.getElementById('videoArea').classList.remove('show');
  document.getElementById('uploadBox').style.display = 'flex';
}

/* ══════════ TABS */
function setTab(name) {
  const names = ['live', 'distance', 'zones', 'history', 'settings'];
  document.querySelectorAll('.stab').forEach((b, i) => b.classList.toggle('act', names[i] === name));
  document.querySelectorAll('.tab-p').forEach(p => p.classList.remove('act'));
  document.getElementById('tab-' + name).classList.add('act');
  if (name === 'zones')    setTimeout(renderZoneMap, 50);
  if (name === 'history')  updateHistoryView();
  if (name === 'distance') { renderNNHistogram(); renderWalkableMaskCanvas(); }
}

/* ══════════ BACKEND STATUS */
(async () => {
  try {
    const d = await fetch(API + '/').then(r => r.json());
    const c = document.getElementById('backendChip');
    c.className = 'status-chip ok';
    const tiers = [];
    if (d.segformer) tiers.push('SegFormer✓');
    if (d.yolo)      tiers.push('YOLO✓');
    if (!d.segformer && !d.yolo) tiers.push('Heuristic');
    if (d.flow)      tiers.push('Flow✓');
    document.getElementById('backendTxt').textContent =
      d.model + ' · ' + d.device + ' · ' + tiers.join(' · ');
  } catch (_) { addLog('✗ Cannot reach ' + API); }
})();

/* ══════════ CANVAS SYNC */
function syncCanvas() {
  const cvs = document.getElementById('overlay');
  const w = vid.offsetWidth, h = vid.offsetHeight;
  if (!w || !h) return;
  if (cvs.width !== w || cvs.height !== h) {
    cvs.width = w; cvs.height = h;
    cvs.style.width = w + 'px'; cvs.style.height = h + 'px';
  }
}
window.addEventListener('resize', () => { syncCanvas(); renderZoneMap(); });

/* ══════════ VIDEO EVENTS */
vid.addEventListener('timeupdate', () => {
  if (!isScrubbing) updateTL(vid.currentTime);
  if (reviewMode && !isPlaying) paintFrame();
});
vid.addEventListener('ended', () => {
  isPlaying = false;
  document.getElementById('playBtn').textContent = '▶';
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  paintFrame();
});
vid.addEventListener('progress', () => {
  if (vid.buffered.length && vid.duration)
    document.getElementById('tlBuf').style.width =
      (vid.buffered.end(vid.buffered.length - 1) / vid.duration * 100) + '%';
});

function updateTL(t) {
  const p = (t / (vid.duration || 1)) * 100;
  document.getElementById('tlPlayed').style.width = p + '%';
  document.getElementById('tlHead').style.left    = p + '%';
  document.getElementById('curTime').textContent  = fmt(t);
}
function setVol(v)    { vid.volume = +v; vid.muted = (+v === 0); }
function toggleMute() {
  vid.muted = !vid.muted;
  document.getElementById('volSl').value = vid.muted ? 0 : vid.volume;
  document.getElementById('muteBtn').textContent = vid.muted ? '🔇' : '🔊';
}
function skip(dt) { vid.currentTime = clamp(vid.currentTime + dt, 0, vid.duration || 0); }
function togglePlay() {
  if (isPlaying) {
    vid.pause(); isPlaying = false;
    document.getElementById('playBtn').textContent = '▶';
    if (animId) { cancelAnimationFrame(animId); animId = null; }
  } else {
    vid.play(); isPlaying = true;
    document.getElementById('playBtn').textContent = '⏸';
    startDraw();
  }
}

/* ══════════ TIMELINE SCRUBBER */
const tlTrack  = document.getElementById('tlTrack');
const pxToTime = cx => clamp((cx - tlTrack.getBoundingClientRect().left) / tlTrack.getBoundingClientRect().width, 0, 1) * (vid.duration || 0);
const pxToPct  = cx => clamp((cx - tlTrack.getBoundingClientRect().left) / tlTrack.getBoundingClientRect().width, 0, 1) * 100;

function showThumb(cx) {
  if (!thumbCache.length) return;
  const t = pxToTime(cx), r = tlTrack.getBoundingClientRect();
  let best = thumbCache[0];
  for (const th of thumbCache) if (Math.abs(th.time - t) < Math.abs(best.time - t)) best = th;
  const pop  = document.getElementById('thumbPop');
  document.getElementById('thumbImg').src = best.dataUrl;
  document.getElementById('thumbTs').textContent = fmt(t);
  const safe = clamp(cx - r.left, THUMB_W / 2, r.width - THUMB_W / 2);
  pop.style.left = safe + 'px'; pop.classList.add('on');
  document.getElementById('tlHover').style.width = pxToPct(cx) + '%';
}
function hideThumb() {
  document.getElementById('thumbPop').classList.remove('on');
  document.getElementById('tlHover').style.width = '0%';
}
tlTrack.addEventListener('mousemove',  e => { showThumb(e.clientX); if (isScrubbing) { const t2 = pxToTime(e.clientX); vid.currentTime = t2; updateTL(t2); paintFrame(); } });
tlTrack.addEventListener('mouseleave', () => { if (!isScrubbing) hideThumb(); });
tlTrack.addEventListener('mousedown',  e => { isScrubbing = true; tlTrack.classList.add('drag'); const t2 = pxToTime(e.clientX); vid.currentTime = t2; updateTL(t2); showThumb(e.clientX); paintFrame(); });
window.addEventListener('mousemove',   e => { if (!isScrubbing) return; showThumb(e.clientX); vid.currentTime = pxToTime(e.clientX); updateTL(vid.currentTime); paintFrame(); });
window.addEventListener('mouseup',     () => { if (!isScrubbing) return; isScrubbing = false; tlTrack.classList.remove('drag'); hideThumb(); });

/* ══════════ THUMBNAILS */
async function generateThumbs() {
  thumbCache = []; thumbReady = false;
  if (!vid.duration) return;
  const dur = vid.duration, total = Math.ceil(dur / THUMB_INTERVAL) + 1;
  const off = document.createElement('canvas'); off.width = THUMB_W; off.height = THUMB_H;
  const ctx = off.getContext('2d');
  const wasPlaying = isPlaying, savedTime = vid.currentTime;
  if (wasPlaying) vid.pause();
  // Hide video + show spinner during thumbnail scrubbing — rapid seeks are visible
  vid.style.visibility = 'hidden';
  document.getElementById('captureOverlay').classList.add('active');
  document.getElementById('captureOverlay').querySelector('.capture-overlay-text').textContent = 'Generating thumbnails…';
  for (let i = 0; i <= total; i++) {
    const t2 = Math.min(i * THUMB_INTERVAL, dur - 0.01);
    vid.currentTime = t2;
    await new Promise(res => { const fn = () => { vid.removeEventListener('seeked', fn); res(); }; vid.addEventListener('seeked', fn); });
    ctx.drawImage(vid, 0, 0, THUMB_W, THUMB_H);
    thumbCache.push({ time: t2, dataUrl: off.toDataURL('image/jpeg', 0.6) });
  }
  vid.currentTime = savedTime;
  vid.style.visibility = '';
  document.getElementById('captureOverlay').classList.remove('active');
  if (wasPlaying) { vid.play(); isPlaying = true; document.getElementById('playBtn').textContent = '⏸'; }
  thumbReady = true;
  addLog('✓ ' + thumbCache.length + ' thumbnails ready');
}

/* ══════════ FILE INPUT */
document.getElementById('fInput').onchange = async function (e) {
  file = e.target.files[0]; if (!file) return;
  reviewMode = false;
  vid.src = URL.createObjectURL(file);
  showVideo();
  document.getElementById('procWrap').classList.add('show');
  // { once: true } is critical — prevents this handler re-firing when
  // enterReviewMode sets vid.currentTime = 0, which causes some browsers
  // to re-dispatch loadedmetadata, re-running generateThumbs() and making
  // the UI appear to "reload" after analysis completes.
  vid.addEventListener('loadedmetadata', async () => {
    document.getElementById('durTime').textContent = fmt(vid.duration);
    addLog(file.name + ' · ' + vid.videoWidth + '×' + vid.videoHeight + ' · ' + vid.duration.toFixed(1) + 's');
    requestAnimationFrame(syncCanvas);
    await generateThumbs();
  }, { once: true });
};

/* ══════════ DRAW LOOP */
function startDraw() {
  function loop() { if (!isPlaying) return; syncCanvas(); paintFrame(); animId = requestAnimationFrame(loop); }
  animId = requestAnimationFrame(loop);
}

function paintFrame() {
  try {
    const cvs = document.getElementById('overlay');
    const ctx = cvs.getContext('2d');
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    if (!frameData.length) return;
    const fps    = parseFloat(document.getElementById('fpsIn').value) || 2;
    const target = vid.currentTime * fps;
    let best = frameData[0], gap = Math.abs(best.frame - target);
    for (const fd of frameData) { const g = Math.abs(fd.frame - target); if (g < gap) { best = fd; gap = g; } }
    if (best !== lastFrameShown) { lastFrameShown = best; onFrameChange(best); }
    drawOverlay(ctx, cvs, best);
    if (reviewMode) updateReviewBadge(best);
  } catch (_) {}
}

/* ══════════ REVIEW MODE */
function enterReviewMode(completionMsg) {
  reviewMode = true;

  const procWrap = document.getElementById('procWrap');
  if (procWrap) {
    const peakBeh  = completionMsg.peak_behavior || 'NORMAL';
    const behColor = peakBeh === 'SURGE' ? 'var(--red)' : peakBeh === 'PRE_SURGE' ? 'var(--orange)' : 'var(--green)';
    const alerts   = completionMsg.alert_count;
    procWrap.innerHTML = `
      <div class="review-ribbon" id="reviewRibbon">
        <div class="review-ribbon-left">
          <span class="review-mode-badge">🔍 REVIEW MODE</span>
          <div class="review-stats">
            <span class="rs-item">📊 <b>${completionMsg.total_frames_processed}</b> frames</span>
            <span class="rs-item">👥 Peak <b>${completionMsg.peak_count}</b></span>
            <span class="rs-item">⟨ Avg <b>${completionMsg.avg_count}</b> ⟩</span>
            <span class="rs-item" style="color:${alerts > 0 ? 'var(--red)' : 'var(--green)'}">🚨 <b>${alerts}</b> alert${alerts !== 1 ? 's' : ''}</span>
            <span class="rs-item" style="color:${behColor}">● <b>${peakBeh}</b></span>
          </div>
        </div>
        <div class="review-ribbon-right">
          <div class="review-frame-badge" id="reviewFrameBadge">
            <span class="rfb-label">Scrub or play to inspect frames</span>
          </div>
        </div>
      </div>`;
    // Keep procWrap visible — it now shows the review ribbon
    procWrap.style.display = 'block';
    procWrap.classList.add('show');
  }

  // #log is now gone from the DOM — addLog() handles that safely.
  // Guarantee video stays visible no matter what.
  showVideo();
  document.getElementById('processBtn').disabled = false;
  // Do NOT reset vid.currentTime here — it causes a visible video jump that
  // looks like a "reload". The overlay paints from frameData[], not video time,
  // so the position does not matter. Let the video stay wherever extractFrames left it.
  syncCanvas();
  setTimeout(paintFrame, 100);
}

function updateReviewBadge(fd) {
  try {
    const badge = document.getElementById('reviewFrameBadge');
    if (!badge) return;
    const fps  = parseFloat(document.getElementById('fpsIn').value) || 2;
    const t    = fd.frame / fps;
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60).toString().padStart(2, '0');
    const isAlert = fd.alert || (fd.count > fd.dynamic_threshold);
    badge.innerHTML = `
      <span class="rfb-frame">Frame ${fd.frame}</span>
      <span class="rfb-time">${mins}:${secs}</span>
      <span class="rfb-count ${isAlert ? 'rfb-alert' : ''}">👥 ${fd.count}${isAlert ? ' 🚨' : ''}</span>
      <span class="rfb-thresh">T=${fd.dynamic_threshold || '—'}</span>
      <span class="rfb-beh">${fd.behavior?.state || ''}</span>`;
  } catch (_) {}
}

/* ══════════ ON FRAME CHANGE */
function onFrameChange(fd) {
  try {
    document.getElementById('sCurrent').textContent = fd.count;
    document.getElementById('sMax').textContent     = Math.max(...frameData.map(f => f.count), 0);
    document.getElementById('sAvg').textContent     = Math.round(frameData.reduce((s, f) => s + f.count, 0) / Math.max(frameData.length, 1));
    updateThreshCard(fd.dynamic_threshold, fd.threshold_mode, fd.threshold_mean, fd.threshold_std, fd.threshold_samples, fd.threshold_signals);
    updateBehaviorBanner(fd.behavior);
    updateLOS(fd.los);
    if (fd.zones)              { currentZoneData = fd.zones; renderZoneMap(); updateZoneStats(fd.zones, fd.zone_aggregate); }
    if (fd.walkable_mask_grid)   currentWalkableMask = fd.walkable_mask_grid;
    if (fd.distance)           { currentDistMeta = fd.distance; updateDistancePanel(fd.distance); }
    if (fd.walkable)           { currentWalkableMeta = fd.walkable; updateWalkableCard(fd.walkable); }
    if (fd.dynamic_threshold) {
      const pct = clamp(Math.round((fd.count / fd.dynamic_threshold) * 100), 0, 100);
      const bf  = document.getElementById('barFil');
      bf.style.width  = pct + '%';
      bf.className    = 'bar-fil' + (pct >= 100 ? ' danger' : pct >= 90 ? ' warn' : '');
      document.getElementById('capPct').textContent = pct + '%';
      document.getElementById('capLabel').style.color =
        pct >= 100 ? 'var(--red)' : pct >= 90 ? 'var(--orange)' : pct >= 75 ? 'var(--amber)' : 'var(--text2)';
      if (!reviewMode) SpeechSystem.evaluate(fd, fd.dynamic_threshold);
    }
    updateHistoryView(); renderNNHistogram(); renderWalkableMaskCanvas();
  } catch (_) {}
}

/* ══════════ FRAME EXTRACTION */
async function extractFrames(total, fps) {
  // Hide the video element during extraction — seeking through frames rapidly
  // is visible to the user and looks like the page is "reloading/restarting".
  vid.style.visibility = 'hidden';
  const cvs = document.getElementById('overlay');
  cvs.style.visibility = 'hidden';
  document.getElementById('captureOverlay').classList.add('active');
  document.getElementById('captureOverlay').querySelector('.capture-overlay-text').textContent = 'Capturing frames…';

  const MAX = 960; let w = vid.videoWidth, h = vid.videoHeight;
  if (Math.max(w, h) > MAX) { const s = MAX / Math.max(w, h); w = Math.floor(w * s); h = Math.floor(h * s); }
  const off = document.createElement('canvas'); off.width = w; off.height = h;
  const ctx = off.getContext('2d'); const out = [];
  for (let i = 0; i < total; i++) {
    vid.currentTime = i / fps;
    await new Promise(res => { const fn = () => { vid.removeEventListener('seeked', fn); res(); }; vid.addEventListener('seeked', fn); });
    ctx.drawImage(vid, 0, 0, w, h);
    out.push(off.toDataURL('image/jpeg', 0.75));
    if (i % 5 === 0 || i === total - 1) addLog('Captured ' + (i + 1) + '/' + total);
  }

  // Restore visibility — analysis overlay will paint on top of video from here
  vid.style.visibility = '';
  cvs.style.visibility = '';
  document.getElementById('captureOverlay').classList.remove('active');
  return out;
}

/* ══════════ PROCESS VIDEO */
async function processVideo() {
  if (!file) { alert('Load a video first.'); return; }

  reviewMode = false;
  vid.pause();
  if (isPlaying) { isPlaying = false; document.getElementById('playBtn').textContent = '▶'; }
  if (animId)    { cancelAnimationFrame(animId); animId = null; }

  frameData = []; lastFrameShown = null;
  currentZoneData = null; currentWalkableMask = null;
  currentDistMeta = null; currentWalkableMeta = null; notableEvents = [];

  document.getElementById('processBtn').disabled = true;

  // Rebuild the progress bar HTML (clears any review ribbon from a previous run)
  const procWrap = document.getElementById('procWrap');
  procWrap.innerHTML = `
    <div class="proc-hdr">
      <span>Analyzing frames</span>
      <span id="pctTxt">0%</span>
    </div>
    <div class="proc-trk"><div class="proc-fil" id="pctFil"></div></div>
    <div class="log" id="log"></div>`;
  procWrap.classList.add('show');

  showVideo(); // keep video visible throughout analysis

  document.getElementById('pctFil').style.width  = '0%';
  document.getElementById('pctTxt').textContent   = '0%';
  document.getElementById('tcV').textContent      = '—';
  document.getElementById('tcSub').textContent    = 'Accumulating baseline…';
  document.getElementById('tcBadge').className    = 'tc-badge b-wu';
  document.getElementById('tcBadge').textContent  = 'Warm-up';
  document.getElementById('barFil').style.width   = '0%';
  document.getElementById('signalStack').style.display = 'none';
  hideAlert('alertBanner'); hideAlert('predictAlert');

  const fps   = clamp(parseFloat(document.getElementById('fpsIn').value) || 2, 1, 10);
  const total = Math.max(1, Math.floor(vid.duration * fps));
  addLog('Starting v3.1 analysis — ' + total + ' frames @ ' + fps + ' FPS');

  const frames = await extractFrames(total, fps);
  addLog('Connecting to backend…');

  const sid   = 'sess_' + Date.now();
  const wsUrl = API.replace(/^https/, 'wss').replace(/^http/, 'ws') + '/ws/process-frames/' + sid;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    try {
      addLog('Connected');
      if (pixelsPerMeter)
        ws.send(JSON.stringify({ type: 'calibration', calibration: { pixels_per_meter: pixelsPerMeter } }));
      let i = 0;
      function next() {
        try {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          if (i >= frames.length) {
            ws.send(JSON.stringify({
              type: 'complete', fps,
              venue_name: document.getElementById('venueName').value || 'Unknown',
              name:       document.getElementById('venueName').value || 'Session ' + sid.slice(5, 13),
            }));
            return;
          }
          ws.send(JSON.stringify({
            type: 'frame', frame_number: i, frame_data: frames[i],
            total_frames: frames.length, confidence: currentConf, z_factor: currentZ,
          }));
          i++; setTimeout(next, 18);
        } catch (e) { addLog('Send error: ' + e.message); }
      }
      next();
    } catch (e) { addLog('onopen error: ' + e.message); }
  };

  // ── Every handler is wrapped in try/catch ────────────────────────────────
  // Any uncaught exception in a WS handler would stop JS mid-execution,
  // leaving the DOM in whatever partial state it was in — which could mean
  // videoArea hidden. The catch block calls showVideo() to guarantee recovery.

  ws.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);

      if (msg.type === 'result') {
        frameData.push(msg);
        document.getElementById('pctFil').style.width = msg.progress + '%';
        document.getElementById('pctTxt').textContent  = msg.progress + '%';
        const counts = frameData.map(f => f.count);
        document.getElementById('sCurrent').textContent = msg.count;
        document.getElementById('sMax').textContent     = Math.max(...counts);
        document.getElementById('sAvg').textContent     = Math.round(counts.reduce((a, b) => a + b, 0) / counts.length);
        updateThreshCard(msg.dynamic_threshold, msg.threshold_mode, msg.threshold_mean,
          msg.threshold_std, msg.threshold_samples, msg.threshold_signals);
        const pct = clamp(Math.round((msg.count / Math.max(msg.dynamic_threshold || 1, 1)) * 100), 0, 100);
        const bf  = document.getElementById('barFil');
        bf.style.width = pct + '%';
        bf.className   = 'bar-fil' + (pct >= 100 ? ' danger' : pct >= 90 ? ' warn' : '');
        document.getElementById('capPct').textContent = pct + '%';
        if (msg.zones)              { currentZoneData = msg.zones; renderZoneMap(); updateZoneStats(msg.zones, msg.zone_aggregate); }
        if (msg.walkable_mask_grid)   currentWalkableMask = msg.walkable_mask_grid;
        if (msg.behavior)             updateBehaviorBanner(msg.behavior);
        if (msg.los)                  updateLOS(msg.los);
        if (msg.distance)           { currentDistMeta = msg.distance; updateDistancePanel(msg.distance); }
        if (msg.walkable)           { currentWalkableMeta = msg.walkable; updateWalkableCard(msg.walkable); }
        if (msg.count > msg.dynamic_threshold) {
          ALERT_HISTORY.push({ time: Date.now(), count: msg.count, threshold: msg.dynamic_threshold });
          document.getElementById('alertBanner').classList.add('show');
          document.getElementById('alertTitle').textContent =
            msg.behavior?.state === 'SURGE' ? '🚨 SURGE EVENT DETECTED' : '⚠️ Density Threshold Exceeded';
          document.getElementById('alertMsg').textContent =
            msg.count + ' people detected — threshold exceeded (' + msg.dynamic_threshold + ')';
          document.getElementById('alertDetail').textContent = msg.threshold_signals
            ? `Baseline: ${msg.threshold_signals.baseline}  Area: ${msg.threshold_signals.area_cap || '—'}  Prox: ×${msg.threshold_signals.prox_factor?.toFixed(2)}  Zone: ×${msg.threshold_signals.zone_factor?.toFixed(2)}`
            : new Date().toLocaleTimeString();
        }
        SpeechSystem.evaluate(msg, msg.dynamic_threshold || 1);
        if (msg.frame % 5 === 0)
          addLog(`Frame ${msg.frame}: ${msg.count}p · T=${msg.dynamic_threshold} · prox=${(msg.distance?.proximity_score || 0).toFixed(2)} · ${msg.behavior?.state}`);

      } else if (msg.type === 'complete') {
        addLog('Done — ' + frameData.length + ' frames · peak=' + msg.peak_count + ' · ' + msg.peak_behavior);
        updateHistoryView();
        enterReviewMode(msg); // removes #log from DOM — addLog stays safe after this
        showVideo();          // re-assert visibility after DOM swap

      } else if (msg.type === 'error') {
        addLog('Backend error: ' + msg.message);
        if (!reviewMode) document.getElementById('processBtn').disabled = false;
      }
    } catch (err) {
      console.error('ws.onmessage exception:', err);
      showVideo(); // recovery: ensure video is never left hidden after a crash
    }
  };

  ws.onerror = () => {
    try {
      addLog('WebSocket error — is backend running at ' + API + '?');
      if (!reviewMode) document.getElementById('processBtn').disabled = false;
      showVideo();
    } catch (_) {}
  };

  // ws.onclose fires AFTER the complete handler runs enterReviewMode().
  // At that point #log is gone. addLog() is null-safe. showVideo() is idempotent.
  ws.onclose = () => {
    try {
      addLog('Connection closed');
      showVideo(); // always re-assert — costs nothing, prevents any lingering hide
    } catch (_) {}
  };
}

/* ══════════ REPORT */
function downloadReport() {
  if (!frameData.length) { alert('Run analysis first.'); return; }
  const counts = frameData.map(f => f.count);
  const report = {
    report_type:  'crowd_intelligence_incident_report_v3',
    generated_at: new Date().toISOString(),
    venue:        document.getElementById('venueName').value || 'Unknown Venue',
    calibration:  pixelsPerMeter ? { pixels_per_meter: pixelsPerMeter } : null,
    statistics: {
      total_frames: frameData.length, peak_count: Math.max(...counts),
      avg_count: Math.round(counts.reduce((a, b) => a + b, 0) / counts.length),
      alert_count: ALERT_HISTORY.length,
    },
    threshold_engine: 'adaptive_v2_multi_signal',
    announcement_log: SpeechSystem.logEntries,
    alert_timeline:   ALERT_HISTORY,
    proximity_summary: {
      peak_score:     Math.max(...frameData.map(f => f.distance?.proximity_score || 0)).toFixed(2),
      avg_violations: Math.round(frameData.reduce((s, f) => s + (f.distance?.violations || 0), 0) / Math.max(frameData.length, 1)),
    },
    walkable_summary: {
      avg_walkable_pct: Math.round(frameData.reduce((s, f) => s + (f.walkable?.walkable_pct || 100), 0) / Math.max(frameData.length, 1)),
      avg_obstacles:    Math.round(frameData.reduce((s, f) => s + (f.walkable?.obstacle_count || 0), 0) / Math.max(frameData.length, 1)),
    },
    recommendation: ALERT_HISTORY.length > 5
      ? 'CRITICAL: Multiple threshold breaches. Reduce capacity immediately.'
      : ALERT_HISTORY.length > 0 ? 'WARNING: Density anomalies detected. Review protocols.'
      : 'NOMINAL: No significant safety incidents detected.',
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'crowd_report_v3_' + Date.now() + '.json'; a.click();
  addLog('Report downloaded');
}

/* ══════════ CALIBRATION */
function enterCalibMode() {
  if (!vid.src || !vid.duration) { alert('Load a video first.'); return; }
  const W = document.getElementById('calibW').value || document.getElementById('calibW2').value;
  const H = document.getElementById('calibH').value || document.getElementById('calibH2').value;
  if (!W || !H) { alert('Enter real-world width and height (meters) first.'); return; }
  document.getElementById('calibW').value = W; document.getElementById('calibH').value = H;
  calibPts = []; calibMode = true;
  document.getElementById('calibOverlay').classList.add('show');
  document.getElementById('calibDotCount').textContent = 'Click point 1 of 4…';
  document.querySelectorAll('.calib-dot').forEach(d => d.remove());
  document.getElementById('videoWrap').addEventListener('click', handleCalibClick);
}
function handleCalibClick(e) {
  if (!calibMode) return;
  const wrap = document.getElementById('videoWrap'), rect = wrap.getBoundingClientRect();
  const px = (e.clientX - rect.left) / rect.width  * vid.videoWidth;
  const py = (e.clientY - rect.top)  / rect.height * vid.videoHeight;
  calibPts.push({ x: px, y: py, clientX: e.clientX - rect.left, clientY: e.clientY - rect.top });
  const dot = document.createElement('div'); dot.className = 'calib-dot';
  dot.style.left = (calibPts[calibPts.length - 1].clientX / rect.width  * 100) + '%';
  dot.style.top  = (calibPts[calibPts.length - 1].clientY / rect.height * 100) + '%';
  wrap.appendChild(dot);
  if (calibPts.length < 4)
    document.getElementById('calibDotCount').textContent = 'Click point ' + (calibPts.length + 1) + ' of 4…';
  else finishCalib();
}
function finishCalib() {
  const W = parseFloat(document.getElementById('calibW').value);
  const H = parseFloat(document.getElementById('calibH').value);
  const dxW = calibPts[1].x - calibPts[0].x, dyW = calibPts[1].y - calibPts[0].y;
  const dxH = calibPts[3].x - calibPts[0].x, dyH = calibPts[3].y - calibPts[0].y;
  const pxW = Math.sqrt(dxW * dxW + dyW * dyW), pxH = Math.sqrt(dxH * dxH + dyH * dyH);
  pixelsPerMeter = (pxW / W + pxH / H) / 2; calibMode = false;
  document.getElementById('calibOverlay').classList.remove('show');
  document.getElementById('videoWrap').removeEventListener('click', handleCalibClick);
  const msg = '✓ Calibrated: ' + pixelsPerMeter.toFixed(1) + ' px/m';
  document.getElementById('calibStatus').textContent = msg;
  document.getElementById('calibStatus').className   = 'calib-status ok';
  addLog(msg);
}
function cancelCalib() {
  calibMode = false; calibPts = [];
  document.getElementById('calibOverlay').classList.remove('show');
  document.getElementById('videoWrap').removeEventListener('click', handleCalibClick);
  document.querySelectorAll('.calib-dot').forEach(d => d.remove());
}

/* ══════════ SEEK */
function seekToFrame(frameNo) {
  const fps = parseFloat(document.getElementById('fpsIn').value) || 2;
  vid.currentTime = frameNo / fps; updateTL(vid.currentTime); paintFrame();
}

/* ══════════ RESET — only place vid.src is cleared and video is hidden */
function resetAll() {
  vid.pause();
  vid.src = ''; // THE only place vid.src is cleared

  if (ws && ws.readyState === WebSocket.OPEN) ws.close(); ws = null;
  if (animId) { cancelAnimationFrame(animId); animId = null; }
  window.speechSynthesis.cancel();

  reviewMode = false; file = null; frameData = []; isPlaying = false; lastFrameShown = null;
  thumbCache = []; thumbReady = false; currentZoneData = null; currentWalkableMask = null;
  currentDistMeta = null; currentWalkableMeta = null;
  calibMode = false; calibPts = []; pixelsPerMeter = null;
  ALERT_HISTORY.length = 0; notableEvents = []; selectedZone = null;
  SpeechSystem.lastLevel = null; SpeechSystem.lastAnnounceTime = 0; SpeechSystem.logEntries = [];

  document.querySelectorAll('.calib-dot').forEach(d => d.remove());
  hideVideo(); // THE only place video is hidden

  const procWrap = document.getElementById('procWrap');
  procWrap.innerHTML = `
    <div class="proc-hdr">
      <span>Analyzing frames</span>
      <span id="pctTxt">0%</span>
    </div>
    <div class="proc-trk"><div class="proc-fil" id="pctFil"></div></div>
    <div class="log" id="log"></div>`;
  procWrap.classList.remove('show');
  procWrap.style.display = '';

  document.getElementById('processBtn').disabled    = false;
  document.getElementById('playBtn').textContent    = '▶';
  ['sCurrent','sMax','sAvg'].forEach(id => document.getElementById(id).textContent = '0');
  document.getElementById('tcV').textContent        = '—';
  document.getElementById('tcSub').textContent      = 'Accumulating baseline…';
  document.getElementById('tcBadge').className      = 'tc-badge b-wu';
  document.getElementById('tcBadge').textContent    = 'Warm-up';
  document.getElementById('barFil').style.width     = '0%';
  document.getElementById('barLbl').textContent     = '—';
  document.getElementById('tlPlayed').style.width   = '0%';
  document.getElementById('tlHead').style.left      = '0%';
  document.getElementById('curTime').textContent    = '0:00';
  document.getElementById('durTime').textContent    = '0:00';
  document.getElementById('capPct').textContent     = '—%';
  document.getElementById('walkableInfo').textContent = 'Walkable area: awaiting data';
  document.getElementById('calibStatus').textContent  = 'Not calibrated — density shown as relative';
  document.getElementById('calibStatus').className    = 'calib-status';
  document.getElementById('losRow').style.display     = 'none';
  document.getElementById('behBanner').className      = 'beh-banner normal';
  document.getElementById('behState').textContent     = 'NORMAL';
  document.getElementById('behConf').textContent      = '—';
  document.getElementById('behDetail').textContent    = 'No anomalies detected';
  document.getElementById('announceLog').innerHTML    = '<div style="color:var(--text3);font-size:11.5px;padding:4px 0;">No announcements yet…</div>';
  document.getElementById('signalStack').style.display = 'none';
  document.getElementById('dViolations').textContent  = '0';
  document.getElementById('dVioRatio').textContent    = '0%';
  document.getElementById('dMedianNN').textContent    = '—';
  document.getElementById('dSafeDist').textContent    = '—';
  document.getElementById('proxBanner').className     = 'prox-banner safe';
  document.getElementById('proxLabel').textContent    = 'SAFE SPACING';
  document.getElementById('proxScore').textContent    = '0%';
  document.getElementById('walkablePct').textContent  = '—%';
  document.getElementById('obstaclePct').textContent  = '—%';
  document.getElementById('obstacleList').innerHTML   = '<div style="color:var(--text3);font-size:11px;">No obstacles detected yet</div>';
  const obsChk = document.getElementById('showObstacles'); if (obsChk) obsChk.checked = true;
  hideAlert('alertBanner'); hideAlert('predictAlert');
  renderZoneMap();
  addLog('Reset');
}

/* ══════════ INIT */
window.addEventListener('load', () => { setTimeout(renderZoneMap, 300); });
if ('speechSynthesis' in window) { window.speechSynthesis.onvoiceschanged = () => {}; window.speechSynthesis.getVoices(); }