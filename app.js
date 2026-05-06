'use strict';
// ════════════════════════════════════════════════════════════════════════════
// app.js  v3.2 — Core application: state, video controls, streaming pipeline
//
// Key changes:
//   • No pre-buffering — frames are captured and sent one at a time (O(1) memory)
//   • Homography calibration: 4-pt DLT solved client-side; pixel_points sent to
//     backend so it can also transform coordinates server-side
//   • Virtual tripwire: two-click line across video; net crossing count
//   • CUSUM banner integration
// ════════════════════════════════════════════════════════════════════════════

/* ══ CONFIG */
const API        = 'http://localhost:8000';
const THUMB_W    = 160, THUMB_H = 90, THUMB_INTERVAL = 1;
const GRID_COLS  = 8, GRID_ROWS = 6;

/* ══ STATE */
const vid = document.getElementById('video');
let file = null, ws = null, frameData = [], isPlaying = false;
let animId = null, isScrubbing = false;
let currentConf = 0.5, currentZ = 3.5, smoothWin = 7;
let thumbCache = [], thumbReady = false;
let lastFrameShown = null;
let calibMode = false, calibPts = [], pixelsPerMeter = null, homographyMatrix = null;
let currentZoneData = null, currentWalkableMask = null;
let currentDistMeta = null, currentWalkableMeta = null;
let histTimeRangeValue = 'all', selectedZone = null;
let notableEvents = [];
let zoneDisplayMode = 'risk';
const ALERT_HISTORY = [];

// Tripwire state
let tripwireMode = false, tripwire = null; // {a:[x,y], b:[x,y]}
let tripwireIn = 0, tripwireOut = 0;
let _prevSides = [];   // side of each person (last frame)

let reviewMode = false;

/* ══ UTILS */
const fmt   = s => { const m = Math.floor(s/60), ss = Math.floor(s%60); return m+':'+String(ss).padStart(2,'0'); };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function addLog(m) {
  try {
    const l = document.getElementById('log'); if (!l) return;
    const d = document.createElement('div');
    d.textContent = new Date().toLocaleTimeString()+'  '+m; l.prepend(d);
  } catch(_) {}
}
function updConf(v)   { currentConf = +v; document.getElementById('confV').textContent = (+v).toFixed(2); }
function updZ(v)      { currentZ    = +v; document.getElementById('zV').textContent    = (+v).toFixed(2)+'σ'; }
function updSmooth(v) { smoothWin   = +v; document.getElementById('smV').textContent   = v; }
function hideAlert(id){ document.getElementById(id).classList.remove('show'); }

/* ══ VIDEO VISIBILITY */
function showVideo() { document.getElementById('videoArea').classList.add('show'); document.getElementById('uploadBox').style.display='none'; }
function hideVideo() { document.getElementById('videoArea').classList.remove('show'); document.getElementById('uploadBox').style.display='flex'; }

/* ══ TABS */
function setTab(name) {
  const names = ['live','distance','zones','history','settings'];
  document.querySelectorAll('.stab').forEach((b,i) => b.classList.toggle('act', names[i]===name));
  document.querySelectorAll('.tab-p').forEach(p => p.classList.remove('act'));
  document.getElementById('tab-'+name).classList.add('act');
  if (name==='zones')    setTimeout(renderZoneMap, 50);
  if (name==='history')  updateHistoryView();
  if (name==='distance') { renderNNHistogram(); renderWalkableMaskCanvas(); }
}

/* ══ BACKEND STATUS */
(async () => {
  try {
    const d = await fetch(API+'/').then(r=>r.json());
    const c = document.getElementById('backendChip');
    c.className = 'status-chip ok';
    const tiers = [];
    if (d.segformer) tiers.push('SegFormer✓');
    if (d.yolo)      tiers.push('YOLO✓');
    if (!d.segformer && !d.yolo) tiers.push('Heuristic');
    if (d.flow)      tiers.push('Flow✓');
    document.getElementById('backendTxt').textContent = d.model+' · '+d.device+' · '+tiers.join(' · ');
  } catch(_) { addLog('✗ Cannot reach '+API); }
})();

/* ══ CANVAS SYNC */
function syncCanvas() {
  const cvs = document.getElementById('overlay');
  const w = vid.offsetWidth, h = vid.offsetHeight;
  if (!w || !h) return;
  if (cvs.width!==w || cvs.height!==h) { cvs.width=w; cvs.height=h; cvs.style.cssText=`width:${w}px;height:${h}px`; }
}
window.addEventListener('resize', () => { syncCanvas(); renderZoneMap(); if (reviewMode||frameData.length) setTimeout(paintFrame,16); });

/* ══ VIDEO EVENTS */
vid.addEventListener('timeupdate', () => { if (!isScrubbing) updateTL(vid.currentTime); if (reviewMode&&!isPlaying) paintFrame(); });
vid.addEventListener('ended', () => { isPlaying=false; document.getElementById('playBtn').textContent='▶'; if (animId){cancelAnimationFrame(animId);animId=null;} paintFrame(); });
vid.addEventListener('progress', () => {
  if (vid.buffered.length && vid.duration)
    document.getElementById('tlBuf').style.width = (vid.buffered.end(vid.buffered.length-1)/vid.duration*100)+'%';
});

function updateTL(t) {
  const p = (t/(vid.duration||1))*100;
  document.getElementById('tlPlayed').style.width = p+'%';
  document.getElementById('tlHead').style.left    = p+'%';
  document.getElementById('curTime').textContent  = fmt(t);
}
function setVol(v) { vid.volume=+v; vid.muted=(+v===0); }
function toggleMute() { vid.muted=!vid.muted; document.getElementById('volSl').value=vid.muted?0:vid.volume; document.getElementById('muteBtn').textContent=vid.muted?'🔇':'🔊'; }
function skip(dt) { vid.currentTime=clamp(vid.currentTime+dt,0,vid.duration||0); }
function togglePlay() {
  if (isPlaying) { vid.pause(); isPlaying=false; document.getElementById('playBtn').textContent='▶'; if(animId){cancelAnimationFrame(animId);animId=null;} }
  else { vid.play(); isPlaying=true; document.getElementById('playBtn').textContent='⏸'; startDraw(); }
}

/* ══ TIMELINE SCRUBBER */
const tlTrack = document.getElementById('tlTrack');
const pxToTime = cx => clamp((cx-tlTrack.getBoundingClientRect().left)/tlTrack.getBoundingClientRect().width,0,1)*(vid.duration||0);
const pxToPct  = cx => clamp((cx-tlTrack.getBoundingClientRect().left)/tlTrack.getBoundingClientRect().width,0,1)*100;

function showThumb(cx) {
  if (!thumbCache.length) return;
  const t=pxToTime(cx), r=tlTrack.getBoundingClientRect();
  let best=thumbCache[0];
  for (const th of thumbCache) if (Math.abs(th.time-t)<Math.abs(best.time-t)) best=th;
  const pop=document.getElementById('thumbPop');
  document.getElementById('thumbImg').src=best.dataUrl;
  document.getElementById('thumbTs').textContent=fmt(t);
  const safe=clamp(cx-r.left,THUMB_W/2,r.width-THUMB_W/2);
  pop.style.left=safe+'px'; pop.classList.add('on');
  document.getElementById('tlHover').style.width=pxToPct(cx)+'%';
}
function hideThumb() { document.getElementById('thumbPop').classList.remove('on'); document.getElementById('tlHover').style.width='0%'; }
tlTrack.addEventListener('mousemove',  e => { showThumb(e.clientX); if (isScrubbing){const t2=pxToTime(e.clientX);vid.currentTime=t2;updateTL(t2);paintFrame();} });
tlTrack.addEventListener('mouseleave', () => { if (!isScrubbing) hideThumb(); });
tlTrack.addEventListener('mousedown',  e => { isScrubbing=true; tlTrack.classList.add('drag'); const t2=pxToTime(e.clientX); vid.currentTime=t2; updateTL(t2); showThumb(e.clientX); paintFrame(); });
window.addEventListener('mousemove',   e => { if (!isScrubbing) return; showThumb(e.clientX); vid.currentTime=pxToTime(e.clientX); updateTL(vid.currentTime); paintFrame(); });
window.addEventListener('mouseup',     () => { if (!isScrubbing) return; isScrubbing=false; tlTrack.classList.remove('drag'); hideThumb(); });

/* ══ THUMBNAILS (small — 160×90, fine to buffer) */
async function generateThumbs() {
  thumbCache=[]; thumbReady=false;
  if (!vid.duration) return;
  const dur=vid.duration, total=Math.ceil(dur/THUMB_INTERVAL)+1;
  const off=document.createElement('canvas'); off.width=THUMB_W; off.height=THUMB_H;
  const ctx=off.getContext('2d');
  const wasPl=isPlaying, saved=vid.currentTime;
  if (wasPl) vid.pause();
  vid.style.visibility='hidden';
  document.getElementById('captureOverlay').classList.add('active');
  document.getElementById('captureOverlay').querySelector('.capture-overlay-text').textContent='Generating thumbnails…';
  for (let i=0; i<=total; i++) {
    const t2=Math.min(i*THUMB_INTERVAL,dur-0.01);
    vid.currentTime=t2;
    await new Promise(res => { const fn=()=>{vid.removeEventListener('seeked',fn);res();}; vid.addEventListener('seeked',fn); });
    ctx.drawImage(vid,0,0,THUMB_W,THUMB_H);
    thumbCache.push({time:t2, dataUrl:off.toDataURL('image/jpeg',0.6)});
  }
  vid.currentTime=saved; vid.style.visibility='';
  document.getElementById('captureOverlay').classList.remove('active');
  if (wasPl){vid.play();isPlaying=true;document.getElementById('playBtn').textContent='⏸';}
  thumbReady=true;
  addLog('✓ '+thumbCache.length+' thumbnails ready');
}

/* ══ FILE INPUT */
document.getElementById('fInput').onchange = async function(e) {
  file=e.target.files[0]; if (!file) return;
  reviewMode=false;
  vid.src=URL.createObjectURL(file); showVideo();
  document.getElementById('procWrap').classList.add('show');
  vid.addEventListener('loadedmetadata', async () => {
    document.getElementById('durTime').textContent=fmt(vid.duration);
    addLog(file.name+' · '+vid.videoWidth+'×'+vid.videoHeight+' · '+vid.duration.toFixed(1)+'s');
    requestAnimationFrame(syncCanvas);
    await generateThumbs();
  }, {once:true});
};

/* ══ DRAW LOOP */
function startDraw() {
  function loop(){ if(!isPlaying)return; syncCanvas(); paintFrame(); animId=requestAnimationFrame(loop); }
  animId=requestAnimationFrame(loop);
}
function paintFrame() {
  try {
    const cvs=document.getElementById('overlay'), ctx=cvs.getContext('2d');
    ctx.clearRect(0,0,cvs.width,cvs.height);
    if (!frameData.length) return;
    const fps=parseFloat(document.getElementById('fpsIn').value)||2;
    const target=vid.currentTime*fps;
    let best=frameData[0], gap=Math.abs(best.frame-target);
    for (const fd of frameData){const g=Math.abs(fd.frame-target);if(g<gap){best=fd;gap=g;}}
    if (best!==lastFrameShown){lastFrameShown=best;onFrameChange(best);}
    drawOverlay(ctx,cvs,best);
    if (reviewMode) updateReviewBadge(best);
  } catch(_){}
}

/* ══ REVIEW MODE */
function enterReviewMode(completionMsg) {
  reviewMode=true;
  document.getElementById('analysisProgress').style.display='none';
  const ribbonWrap=document.getElementById('reviewRibbonWrap');
  const peakBeh=completionMsg.peak_behavior||'NORMAL';
  const behColor=peakBeh==='SURGE'?'var(--red)':peakBeh==='PRE_SURGE'?'var(--orange)':'var(--green)';
  const alerts=completionMsg.alert_count;
  ribbonWrap.innerHTML=`
    <div class="review-ribbon" id="reviewRibbon">
      <div class="review-ribbon-left">
        <span class="review-mode-badge">🔍 REVIEW MODE</span>
        <div class="review-stats">
          <span class="rs-item">📊 <b>${completionMsg.total_frames_processed}</b> frames</span>
          <span class="rs-item">👥 Peak <b>${completionMsg.peak_count}</b></span>
          <span class="rs-item">⟨ Avg <b>${completionMsg.avg_count}</b> ⟩</span>
          <span class="rs-item" style="color:${alerts>0?'var(--red)':'var(--green)'}">🚨 <b>${alerts}</b> alert${alerts!==1?'s':''}</span>
          <span class="rs-item" style="color:${behColor}">● <b>${peakBeh}</b></span>
        </div>
      </div>
      <div class="review-ribbon-right">
        <div class="review-frame-badge" id="reviewFrameBadge"><span class="rfb-label">Scrub or play to inspect frames</span></div>
      </div>
    </div>`;
  ribbonWrap.style.display='block';
  const procWrap=document.getElementById('procWrap');
  procWrap.style.display='block'; procWrap.classList.add('show');
  showVideo(); document.getElementById('processBtn').disabled=false;
  syncCanvas(); setTimeout(paintFrame,80);
}

function updateReviewBadge(fd) {
  try {
    const badge=document.getElementById('reviewFrameBadge'); if(!badge) return;
    const fps=parseFloat(document.getElementById('fpsIn').value)||2;
    const t=fd.frame/fps, mins=Math.floor(t/60), secs=Math.floor(t%60).toString().padStart(2,'0');
    const isAlert=fd.alert||(fd.count>fd.dynamic_threshold);
    badge.innerHTML=`<span class="rfb-frame">Frame ${fd.frame}</span><span class="rfb-time">${mins}:${secs}</span>
      <span class="rfb-count ${isAlert?'rfb-alert':''}">👥 ${fd.count}${isAlert?' 🚨':''}</span>
      <span class="rfb-thresh">T=${fd.dynamic_threshold||'—'}</span>
      <span class="rfb-beh">${fd.behavior?.state||''}</span>`;
  } catch(_){}
}

/* ══ ON FRAME CHANGE */
function onFrameChange(fd) {
  try {
    document.getElementById('sCurrent').textContent = fd.count;
    document.getElementById('sMax').textContent     = Math.max(...frameData.map(f=>f.count),0);
    document.getElementById('sAvg').textContent     = Math.round(frameData.reduce((s,f)=>s+f.count,0)/Math.max(frameData.length,1));
    updateThreshCard(fd.dynamic_threshold,fd.threshold_mode,fd.threshold_mean,fd.threshold_std,fd.threshold_samples,fd.threshold_signals);
    updateBehaviorBanner(fd.behavior);
    updateLOS(fd.los);
    if (fd.zones){currentZoneData=fd.zones;renderZoneMap();updateZoneStats(fd.zones,fd.zone_aggregate);}
    if (fd.walkable_mask_grid) currentWalkableMask=fd.walkable_mask_grid;
    if (fd.distance){currentDistMeta=fd.distance;updateDistancePanel(fd.distance);}
    if (fd.walkable){currentWalkableMeta=fd.walkable;updateWalkableCard(fd.walkable);}
    if (fd.cusum)    updateCUSUMBadge(fd.cusum);
    if (fd.trajectories) updateTrajectoryStats(fd.trajectories);
    // Tripwire crossing update
    if (fd.coordinates && tripwire) processTripwire(fd.coordinates, fd.original_size||[1920,1080]);
    if (fd.dynamic_threshold) {
      const pct=clamp(Math.round((fd.count/fd.dynamic_threshold)*100),0,100);
      const bf=document.getElementById('barFil');
      bf.style.width=pct+'%';
      bf.className='bar-fil'+(pct>=100?' danger':pct>=92?' warn':'');
      document.getElementById('capPct').textContent=pct+'%';
    }
    updateThreshCard(fd.dynamic_threshold,fd.threshold_mode,fd.threshold_mean,fd.threshold_std,fd.threshold_samples,fd.threshold_signals);
  } catch(_){}
}

/* ══ TRIPWIRE ══════════════════════════════════════════════════════════════ */
function enterTripwireMode() {
  if (!vid.src||!vid.duration){alert('Load a video first.');return;}
  tripwireMode=true; tripwire=null; tripwireIn=0; tripwireOut=0; _prevSides=[];
  document.getElementById('twStatus').textContent='Click two points on video to draw line…';
  document.getElementById('videoWrap').addEventListener('click',_twClick);
  document.getElementById('twBtn').textContent='Cancel tripwire';
  document.getElementById('twBtn').onclick=cancelTripwire;
}
function cancelTripwire() {
  tripwireMode=false; tripwire=null; _prevSides=[]; tripwireIn=0; tripwireOut=0;
  document.getElementById('videoWrap').removeEventListener('click',_twClick);
  document.getElementById('twStatus').textContent='No tripwire set';
  document.getElementById('twBtn').textContent='Draw tripwire';
  document.getElementById('twBtn').onclick=enterTripwireMode;
  document.getElementById('twCounts').textContent='';
  paintFrame();
}
let _twPts=[];
function _twClick(e) {
  if (!tripwireMode) return;
  const wrap=document.getElementById('videoWrap'), rect=wrap.getBoundingClientRect();
  const nx=(e.clientX-rect.left)/rect.width, ny=(e.clientY-rect.top)/rect.height;
  _twPts.push([nx,ny]);
  if (_twPts.length>=2) {
    tripwire={a:_twPts[0],b:_twPts[1]};
    _twPts=[]; tripwireMode=false;
    document.getElementById('videoWrap').removeEventListener('click',_twClick);
    document.getElementById('twStatus').textContent='Tripwire active';
    paintFrame();
  } else {
    document.getElementById('twStatus').textContent='Click second point…';
  }
}

function _tripSide(px, py, ax, ay, bx, by) {
  // Signed cross product: positive = left side, negative = right side
  return (bx-ax)*(py-ay)-(by-ay)*(px-ax);
}

function processTripwire(coords, origSize) {
  if (!tripwire) return;
  const [ow,oh]=origSize;
  const {a,b}=tripwire;
  // a and b are normalised [0,1]; coords are in pixel space [0,ow]×[0,oh]
  const ax=a[0]*ow, ay=a[1]*oh, bx=b[0]*ow, by=b[1]*oh;
  const sides=coords.map(([cx,cy])=>_tripSide(cx,cy,ax,ay,bx,by)>=0?1:-1);

  if (_prevSides.length===coords.length) {
    for (let i=0;i<sides.length;i++) {
      if (_prevSides[i]!==0 && sides[i]!==0 && _prevSides[i]!==sides[i]) {
        if (sides[i]>0) tripwireIn++;
        else            tripwireOut++;
      }
    }
  }
  _prevSides=sides;
  document.getElementById('twCounts').textContent=
    `↓ In: ${tripwireIn}  ↑ Out: ${tripwireOut}  Net: ${tripwireIn-tripwireOut}`;
}

/* ══ HOMOGRAPHY — client-side 4-point DLT ═════════════════════════════════ */
function _gaussSolve(A, b) {
  const n=A.length, M=A.map((r,i)=>[...r,b[i]]);
  for (let c=0;c<n;c++) {
    let mx=c;
    for (let r=c+1;r<n;r++) if (Math.abs(M[r][c])>Math.abs(M[mx][c])) mx=r;
    [M[c],M[mx]]=[M[mx],M[c]];
    for (let r=c+1;r<n;r++){const f=M[r][c]/M[c][c];for(let k=c;k<=n;k++)M[r][k]-=f*M[c][k];}
  }
  const x=new Array(n).fill(0);
  for (let r=n-1;r>=0;r--){x[r]=M[r][n];for(let k=r+1;k<n;k++)x[r]-=M[r][k]*x[k];x[r]/=M[r][r];}
  return x;
}

function computeHomography(srcPts, dstPts) {
  // srcPts, dstPts: 4×[x,y]; solves 8×8 system with h22=1 fixed
  const A=[], b=[];
  for (let i=0;i<4;i++){
    const [sx,sy]=srcPts[i],[dx,dy]=dstPts[i];
    A.push([-sx,-sy,-1,  0,  0, 0, sx*dx, sy*dx]); b.push(-dx);
    A.push([  0,  0, 0,-sx,-sy,-1, sx*dy, sy*dy]); b.push(-dy);
  }
  return [..._gaussSolve(A,b),1];  // h22=1
}

function applyH(H,x,y) {
  const w=H[6]*x+H[7]*y+H[8];
  return [(H[0]*x+H[1]*y+H[2])/w,(H[3]*x+H[4]*y+H[5])/w];
}

/* ══ CALIBRATION */
function enterCalibMode() {
  if (!vid.src||!vid.duration){alert('Load a video first.');return;}
  const W=document.getElementById('calibW').value||document.getElementById('calibW2').value;
  const H=document.getElementById('calibH').value||document.getElementById('calibH2').value;
  if (!W||!H){alert('Enter real-world width and height (metres) first.');return;}
  document.getElementById('calibW').value=W;
  calibPts=[]; calibMode=true;
  document.getElementById('calibOverlay').classList.add('show');
  document.getElementById('calibDotCount').textContent='Click corner 1 of 4 (top-left)…';
  document.querySelectorAll('.calib-dot').forEach(d=>d.remove());
  document.getElementById('videoWrap').addEventListener('click',handleCalibClick);
}
function handleCalibClick(e) {
  if (!calibMode) return;
  const wrap=document.getElementById('videoWrap'), rect=wrap.getBoundingClientRect();
  const px=(e.clientX-rect.left)/rect.width*vid.videoWidth;
  const py=(e.clientY-rect.top) /rect.height*vid.videoHeight;
  calibPts.push({x:px,y:py,clientX:e.clientX-rect.left,clientY:e.clientY-rect.top});
  const labels=['top-left','top-right','bottom-right','bottom-left'];
  const dot=document.createElement('div'); dot.className='calib-dot';
  dot.style.left=(calibPts[calibPts.length-1].clientX/rect.width*100)+'%';
  dot.style.top =(calibPts[calibPts.length-1].clientY/rect.height*100)+'%';
  wrap.appendChild(dot);
  if (calibPts.length<4)
    document.getElementById('calibDotCount').textContent='Click corner '+(calibPts.length+1)+' of 4 ('+labels[calibPts.length]+')…';
  else finishCalib();
}
function finishCalib() {
  const W=parseFloat(document.getElementById('calibW').value);
  const H=parseFloat(document.getElementById('calibH').value);

  // Pixel points of 4 corners
  const srcPts=calibPts.map(p=>[p.x,p.y]);
  // Corresponding world points (TL, TR, BR, BL)
  const dstPts=[[0,0],[W,0],[W,H],[0,H]];

  // Client-side homography — used for display and sent to backend
  homographyMatrix = computeHomography(srcPts, dstPts);

  // Legacy px/m (mean of two edge lengths) — still useful for uncalibrated fallback
  const dx0=calibPts[1].x-calibPts[0].x, dy0=calibPts[1].y-calibPts[0].y;
  const dx1=calibPts[3].x-calibPts[0].x, dy1=calibPts[3].y-calibPts[0].y;
  pixelsPerMeter=(Math.sqrt(dx0*dx0+dy0*dy0)/W + Math.sqrt(dx1*dx1+dy1*dy1)/H)/2;

  calibMode=false;
  document.getElementById('calibOverlay').classList.remove('show');
  document.getElementById('videoWrap').removeEventListener('click',handleCalibClick);
  const msg='✓ Calibrated: '+pixelsPerMeter.toFixed(1)+' px/m (homography ready)';
  document.getElementById('calibStatus').textContent=msg;
  document.getElementById('calibStatus').className='calib-status ok';
  addLog(msg);
}
function cancelCalib() {
  calibMode=false; calibPts=[];
  document.getElementById('calibOverlay').classList.remove('show');
  document.getElementById('videoWrap').removeEventListener('click',handleCalibClick);
  document.querySelectorAll('.calib-dot').forEach(d=>d.remove());
}

/* ══ SEEK */
function seekToFrame(frameNo) {
  const fps=parseFloat(document.getElementById('fpsIn').value)||2;
  vid.currentTime=frameNo/fps; updateTL(vid.currentTime); paintFrame();
}

/* ══ CUSUM BADGE */
function updateCUSUMBadge(cusum) {
  try {
    const el=document.getElementById('cusumBadge'); if(!el) return;
    if (cusum.mode==='warmup'){el.textContent='CUSUM warm-up';el.className='cusum-badge warmup';return;}
    if (cusum.alert){
      el.textContent='CUSUM: '+cusum.direction.toUpperCase()+' detected';
      el.className='cusum-badge alert';
    } else {
      el.textContent='CUSUM OK  S⁺='+cusum.S_pos+'  z='+cusum.z;
      el.className='cusum-badge ok';
    }
  } catch(_){}
}

/* ══ TRAJECTORY STATS */
function updateTrajectoryStats(tracks) {
  try {
    const el=document.getElementById('trajStats'); if(!el||!tracks) return;
    const n=tracks.length;
    if (!n){el.textContent='No tracks';return;}
    const speeds=tracks.map(t=>t.speed);
    const avg=speeds.reduce((a,b)=>a+b,0)/n;
    const fast=speeds.filter(s=>s>15).length;
    el.textContent=`${n} tracks · avg vel ${avg.toFixed(1)}px/f · ${fast} fast-moving`;
  } catch(_){}
}

/* ══ STREAMING VIDEO ANALYSIS ══════════════════════════════════════════════
   v3.2: frames are captured and sent one-at-a-time — O(1) browser memory.
   No pre-buffering regardless of video length or FPS setting.
 ════════════════════════════════════════════════════════════════════════════ */
async function processVideo() {
  if (!file){alert('Load a video first.');return;}

  reviewMode=false; vid.pause();
  if (isPlaying){isPlaying=false;document.getElementById('playBtn').textContent='▶';}
  if (animId){cancelAnimationFrame(animId);animId=null;}

  frameData=[]; lastFrameShown=null;
  currentZoneData=null; currentWalkableMask=null;
  currentDistMeta=null; currentWalkableMeta=null; notableEvents=[];
  tripwireIn=0; tripwireOut=0; _prevSides=[];

  document.getElementById('processBtn').disabled=true;
  document.getElementById('reviewRibbonWrap').style.display='none';
  document.getElementById('reviewRibbonWrap').innerHTML='';
  document.getElementById('analysisProgress').style.display='';
  document.getElementById('log').innerHTML='';
  document.getElementById('pctFil').style.width='0%';
  document.getElementById('pctTxt').textContent='0%';
  document.getElementById('procWrap').classList.add('show');
  showVideo();
  hideAlert('alertBanner'); hideAlert('predictAlert');

  const fps   = clamp(parseFloat(document.getElementById('fpsIn').value)||2, 1, 10);
  const total = Math.max(1, Math.floor(vid.duration*fps));
  addLog('Starting v3.2 streaming analysis — '+total+' frames @ '+fps+' FPS');

  // Reusable off-screen capture canvas (full resolution, single allocation)
  const cap=document.createElement('canvas');
  cap.width=vid.videoWidth; cap.height=vid.videoHeight;
  const capCtx=cap.getContext('2d');

  const sid  = 'sess_'+Date.now();
  const wsUrl= API.replace(/^https/,'wss').replace(/^http/,'ws')+'/ws/process-frames/'+sid;
  ws=new WebSocket(wsUrl);

  // pendingResolve: resolves the current await-for-result promise
  let pendingResolve=null;

  ws.onmessage=e=>{
    try {
      const msg=JSON.parse(e.data);
      if (msg.type==='result'){
        // Accumulate result for review mode
        frameData.push(msg);

        // Real-time UI update
        document.getElementById('sCurrent').textContent=msg.count;
        document.getElementById('sMax').textContent=Math.max(...frameData.map(f=>f.count));
        document.getElementById('sAvg').textContent=Math.round(frameData.reduce((s,f)=>s+f.count,0)/frameData.length);
        updateThreshCard(msg.dynamic_threshold,msg.threshold_mode,msg.threshold_mean,msg.threshold_std,msg.threshold_samples,msg.threshold_signals);
        const pct=clamp(Math.round((msg.count/Math.max(msg.dynamic_threshold||1,1))*100),0,100);
        const bf=document.getElementById('barFil');
        bf.style.width=pct+'%';
        bf.className='bar-fil'+(pct>=100?' danger':pct>=92?' warn':'');
        document.getElementById('capPct').textContent=pct+'%';
        if (msg.zones){currentZoneData=msg.zones;renderZoneMap();updateZoneStats(msg.zones,msg.zone_aggregate);}
        if (msg.walkable_mask_grid) currentWalkableMask=msg.walkable_mask_grid;
        if (msg.behavior)  updateBehaviorBanner(msg.behavior);
        if (msg.los)       updateLOS(msg.los);
        if (msg.distance){ currentDistMeta=msg.distance;updateDistancePanel(msg.distance);}
        if (msg.walkable){ currentWalkableMeta=msg.walkable;updateWalkableCard(msg.walkable);}
        if (msg.cusum)     updateCUSUMBadge(msg.cusum);
        if (msg.trajectories) updateTrajectoryStats(msg.trajectories);
        if (msg.coordinates&&tripwire) processTripwire(msg.coordinates,msg.original_size||[1920,1080]);
        if (msg.count>msg.dynamic_threshold){
          ALERT_HISTORY.push({time:Date.now(),count:msg.count,threshold:msg.dynamic_threshold});
          document.getElementById('alertBanner').classList.add('show');
          document.getElementById('alertTitle').textContent=
            msg.behavior?.state==='SURGE'?'🚨 SURGE EVENT DETECTED':'⚠️ Density Threshold Exceeded';
          document.getElementById('alertMsg').textContent=
            msg.count+' people detected — threshold exceeded ('+msg.dynamic_threshold+')';
        }
        SpeechSystem.evaluate(msg,msg.dynamic_threshold||1);
        if (msg.frame%5===0)
          addLog(`Frame ${msg.frame}: ${msg.count}p · T=${msg.dynamic_threshold} · ${msg.behavior?.state} · cusum=${msg.cusum?.alert?'⚠':'OK'}`);

        // Resolve the await in the frame loop
        if (pendingResolve){const r=pendingResolve;pendingResolve=null;r(msg);}

      } else if (msg.type==='calibration_ack'){
        if (msg.H_matrix) addLog('✓ Server H-matrix received');
        if (pendingResolve){const r=pendingResolve;pendingResolve=null;r(msg);}

      } else if (msg.type==='complete'){
        addLog('Done — '+frameData.length+' frames · peak='+msg.peak_count+' · '+msg.peak_behavior);
        updateHistoryView(); enterReviewMode(msg); showVideo();

      } else if (msg.type==='error'){
        addLog('Backend error: '+msg.message);
        document.getElementById('processBtn').disabled=false;
        if (pendingResolve){const r=pendingResolve;pendingResolve=null;r(null);}
      }
    } catch(err){console.error('ws.onmessage:',err);showVideo();}
  };
  ws.onerror=()=>{addLog('WebSocket error — is backend running?');document.getElementById('processBtn').disabled=false;showVideo();};
  ws.onclose=()=>{addLog('Connection closed');showVideo();};

  // Wait for connection
  await new Promise((res,rej)=>{ws.onopen=res;setTimeout(()=>rej(new Error('WS timeout')),10000);});
  addLog('Connected — streaming '+total+' frames');

  // Send calibration (with pixel points for server-side homography)
  if (calibPts.length===4||homographyMatrix) {
    const calMsg={
      type:'calibration',
      calibration:{
        pixels_per_meter: pixelsPerMeter,
        H_matrix:         homographyMatrix,
        pixel_points:     calibPts.length===4?calibPts.map(p=>[p.x,p.y]):null,
        world_width:      parseFloat(document.getElementById('calibW').value)||1,
        world_height:     parseFloat(document.getElementById('calibH').value)||1,
      }
    };
    ws.send(JSON.stringify(calMsg));
    // Wait for ack (or timeout 2s)
    await Promise.race([
      new Promise(res=>{pendingResolve=res;}),
      new Promise(res=>setTimeout(res,2000)),
    ]);
  }

  // ── MAIN FRAME LOOP ──────────────────────────────────────────────────────
  // Seek → capture → send → await result, one frame at a time.
  // Browser never holds more than one raw JPEG in memory.

  for (let i=0; i<total; i++) {
    if (!ws||ws.readyState!==WebSocket.OPEN) break;

    // Seek video to frame timestamp
    await new Promise(res=>{
      const fn=()=>{vid.removeEventListener('seeked',fn);res();};
      vid.addEventListener('seeked',fn);
      vid.currentTime=i/fps;
    });

    // Capture full-resolution frame
    capCtx.drawImage(vid,0,0,cap.width,cap.height);
    const dataUrl=cap.toDataURL('image/jpeg',0.82);

    // Update progress bar immediately
    document.getElementById('pctFil').style.width=Math.round(i/total*100)+'%';
    document.getElementById('pctTxt').textContent=Math.round(i/total*100)+'%';

    // Send frame and wait for result (backpressure — prevents OOM on slow backend)
    const sent=new Promise(res=>{
      pendingResolve=res;
      ws.send(JSON.stringify({
        type:'frame', frame_number:i, frame_data:dataUrl,
        total_frames:total, confidence:currentConf, z_factor:currentZ,
      }));
    });
    await Promise.race([sent, new Promise(res=>setTimeout(res,120000))]);
  }

  // Send completion signal
  if (ws&&ws.readyState===WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type:'complete', fps,
      venue_name:document.getElementById('venueName').value||'Unknown',
      name:document.getElementById('venueName').value||'Session '+sid.slice(5,13),
    }));
  }
}

/* ══ REPORT */
function downloadReport() {
  if (!frameData.length){alert('Run analysis first.');return;}
  const counts=frameData.map(f=>f.count);
  const report={
    report_type:'crowd_intelligence_incident_report_v3',
    generated_at:new Date().toISOString(),
    venue:document.getElementById('venueName').value||'Unknown Venue',
    calibration:pixelsPerMeter?{pixels_per_meter:pixelsPerMeter,H_matrix:homographyMatrix}:null,
    statistics:{
      total_frames:frameData.length,
      peak_count:Math.max(...counts),
      avg_count:Math.round(counts.reduce((a,b)=>a+b,0)/counts.length),
      alert_count:ALERT_HISTORY.length,
    },
    tripwire_summary:tripwire?{in:tripwireIn,out:tripwireOut,net:tripwireIn-tripwireOut}:null,
    threshold_engine:'adaptive_v2_multi_signal',
    announcement_log:SpeechSystem.logEntries,
    alert_timeline:ALERT_HISTORY,
    proximity_summary:{
      peak_score:Math.max(...frameData.map(f=>f.distance?.proximity_score||0)).toFixed(2),
      avg_violations:Math.round(frameData.reduce((s,f)=>s+(f.distance?.violations||0),0)/Math.max(frameData.length,1)),
    },
    cusum_alerts:frameData.filter(f=>f.cusum?.alert).length,
    recommendation:ALERT_HISTORY.length>5
      ?'CRITICAL: Multiple threshold breaches.'
      :ALERT_HISTORY.length>0?'WARNING: Density anomalies detected.'
      :'NOMINAL: No significant safety incidents.',
  };
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  a.download='crowd_report_v3_'+Date.now()+'.json'; a.click();
  addLog('Report downloaded');
}

/* ══ RESET */
function resetAll() {
  vid.pause(); vid.src='';
  if (ws&&ws.readyState===WebSocket.OPEN) ws.close(); ws=null;
  if (animId){cancelAnimationFrame(animId);animId=null;}
  window.speechSynthesis.cancel();

  reviewMode=false; file=null; frameData=[]; isPlaying=false; lastFrameShown=null;
  thumbCache=[]; thumbReady=false; currentZoneData=null; currentWalkableMask=null;
  currentDistMeta=null; currentWalkableMeta=null;
  calibMode=false; calibPts=[]; pixelsPerMeter=null; homographyMatrix=null;
  ALERT_HISTORY.length=0; notableEvents=[]; selectedZone=null;
  tripwire=null; tripwireMode=false; tripwireIn=0; tripwireOut=0; _prevSides=[];
  SpeechSystem.lastLevel=null; SpeechSystem.lastAnnounceTime=0; SpeechSystem.logEntries=[];

  document.querySelectorAll('.calib-dot').forEach(d=>d.remove());
  hideVideo();

  document.getElementById('reviewRibbonWrap').style.display='none';
  document.getElementById('reviewRibbonWrap').innerHTML='';
  document.getElementById('analysisProgress').style.display='';
  document.getElementById('log').innerHTML='';
  document.getElementById('pctFil').style.width='0%';
  document.getElementById('pctTxt').textContent='0%';
  const procWrap=document.getElementById('procWrap');
  procWrap.classList.remove('show'); procWrap.style.display='';

  document.getElementById('processBtn').disabled=false;
  document.getElementById('playBtn').textContent='▶';
  ['sCurrent','sMax','sAvg'].forEach(id=>document.getElementById(id).textContent='0');
  const els={
    tcV:'—',capPct:'—%',walkableInfo:'Walkable area: awaiting data',
    calibStatus:'Not calibrated — density shown as relative',
    behState:'NORMAL',behConf:'—',behDetail:'No anomalies detected',
    dViolations:'0',dVioRatio:'0%',dMedianNN:'—',dSafeDist:'—',
    proxScore:'0%',proxLabel:'SAFE SPACING',walkablePct:'—%',obstaclePct:'—%',
    twStatus:'No tripwire set',twCounts:'',
  };
  Object.entries(els).forEach(([id,v])=>{const e=document.getElementById(id);if(e)e.textContent=v;});
  const twBtn=document.getElementById('twBtn');if(twBtn){twBtn.textContent='Draw tripwire';twBtn.onclick=enterTripwireMode;}
  document.getElementById('calibStatus').className='calib-status';
  document.getElementById('losRow').style.display='none';
  document.getElementById('behBanner').className='beh-banner normal';
  document.getElementById('proxBanner').className='prox-banner safe';
  document.getElementById('announceLog').innerHTML='<div style="color:var(--text3);font-size:11.5px;padding:4px 0;">No announcements yet…</div>';
  document.getElementById('obstacleList').innerHTML='<div style="color:var(--text3);font-size:11px;">No obstacles detected yet</div>';
  document.getElementById('signalStack').style.display='none';
  document.getElementById('tcBadge').className='tc-badge b-wu';
  document.getElementById('tcBadge').textContent='Warm-up';
  document.getElementById('tcSub').textContent='Accumulating baseline…';
  document.getElementById('barFil').style.width='0%';
  document.getElementById('barLbl').textContent='—';
  document.getElementById('tlPlayed').style.width='0%';
  document.getElementById('tlHead').style.left='0%';
  document.getElementById('curTime').textContent='0:00';
  document.getElementById('durTime').textContent='0:00';
  document.getElementById('behBanner').className='beh-banner normal';
  hideAlert('alertBanner'); hideAlert('predictAlert');
  renderZoneMap(); addLog('Reset');
}

/* ══ INIT */
window.addEventListener('load', ()=>{ setTimeout(renderZoneMap,300); });
if ('speechSynthesis' in window){window.speechSynthesis.onvoiceschanged=()=>{};window.speechSynthesis.getVoices();}