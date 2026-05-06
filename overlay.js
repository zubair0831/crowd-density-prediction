'use strict';
// ════════════════════════════════════════════════════════════════════════════
// overlay.js  v3.2 — Canvas overlay rendering
// New in v3.2:
//   • Tripwire line + crossing count badge
//   • Trajectory velocity vectors (from TrajectoryTracker)
// ════════════════════════════════════════════════════════════════════════════

function rrect(ctx, x, y, w, h, r) {
  if (typeof r==='number') r=[r,r,r,r];
  ctx.beginPath();
  ctx.moveTo(x+r[0],y); ctx.lineTo(x+w-r[1],y);
  ctx.quadraticCurveTo(x+w,y,x+w,y+r[1]);
  ctx.lineTo(x+w,y+h-r[2]);
  ctx.quadraticCurveTo(x+w,y+h,x+w-r[2],y+h);
  ctx.lineTo(x+r[3],y+h);
  ctx.quadraticCurveTo(x,y+h,x,y+h-r[3]);
  ctx.lineTo(x,y+r[0]);
  ctx.quadraticCurveTo(x,y,x+r[0],y);
  ctx.closePath();
}

function drawOverlay(ctx, cvs, fd) {
  const coords  = fd.coordinates||[];
  const bW      = fd.original_size?fd.original_size[0]:vid.videoWidth;
  const bH      = fd.original_size?fd.original_size[1]:vid.videoHeight;
  const sx      = cvs.width/bW, sy=cvs.height/bH;
  const t       = Date.now();

  // ── Layer 0: Walkable area hatching ──────────────────────────────────────
  if (document.getElementById('showWalkable').checked && fd.walkable_mask_grid) {
    const cw=cvs.width/GRID_COLS, ch=cvs.height/GRID_ROWS;
    fd.walkable_mask_grid.forEach((row,r)=>row.forEach((v,c)=>{
      if (v<0.10){ctx.fillStyle='rgba(248,113,113,.11)';ctx.fillRect(c*cw,r*ch,cw,ch);}
    }));
  }

  // ── Layer 0b: Obstacle bounding boxes ────────────────────────────────────
  if (document.getElementById('showObstacles').checked && fd.walkable?.obstacles_detected?.length) {
    const CAT_COLORS={
      vehicle:   ['rgba(251,146,60,0.9)',  'rgba(251,146,60,0.10)', '#fb923c'],
      nature:    ['rgba(74,222,128,0.85)', 'rgba(74,222,128,0.08)', '#4ade80'],
      furniture: ['rgba(167,139,250,0.85)','rgba(167,139,250,0.08)','#a78bfa'],
      structure: ['rgba(248,113,113,0.85)','rgba(248,113,113,0.08)','#f87171'],
    };
    const pulse=0.65+Math.sin(t/600)*0.35;
    fd.walkable.obstacles_detected.forEach(ob=>{
      let bx1,by1,bx2,by2;
      if (ob.bbox_norm){bx1=ob.bbox_norm[0]*cvs.width;by1=ob.bbox_norm[1]*cvs.height;bx2=ob.bbox_norm[2]*cvs.width;by2=ob.bbox_norm[3]*cvs.height;}
      else{bx1=ob.bbox[0]*sx;by1=ob.bbox[1]*sy;bx2=ob.bbox[2]*sx;by2=ob.bbox[3]*sy;}
      const bw=bx2-bx1,bh=by2-by1;
      const cat=ob.category||'structure';
      const [borderCol,fillCol,textCol]=CAT_COLORS[cat]||CAT_COLORS.structure;
      ctx.fillStyle=fillCol; ctx.fillRect(bx1,by1,bw,bh);
      ctx.strokeStyle=borderCol.replace('0.9',String(pulse)); ctx.lineWidth=1.8;
      const corner=Math.min(18,bw*0.18,bh*0.18);
      ctx.beginPath();
      ctx.moveTo(bx1,by1+corner);ctx.lineTo(bx1,by1);ctx.lineTo(bx1+corner,by1);
      ctx.moveTo(bx2-corner,by1);ctx.lineTo(bx2,by1);ctx.lineTo(bx2,by1+corner);
      ctx.moveTo(bx2,by2-corner);ctx.lineTo(bx2,by2);ctx.lineTo(bx2-corner,by2);
      ctx.moveTo(bx1+corner,by2);ctx.lineTo(bx1,by2);ctx.lineTo(bx1,by2-corner);
      ctx.stroke();
      ctx.strokeStyle=borderCol.replace('0.9','0.28'); ctx.lineWidth=0.8; ctx.strokeRect(bx1,by1,bw,bh);
      const label=ob.class.toUpperCase(), confLabel=Math.round(ob.conf*100)+'%', fullLabel=label+'  '+confLabel;
      const fontSize=Math.max(9,Math.min(12,bw/8));
      ctx.font=`700 ${fontSize}px "JetBrains Mono",monospace`;
      const tw=ctx.measureText(fullLabel).width, ph=fontSize+8, pw=tw+14;
      const px=clamp(bx1,2,cvs.width-pw-2), py=by1>ph+4?by1-ph-2:by1+2;
      ctx.fillStyle='rgba(7,11,18,0.88)'; rrect(ctx,px,py,pw,ph,4); ctx.fill();
      ctx.fillStyle=textCol; rrect(ctx,px,py,3,ph,[4,0,0,4]); ctx.fill();
      ctx.fillStyle=textCol; ctx.font=`700 ${fontSize}px "JetBrains Mono",monospace`;
      ctx.fillText(label,px+8,py+ph-5);
      ctx.fillStyle='rgba(255,255,255,0.5)';
      const lw=ctx.measureText(label+'  ').width;
      ctx.font=`500 ${fontSize-1}px "JetBrains Mono",monospace`;
      ctx.fillText(confLabel,px+8+lw,py+ph-5);
    });
  }

  // ── Layer 1: Zone risk overlay ────────────────────────────────────────────
  if (document.getElementById('showZones').checked && fd.zones) {
    const cw=cvs.width/GRID_COLS, ch=cvs.height/GRID_ROWS;
    fd.zones.forEach(z=>{
      const x=z.col*cw, y=z.row*ch; if (!z.is_walkable) return;
      const r=z.risk;
      ctx.fillStyle=`rgba(${Math.round(r<0.5?r*2*255:255)},${Math.round(r<0.5?255:(1-r)*2*255)},0,${0.06+r*0.28})`;
      ctx.fillRect(x,y,cw,ch);
      if (r>0.68){
        const p=0.5+Math.sin(t/280)*0.4;
        ctx.strokeStyle=`rgba(248,113,113,${p})`; ctx.lineWidth=1.5; ctx.strokeRect(x+1,y+1,cw-2,ch-2);
      }
      if (document.getElementById('showCount').checked && z.count>0) {
        ctx.font=`bold ${Math.max(10,Math.min(14,cw/4))}px 'JetBrains Mono',monospace`;
        ctx.textAlign='center';
        const lx=x+cw/2,ly=y+ch/2+5;
        ctx.fillStyle='rgba(0,0,0,.55)'; ctx.fillText(z.count,lx+1,ly+1);
        ctx.fillStyle=r>0.68?'#f87171':r>0.4?'#fbbf24':'rgba(255,255,255,.85)';
        ctx.fillText(z.count,lx,ly); ctx.textAlign='left';
      }
    });
  }

  // ── Layer 2: Pressure field ───────────────────────────────────────────────
  if (document.getElementById('showPres').checked && fd.pressure?.cells) {
    const cw=cvs.width/GRID_COLS, ch=cvs.height/GRID_ROWS;
    fd.pressure.cells.forEach(c=>{
      if (c.pressure<0.15) return;
      ctx.fillStyle=`rgba(248,113,113,${c.pressure*0.28})`; ctx.fillRect(c.col*cw,c.row*ch,cw,ch);
    });
  }

  // ── Layer 3: Density heatmap ──────────────────────────────────────────────
  if (document.getElementById('showHeat').checked && coords.length) {
    coords.forEach(([cx,cy])=>{
      const x=cx*sx, y=cy*sy;
      const g=ctx.createRadialGradient(x,y,0,x,y,36);
      g.addColorStop(0,'rgba(232,121,160,.38)'); g.addColorStop(.5,'rgba(251,191,36,.14)'); g.addColorStop(1,'rgba(0,0,0,0)');
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,36,0,Math.PI*2); ctx.fill();
    });
  }

  // ── Layer 4: Optical flow vectors ─────────────────────────────────────────
  if (document.getElementById('showFlow').checked && fd.zones) {
    const cw=cvs.width/GRID_COLS, ch=cvs.height/GRID_ROWS;
    fd.zones.forEach(z=>{
      if (z.magnitude<0.35||!z.is_walkable) return;
      const cx=z.col*cw+cw/2, cy=z.row*ch+ch/2;
      const mag=Math.min(z.magnitude,8)/8, scale=Math.min(cw,ch)*0.38;
      const ex=cx+z.dx*scale*mag, ey=cy+z.dy*scale*mag;
      const alpha=0.3+mag*0.5;
      ctx.strokeStyle=`rgba(34,211,238,${alpha})`; ctx.lineWidth=1+mag*1.5;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(ex,ey); ctx.stroke();
      const ang=Math.atan2(ey-cy,ex-cx), aL=5+mag*3;
      ctx.fillStyle=`rgba(34,211,238,${alpha})`;
      ctx.beginPath(); ctx.moveTo(ex,ey);
      ctx.lineTo(ex-aL*Math.cos(ang-0.45),ey-aL*Math.sin(ang-0.45));
      ctx.lineTo(ex-aL*Math.cos(ang+0.45),ey-aL*Math.sin(ang+0.45));
      ctx.closePath(); ctx.fill();
    });
  }

  // ── Layer 4b: Trajectory velocity vectors ─────────────────────────────────
  if (document.getElementById('showVelocity')?.checked && fd.trajectories?.length) {
    fd.trajectories.forEach(tr=>{
      if (tr.speed<3) return;   // skip near-stationary
      const [px,py]=[tr.pos[0]*sx, tr.pos[1]*sy];
      const [vx,vy]=[tr.vel[0]*sx*3, tr.vel[1]*sy*3];  // scaled 3× for visibility
      const fast=tr.speed>20;
      ctx.strokeStyle=fast?'rgba(248,113,113,0.75)':'rgba(167,139,250,0.6)';
      ctx.lineWidth=fast?2:1.2;
      ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+vx,py+vy); ctx.stroke();
      const ang=Math.atan2(vy,vx), aL=4+Math.min(tr.speed/10,4);
      ctx.fillStyle=fast?'rgba(248,113,113,0.75)':'rgba(167,139,250,0.6)';
      ctx.beginPath(); ctx.moveTo(px+vx,py+vy);
      ctx.lineTo(px+vx-aL*Math.cos(ang-0.5),py+vy-aL*Math.sin(ang-0.5));
      ctx.lineTo(px+vx-aL*Math.cos(ang+0.5),py+vy-aL*Math.sin(ang+0.5));
      ctx.closePath(); ctx.fill();
    });
  }

  // ── Layer 5: Detection dots ───────────────────────────────────────────────
  if (document.getElementById('showDots').checked && coords.length) {
    const safePx=fd.distance?.min_safe_px, nnDists=fd.distance?.nn_distances||[];
    coords.forEach(([cx,cy],i)=>{
      const x=cx*sx, y=cy*sy;
      const isVio=safePx&&nnDists[i]!=null&&nnDists[i]<safePx;
      ctx.beginPath(); ctx.arc(x,y,5.5,0,Math.PI*2);
      ctx.fillStyle=isVio?'rgba(248,113,113,.18)':'rgba(34,211,238,.14)'; ctx.fill();
      ctx.beginPath(); ctx.arc(x,y,2.8,0,Math.PI*2);
      ctx.fillStyle=isVio?'rgba(248,113,113,.9)':'rgba(34,211,238,.88)'; ctx.fill();
      ctx.beginPath(); ctx.arc(x,y,1.1,0,Math.PI*2);
      ctx.fillStyle='rgba(255,255,255,.9)'; ctx.fill();
    });
  }

  // ── Layer 5b: Tripwire line + badge ───────────────────────────────────────
  if (tripwire) {
    const {a,b}=tripwire;
    const ax=a[0]*cvs.width, ay=a[1]*cvs.height;
    const bx=b[0]*cvs.width, by=b[1]*cvs.height;
    // Animated dashed line
    const pulse=0.55+Math.sin(t/400)*0.35;
    ctx.save();
    ctx.setLineDash([8,5]);
    ctx.lineDashOffset=-(t/40)%13;
    ctx.strokeStyle=`rgba(251,191,36,${pulse})`; ctx.lineWidth=2.5;
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
    ctx.restore();
    // Endpoint circles
    for (const [ex,ey] of [[ax,ay],[bx,by]]) {
      ctx.beginPath(); ctx.arc(ex,ey,5,0,Math.PI*2);
      ctx.fillStyle='rgba(251,191,36,.9)'; ctx.fill();
    }
    // Crossing badge
    const net=tripwireIn-tripwireOut;
    const badge=`↓${tripwireIn}  ↑${tripwireOut}  NET ${net>=0?'+':''}${net}`;
    const midX=(ax+bx)/2, midY=(ay+by)/2;
    ctx.font='bold 12px "JetBrains Mono",monospace';
    const tw=ctx.measureText(badge).width, bh=22, bwd=tw+18;
    const bx0=clamp(midX-bwd/2,4,cvs.width-bwd-4), by0=midY-bh-6;
    ctx.fillStyle='rgba(7,11,18,.88)'; rrect(ctx,bx0,by0,bwd,bh,5); ctx.fill();
    ctx.fillStyle='rgba(251,191,36,.8)'; rrect(ctx,bx0,by0,3,bh,[5,0,0,5]); ctx.fill();
    ctx.fillStyle='#fbbf24'; ctx.fillText(badge,bx0+8,by0+15);
  }

  // ── Layer 6: Count pill ───────────────────────────────────────────────────
  if (document.getElementById('showCount').checked) {
    const lbl=String(fd.count);
    ctx.font='bold 17px "Epilogue",Arial'; const tw=ctx.measureText(lbl).width;
    ctx.fillStyle='rgba(7,11,18,.87)'; rrect(ctx,14,14,tw+52,38,9); ctx.fill();
    ctx.fillStyle='#22d3ee'; rrect(ctx,14,14,3,38,[9,0,0,9]); ctx.fill();
    ctx.fillStyle='rgba(34,211,238,.55)'; ctx.font='9.5px "JetBrains Mono",monospace'; ctx.fillText('PEOPLE',24,28);
    ctx.fillStyle='#e8eef8'; ctx.font='bold 17px "Epilogue",Arial'; ctx.fillText(lbl,24,44);
  }

  // ── Layer 7: Threshold badge ──────────────────────────────────────────────
  if (document.getElementById('showThresh').checked && fd.dynamic_threshold!=null) {
    const exc=fd.count>=fd.dynamic_threshold, tl=String(fd.dynamic_threshold);
    ctx.font='bold 17px "Epilogue",Arial'; const tw2=ctx.measureText(tl).width, bx2=cvs.width-tw2-58;
    ctx.fillStyle=exc?'rgba(248,113,113,.87)':'rgba(7,11,18,.87)'; rrect(ctx,bx2,14,tw2+52,38,9); ctx.fill();
    ctx.fillStyle=exc?'#f87171':'#fbbf24'; rrect(ctx,bx2+tw2+49,14,3,38,[0,9,9,0]); ctx.fill();
    ctx.fillStyle=exc?'rgba(255,200,200,.6)':'rgba(251,191,36,.6)'; ctx.font='9.5px "JetBrains Mono",monospace'; ctx.fillText('THRESH',bx2+9,28);
    ctx.fillStyle='#e8eef8'; ctx.font='bold 17px "Epilogue",Arial'; ctx.fillText(tl,bx2+9,44);
  }

  // ── Layer 8: Behavior label ───────────────────────────────────────────────
  if (document.getElementById('showBehav').checked && fd.behavior) {
    const s=fd.behavior.state, lbl=s.replace('_',' ');
    const cols={NORMAL:'#4ade80',PRE_SURGE:'#fbbf24',SURGE:'#f87171',DISPERSING:'#60a5fa'};
    const col=cols[s]||'#94a3b8';
    ctx.font='bold 14px "Epilogue",Arial'; const tw3=ctx.measureText(lbl).width;
    const bx3=14, by3=cvs.height-52;
    ctx.fillStyle='rgba(7,11,18,.87)'; rrect(ctx,bx3,by3,tw3+52,34,9); ctx.fill();
    ctx.fillStyle=col; rrect(ctx,bx3,by3,3,34,[9,0,0,9]); ctx.fill();
    ctx.fillStyle=col+'99'; ctx.font='9px "JetBrains Mono",monospace'; ctx.fillText('BEHAVIOR',bx3+10,by3+14);
    ctx.fillStyle='#e8eef8'; ctx.font='bold 14px "Epilogue",Arial'; ctx.fillText(lbl,bx3+10,by3+27);
  }

  // ── Layer 9: Proximity score badge ────────────────────────────────────────
  if (document.getElementById('showProx').checked && fd.distance) {
    const ps=fd.distance.proximity_score||0, psLbl=Math.round(ps*100)+'%';
    const col=ps>0.65?'#f87171':ps>0.30?'#fbbf24':'#4ade80';
    ctx.font='bold 13px "Epilogue",Arial'; const tw4=ctx.measureText(psLbl).width;
    const bx4=14, by4=cvs.height-98;
    ctx.fillStyle='rgba(7,11,18,.87)'; rrect(ctx,bx4,by4,tw4+52,30,7); ctx.fill();
    ctx.fillStyle=col; rrect(ctx,bx4,by4,3,30,[7,0,0,7]); ctx.fill();
    ctx.fillStyle=col+'99'; ctx.font='9px "JetBrains Mono",monospace'; ctx.fillText('PROXIMITY',bx4+10,by4+11);
    ctx.fillStyle='#e8eef8'; ctx.font='bold 13px "Epilogue",Arial'; ctx.fillText(psLbl,bx4+10,by4+24);
  }

  // ── Layer 10: CUSUM alert flash ───────────────────────────────────────────
  if (fd.cusum?.alert) {
    const flash=0.12+Math.abs(Math.sin(t/120))*0.18;
    const dir=fd.cusum.direction==='surge'?'rgba(248,113,113,':'rgba(96,165,250,';
    ctx.fillStyle=dir+flash+')'; ctx.fillRect(0,0,cvs.width,cvs.height);
    const label='CUSUM: '+(fd.cusum.direction||'').toUpperCase();
    ctx.font='bold 13px "Epilogue",Arial';
    const tw5=ctx.measureText(label).width, bx5=cvs.width/2-tw5/2-10;
    ctx.fillStyle='rgba(7,11,18,.9)'; rrect(ctx,bx5,cvs.height-140,tw5+20,26,6); ctx.fill();
    ctx.fillStyle=fd.cusum.direction==='surge'?'#f87171':'#60a5fa';
    ctx.fillText(label,bx5+10,cvs.height-122);
  }
}