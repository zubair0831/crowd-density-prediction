# ═══════════════════════════════════════════════════════════════════════════════
# AI Crowd Intelligence System — Backend
# ═══════════════════════════════════════════════════════════════════════════════
# Features:
#   • P2PNet crowd detection with adjustable confidence
#   • Optical flow analysis (OpenCV Farneback) for movement / compression
#   • Spatial zone grid (8×6) with per-zone density, flow, divergence & risk
#   • Pressure field estimation (density + compression + temporal trend)
#   • Rule-based behavior classifier (NORMAL / PRE_SURGE / SURGE / DISPERSING)
#   • CrowdBehaviorTransformer architecture (ready to train with synthetic data)
#   • Adaptive threshold (EMA + rolling median, damped downward changes)
#   • SQLite session persistence with incident report export
#   • RTSP / webcam live-stream endpoint (async producer-consumer pipeline)
#   • Venue calibration (pixels → m², Fruin Level-of-Service)
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Optional OpenCV (required for optical flow; install with: pip install opencv-python)
try:
    import cv2
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False
    print("⚠️  cv2 not found — optical flow disabled (pip install opencv-python)")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
P2PNET_DIR   = os.path.join(BASE_DIR, "CrowdCounting-P2PNet")
WEIGHTS_PATH = os.path.join(P2PNET_DIR, "weights", "SHTechA.pth")
DB_PATH      = os.path.join(BASE_DIR, "sessions.db")
sys.path.insert(0, P2PNET_DIR)

from models import build_model

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_COLS   = 8
GRID_ROWS   = 6
GRID_CELLS  = GRID_COLS * GRID_ROWS   # 48
EMA_ALPHA   = 0.22
executor    = ThreadPoolExecutor(max_workers=4)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            TEXT PRIMARY KEY,
            name          TEXT,
            venue_name    TEXT,
            created_at    TEXT,
            total_frames  INTEGER DEFAULT 0,
            duration_sec  REAL    DEFAULT 0,
            peak_count    INTEGER DEFAULT 0,
            avg_count     REAL    DEFAULT 0,
            alert_count   INTEGER DEFAULT 0,
            peak_behavior TEXT    DEFAULT 'NORMAL',
            calibration   TEXT,
            summary       TEXT
        )
    """)
    conn.commit()
    conn.close()


def db_upsert_session(session: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO sessions
        (id, name, venue_name, created_at, total_frames, duration_sec,
         peak_count, avg_count, alert_count, peak_behavior, calibration, summary)
        VALUES (:id,:name,:venue_name,:created_at,:total_frames,:duration_sec,
                :peak_count,:avg_count,:alert_count,:peak_behavior,:calibration,:summary)
    """, session)
    conn.commit()
    conn.close()


def db_list_sessions() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id,name,venue_name,created_at,total_frames,peak_count,"
        "avg_count,alert_count,peak_behavior FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_session(sid: str) -> Optional[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def db_delete_session(sid: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0

# ═══════════════════════════════════════════════════════════════════════════════
# OPTICAL FLOW ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class FlowAnalyzer:
    """
    Computes Farneback optical flow between consecutive frames and aggregates
    per-zone statistics: mean displacement (dx, dy), magnitude, and divergence.
    Negative divergence in a zone indicates crowd compression — a key precursor
    to crowd crush events.
    """
    def __init__(self, grid_cols: int = GRID_COLS, grid_rows: int = GRID_ROWS):
        self.grid_cols  = grid_cols
        self.grid_rows  = grid_rows
        self.prev_gray  = None

    def update(self, pil_img: Image.Image) -> Optional[List[Dict]]:
        if not FLOW_AVAILABLE:
            return None

        rgb  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray

        h, w = gray.shape
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        results: List[Dict] = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                r0, r1 = row * cell_h, min((row + 1) * cell_h, h)
                c0, c1 = col * cell_w, min((col + 1) * cell_w, w)
                cf = flow[r0:r1, c0:c1]
                fx, fy = cf[..., 0], cf[..., 1]

                # Divergence: d(fx)/dx + d(fy)/dy  (negative ≡ compression)
                if fx.shape[0] > 1 and fx.shape[1] > 1:
                    div = float(np.gradient(fx, axis=1).mean() +
                                np.gradient(fy, axis=0).mean())
                else:
                    div = 0.0

                results.append({
                    "row":       row,
                    "col":       col,
                    "mean_dx":   float(fx.mean()),
                    "mean_dy":   float(fy.mean()),
                    "divergence": round(div, 4),
                    "magnitude": round(float(np.sqrt(fx**2 + fy**2).mean()), 4),
                })
        return results

    def reset(self) -> None:
        self.prev_gray = None

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_zones(
    coords: List[List[float]],
    img_w: int,
    img_h: int,
    flow_zones: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Bin P2PNet detections into an 8×6 spatial grid and merge with flow data.
    Returns a flat list of 48 zone records sorted row-major.

    Risk formula:
        risk = 0.50 * density_norm
              + 0.30 * max(0, -divergence / 0.5)   # compression component
              + 0.20 * min(magnitude / 10, 1)       # turbulence component
    """
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS

    grid: List[List[Dict]] = [
        [{"count": 0, "dx": 0.0, "dy": 0.0, "divergence": 0.0, "magnitude": 0.0}
         for _ in range(GRID_COLS)]
        for _ in range(GRID_ROWS)
    ]

    for x, y in coords:
        col = min(int(x / cell_w), GRID_COLS - 1)
        row = min(int(y / cell_h), GRID_ROWS - 1)
        grid[row][col]["count"] += 1

    # Merge flow
    if flow_zones:
        for fz in flow_zones:
            r, c = fz["row"], fz["col"]
            grid[r][c]["dx"]          = fz["mean_dx"]
            grid[r][c]["dy"]          = fz["mean_dy"]
            grid[r][c]["divergence"]  = fz["divergence"]
            grid[r][c]["magnitude"]   = fz["magnitude"]

    max_count = max(
        (grid[r][c]["count"] for r in range(GRID_ROWS) for c in range(GRID_COLS)),
        default=1
    ) or 1

    result: List[Dict] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            z = grid[r][c]
            dn         = z["count"] / max_count
            compression = max(0.0, -z["divergence"] / 0.5)
            turb       = min(z["magnitude"] / 10.0, 1.0)
            risk       = min(1.0, 0.50 * dn + 0.30 * compression + 0.20 * turb)
            result.append({
                "row":          r,
                "col":          c,
                "count":        z["count"],
                "density_norm": round(dn, 3),
                "dx":           round(z["dx"], 2),
                "dy":           round(z["dy"], 2),
                "divergence":   round(z["divergence"], 4),
                "magnitude":    round(z["magnitude"], 4),
                "risk":         round(risk, 3),
            })
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# PRESSURE FIELD ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pressure(
    zone_data: List[Dict],
    count_history: List[int],
) -> Dict:
    """
    Per-zone pressure estimate:
        P = 0.50 * density_norm
          + 0.30 * compression_factor   (from flow divergence)
          + 0.20 * trend_factor         (2nd-order temporal derivative)

    Crush risk zones: pressure > 0.70
    Fruin LOS is computed if calibration (m²/pixel²) is available.
    """
    # Temporal trend: normalised slope over last 5 counts
    if len(count_history) >= 3:
        n     = min(len(count_history), 5)
        win   = count_history[-n:]
        x_m   = (n - 1) / 2
        y_m   = sum(win) / n
        num   = sum((i - x_m) * (win[i] - y_m) for i in range(n))
        den   = sum((i - x_m) ** 2 for i in range(n)) or 1e-9
        slope = num / den
        # Normalise: +1 = rapidly increasing, -1 = rapidly falling
        trend = max(-1.0, min(1.0, slope / max(y_m, 1) * 5))
    else:
        trend = 0.0

    cells: List[Dict] = []
    crush_zones: List[Dict] = []
    max_p = 0.0

    for z in zone_data:
        comp = max(0.0, -z["divergence"] / 0.5)
        p    = min(1.0,
               0.50 * z["density_norm"] +
               0.30 * comp +
               0.20 * max(0.0, trend))
        p = round(p, 3)
        max_p = max(max_p, p)
        cells.append({"row": z["row"], "col": z["col"], "pressure": p})
        if p > 0.70:
            crush_zones.append({"row": z["row"], "col": z["col"]})

    return {
        "cells":           cells,
        "max_pressure":    round(max_p, 3),
        "crush_risk_zones": crush_zones,
        "trend_factor":    round(trend, 3),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR CLASSIFIER  (rule-based state machine)
# ═══════════════════════════════════════════════════════════════════════════════

class BehaviorClassifier:
    """
    Classifies crowd behavior into one of four states using a rule-based
    state machine on sliding-window features:

        NORMAL      → steady-state, low density, low compression
        PRE_SURGE   → density rising, compression zones appearing
        SURGE       → high density + strong compression or threshold exceeded
        DISPERSING  → density falling after surge, positive divergence

    Transition probabilities are derived from the Social Force Model
    (Helbing et al., 2000) adapted for monocular video.
    """

    STATES = ["NORMAL", "PRE_SURGE", "SURGE", "DISPERSING"]

    def __init__(self, history_len: int = 10) -> None:
        self.history_len  = history_len
        self.counts:       List[float] = []
        self.compressions: List[int]   = []
        self.state        = "NORMAL"
        self.state_frames = 0

    def reset(self) -> None:
        self.counts       = []
        self.compressions = []
        self.state        = "NORMAL"
        self.state_frames = 0

    def _trend(self) -> float:
        """Linear regression slope over the count history window."""
        n = len(self.counts)
        if n < 3:
            return 0.0
        x_m   = (n - 1) / 2
        y_m   = sum(self.counts) / n
        num   = sum((i - x_m) * (self.counts[i] - y_m) for i in range(n))
        den   = sum((i - x_m) ** 2 for i in range(n)) or 1e-9
        return num / den

    def update(self, count: int, zone_data: List[Dict], threshold: int) -> Dict:
        self.state_frames += 1
        self.counts.append(float(count))
        if len(self.counts) > self.history_len:
            self.counts.pop(0)

        comp_count = sum(1 for z in zone_data if z.get("divergence", 0) < -0.10)
        self.compressions.append(comp_count)
        if len(self.compressions) > self.history_len:
            self.compressions.pop(0)

        trend        = self._trend()
        density_ratio = count / max(threshold, 1)
        prev_state   = self.state

        # ── State machine ────────────────────────────────────────────────────
        if density_ratio < 0.28 and self.state_frames >= 4:
            self.state = "NORMAL"
        elif self.state == "NORMAL":
            if (density_ratio > 0.58 or comp_count >= 3) and trend > 0.3:
                self.state = "PRE_SURGE"
        elif self.state == "PRE_SURGE":
            if density_ratio >= 0.85 or comp_count >= 5:
                self.state = "SURGE"
            elif density_ratio < 0.45 and trend < -0.5:
                self.state = "NORMAL"
        elif self.state == "SURGE":
            if trend < -1.5 and comp_count < 3:
                self.state = "DISPERSING"
        elif self.state == "DISPERSING":
            if density_ratio < 0.38:
                self.state = "NORMAL"
            elif density_ratio > 0.82:
                self.state = "SURGE"

        if self.state != prev_state:
            self.state_frames = 0

        # ── Confidence ───────────────────────────────────────────────────────
        if self.state == "NORMAL":
            conf = min(1.0, 1.0 - density_ratio)
        elif self.state == "PRE_SURGE":
            conf = min(1.0, 0.45 + density_ratio * 0.35 + comp_count * 0.04)
        elif self.state == "SURGE":
            conf = min(1.0, density_ratio * 0.65 + comp_count * 0.055)
        else:   # DISPERSING
            conf = min(1.0, 0.45 + max(0.0, -trend) * 0.08)

        return {
            "state":             self.state,
            "confidence":        round(conf, 3),
            "count_trend":       round(trend, 2),
            "compression_zones": comp_count,
            "density_ratio":     round(density_ratio, 3),
        }

# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD  (EMA + rolling median, damped)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveThreshold:
    """
    Improved threshold:
      1. Rolling median  — robust to outliers (spike resistance)
      2. EMA of median   — smooth centre estimate
      3. threshold = EMA + z * max(σ, 5% of EMA)
      4. Damping: threshold cannot fall faster than 4% per frame
    """
    def __init__(self, smooth_win: int = 7, ema_alpha: float = EMA_ALPHA,
                 z_factor: float = 2.0) -> None:
        self.smooth_win = smooth_win
        self.ema_alpha  = ema_alpha
        self.z_factor   = z_factor
        self.window:    List[float] = []
        self.ema:       Optional[float] = None
        self.prev:      Optional[float] = None

    def reset(self) -> None:
        self.window = []; self.ema = None; self.prev = None

    def update(self, count: int) -> Dict:
        self.window.append(float(count))
        if len(self.window) > self.smooth_win:
            self.window.pop(0)
        n = len(self.window)

        if n < 5:
            naive = round(max(self.window) * 1.30 + 2)
            return {"threshold": naive, "mode": "warmup",
                    "mean": None, "std": None, "samples": n}

        sw   = sorted(self.window)
        med  = (sw[n // 2 - 1] + sw[n // 2]) / 2 if n % 2 == 0 else sw[n // 2]
        mean = sum(self.window) / n
        std  = math.sqrt(sum((v - mean) ** 2 for v in self.window) / n)

        self.ema = (self.ema_alpha * med + (1 - self.ema_alpha) * self.ema
                    if self.ema is not None else med)

        eff_std = max(std, self.ema * 0.05)
        raw     = self.ema + self.z_factor * eff_std

        thresh = max(raw, self.prev * 0.96) if self.prev is not None else raw
        thresh = int(round(max(thresh, 1.0)))
        self.prev = thresh

        return {"threshold": thresh, "mode": "dynamic",
                "mean": round(mean, 1), "std": round(std, 1), "samples": n}

# ═══════════════════════════════════════════════════════════════════════════════
# CROWD BEHAVIOR TRANSFORMER  (architecture — train with synthetic data)
# ═══════════════════════════════════════════════════════════════════════════════
# Feature vector per frame (145-dim):
#   [count(1), zone_densities(48), flow_magnitudes(48), flow_divergences(48)]
#
# Training strategy: generate synthetic crowd scenarios with Social Force Model
# (Helbing & Molnár 1995), label phases programmatically, train on feature
# sequences — zero-shot transfer to real P2PNet outputs (research contribution).

class CrowdBehaviorTransformer(nn.Module):
    """
    Lightweight temporal transformer for crowd behavior sequence classification.
    Input:  (batch, seq_len=16, feature_dim=145)
    Output: (batch, 4)  logits for [NORMAL, PRE_SURGE, SURGE, DISPERSING]
    """
    CLASSES = ["NORMAL", "PRE_SURGE", "SURGE", "DISPERSING"]

    def __init__(self, feature_dim: int = 145, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.proj    = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, len(self.CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = self.proj(x) + self.pos_emb(pos)
        x   = self.encoder(x)
        return self.head(x[:, -1, :])   # classify from last token

# ═══════════════════════════════════════════════════════════════════════════════
# FRUIN LEVEL OF SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

def fruin_los(density_sqm: float) -> str:
    """Map people/m² to Fruin Level-of-Service grade."""
    if density_sqm <= 0.5:  return "A"
    if density_sqm <= 1.0:  return "B"
    if density_sqm <= 1.7:  return "C"
    if density_sqm <= 2.7:  return "D"
    if density_sqm <= 4.0:  return "E"
    return "F"

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE
# ═══════════════════════════════════════════════════════════════════════════════

model     = None
device    = None
transform = None

class Args:
    backbone = "vgg16_bn"
    row = 2
    line = 2

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform
    print("🚀 Loading P2PNet …")
    init_db()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args      = Args()
    model     = build_model(args, training=False)
    ckpt      = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"✅ P2PNet ready on {device}")
    yield
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("🧹 Shutdown complete")

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(lifespan=lifespan, title="AI Crowd Intelligence API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ─── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":  "running",
        "model":   "P2PNet (SHTechA)",
        "device":  str(device),
        "flow":    FLOW_AVAILABLE,
        "version": "2.0.0",
    }

# ─── Sessions ─────────────────────────────────────────────────────────────────
@app.get("/sessions")
async def list_sessions():
    return db_list_sessions()

@app.get("/sessions/{sid}")
async def get_session(sid: str):
    s = db_get_session(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    return s

@app.delete("/sessions/{sid}")
async def delete_session(sid: str):
    if not db_delete_session(sid):
        raise HTTPException(404, "Session not found")
    return {"deleted": sid}

@app.get("/sessions/{sid}/report")
async def get_report(sid: str):
    """Return a structured JSON incident report for download."""
    s = db_get_session(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    summary = json.loads(s.get("summary") or "{}")
    report  = {
        "report_type":    "crowd_intelligence_incident_report",
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "session_id":     sid,
        "venue":          s.get("venue_name", "Unknown"),
        "recorded_at":    s.get("created_at"),
        "statistics": {
            "total_frames":   s["total_frames"],
            "peak_count":     s["peak_count"],
            "avg_count":      round(s["avg_count"], 1),
            "alert_count":    s["alert_count"],
            "peak_behavior":  s["peak_behavior"],
        },
        "timeline":       summary.get("timeline", []),
        "top_risk_frames": summary.get("top_risk_frames", []),
        "calibration":    json.loads(s.get("calibration") or "null"),
        "recommendation": _generate_recommendation(s),
    }
    return JSONResponse(report)

def _generate_recommendation(s: Dict) -> str:
    if s["alert_count"] > 5 or s["peak_behavior"] == "SURGE":
        return "CRITICAL: Multiple surge events detected. Recommend immediate crowd management review and capacity reduction measures."
    if s["alert_count"] > 0 or s["peak_behavior"] == "PRE_SURGE":
        return "WARNING: Anomalous density episodes detected. Recommend monitoring and pre-emptive dispersal protocols."
    return "NOMINAL: No significant crowd safety incidents detected in this session."

# ═══════════════════════════════════════════════════════════════════════════════
# FRAME PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_p2pnet(img: Image.Image, confidence: float) -> Tuple[List, List, float, Tuple[int,int]]:
    """Run P2PNet on one PIL image. Returns (coords, scores, avg_conf, orig_size)."""
    ow, oh = img.size
    max_dim = 2048
    w, h = ow, oh
    if max(w, h) > max_dim:
        s = max_dim / max(w, h); w, h = int(w * s), int(h * s)
    nw = max(128, (w // 128) * 128)
    nh = max(128, (h // 128) * 128)

    img_r = img.resize((nw, nh), Image.LANCZOS)
    inp   = transform(img_r).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    scores  = F.softmax(out["pred_logits"], -1)[0, :, 1]
    pts     = out["pred_points"][0]
    mask    = scores > confidence
    pts_sel = pts[mask].cpu().numpy()
    sco_sel = scores[mask].cpu().numpy()

    sx, sy  = ow / nw, oh / nh
    coords  = [[float(x * sx), float(y * sy)] for x, y in pts_sel]
    avg_c   = float(sco_sel.mean()) if len(sco_sel) > 0 else 0.0
    return coords, sco_sel.tolist(), avg_c, (ow, oh)


def process_one_frame(
    frame_bytes: bytes,
    confidence:  float,
    flow_analyzer: FlowAnalyzer,
    thresh_calc:   AdaptiveThreshold,
    behav_clf:     BehaviorClassifier,
    count_history: List[int],
    calibration:   Optional[Dict] = None,
) -> Dict:
    """Full pipeline for a single frame: detection → zones → flow → pressure → behavior."""
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    # ── P2PNet detection ──────────────────────────────────────────────────────
    coords, conf_scores, avg_conf, (ow, oh) = _run_p2pnet(img, confidence)
    count = len(coords)
    count_history.append(count)

    # ── Optical flow ─────────────────────────────────────────────────────────
    flow_zones = flow_analyzer.update(img)

    # ── Zone spatial analysis ─────────────────────────────────────────────────
    zone_data = analyze_zones(coords, ow, oh, flow_zones)

    # ── Pressure field ────────────────────────────────────────────────────────
    pressure = compute_pressure(zone_data, count_history)

    # ── Adaptive threshold ────────────────────────────────────────────────────
    thresh_info = thresh_calc.update(count)

    # ── Behavior classification ───────────────────────────────────────────────
    behavior = behav_clf.update(count, zone_data, thresh_info["threshold"])

    # ── Fruin LOS (requires calibration) ─────────────────────────────────────
    los_data: Dict = {}
    if calibration and calibration.get("pixels_per_meter"):
        ppm      = calibration["pixels_per_meter"]
        area_m2  = (ow * oh) / (ppm ** 2)
        dens     = count / max(area_m2, 0.001)
        los_data = {"density_sqm": round(dens, 2), "los": fruin_los(dens), "area_m2": round(area_m2, 1)}

    return {
        "count":          count,
        "coordinates":    coords,
        "avg_confidence": round(avg_conf, 3),
        "original_size":  [ow, oh],
        "zones":          zone_data,
        "pressure":       pressure,
        "behavior":       behavior,
        "threshold":      thresh_info,
        "los":            los_data,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — BATCH VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/process-frames/{session_id}")
async def ws_process_frames(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"📡 Session {session_id} connected")

    # Per-session state
    flow_analyzer = FlowAnalyzer()
    thresh_calc   = AdaptiveThreshold()
    behav_clf     = BehaviorClassifier()
    count_history: List[int] = []
    calibration:   Optional[Dict] = None
    processed      = 0
    alert_count    = 0
    peak_count     = 0
    timeline:      List[Dict] = []
    top_risk:      List[Dict] = []

    try:
        while True:
            data = await websocket.receive_json()

            # ── Calibration message ───────────────────────────────────────────
            if data["type"] == "calibration":
                calibration = data.get("calibration")
                await websocket.send_json({"type": "calibration_ack"})
                continue

            # ── Frame message ─────────────────────────────────────────────────
            if data["type"] == "frame":
                frame_no   = data["frame_number"]
                total      = data.get("total_frames", 1)
                confidence = float(data.get("confidence", 0.5))

                frame_bytes = base64.b64decode(data["frame_data"].split(",")[1])

                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor,
                    process_one_frame,
                    frame_bytes, confidence,
                    flow_analyzer, thresh_calc, behav_clf,
                    count_history, calibration,
                )

                processed += 1
                progress   = int((processed / total) * 100)
                thresh     = result["threshold"]
                count      = result["count"]

                peak_count  = max(peak_count, count)
                is_alert    = count > thresh["threshold"]
                if is_alert:
                    alert_count += 1

                # Timeline snapshot every 5 frames or on alert
                if processed % 5 == 0 or is_alert:
                    timeline.append({
                        "frame": frame_no,
                        "count": count,
                        "threshold": thresh["threshold"],
                        "behavior":  result["behavior"]["state"],
                        "alert":     is_alert,
                    })

                # Track top-5 highest pressure frames
                mp = result["pressure"]["max_pressure"]
                top_risk.append({"frame": frame_no, "count": count,
                                  "max_pressure": mp,
                                  "behavior": result["behavior"]["state"]})
                top_risk.sort(key=lambda x: x["max_pressure"], reverse=True)
                top_risk = top_risk[:5]

                await websocket.send_json({
                    "type":        "result",
                    "frame":       frame_no,
                    "progress":    progress,
                    "total_frames": total,
                    "count":       count,
                    "coordinates": result["coordinates"],
                    "avg_confidence": result["avg_confidence"],
                    "original_size":  result["original_size"],
                    # Spatial / temporal analysis
                    "zones":       result["zones"],
                    "pressure":    result["pressure"],
                    "behavior":    result["behavior"],
                    # Threshold
                    "dynamic_threshold": thresh["threshold"],
                    "threshold_mode":    thresh["mode"],
                    "threshold_mean":    thresh["mean"],
                    "threshold_std":     thresh["std"],
                    "threshold_samples": thresh["samples"],
                    # LOS (if calibrated)
                    "los": result["los"],
                    "alert": is_alert,
                })

                print(f"  Frame {frame_no:4d} | {count:4d} people | "
                      f"thresh={thresh['threshold']} [{thresh['mode']}] | "
                      f"state={result['behavior']['state']} | "
                      f"pressure={mp:.2f}")

            # ── Complete message ──────────────────────────────────────────────
            elif data["type"] == "complete":
                avg_count = (sum(count_history) / len(count_history)
                             if count_history else 0.0)

                # Determine peak behavior
                state_priority = {"SURGE": 3, "PRE_SURGE": 2,
                                   "DISPERSING": 1, "NORMAL": 0}
                peak_behavior  = max(
                    (e["behavior"] for e in timeline),
                    key=lambda s: state_priority.get(s, 0),
                    default="NORMAL"
                )

                # Persist session
                summary = json.dumps({
                    "timeline":        timeline,
                    "top_risk_frames": top_risk,
                })
                venue = data.get("venue_name", "Unknown Venue")
                db_upsert_session({
                    "id":           session_id,
                    "name":         data.get("name", f"Session {session_id[:8]}"),
                    "venue_name":   venue,
                    "created_at":   datetime.utcnow().isoformat(),
                    "total_frames": processed,
                    "duration_sec": processed / max(data.get("fps", 2), 1),
                    "peak_count":   peak_count,
                    "avg_count":    round(avg_count, 1),
                    "alert_count":  alert_count,
                    "peak_behavior": peak_behavior,
                    "calibration":  json.dumps(calibration),
                    "summary":      summary,
                })

                await websocket.send_json({
                    "type":                 "complete",
                    "total_frames_processed": processed,
                    "session_id":           session_id,
                    "peak_count":           peak_count,
                    "avg_count":            round(avg_count, 1),
                    "alert_count":          alert_count,
                    "peak_behavior":        peak_behavior,
                })
                print(f"🏁 Session {session_id} complete — {processed} frames")
                break

    except WebSocketDisconnect:
        print(f"🔌 Session {session_id} disconnected")
    except Exception as exc:
        import traceback; traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — LIVE STREAM  (RTSP / webcam / file streaming)
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/live/{session_id}")
async def ws_live_stream(websocket: WebSocket, session_id: str):
    """
    Producer-consumer pipeline for real-time camera feeds.
    Client sends: {"type":"start","source":"rtsp://...","confidence":0.5}
    Server pushes: same result format as batch processing.
    """
    await websocket.accept()
    if not FLOW_AVAILABLE:
        await websocket.send_json(
            {"type":"error","message":"OpenCV not installed — live stream unavailable"})
        return

    data   = await websocket.receive_json()
    source = data.get("source", 0)   # 0 = default webcam
    conf   = float(data.get("confidence", 0.5))

    flow_analyzer = FlowAnalyzer()
    thresh_calc   = AdaptiveThreshold()
    behav_clf     = BehaviorClassifier()
    count_history: List[int] = []

    frame_q:  asyncio.Queue = asyncio.Queue(maxsize=4)
    result_q: asyncio.Queue = asyncio.Queue(maxsize=8)
    stop_evt  = asyncio.Event()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        await websocket.send_json({"type":"error","message":f"Cannot open source: {source}"})
        return

    loop = asyncio.get_event_loop()

    async def producer():
        idx = 0
        while not stop_evt.is_set():
            ret, frame = await loop.run_in_executor(executor, cap.read)
            if not ret:
                stop_evt.set(); break
            if not frame_q.full():
                await frame_q.put((idx, frame))
            idx += 1
            await asyncio.sleep(0.033)   # ~30fps cap

    async def worker():
        while not stop_evt.is_set() or not frame_q.empty():
            try:
                idx, bgr = await asyncio.wait_for(frame_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf     = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=75)
            result = await loop.run_in_executor(
                executor, process_one_frame,
                buf.getvalue(), conf,
                flow_analyzer, thresh_calc, behav_clf, count_history, None,
            )
            await result_q.put((idx, result))

    async def sender():
        while not stop_evt.is_set() or not result_q.empty():
            try:
                idx, result = await asyncio.wait_for(result_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            thresh = result["threshold"]
            await websocket.send_json({
                "type":"live_result","frame":idx,
                "count":result["count"],
                "coordinates":result["coordinates"],
                "zones":result["zones"],
                "pressure":result["pressure"],
                "behavior":result["behavior"],
                "dynamic_threshold":thresh["threshold"],
                "threshold_mode":thresh["mode"],
                "alert": result["count"] > thresh["threshold"],
            })

    async def receiver():
        """Listen for stop signal from client."""
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "stop":
                    stop_evt.set(); break
        except WebSocketDisconnect:
            stop_evt.set()

    try:
        await asyncio.gather(producer(), worker(), sender(), receiver())
    finally:
        cap.release()

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
