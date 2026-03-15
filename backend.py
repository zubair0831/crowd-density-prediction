# ═══════════════════════════════════════════════════════════════════════════════
# backend.py — AI Crowd Intelligence System v3  (FastAPI entry point)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Run:
#   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
#
# Modules in same directory:
#   constants.py, database.py, analyzers.py, zones.py,
#   walkable_area_estimator.py, segformer_walkable.py
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
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

# ─── Local modules ────────────────────────────────────────────────────────────
from constants import GRID_COLS, GRID_ROWS
from database import (
    init_db, db_upsert_session, db_list_sessions, db_get_session, db_delete_session,
)
from analyzers import DistanceAnalyzer, FlowAnalyzer, BehaviorClassifier, fruin_los
from zones import DynamicZoneEngine, AdaptiveThresholdV2, compute_pressure
from walkable_area_estimator import WalkableAreaEstimator

# ─── Optional OpenCV ──────────────────────────────────────────────────────────
try:
    import cv2
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False
    print("⚠️  cv2 not found — optical flow disabled")

# ─── Optional YOLOv8 ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not found — YOLO tier using heuristics only")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
P2PNET_DIR   = os.path.join(BASE_DIR, "CrowdCounting-P2PNet")
WEIGHTS_PATH = os.path.join(P2PNET_DIR, "weights", "SHTechA.pth")
sys.path.insert(0, P2PNET_DIR)
from models import build_model  # noqa: E402  (P2PNet repo)

executor = ThreadPoolExecutor(max_workers=4)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE
# ═══════════════════════════════════════════════════════════════════════════════

model:               Any = None
device:              Any = None
transform:           Any = None
walkable_estimator: Optional[WalkableAreaEstimator] = None


class Args:
    backbone = "vgg16_bn"
    row      = 2
    line     = 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform, walkable_estimator
    print("🚀 Loading P2PNet …")
    init_db()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(Args(), training=False)
    ckpt      = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    walkable_estimator = WalkableAreaEstimator()
    print(
        f"✅ P2PNet ready on {device} | "
        f"YOLO={YOLO_AVAILABLE} | Flow={FLOW_AVAILABLE} | "
        f"SegFormer={walkable_estimator.segformer.is_loaded()}"
    )
    yield
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(lifespan=lifespan, title="AI Crowd Intelligence API v3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status":           "running",
        "model":            "P2PNet (SHTechA)",
        "device":           str(device),
        "flow":             FLOW_AVAILABLE,
        "yolo":             YOLO_AVAILABLE,
        "segformer":        walkable_estimator.segformer.is_loaded() if walkable_estimator else False,
        "version":          "3.1.0",
        "threshold_engine": "adaptive_v2_multi_signal",
    }


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
        raise HTTPException(404, "Not found")
    return {"deleted": sid}


@app.get("/sessions/{sid}/report")
async def get_report(sid: str):
    s = db_get_session(sid)
    if not s:
        raise HTTPException(404, "Not found")
    summary = json.loads(s.get("summary") or "{}")
    report  = {
        "report_type":  "crowd_intelligence_incident_report_v3",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "session_id":   sid,
        "venue":        s.get("venue_name", "Unknown"),
        "recorded_at":  s.get("created_at"),
        "statistics": {
            "total_frames":  s["total_frames"],
            "peak_count":    s["peak_count"],
            "avg_count":     round(s["avg_count"], 1),
            "alert_count":   s["alert_count"],
            "peak_behavior": s["peak_behavior"],
        },
        "timeline":                  summary.get("timeline", []),
        "top_risk_frames":           summary.get("top_risk_frames", []),
        "calibration":               json.loads(s.get("calibration") or "null"),
        "threshold_signals_sample":  summary.get("threshold_signals_sample"),
        "recommendation":            _recommendation(s),
    }
    return JSONResponse(report)


def _recommendation(s: Dict) -> str:
    if s["alert_count"] > 5 or s["peak_behavior"] == "SURGE":
        return "CRITICAL: Multiple surge events detected. Recommend immediate review."
    if s["alert_count"] > 0 or s["peak_behavior"] == "PRE_SURGE":
        return "WARNING: Density anomalies detected. Review crowd management protocols."
    return "NOMINAL: No significant crowd safety incidents detected."


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_p2pnet(img: Image.Image, confidence: float):
    ow, oh = img.size
    max_dim = 2048
    w, h    = ow, oh
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        w, h = int(w * s), int(h * s)
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


def _encode_walkable_mask(mask: np.ndarray, cols: int, rows: int) -> List[List[float]]:
    """Compress walkable mask to GRID_ROWS × GRID_COLS grid of walkable fractions."""
    H, W = mask.shape
    grid: List[List[float]] = []
    for r in range(rows):
        row_data: List[float] = []
        for c in range(cols):
            r0 = int(r * H / rows);     r1 = int((r + 1) * H / rows)
            c0 = int(c * W / cols);     c1 = int((c + 1) * W / cols)
            row_data.append(round(float(mask[r0:r1, c0:c1].mean()), 3))
        grid.append(row_data)
    return grid


def process_one_frame(
    frame_bytes:   bytes,
    confidence:    float,
    flow_analyzer: FlowAnalyzer,
    thresh_calc:   AdaptiveThresholdV2,
    behav_clf:     BehaviorClassifier,
    dist_analyzer: DistanceAnalyzer,
    zone_engine:   DynamicZoneEngine,
    count_history: List[int],
    calibration:   Optional[Dict] = None,
) -> Dict:
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    ow, oh = img.size

    # 1. P2PNet crowd detection
    coords, conf_scores, avg_conf, _ = _run_p2pnet(img, confidence)
    count = len(coords)
    count_history.append(count)

    ppm = calibration.get("pixels_per_meter") if calibration else None

    # 2. Walkable area estimation (three-tier: SegFormer + YOLO + heuristic)
    walkable_mask, walkable_meta = walkable_estimator.estimate(img)

    # 3. Proximity / distance analysis
    dist_meta = dist_analyzer.analyze(coords, ow, oh, ppm)

    # 4. Optical flow
    flow_zones = flow_analyzer.update(img)

    # 5. Dynamic zone computation (walkable-area-aware)
    zone_engine.pixels_per_meter = ppm
    zone_list, zone_agg = zone_engine.compute(coords, ow, oh, walkable_mask, flow_zones)

    # 6. Pressure field
    pressure = compute_pressure(zone_list, count_history)

    # 7. Adaptive threshold v2 (multi-signal)
    thresh_info = thresh_calc.update(
        count, walkable_meta, dist_meta, zone_agg, zone_list, ppm
    )

    # 8. Behavior classification (proximity-aware)
    behavior = behav_clf.update(
        count, zone_list, thresh_info["threshold"],
        dist_meta.get("proximity_score", 0.0),
    )

    # 9. Global Fruin LOS
    los_data: Dict = {}
    if ppm and ppm > 0 and zone_agg.get("total_walkable_m2", 0) > 0:
        walkable_m2 = zone_agg["total_walkable_m2"]
        global_dens = count / max(walkable_m2, 0.001)
        los_data = {
            "density_sqm": round(global_dens, 2),
            "los":         fruin_los(global_dens),
            "area_m2":     round(walkable_m2, 1),
        }

    # 10. Compact walkable mask for frontend
    mask_data = _encode_walkable_mask(walkable_mask, GRID_COLS, GRID_ROWS)

    return {
        "count":              count,
        "coordinates":        coords,
        "avg_confidence":     round(avg_conf, 3),
        "original_size":      [ow, oh],
        "zones":              zone_list,
        "zone_aggregate":     zone_agg,
        "pressure":           pressure,
        "behavior":           behavior,
        "threshold":          thresh_info,
        "los":                los_data,
        "distance":           dist_meta,
        "walkable":           walkable_meta,
        "walkable_mask_grid": mask_data,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — FRAME BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/process-frames/{session_id}")
async def ws_process_frames(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"📡 Session {session_id} connected")

    flow_analyzer = FlowAnalyzer()
    thresh_calc   = AdaptiveThresholdV2(z_factor=3.0)  # 3σ → far fewer false positives
    behav_clf     = BehaviorClassifier()
    dist_analyzer = DistanceAnalyzer()
    zone_engine   = DynamicZoneEngine()
    count_history: List[int] = []
    calibration:   Optional[Dict] = None
    _z_factor_applied = False  # track whether client z_factor has been applied

    processed = 0;  alert_count = 0;  peak_count = 0
    timeline:               List[Dict] = []
    top_risk:               List[Dict] = []
    thresh_signals_samples: List[Dict] = []

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "calibration":
                calibration = data.get("calibration")
                await websocket.send_json({"type": "calibration_ack"})
                continue

            if data["type"] == "frame":
                frame_no    = data["frame_number"]
                total       = data.get("total_frames", 1)
                confidence  = float(data.get("confidence", 0.5))
                frame_bytes = base64.b64decode(data["frame_data"].split(",")[1])

                # Apply client z_factor on first frame (clamped to sane range)
                if not _z_factor_applied:
                    client_z = float(data.get("z_factor", 3.0))
                    thresh_calc.z_factor = max(2.5, min(5.0, client_z))
                    _z_factor_applied = True

                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor, process_one_frame,
                    frame_bytes, confidence,
                    flow_analyzer, thresh_calc, behav_clf,
                    dist_analyzer, zone_engine,
                    count_history, calibration,
                )

                processed  += 1
                progress    = int((processed / total) * 100)
                thresh      = result["threshold"]
                count       = result["count"]
                peak_count  = max(peak_count, count)
                is_alert    = count > thresh["threshold"]
                if is_alert:
                    alert_count += 1

                if processed % 5 == 0 or is_alert:
                    timeline.append({
                        "frame":     frame_no,
                        "count":     count,
                        "threshold": thresh["threshold"],
                        "behavior":  result["behavior"]["state"],
                        "alert":     is_alert,
                    })
                if processed % 10 == 0:
                    thresh_signals_samples.append({
                        "frame": frame_no,
                        **thresh.get("signals", {}),
                    })

                mp = result["pressure"]["max_pressure"]
                top_risk.append({
                    "frame":        frame_no,
                    "count":        count,
                    "max_pressure": mp,
                    "behavior":     result["behavior"]["state"],
                })
                top_risk.sort(key=lambda x: x["max_pressure"], reverse=True)
                top_risk = top_risk[:5]

                await websocket.send_json({
                    "type":               "result",
                    "frame":              frame_no,
                    "progress":           progress,
                    "total_frames":       total,
                    "count":              count,
                    "coordinates":        result["coordinates"],
                    "avg_confidence":     result["avg_confidence"],
                    "original_size":      result["original_size"],
                    "zones":              result["zones"],
                    "zone_aggregate":     result["zone_aggregate"],
                    "pressure":           result["pressure"],
                    "behavior":           result["behavior"],
                    "distance":           result["distance"],
                    "walkable":           result["walkable"],
                    "walkable_mask_grid": result["walkable_mask_grid"],
                    "dynamic_threshold":  thresh["threshold"],
                    "threshold_mode":     thresh["mode"],
                    "threshold_mean":     thresh["mean"],
                    "threshold_std":      thresh["std"],
                    "threshold_samples":  thresh["samples"],
                    "threshold_signals":  thresh.get("signals"),
                    "los":                result["los"],
                    "alert":              is_alert,
                })

                print(
                    f"  Frame {frame_no:4d} | {count:4d}p | "
                    f"T={thresh['threshold']} [{thresh['mode']}] | "
                    f"prox={result['distance'].get('proximity_score', 0):.2f} | "
                    f"{result['behavior']['state']}"
                )

            elif data["type"] == "complete":
                avg_count = (
                    sum(count_history) / len(count_history) if count_history else 0.0
                )
                state_priority = {"SURGE": 3, "PRE_SURGE": 2, "DISPERSING": 1, "NORMAL": 0}
                peak_behavior  = max(
                    (e["behavior"] for e in timeline),
                    key=lambda s: state_priority.get(s, 0),
                    default="NORMAL",
                )
                summary = json.dumps({
                    "timeline":                  timeline,
                    "top_risk_frames":           top_risk,
                    "threshold_signals_sample":  thresh_signals_samples[:10],
                })
                db_upsert_session({
                    "id":           session_id,
                    "name":         data.get("name", f"Session {session_id[:8]}"),
                    "venue_name":   data.get("venue_name", "Unknown"),
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
                    "type":                  "complete",
                    "total_frames_processed": processed,
                    "session_id":            session_id,
                    "peak_count":            peak_count,
                    "avg_count":             round(avg_count, 1),
                    "alert_count":           alert_count,
                    "peak_behavior":         peak_behavior,
                })
                print(f"🏁 Session {session_id} complete — {processed} frames")
                break

    except WebSocketDisconnect:
        print(f"🔌 Session {session_id} disconnected")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)