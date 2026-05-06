# ═══════════════════════════════════════════════════════════════════════════════
# backend.py — AI Crowd Intelligence System v3.2   (FastAPI entry point)
#
#   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
#
# Key changes vs v3.1
#   • SessionState dataclass — all per-session objects in one place
#   • Perspective homography calibration (4-pt DLT, numpy only)
#   • Walkable mask frame-diff cache — SegFormer called only on scene change
#   • P2PNet 128-quantisation rounds UP to preserve aspect ratio
#   • BehaviorClassifier now receives EMA mean, not alert threshold
#   • TrajectoryTracker + CUSUMDetector integrated into session pipeline
#   • Optical flow restricted to walkable zone
#   • ThreadPoolExecutor max_workers=1 per server (sequential safety)
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio, base64, io, json, os, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from constants import GRID_COLS, GRID_ROWS, SCENE_CHANGE_THRESH
from database import init_db, db_upsert_session, db_list_sessions, db_get_session, db_delete_session
from analyzers import DistanceAnalyzer, FlowAnalyzer, BehaviorClassifier, fruin_los, TrajectoryTracker, CUSUMDetector
from zones import DynamicZoneEngine, AdaptiveThresholdV2, compute_pressure
from walkable_area_estimator import WalkableAreaEstimator

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
P2PNET_DIR   = os.path.join(BASE_DIR, "CrowdCounting-P2PNet")
WEIGHTS_PATH = os.path.join(P2PNET_DIR, "weights", "SHTechA.pth")
sys.path.insert(0, P2PNET_DIR)
from models import build_model  # noqa: E402

executor = ThreadPoolExecutor(max_workers=2)   # 2 allows overlap of IO and CPU


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE
# ═══════════════════════════════════════════════════════════════════════════════

model: Any = None; device: Any = None; transform: Any = None
walkable_estimator: Optional[WalkableAreaEstimator] = None


class _Args:
    backbone = "vgg16_bn"; row = 2; line = 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform, walkable_estimator
    print("🚀 Loading P2PNet …")
    init_db()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(_Args(), training=False)
    ckpt      = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    walkable_estimator = WalkableAreaEstimator()
    print(f"✅ P2PNet ready on {device} | "
          f"SegFormer={walkable_estimator.segformer.is_loaded()} | "
          f"YOLO={YOLO_AVAILABLE} | Flow={CV2_AVAILABLE}")
    yield
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(lifespan=lifespan, title="AI Crowd Intelligence API v3.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {
        "status": "running", "model": "P2PNet (SHTechA)",
        "device": str(device), "flow": CV2_AVAILABLE,
        "yolo": YOLO_AVAILABLE,
        "segformer": walkable_estimator.segformer.is_loaded() if walkable_estimator else False,
        "version": "3.2.0",
    }

@app.get("/sessions")
async def list_sessions():
    return db_list_sessions()

@app.get("/sessions/{sid}")
async def get_session(sid: str):
    s = db_get_session(sid)
    if not s: raise HTTPException(404, "Not found")
    return s

@app.delete("/sessions/{sid}")
async def delete_session(sid: str):
    if not db_delete_session(sid): raise HTTPException(404, "Not found")
    return {"deleted": sid}

@app.get("/sessions/{sid}/report")
async def get_report(sid: str):
    s = db_get_session(sid)
    if not s: raise HTTPException(404, "Not found")
    summary = json.loads(s.get("summary") or "{}")
    report  = {
        "report_type":  "crowd_intelligence_incident_report_v3",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "session_id":   sid,
        "venue":        s.get("venue_name","Unknown"),
        "recorded_at":  s.get("created_at"),
        "statistics":   {k: s[k] for k in
                         ("total_frames","peak_count","avg_count","alert_count","peak_behavior")},
        "timeline":              summary.get("timeline", []),
        "top_risk_frames":       summary.get("top_risk_frames", []),
        "calibration":           json.loads(s.get("calibration") or "null"),
        "recommendation":        _recommendation(s),
    }
    return JSONResponse(report)

def _recommendation(s: Dict) -> str:
    if s["alert_count"] > 5 or s["peak_behavior"] == "SURGE":
        return "CRITICAL: Multiple surge events detected. Recommend immediate review."
    if s["alert_count"] > 0 or s["peak_behavior"] == "PRE_SURGE":
        return "WARNING: Density anomalies detected. Review crowd management protocols."
    return "NOMINAL: No significant crowd safety incidents detected."


# ═══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_homography(pixel_pts: list, world_w: float, world_h: float) -> list:
    """
    4-point DLT homography: pixel space → world space (metres).
    pixel_pts: [[x0,y0],…,[x3,y3]] top-left, top-right, bottom-right, bottom-left.
    Returns 3×3 H as a flat 9-element list (row-major).
    """
    src = np.float64(pixel_pts)
    dst = np.float64([[0,0],[world_w,0],[world_w,world_h],[0,world_h]])
    A = []
    for (sx,sy),(dx,dy) in zip(src, dst):
        A += [[-sx,-sy,-1, 0, 0, 0, sx*dx, sy*dx, dx],
              [  0,  0, 0,-sx,-sy,-1, sx*dy, sy*dy, dy]]
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    return H.ravel().tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    """All per-session stateful objects in one place — eliminates threading risk."""
    flow_analyzer:  FlowAnalyzer       = field(default_factory=FlowAnalyzer)
    thresh_calc:    AdaptiveThresholdV2 = field(default_factory=lambda: AdaptiveThresholdV2(z_factor=3.5))
    behav_clf:      BehaviorClassifier = field(default_factory=BehaviorClassifier)
    dist_analyzer:  DistanceAnalyzer   = field(default_factory=DistanceAnalyzer)
    zone_engine:    DynamicZoneEngine  = field(default_factory=DynamicZoneEngine)
    traj_tracker:   TrajectoryTracker  = field(default_factory=TrajectoryTracker)
    cusum:          CUSUMDetector      = field(default_factory=CUSUMDetector)
    count_history:  List[int]          = field(default_factory=list)
    # Walkable mask frame-diff cache
    prev_gray:      Optional[np.ndarray] = field(default=None)
    cached_mask:    Optional[np.ndarray] = field(default=None)
    cached_walk_meta: Optional[Dict]     = field(default=None)
    # Calibration
    calibration:    Optional[Dict]       = field(default=None)


# ═══════════════════════════════════════════════════════════════════════════════
# P2PNET INFERENCE  (v3.2: rounds UP to nearest 128 to preserve aspect ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_p2pnet(img: Image.Image, confidence: float):
    ow, oh = img.size
    w, h   = ow, oh
    if max(w, h) > 2048:
        s = 2048 / max(w, h)
        w, h = int(w * s), int(h * s)
    # Round UP to nearest 128 — preserves aspect ratio better than truncation
    nw = max(128, ((w + 127) // 128) * 128)
    nh = max(128, ((h + 127) // 128) * 128)
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
    avg_c   = float(sco_sel.mean()) if len(sco_sel) else 0.0
    return coords, sco_sel.tolist(), avg_c, (ow, oh)


# ═══════════════════════════════════════════════════════════════════════════════
# WALKABLE MASK WITH SCENE-CHANGE CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_walkable(img: Image.Image, state: SessionState):
    """Return walkable mask+meta; re-estimate only when the scene changes."""
    if CV2_AVAILABLE:
        rgb  = np.array(img.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.array(img.convert("L"), dtype=np.uint8)

    recompute = True
    if state.prev_gray is not None and state.cached_mask is not None:
        if state.prev_gray.shape == gray.shape:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - state.prev_gray.astype(np.float32)))) / 255.0
            recompute = diff > SCENE_CHANGE_THRESH

    state.prev_gray = gray
    if recompute:
        state.cached_mask, state.cached_walk_meta = walkable_estimator.estimate(img)

    return state.cached_mask, state.cached_walk_meta


def _encode_walkable_mask(mask: np.ndarray, cols: int, rows: int) -> List[List[float]]:
    H, W = mask.shape
    return [
        [round(float(mask[int(r*H/rows):int((r+1)*H/rows), int(c*W/cols):int((c+1)*W/cols)].mean()), 3)
         for c in range(cols)]
        for r in range(rows)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-FRAME PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_one_frame(
    frame_bytes: bytes,
    confidence:  float,
    state:       SessionState,
) -> Dict:
    img    = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    ow, oh = img.size
    cal    = state.calibration or {}
    ppm    = cal.get("pixels_per_meter")
    H_mat  = cal.get("H_matrix")        # flat 9-element list or None

    # 1. P2PNet crowd detection
    coords, conf_scores, avg_conf, _ = _run_p2pnet(img, confidence)
    count = len(coords)
    state.count_history.append(count)

    # 2. Walkable area (cached per scene change)
    walkable_mask, walkable_meta = _get_walkable(img, state)

    # 3. Proximity / distance (homography-aware)
    dist_meta = state.dist_analyzer.analyze(coords, ow, oh, ppm, H_mat)

    # 4. Optical flow — restricted to walkable zone
    flow_zones = state.flow_analyzer.update(img, walkable_mask)

    # 5. Dynamic zone computation
    state.zone_engine.pixels_per_meter = ppm
    zone_list, zone_agg = state.zone_engine.compute(coords, ow, oh, walkable_mask, flow_zones)

    # 6. Pressure field
    pressure = compute_pressure(zone_list, state.count_history)

    # 7. Adaptive threshold v2.2
    thresh_info = state.thresh_calc.update(count, walkable_meta, dist_meta, zone_agg, zone_list, ppm)

    # 8. Behavior classifier — pass EMA mean, not alert threshold (v3.2 fix)
    mean_baseline = thresh_info.get("mean") or float(thresh_info["threshold"])
    behavior      = state.behav_clf.update(
        count, zone_list, mean_baseline,
        dist_meta.get("proximity_score", 0.0),
    )

    # 9. Fruin global LOS
    los_data: Dict = {}
    if ppm and ppm > 0 and zone_agg.get("total_walkable_m2", 0) > 0:
        wm2  = zone_agg["total_walkable_m2"]
        dens = count / max(wm2, 0.001)
        los_data = {"density_sqm": round(dens, 2), "los": fruin_los(dens), "area_m2": round(wm2, 1)}

    # 10. Trajectory tracking
    trajectories = state.traj_tracker.update(coords)

    # 11. CUSUM anomaly detection
    cusum_result = state.cusum.update(float(count))

    # 12. Compact walkable mask for frontend
    mask_grid = _encode_walkable_mask(walkable_mask, GRID_COLS, GRID_ROWS)

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
        "walkable_mask_grid": mask_grid,
        "trajectories":       trajectories,
        "cusum":              cusum_result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — STREAMING FRAME PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/process-frames/{session_id}")
async def ws_process_frames(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"📡 Session {session_id} connected")

    state = SessionState()

    processed = 0; alert_count = 0; peak_count = 0
    timeline:  List[Dict] = []
    top_risk:  List[Dict] = []
    thresh_samples: List[Dict] = []
    _z_applied = False

    try:
        while True:
            data = await websocket.receive_json()

            # ── Calibration ──────────────────────────────────────────────────
            if data["type"] == "calibration":
                cal = data.get("calibration", {})
                pts = cal.get("pixel_points")
                if pts and len(pts) == 4:
                    try:
                        H = _compute_homography(
                            pts, cal.get("world_width", 1), cal.get("world_height", 1)
                        )
                        cal = {**cal, "H_matrix": H}
                    except Exception as e:
                        print(f"⚠️ Homography failed: {e}")
                state.calibration = cal
                await websocket.send_json({
                    "type":     "calibration_ack",
                    "H_matrix": state.calibration.get("H_matrix"),
                })
                continue

            # ── Frame ─────────────────────────────────────────────────────────
            if data["type"] == "frame":
                frame_no   = data["frame_number"]
                total      = data.get("total_frames", 1)
                confidence = float(data.get("confidence", 0.5))

                if not _z_applied:
                    state.thresh_calc.z_factor = max(3.0, min(5.0, float(data.get("z_factor", 3.5))))
                    _z_applied = True

                frame_bytes = base64.b64decode(data["frame_data"].split(",")[1])
                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor, process_one_frame, frame_bytes, confidence, state
                )

                processed += 1
                count     = result["count"]
                thresh    = result["threshold"]
                peak_count = max(peak_count, count)
                is_alert  = count > thresh["threshold"]
                if is_alert: alert_count += 1

                if processed % 5 == 0 or is_alert:
                    timeline.append({
                        "frame":     frame_no,
                        "count":     count,
                        "threshold": thresh["threshold"],
                        "behavior":  result["behavior"]["state"],
                        "alert":     is_alert,
                        "cusum":     result["cusum"].get("alert", False),
                    })
                if processed % 10 == 0:
                    thresh_samples.append({"frame": frame_no, **thresh.get("signals", {})})

                mp = result["pressure"]["max_pressure"]
                top_risk.append({"frame": frame_no, "count": count,
                                "max_pressure": mp, "behavior": result["behavior"]["state"]})
                top_risk.sort(key=lambda x: x["max_pressure"], reverse=True)
                top_risk = top_risk[:5]

                await websocket.send_json({
                    "type":               "result",
                    "frame":              frame_no,
                    "progress":           int(processed / total * 100),
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
                    "trajectories":       result["trajectories"],
                    "cusum":              result["cusum"],
                })

                print(f"  Frame {frame_no:4d} | {count:4d}p | "
                    f"T={thresh['threshold']} | {result['behavior']['state']} | "
                    f"cusum={result['cusum']['alert']}")

            # ── Session complete ──────────────────────────────────────────────
            elif data["type"] == "complete":
                avg_count = sum(state.count_history) / max(len(state.count_history), 1)
                prio = {"SURGE":3,"PRE_SURGE":2,"DISPERSING":1,"NORMAL":0}
                peak_beh = max((e["behavior"] for e in timeline),
                            key=lambda s: prio.get(s, 0), default="NORMAL")
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
                    "peak_behavior": peak_beh,
                    "calibration":  json.dumps(state.calibration),
                    "summary":      json.dumps({
                        "timeline":             timeline,
                        "top_risk_frames":      top_risk,
                        "threshold_signals":    thresh_samples[:10],
                    }),
                })
                await websocket.send_json({
                    "type":                   "complete",
                    "total_frames_processed": processed,
                    "session_id":             session_id,
                    "peak_count":             peak_count,
                    "avg_count":              round(avg_count, 1),
                    "alert_count":            alert_count,
                    "peak_behavior":          peak_beh,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)