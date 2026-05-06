# ═══════════════════════════════════════════════════════════════════════════════
# analyzers.py — Distance, Flow, Behavior, Trajectory & CUSUM modules  v3.2
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from PIL import Image

from constants import (
    GRID_COLS, GRID_ROWS,
    CUSUM_K, CUSUM_H, CUSUM_WARMUP,
    TRAJ_MAX_DIST_PX, TRAJ_MAX_LOST,
)

try:
    import cv2
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False
    print("⚠️  cv2 not found — optical flow disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# FRUIN LEVEL OF SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

def fruin_los(density_sqm: Optional[float]) -> str:
    if density_sqm is None: return "?"
    thresholds = [(0.5,"A"),(1.0,"B"),(1.7,"C"),(2.7,"D"),(4.0,"E")]
    for t, grade in thresholds:
        if density_sqm <= t: return grade
    return "F"


# ═══════════════════════════════════════════════════════════════════════════════
# DISTANCE ANALYZER  —  KD-Tree KNN proximity (homography-aware)
# ═══════════════════════════════════════════════════════════════════════════════

class DistanceAnalyzer:
    """
    Computes nearest-neighbour distances and proximity metrics.
    When H_matrix (perspective homography pixel→world/metres) is supplied,
    all distances are computed in metric space; otherwise in pixel space.
    """

    MIN_SAFE_FRAC = 0.035   # safe dist as fraction of image diagonal (no calibration)
    MIN_SAFE_M    = 0.8     # LOS-B boundary in metres (when calibrated)

    @staticmethod
    def _to_world(coords: List[List[float]], H: np.ndarray) -> np.ndarray:
        pts = np.array(coords, dtype=np.float64)
        h   = np.hstack([pts, np.ones((len(pts), 1))])
        w   = (H @ h.T).T
        return w[:, :2] / w[:, 2:3]

    def analyze(
        self,
        coords: List[List[float]],
        img_w: int,
        img_h: int,
        pixels_per_meter: Optional[float] = None,
        H_matrix: Optional[List] = None,
    ) -> Dict:
        n = len(coords)
        if n == 0:
            return self._empty()
        if n == 1:
            return {"count":1,"median_nn_dist":None,"mean_nn_dist":None,
                    "min_safe_px":None,"violations":0,
                    "proximity_violation_ratio":0.0,"proximity_score":0.0,
                    "nn_distances":[],"k_used":0,"space":"pixel"}

        # Choose coordinate space
        if H_matrix is not None:
            H   = np.array(H_matrix, dtype=np.float64).reshape(3, 3)
            pts = self._to_world(coords, H)
            min_safe = self.MIN_SAFE_M
            space = "world_m"
        elif pixels_per_meter:
            pts      = np.array(coords, dtype=np.float32) / pixels_per_meter
            min_safe = self.MIN_SAFE_M
            space    = "world_m"
        else:
            pts      = np.array(coords, dtype=np.float32)
            diag     = math.sqrt(img_w**2 + img_h**2)
            min_safe = diag * self.MIN_SAFE_FRAC
            space    = "pixel"

        k = min(5, n - 1)
        dists, _ = KDTree(pts).query(pts, k=k + 1)
        nn       = dists[:, 1]

        median_d  = float(np.median(nn))
        mean_d    = float(nn.mean())
        violations = int((nn < min_safe).sum())

        lo, hi    = min_safe * 0.3, min_safe * 2.0
        prox      = 0.0 if median_d >= hi else (1.0 if median_d <= lo
                    else 1.0 - (median_d - lo) / (hi - lo))
        prox      = round(max(0.0, min(1.0, prox)), 3)

        return {
            "count":                     n,
            "median_nn_dist":            round(median_d, 4),
            "mean_nn_dist":              round(mean_d, 4),
            "min_safe_px":               round(min_safe, 4),
            "violations":                violations,
            "proximity_violation_ratio": round(violations / n, 3),
            "proximity_score":           prox,
            "nn_distances":              [round(d, 3) for d in nn.tolist()],
            "k_used":                    k,
            "space":                     space,
        }

    @staticmethod
    def _empty() -> Dict:
        return {"count":0,"median_nn_dist":None,"mean_nn_dist":None,
                "min_safe_px":None,"violations":0,
                "proximity_violation_ratio":0.0,"proximity_score":0.0,
                "nn_distances":[],"k_used":0,"space":"pixel"}


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW ANALYZER  —  Farneback optical flow, masked to walkable region
# ═══════════════════════════════════════════════════════════════════════════════

class FlowAnalyzer:
    def __init__(self, grid_cols: int = GRID_COLS, grid_rows: int = GRID_ROWS) -> None:
        self.grid_cols  = grid_cols
        self.grid_rows  = grid_rows
        self.prev_gray  = None

    def update(
        self,
        pil_img: Image.Image,
        walkable_mask: Optional[np.ndarray] = None,   # (H,W) uint8 — restrict flow
    ) -> Optional[List[Dict]]:
        if not FLOW_AVAILABLE:
            return None
        rgb  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        self.prev_gray = gray

        # Mask flow to walkable region — prevents background / obstacle motion noise
        if walkable_mask is not None:
            msk = cv2.resize(walkable_mask, (gray.shape[1], gray.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            flow[~msk] = 0.0

        h, w = gray.shape
        results: List[Dict] = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                r0 = int(row * h / self.grid_rows); r1 = int((row+1) * h / self.grid_rows)
                c0 = int(col * w / self.grid_cols); c1 = int((col+1) * w / self.grid_cols)
                cf = flow[r0:r1, c0:c1]
                fx, fy = cf[..., 0], cf[..., 1]
                div = 0.0
                if fx.shape[0] > 1 and fx.shape[1] > 1:
                    div = float(np.gradient(fx, axis=1).mean() + np.gradient(fy, axis=0).mean())
                results.append({
                    "row": row, "col": col,
                    "mean_dx":   float(fx.mean()),
                    "mean_dy":   float(fy.mean()),
                    "divergence": round(div, 4),
                    "magnitude":  round(float(np.sqrt(fx**2 + fy**2).mean()), 4),
                })
        return results

    def reset(self) -> None:
        self.prev_gray = None


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY TRACKER  —  Hungarian-matched person tracks
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryTracker:
    """
    Lightweight multi-object tracker.  Links P2PNet point detections frame-to-frame
    using the Hungarian algorithm on L2 distance.  Returns per-track velocity vectors
    for anomalous-velocity detection and tripwire crossing.
    """

    def __init__(
        self,
        max_dist: float = TRAJ_MAX_DIST_PX,
        max_lost: int   = TRAJ_MAX_LOST,
    ) -> None:
        self.max_dist = max_dist
        self.max_lost = max_lost
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0

    def update(self, coords: List[List[float]]) -> List[Dict]:
        """
        Returns list of active tracks:
          {id, pos:[x,y], vel:[dx,dy], age(frames), speed(px/frame)}
        """
        # Age + prune lost tracks
        for tid in list(self.tracks):
            self.tracks[tid]["lost"] += 1
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]

        if not coords:
            return []

        curr = np.array(coords, dtype=np.float64)

        active_ids  = [tid for tid, t in self.tracks.items() if t["lost"] == 1]  # just incremented
        if not active_ids:
            for pt in curr:
                self._new_track(pt)
            return self._active_list()

        prev = np.array([self.tracks[tid]["pos"] for tid in active_ids])
        cost = np.linalg.norm(prev[:, None, :] - curr[None, :, :], axis=2)   # (T, C)

        ri, ci = linear_sum_assignment(cost)
        matched_curr = set()

        for r, c in zip(ri, ci):
            if cost[r, c] < self.max_dist:
                tid = active_ids[r]
                prev_pos = np.array(self.tracks[tid]["pos"])
                new_pos  = curr[c]
                vel      = (new_pos - prev_pos).tolist()
                self.tracks[tid].update({
                    "pos":  new_pos.tolist(),
                    "vel":  vel,
                    "lost": 0,
                    "age":  self.tracks[tid]["age"] + 1,
                    "speed": float(np.linalg.norm(vel)),
                })
                matched_curr.add(c)

        for c in range(len(curr)):
            if c not in matched_curr:
                self._new_track(curr[c])

        return self._active_list()

    def _new_track(self, pos: np.ndarray) -> None:
        self.tracks[self.next_id] = {
            "pos": pos.tolist(), "vel": [0.0, 0.0],
            "lost": 0, "age": 0, "speed": 0.0, "id": self.next_id,
        }
        self.next_id += 1

    def _active_list(self) -> List[Dict]:
        return [
            {"id": t["id"], "pos": t["pos"], "vel": t["vel"],
             "age": t["age"], "speed": round(t["speed"], 2)}
            for t in self.tracks.values() if t["lost"] == 0
        ]

    def reset(self) -> None:
        self.tracks  = {}
        self.next_id = 0


# ═══════════════════════════════════════════════════════════════════════════════
# CUSUM DETECTOR  —  Cumulative Sum for sustained crowd-count shifts
# ═══════════════════════════════════════════════════════════════════════════════

class CUSUMDetector:
    """
    Adaptive two-sided CUSUM.  Detects *sustained* increases (pre-crush) and
    decreases (rapid dispersal) that a threshold spike-test misses.

    Warm-up phase (first CUSUM_WARMUP frames) learns μ and σ.
    Active phase standardises each count as z = (x − μ) / σ, then accumulates:
        S⁺ = max(0, S⁺ + z − k)    ← upper (surge) arm
        S⁻ = max(0, S⁻ − z − k)    ← lower (dispersal) arm
    Alert when S⁺ > h or S⁻ > h; statistics reset on alert.
    """

    def __init__(
        self,
        k: float = CUSUM_K,
        h: float = CUSUM_H,
        warmup: int = CUSUM_WARMUP,
    ) -> None:
        self.k = k; self.h = h; self.warmup = warmup
        self.S_pos = 0.0; self.S_neg = 0.0
        self.mu: Optional[float] = None
        self.sigma: Optional[float] = None
        self._buf: List[float] = []
        self._n = 0

    def update(self, x: float) -> Dict:
        self._n += 1
        self._buf.append(x)
        if len(self._buf) > 30:
            self._buf.pop(0)

        # Warm-up: build initial μ/σ from a stable window
        if self._n <= self.warmup:
            self.mu    = float(np.mean(self._buf))
            self.sigma = max(float(np.std(self._buf)), 0.1)
            return {"alert": False, "direction": None,
                    "S_pos": 0.0, "S_neg": 0.0, "mode": "warmup"}

        # EMA update of μ/σ so the baseline tracks slow drift
        alpha      = 0.04
        self.mu    = (1 - alpha) * self.mu    + alpha * x
        self.sigma = max((1 - alpha) * self.sigma + alpha * abs(x - self.mu), 0.1)

        z = (x - self.mu) / self.sigma
        self.S_pos = max(0.0, self.S_pos + z - self.k)
        self.S_neg = max(0.0, self.S_neg - z - self.k)

        surge     = self.S_pos > self.h
        dispersal = self.S_neg > self.h
        alert     = surge or dispersal

        if alert:
            self.S_pos = 0.0
            self.S_neg = 0.0

        return {
            "alert":     alert,
            "direction": "surge" if surge else ("dispersal" if dispersal else None),
            "S_pos":     round(self.S_pos, 3),
            "S_neg":     round(self.S_neg, 3),
            "z":         round(z, 3),
            "mu":        round(self.mu, 1),
            "mode":      "active",
        }

    def reset(self) -> None:
        self.S_pos = self.S_neg = 0.0
        self.mu = self.sigma = None
        self._buf = []; self._n = 0


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR CLASSIFIER  —  State machine keyed on baseline-relative density
# ═══════════════════════════════════════════════════════════════════════════════

class BehaviorClassifier:
    """
    NORMAL → PRE_SURGE → SURGE → DISPERSING state machine.

    FIX v3.2: density_ratio is now count / mean_baseline (not count / alert_threshold).
    The alert threshold is always >> current count by design, so the old ratio was
    structurally capped at ~0.80 and PRE_SURGE was unreachable.  With the EMA mean
    as denominator, 1.0 = at normal baseline; > 1.2 = clearly elevated.
    """

    def __init__(self, history_len: int = 10) -> None:
        self.history_len  = history_len
        self.counts:       List[float] = []
        self.compressions: List[int]   = []
        self.state       = "NORMAL"
        self.state_frames = 0

    def reset(self) -> None:
        self.counts = []; self.compressions = []
        self.state = "NORMAL"; self.state_frames = 0

    def _trend(self) -> float:
        n = len(self.counts)
        if n < 3: return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(self.counts)
        # Least-squares slope via normal equations (faster than polyfit for n≤10)
        x -= x.mean(); y_m = y.mean()
        denom = float((x * x).sum()) or 1e-9
        return float((x * (y - y_m)).sum() / denom)

    def update(
        self,
        count:           int,
        zone_data:       List[Dict],
        mean_baseline:   float,          # EMA rolling mean from AdaptiveThresholdV2
        proximity_score: float = 0.0,
    ) -> Dict:
        self.state_frames += 1
        self.counts.append(float(count))
        if len(self.counts) > self.history_len: self.counts.pop(0)

        comp_count = sum(1 for z in zone_data if z.get("divergence", 0) < -0.10)
        self.compressions.append(comp_count)
        if len(self.compressions) > self.history_len: self.compressions.pop(0)

        trend         = self._trend()
        density_ratio = count / max(mean_baseline, 1.0)
        prev_state    = self.state
        boost         = proximity_score * 0.06

        # ── Transitions ───────────────────────────────────────────────────────
        if density_ratio < 0.65 and self.state_frames >= 5 and proximity_score <= 0.55:
            self.state = "NORMAL"
        elif self.state == "NORMAL":
            if (density_ratio > 1.20 - boost or comp_count >= 6) and trend > 0.6:
                self.state = "PRE_SURGE"
        elif self.state == "PRE_SURGE":
            if density_ratio >= 1.50 - boost or comp_count >= 8:
                self.state = "SURGE"
            elif density_ratio < 0.90 and trend < -0.4:
                self.state = "NORMAL"
        elif self.state == "SURGE":
            if trend < -1.5 and comp_count < 3:
                self.state = "DISPERSING"
        elif self.state == "DISPERSING":
            if density_ratio < 0.75: self.state = "NORMAL"
            elif density_ratio > 1.45: self.state = "SURGE"

        if self.state != prev_state: self.state_frames = 0

        # ── Confidence ───────────────────────────────────────────────────────
        if   self.state == "NORMAL":     conf = min(1.0, 1.5 - density_ratio)
        elif self.state == "PRE_SURGE":  conf = min(1.0, 0.35 + (density_ratio - 1.0) * 0.5 + comp_count * 0.04)
        elif self.state == "SURGE":      conf = min(1.0, (density_ratio - 1.0) * 0.5 + comp_count * 0.055)
        else:                            conf = min(1.0, 0.45 + max(0.0, -trend) * 0.08)

        return {
            "state":             self.state,
            "confidence":        round(max(0.0, conf), 3),
            "count_trend":       round(trend, 2),
            "compression_zones": comp_count,
            "density_ratio":     round(density_ratio, 3),
        }