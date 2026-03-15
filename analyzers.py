# ═══════════════════════════════════════════════════════════════════════════════
# analyzers.py — Distance, Flow, and Behavior analysis modules
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import KDTree
from PIL import Image

from constants import GRID_COLS, GRID_ROWS

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
    if density_sqm <= 0.5:  return "A"
    if density_sqm <= 1.0:  return "B"
    if density_sqm <= 1.7:  return "C"
    if density_sqm <= 2.7:  return "D"
    if density_sqm <= 4.0:  return "E"
    return "F"


# ═══════════════════════════════════════════════════════════════════════════════
# DISTANCE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class DistanceAnalyzer:
    """
    Computes pairwise k-nearest-neighbour distances between detected persons
    and derives crowd proximity metrics.

    Key outputs
    -----------
    median_nn_dist             : median nearest-neighbour distance (pixels)
    mean_nn_dist               : mean NN distance
    proximity_violation_ratio  : fraction of persons closer than min_safe_px
    proximity_score            : 0–1, 1 = extreme crowding
    nn_distances               : each person's nearest-neighbour distance
    """

    MIN_SAFE_FRAC = 0.035  # min safe distance as fraction of image diagonal (~0.8 m)

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        coords: List[List[float]],
        img_w: int,
        img_h: int,
        pixels_per_meter: Optional[float] = None,
    ) -> Dict:
        n = len(coords)
        if n == 0:
            return self._empty()
        if n == 1:
            return {
                "count": 1, "median_nn_dist": None, "mean_nn_dist": None,
                "proximity_violation_ratio": 0.0, "proximity_score": 0.0,
                "nn_distances": [], "min_safe_px": None,
                "violations": 0, "k_used": 0,
            }

        pts  = np.array(coords, dtype=np.float32)
        diag = math.sqrt(img_w ** 2 + img_h ** 2)

        if pixels_per_meter is not None:
            min_safe_px = pixels_per_meter * 0.8  # 0.8 m safe spacing (LOS-B boundary)
        else:
            min_safe_px = diag * self.MIN_SAFE_FRAC

        k = min(5, n - 1)
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)  # +1 because first result is self
        nn_dists = dists[:, 1]

        median_d   = float(np.median(nn_dists))
        mean_d     = float(nn_dists.mean())
        violations = int((nn_dists < min_safe_px).sum())
        vio_ratio  = violations / n

        if median_d >= min_safe_px * 2:
            prox_score = 0.0
        elif median_d <= min_safe_px * 0.3:
            prox_score = 1.0
        else:
            prox_score = 1.0 - (median_d - min_safe_px * 0.3) / (min_safe_px * 1.7)
        prox_score = round(max(0.0, min(1.0, prox_score)), 3)

        return {
            "count":                     n,
            "median_nn_dist":            round(median_d, 1),
            "mean_nn_dist":              round(mean_d, 1),
            "min_safe_px":               round(min_safe_px, 1),
            "violations":                violations,
            "proximity_violation_ratio": round(vio_ratio, 3),
            "proximity_score":           prox_score,
            "nn_distances":              [round(d, 1) for d in nn_dists.tolist()],
            "k_used":                    k,
        }

    @staticmethod
    def _empty() -> Dict:
        return {
            "count": 0, "median_nn_dist": None, "mean_nn_dist": None,
            "min_safe_px": None, "violations": 0,
            "proximity_violation_ratio": 0.0, "proximity_score": 0.0,
            "nn_distances": [], "k_used": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class FlowAnalyzer:
    """Optical flow via Farneback; per-grid-cell divergence and magnitude."""

    def __init__(self, grid_cols: int = GRID_COLS, grid_rows: int = GRID_ROWS) -> None:
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.prev_gray = None

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
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
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
                div = 0.0
                if fx.shape[0] > 1 and fx.shape[1] > 1:
                    div = float(
                        np.gradient(fx, axis=1).mean() + np.gradient(fy, axis=0).mean()
                    )
                results.append({
                    "row":       row,
                    "col":       col,
                    "mean_dx":   float(fx.mean()),
                    "mean_dy":   float(fy.mean()),
                    "divergence": round(div, 4),
                    "magnitude":  round(float(np.sqrt(fx ** 2 + fy ** 2).mean()), 4),
                })
        return results

    def reset(self) -> None:
        self.prev_gray = None


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class BehaviorClassifier:
    """State machine: NORMAL → PRE_SURGE → SURGE → DISPERSING."""

    STATES = ["NORMAL", "PRE_SURGE", "SURGE", "DISPERSING"]

    def __init__(self, history_len: int = 10) -> None:
        self.history_len = history_len
        self.counts: List[float] = []
        self.compressions: List[int] = []
        self.state = "NORMAL"
        self.state_frames = 0

    def reset(self) -> None:
        self.counts = []
        self.compressions = []
        self.state = "NORMAL"
        self.state_frames = 0

    def _trend(self) -> float:
        n = len(self.counts)
        if n < 3:
            return 0.0
        x_m = (n - 1) / 2
        y_m = sum(self.counts) / n
        num = sum((i - x_m) * (self.counts[i] - y_m) for i in range(n))
        den = sum((i - x_m) ** 2 for i in range(n)) or 1e-9
        return num / den

    def update(
        self,
        count: int,
        zone_data: List[Dict],
        threshold: int,
        proximity_score: float = 0.0,
    ) -> Dict:
        self.state_frames += 1
        self.counts.append(float(count))
        if len(self.counts) > self.history_len:
            self.counts.pop(0)

        comp_count = sum(1 for z in zone_data if z.get("divergence", 0) < -0.10)
        self.compressions.append(comp_count)
        if len(self.compressions) > self.history_len:
            self.compressions.pop(0)

        trend         = self._trend()
        density_ratio = count / max(threshold, 1)
        prev_state    = self.state
        boost         = proximity_score * 0.15  # proximity boosts surge sensitivity

        if density_ratio < 0.30 and self.state_frames >= 5 and not proximity_score > 0.65:
            self.state = "NORMAL"
        elif self.state == "NORMAL":
            if (density_ratio > 0.72 - boost or comp_count >= 5) and trend > 0.5:
                self.state = "PRE_SURGE"
        elif self.state == "PRE_SURGE":
            if density_ratio >= 0.92 - boost or comp_count >= 7:
                self.state = "SURGE"
            elif density_ratio < 0.55 and trend < -0.5:
                self.state = "NORMAL"
        elif self.state == "SURGE":
            if trend < -1.5 and comp_count < 3:
                self.state = "DISPERSING"
        elif self.state == "DISPERSING":
            if density_ratio < 0.42:
                self.state = "NORMAL"
            elif density_ratio > 0.88:
                self.state = "SURGE"

        if self.state != prev_state:
            self.state_frames = 0

        if self.state == "NORMAL":
            conf = min(1.0, 1.0 - density_ratio)
        elif self.state == "PRE_SURGE":
            conf = min(1.0, 0.45 + density_ratio * 0.35 + comp_count * 0.04)
        elif self.state == "SURGE":
            conf = min(1.0, density_ratio * 0.65 + comp_count * 0.055)
        else:
            conf = min(1.0, 0.45 + max(0.0, -trend) * 0.08)

        return {
            "state":              self.state,
            "confidence":         round(conf, 3),
            "count_trend":        round(trend, 2),
            "compression_zones":  comp_count,
            "density_ratio":      round(density_ratio, 3),
        }