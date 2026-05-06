# ═══════════════════════════════════════════════════════════════════════════════
# walkable_area_estimator.py — Three-tier walkable area estimation  v3.2
# ═══════════════════════════════════════════════════════════════════════════════
# Changes vs v3.1:
#   • Structuring elements pre-allocated once in __init__ (no per-call cv2.getStructuringElement)
#   • YOLO result parsed with vectorised numpy operations instead of per-box Python loop
#   • Morphological cleanup extracted to _clean_mask (unchanged logic, same perf)
#   • Category lookup table replaces repeated str.find chains
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
from PIL import Image

from constants import NON_WALKABLE_COCO_IDS, NON_WALKABLE_COCO_NAMES
from segformer_walkable import SegFormerWalkable

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  cv2 not found — morphological cleanup disabled")

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not found — YOLO tier disabled")

# Category keyword lookup (checked in priority order)
_CAT_KEYS = [
    ("vehicle",   ("car","truck","bus","motorcycle","train","bicycle","boat")),
    ("nature",    ("plant","tree","bench","grass")),
    ("furniture", ("chair","couch","bed","dining","table","desk")),
]

def _yolo_category(name: str) -> str:
    n = name.lower()
    for cat, keys in _CAT_KEYS:
        if any(k in n for k in keys):
            return cat
    return "structure"


class WalkableAreaEstimator:
    """
    Three-tier walkable area estimation.
    Tier 1: SegFormer-B2  (semantic segmentation, ADE20K)
    Tier 2: YOLOv8n       (instance detection, COCO)
    Tier 3: HSV heuristic (colour-based, no ML)

    estimate() returns (mask, meta) — identical signature to v3.1.
    """

    OBSTACLE_AREA_THRESH = 0.015   # ignore boxes < 1.5 % of image area

    def __init__(self) -> None:
        self.segformer = SegFormerWalkable()

        self.yolo: Optional[Any] = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = _YOLO(os.path.join(_HERE, "yolov8n.pt"))
                print("✅ YOLOv8n loaded")
            except Exception as e:
                print(f"⚠️  YOLO init failed: {e}")

        # Pre-allocate structuring elements (reused every call)
        if CV2_AVAILABLE:
            self._k_obs  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            self._k_walk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))

        print(
            f"🗺  WalkableAreaEstimator v3.2 | "
            f"SegFormer={'✓' if self.segformer.is_loaded() else '✗'} | "
            f"YOLO={'✓' if self.yolo else '✗'} | Heuristic=✓"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def estimate(self, pil_img: Image.Image) -> Tuple[np.ndarray, Dict]:
        W, H = pil_img.size
        obstacles: List[Dict] = []
        segment_info: List[Dict] = []

        # Tier 1: SegFormer
        if self.segformer.is_loaded():
            mask, segment_info = self.segformer.segment(pil_img)
            tier = "segformer"
        else:
            mask = np.ones((H, W), dtype=np.uint8)
            tier = "heuristic"

        # Tier 2: YOLO obstacle removal
        if self.yolo is not None:
            yolo_mask = np.zeros((H, W), dtype=np.uint8)
            obstacles  = self._yolo_obstacles(pil_img, yolo_mask, W, H)
            mask[yolo_mask == 1] = 0
            tier = "segformer+yolo" if tier == "segformer" else "yolo"
        elif tier == "heuristic":
            # Tier 3: HSV heuristic (only when both ML tiers unavailable)
            self._heuristic_obstacles(pil_img, mask, W, H)

        mask = self._clean_mask(mask)

        walkable_px = int(mask.sum())
        total_px    = W * H
        return mask, {
            "walkable_pixels":    walkable_px,
            "total_pixels":       total_px,
            "walkable_pct":       round(walkable_px / total_px * 100, 1),
            "obstacles_detected": obstacles,
            "obstacle_count":     len(obstacles),
            "segment_classes":    segment_info,
            "tier_used":          tier,
        }

    # ── Morphological cleanup ──────────────────────────────────────────────────

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return mask

        # Dilate obstacle edges
        obs = (mask == 0).astype(np.uint8)
        mask[cv2.dilate(obs, self._k_obs) == 1] = 0

        # Remove specks from walkable region
        mask = cv2.erode(mask,  self._k_walk)
        mask = cv2.dilate(mask, self._k_walk)

        # Keep largest connected components (+ any ≥ 5 % of largest)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 2:
            areas    = stats[1:, cv2.CC_STAT_AREA]
            largest  = int(np.argmax(areas)) + 1
            min_area = max(areas[largest - 1] * 0.05,
                           mask.shape[0] * mask.shape[1] * 0.01)
            keep = np.zeros_like(mask)
            for lbl in range(1, n):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    keep[labels == lbl] = 1
            mask = keep

        return mask.astype(np.uint8)

    # ── YOLO obstacle detection ────────────────────────────────────────────────

    def _yolo_obstacles(
        self, img: Image.Image, yolo_mask: np.ndarray, W: int, H: int
    ) -> List[Dict]:
        results  = self.yolo(img, verbose=False, conf=0.30)[0]
        obstacles: List[Dict] = []
        area_img = W * H

        if not len(results.boxes):
            return obstacles

        boxes    = results.boxes
        cls_ids  = boxes.cls.int().cpu().numpy()          # (N,)
        confs    = boxes.conf.cpu().numpy()               # (N,)
        xyxys    = boxes.xyxy.int().cpu().numpy()         # (N,4)
        names    = results.names

        for i in range(len(cls_ids)):
            cid  = int(cls_ids[i])
            name = names.get(cid, "")
            if cid == 0: continue   # skip persons
            x1, y1, x2, y2 = xyxys[i]
            area = int((x2 - x1) * (y2 - y1))
            if area / area_img < self.OBSTACLE_AREA_THRESH: continue
            if cid not in NON_WALKABLE_COCO_IDS and name.lower() not in NON_WALKABLE_COCO_NAMES:
                continue

            x1c,y1c = max(0,int(x1)),max(0,int(y1))
            x2c,y2c = min(W,int(x2)),min(H,int(y2))
            yolo_mask[y1c:y2c, x1c:x2c] = 1
            obstacles.append({
                "class":     name,
                "conf":      round(float(confs[i]), 2),
                "bbox_norm": [round(int(x1)/W,4),round(int(y1)/H,4),
                              round(int(x2)/W,4),round(int(y2)/H,4)],
                "bbox":      [int(x1),int(y1),int(x2),int(y2)],
                "area_pct":  round(area / area_img * 100, 1),
                "category":  _yolo_category(name),
                "source":    "yolo",
            })
        return obstacles

    # ── HSV heuristic fallback ─────────────────────────────────────────────────

    def _heuristic_obstacles(
        self, img: Image.Image, mask: np.ndarray, W: int, H: int
    ) -> None:
        if not CV2_AVAILABLE:
            return
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Sky: upper 30 %, low saturation, high brightness
        sky = (s < 30) & (v > 200)
        sky[int(H * 0.30):, :] = False
        mask[sky] = 0

        # Dense foliage: green hues, high saturation
        mask[(h >= 35) & (h <= 90) & (s > 70) & (v > 35)] = 0

        # Bright near-white upper half (sky/fog)
        brt = (v > 230) & (s < 25)
        brt[int(H * 0.55):, :] = False
        mask[brt] = 0

        # Vehicles: saturated non-green
        mask[(s > 140) & ((h < 25) | (h > 95))] = 0