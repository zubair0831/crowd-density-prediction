# ═══════════════════════════════════════════════════════════════════════════════
# walkable_area_estimator.py — Three-tier walkable area estimation (v3.1)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tier 1 — SegFormer-B2  (semantic seg, ADE20K 150 classes)
#           Best for: trees, buildings, sky, walls, fences, grass, roads …
# Tier 2 — YOLOv8n       (instance detection, COCO 80 classes)
#           Best for: vehicles, bicycles, chairs, benches, potted plants
# Tier 3 — HSV heuristic (colour-based fallback, no model needed)
#
# Fusion:
#   final_mask = SegFormer_walkable AND NOT(YOLO_obstacles)
#   + morphological cleanup (dilate obstacles → erode/redilate walkable
#     → keep largest connected component)
#
# Dependencies:
#   constants.py, segformer_walkable.py  (same package)
#   pip install ultralytics opencv-python-headless pillow numpy
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from constants import NON_WALKABLE_COCO_IDS, NON_WALKABLE_COCO_NAMES
from segformer_walkable import SegFormerWalkable

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


class WalkableAreaEstimator:
    """
    Three-tier walkable area estimation.  Drop-in replacement for the v3.0
    WalkableAreaEstimator in backend.py — the estimate() signature is identical.

    New fields in the returned meta dict:
        segment_classes : list of top-20 ADE20K classes detected by SegFormer,
                          each {class_id, class_name, is_walkable, pixel_count, pct}
        tier_used       : "segformer+yolo" | "segformer" | "yolo" | "heuristic"
    """

    OBSTACLE_AREA_THRESHOLD = 0.015  # ignore YOLO boxes < 1.5 % of image area

    def __init__(self) -> None:
        # Tier 1: SegFormer
        self.segformer = SegFormerWalkable()

        # Tier 2: YOLOv8n
        self.yolo: Optional[Any] = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = _YOLO("yolov8n.pt")  # auto-downloads ~6 MB
                print("✅ YOLOv8n loaded")
            except Exception as e:
                print(f"⚠️  YOLOv8 init failed: {e}")

        print(
            f"🗺  WalkableAreaEstimator ready | "
            f"SegFormer={'✓' if self.segformer.is_loaded() else '✗'} | "
            f"YOLO={'✓' if self.yolo else '✗'} | "
            f"Heuristic=✓(fallback)"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def estimate(self, pil_img: Image.Image) -> Tuple[np.ndarray, Dict]:
        """
        Returns:
            mask : np.ndarray (H, W) uint8 — 1 = walkable, 0 = obstacle/blocked
            meta : dict with walkable_pixels, total_pixels, walkable_pct,
                   obstacles_detected, obstacle_count, segment_classes, tier_used
        """
        W, H = pil_img.size
        obstacles: List[Dict] = []
        segment_info: List[Dict] = []

        # ── Tier 1: SegFormer ─────────────────────────────────────────────────
        if self.segformer.is_loaded():
            mask, segment_info = self.segformer.segment(pil_img)
            tier_used = "segformer"
        else:
            # Start fully walkable; lower tiers carve out obstacles
            mask = np.ones((H, W), dtype=np.uint8)
            tier_used = "heuristic"

        # ── Tier 2: YOLO obstacle detection ──────────────────────────────────
        if self.yolo is not None:
            yolo_mask = np.zeros((H, W), dtype=np.uint8)
            obstacles = self._yolo_obstacles(pil_img, yolo_mask, W, H)
            # Union: any pixel flagged by YOLO becomes non-walkable
            mask[yolo_mask == 1] = 0
            tier_used = "segformer+yolo" if tier_used == "segformer" else "yolo"
        elif tier_used == "heuristic":
            # Tier 3: pure HSV heuristic (only when both ML tiers unavailable)
            self._heuristic_obstacles(pil_img, mask, W, H)

        # ── Morphological cleanup ─────────────────────────────────────────────
        mask = self._clean_mask(mask, W, H)

        walkable_px  = int(mask.sum())
        total_px     = W * H
        walkable_pct = round(walkable_px / total_px * 100, 1)

        return mask, {
            "walkable_pixels":    walkable_px,
            "total_pixels":       total_px,
            "walkable_pct":       walkable_pct,
            "obstacles_detected": obstacles,
            "obstacle_count":     len(obstacles),
            "segment_classes":    segment_info,   # new in v3.1
            "tier_used":          tier_used,       # new in v3.1
        }

    # ── Morphological cleanup ──────────────────────────────────────────────────

    def _clean_mask(self, mask: np.ndarray, W: int, H: int) -> np.ndarray:
        """
        1. Dilate obstacle regions ~15 px to capture edges around objects.
        2. Erode then re-dilate walkable region to remove tiny specks.
        3. Keep largest connected walkable component + any > 5 % of largest.
        """
        if not CV2_AVAILABLE:
            return mask

        obstacle_layer = (mask == 0).astype(np.uint8)
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_obs = cv2.dilate(obstacle_layer, k1, iterations=1)
        mask[dilated_obs == 1] = 0

        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.erode(mask, k2, iterations=1)
        mask = cv2.dilate(mask, k2, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = np.argmax(areas) + 1
            threshold_area = max(areas[largest - 1] * 0.05, W * H * 0.01)
            keep = np.zeros_like(mask)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= threshold_area:
                    keep[labels == lbl] = 1
            mask = keep

        return mask.astype(np.uint8)

    # ── YOLO obstacle detection ────────────────────────────────────────────────

    def _yolo_obstacles(
        self, img: Image.Image, yolo_mask: np.ndarray, W: int, H: int
    ) -> List[Dict]:
        """Paint detected obstacles into yolo_mask. Returns obstacle list."""
        results  = self.yolo(img, verbose=False, conf=0.30)[0]
        obstacles: List[Dict] = []
        area_img = W * H

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = results.names.get(cls_id, "")
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_area = (x2 - x1) * (y2 - y1)

            if cls_id == 0:  # skip persons
                continue
            if box_area / area_img < self.OBSTACLE_AREA_THRESHOLD:
                continue
            if (
                cls_id not in NON_WALKABLE_COCO_IDS
                and cls_name.lower() not in NON_WALKABLE_COCO_NAMES
            ):
                continue

            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            yolo_mask[y1c:y2c, x1c:x2c] = 1

            name_lower = cls_name.lower()
            if any(v in name_lower for v in ("car", "truck", "bus", "motorcycle", "train", "bicycle", "boat")):
                category = "vehicle"
            elif any(v in name_lower for v in ("plant", "tree", "bench", "grass")):
                category = "nature"
            elif any(v in name_lower for v in ("chair", "couch", "bed", "dining", "table", "desk")):
                category = "furniture"
            else:
                category = "structure"

            obstacles.append({
                "class":     cls_name,
                "conf":      round(conf, 2),
                "bbox_norm": [
                    round(x1 / W, 4), round(y1 / H, 4),
                    round(x2 / W, 4), round(y2 / H, 4),
                ],
                "bbox":      [x1, y1, x2, y2],
                "area_pct":  round(box_area / area_img * 100, 1),
                "category":  category,
                "source":    "yolo",
            })
        return obstacles

    # ── HSV heuristic fallback ─────────────────────────────────────────────────

    def _heuristic_obstacles(
        self, img: Image.Image, mask: np.ndarray, W: int, H: int
    ) -> None:
        """Conservative HSV thresholding when no ML models are available."""
        if not CV2_AVAILABLE:
            return
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Sky: very low saturation + high brightness in upper 30 % of frame
        sky = (s < 30) & (v > 200)
        sky[int(H * 0.30):, :] = False
        mask[sky] = 0

        # Dense foliage: green hues with high saturation
        foliage = (h >= 35) & (h <= 90) & (s > 70) & (v > 35)
        mask[foliage] = 0

        # Very bright / near-white areas in upper half (sky / fog)
        bright_upper = (v > 230) & (s < 25)
        bright_upper[int(H * 0.55):, :] = False
        mask[bright_upper] = 0

        # Vehicles: highly saturated non-green pixels (red / blue / yellow)
        vehicles = (s > 140) & ((h < 25) | (h > 95))
        mask[vehicles] = 0