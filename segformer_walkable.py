# ═══════════════════════════════════════════════════════════════════════════════
# segformer_walkable.py — SegFormer semantic segmentation wrapper (Tier 1)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Wraps nvidia/segformer-b2 (or b0 fallback) to produce binary walkable masks
# from ADE20K 150-class semantic segmentation.
#
# Install:  pip install transformers accelerate timm
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from constants import ADE20K_NON_WALKABLE_NAMES, ADE20K_WALKABLE_NAMES

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    import torch.nn.functional as F
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False
    print("⚠️  transformers not found — SegFormer tier disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class SegFormerWalkable:
    """
    Wraps HuggingFace SegFormer to produce binary walkable masks.

    Model choice (priority order):
      1. nvidia/segformer-b2-finetuned-ade-512-512   (~100 MB, better accuracy)
      2. nvidia/segformer-b0-finetuned-ade-512-512   (~15 MB,  faster)

    Each pixel is classified as:
      WALKABLE     : floor, road, sidewalk, carpet, ground, path, pavement, …
      NON-WALKABLE : tree, building, wall, sky, fence, grass, vehicle, …
      AMBIGUOUS    : everything else → treated as walkable (conservative)
    """

    MODEL_PRIORITY = [
        "nvidia/segformer-b2-finetuned-ade-512-512",
        "nvidia/segformer-b0-finetuned-ade-512-512",
    ]
    PROCESS_SIZE = 512  # inference resolution; mask upsampled back to original

    def __init__(self) -> None:
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.id2label: Dict[int, str] = {}
        self.non_walkable_ids: set = set()
        self.walkable_ids: set = set()
        self.device: str = (
            "cuda"
            if SEGFORMER_AVAILABLE and __import__("torch").cuda.is_available()
            else "cpu"
        )
        self._load()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not SEGFORMER_AVAILABLE:
            return
        for model_id in self.MODEL_PRIORITY:
            try:
                print(f"⏳ Loading SegFormer: {model_id} …")
                self.processor = SegformerImageProcessor.from_pretrained(model_id)
                self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
                self.model.to(self.device).eval()
                self.id2label = self.model.config.id2label  # {int: str}
                self._build_class_sets()
                print(f"✅ SegFormer loaded ({model_id}) on {self.device}")
                print(f"   Non-walkable classes matched: {len(self.non_walkable_ids)}")
                print(f"   Walkable classes matched:     {len(self.walkable_ids)}")
                return
            except Exception as exc:
                print(f"⚠️  Failed to load {model_id}: {exc}")
        print("❌ SegFormer unavailable — falling back to YOLO + heuristic only")

    def _build_class_sets(self) -> None:
        """Match ADE20K class names against our non-walkable / walkable sets."""
        self.non_walkable_ids = set()
        self.walkable_ids = set()
        for idx, name in self.id2label.items():
            name_lower = name.lower().strip()
            if any(nw in name_lower for nw in ADE20K_NON_WALKABLE_NAMES):
                self.non_walkable_ids.add(idx)
            # Explicit walkable override — road/floor beats a broad nw match
            if any(w in name_lower for w in ADE20K_WALKABLE_NAMES):
                self.walkable_ids.add(idx)

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment(self, pil_img: Image.Image) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Returns:
            mask          : np.ndarray (H, W) uint8 — 1=walkable, 0=blocked
                            None if model not loaded.
            segment_info  : list of top-20 {class_id, class_name, is_walkable,
                            pixel_count, pct} sorted by coverage descending.
        """
        if not self.is_loaded():
            return None, []

        import torch
        W, H = pil_img.size

        img_r = pil_img.resize((self.PROCESS_SIZE, self.PROCESS_SIZE), Image.LANCZOS)
        inputs = self.processor(images=img_r, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Upsample logits (1, C, H/4, W/4) → (H, W)
        logits = outputs.logits
        upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W) int

        # Build walkable mask:
        #   explicit non-walkable  → 0
        #   explicit walkable      → 1  (overrides broad non-walkable match)
        #   ambiguous              → 1  (conservative: don't remove)
        mask = np.ones((H, W), dtype=np.uint8)
        for nw_id in self.non_walkable_ids:
            mask[pred == nw_id] = 0
        for w_id in self.walkable_ids:
            mask[pred == w_id] = 1

        segment_info = self._build_segment_info(pred, H * W)
        return mask, segment_info

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_segment_info(self, pred: np.ndarray, total_px: int) -> List[Dict]:
        unique, counts = np.unique(pred, return_counts=True)
        info = []
        for cls_id, cnt in zip(unique.tolist(), counts.tolist()):
            name = self.id2label.get(cls_id, f"class_{cls_id}")
            info.append({
                "class_id":    cls_id,
                "class_name":  name,
                "is_walkable": cls_id not in self.non_walkable_ids,
                "pixel_count": int(cnt),
                "pct":         round(cnt / total_px * 100, 1),
            })
        info.sort(key=lambda x: x["pixel_count"], reverse=True)
        return info[:20]