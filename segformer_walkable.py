# ═══════════════════════════════════════════════════════════════════════════════
# segformer_walkable.py — SegFormer semantic segmentation wrapper  v3.2
# ═══════════════════════════════════════════════════════════════════════════════
# Changes vs v3.1:
#   • torch.inference_mode() (faster than no_grad)
#   • FP16 on CUDA to halve VRAM and latency
#   • Resize with processor's own resize instead of PIL pre-resize (one step)
#   • _build_class_sets caches sets as frozensets for O(1) lookup
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
    MODEL_PRIORITY = [
        "nvidia/segformer-b2-finetuned-ade-512-512",
        "nvidia/segformer-b0-finetuned-ade-512-512",
    ]

    def __init__(self) -> None:
        self.processor: Optional[Any] = None
        self.model:     Optional[Any] = None
        self.id2label:  Dict[int, str]  = {}
        self.non_walkable_ids: frozenset = frozenset()
        self.walkable_ids:     frozenset = frozenset()
        self._use_fp16 = False
        self.device = "cpu"
        self._load()

    def _load(self) -> None:
        if not SEGFORMER_AVAILABLE:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        for mid in self.MODEL_PRIORITY:
            try:
                print(f"⏳ Loading SegFormer: {mid} …")
                self.processor = SegformerImageProcessor.from_pretrained(mid)
                self.model     = SegformerForSemanticSegmentation.from_pretrained(mid)
                # FP16 on CUDA: ~2× throughput, negligible accuracy loss for segmentation
                if self.device == "cuda":
                    self.model = self.model.half()
                    self._use_fp16 = True
                self.model.to(self.device).eval()
                self.id2label = self.model.config.id2label
                self._build_class_sets()
                print(f"✅ SegFormer ready ({mid}) fp16={self._use_fp16} on {self.device}")
                return
            except Exception as e:
                print(f"⚠️  {mid}: {e}")
        print("❌ SegFormer unavailable")

    def _build_class_sets(self) -> None:
        nw, wk = set(), set()
        for idx, name in self.id2label.items():
            n = name.lower().strip()
            if any(k in n for k in ADE20K_NON_WALKABLE_NAMES): nw.add(idx)
            if any(k in n for k in ADE20K_WALKABLE_NAMES):     wk.add(idx)
        self.non_walkable_ids = frozenset(nw)
        self.walkable_ids     = frozenset(wk)

    def is_loaded(self) -> bool:
        return self.model is not None

    def segment(self, pil_img: Image.Image) -> Tuple[Optional[np.ndarray], List[Dict]]:
        if not self.is_loaded():
            return None, []

        W, H = pil_img.size
        inputs = self.processor(images=pil_img, return_tensors="pt")
        if self._use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits          # (1, C, H/4, W/4)

        # Upsample to original size
        up   = F.interpolate(logits.float(), size=(H, W),
                             mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        mask = np.ones((H, W), dtype=np.uint8)
        # Vectorised mask assignment — avoids a Python loop per class
        nw_arr = np.array(sorted(self.non_walkable_ids), dtype=np.int32)
        wk_arr = np.array(sorted(self.walkable_ids),     dtype=np.int32)
        if nw_arr.size: mask[np.isin(pred, nw_arr)] = 0
        if wk_arr.size: mask[np.isin(pred, wk_arr)] = 1   # walkable overrides

        return mask, self._segment_info(pred, H * W)

    def _segment_info(self, pred: np.ndarray, total_px: int) -> List[Dict]:
        unique, counts = np.unique(pred, return_counts=True)
        info = [
            {
                "class_id":    int(cid),
                "class_name":  self.id2label.get(int(cid), f"class_{cid}"),
                "is_walkable": int(cid) not in self.non_walkable_ids,
                "pixel_count": int(cnt),
                "pct":         round(int(cnt) / total_px * 100, 1),
            }
            for cid, cnt in zip(unique, counts)
        ]
        info.sort(key=lambda x: x["pixel_count"], reverse=True)
        return info[:20]