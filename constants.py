# ═══════════════════════════════════════════════════════════════════════════════
# constants.py — Shared constants  v3.2
# ═══════════════════════════════════════════════════════════════════════════════

GRID_COLS  = 8
GRID_ROWS  = 6
GRID_CELLS = GRID_COLS * GRID_ROWS
EMA_ALPHA  = 0.22

# ─── CUSUM anomaly detection ──────────────────────────────────────────────────
CUSUM_K        = 0.5   # allowance — half the minimum detectable shift (σ units)
CUSUM_H        = 4.0   # decision threshold (σ units); lower = more sensitive
CUSUM_WARMUP   = 12    # frames before CUSUM activates

# ─── Walkable mask scene-change gate ─────────────────────────────────────────
SCENE_CHANGE_THRESH = 0.025  # mean|ΔI|/255 to force mask re-estimation

# ─── Trajectory tracker ───────────────────────────────────────────────────────
TRAJ_MAX_DIST_PX = 90   # max pixel distance to link detections across frames
TRAJ_MAX_LOST    = 4    # frames a track may be missing before it is dropped

# ─── ADE20K class sets (SegFormer tier) ──────────────────────────────────────
ADE20K_NON_WALKABLE_NAMES = {
    "building", "house", "skyscraper", "tower", "hovel", "booth",
    "wall", "fence", "railing", "bannister", "balustrade", "column",
    "pillar", "arch", "bridge", "runway",
    "tree", "palm tree", "plant", "flower", "grass", "land",
    "earth", "hill", "mountain", "rock", "stone", "wood", "bush", "shrub",
    "sky",
    "water", "sea", "lake", "river", "waterfall", "swimming pool",
    "car", "bus", "truck", "van", "minibike", "bicycle", "boat", "ship",
    "airplane", "train",
    "chair", "armchair", "swivel chair", "bench", "sofa", "table",
    "dining table", "coffee table", "desk", "wardrobe", "bookcase",
    "cabinet", "shelf", "buffet",
    "awning", "canopy", "sign", "signboard", "streetlight", "pole", "antenna",
    "escalator", "stairway", "step",
    "ceiling",
}


ADE20K_WALKABLE_NAMES = {
    "floor", "floor, flooring", "floor, flooring, floor covering",
    "rug", "mat", "carpet",
    "road", "sidewalk", "path", "pavement", "crosswalk",
    "parking", "plaza", "field", "ground",
    "corridor", "lobby", "hallway",
}

# ─── COCO class sets (YOLO tier) ─────────────────────────────────────────────
NON_WALKABLE_COCO_IDS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75,
}
NON_WALKABLE_COCO_NAMES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "train", "boat",
    "potted plant", "tree", "bench",
    "chair", "couch", "bed", "dining table",
    "traffic light", "stop sign", "parking meter", "fire hydrant",
}