# ═══════════════════════════════════════════════════════════════════════════════
# constants.py — Shared constants for Crowd Intelligence System v3
# ═══════════════════════════════════════════════════════════════════════════════

GRID_COLS  = 8
GRID_ROWS  = 6
GRID_CELLS = GRID_COLS * GRID_ROWS
EMA_ALPHA  = 0.22

# ─── ADE20K class sets (used by SegFormer tier) ───────────────────────────────

ADE20K_NON_WALKABLE_NAMES = {
    # Structural / architectural
    "building", "house", "skyscraper", "tower", "hovel", "booth",
    "wall", "fence", "railing", "bannister", "balustrade", "column",
    "pillar", "arch", "bridge", "runway",
    # Vegetation / nature
    "tree", "palm tree", "plant", "flower", "grass", "land",
    "earth", "hill", "mountain", "rock", "stone", "wood",
    "bush", "shrub",
    # Sky / overhead
    "sky",
    # Water
    "water", "sea", "lake", "river", "waterfall", "swimming pool",
    # Vehicles / transport
    "car", "bus", "truck", "van", "minibike", "bicycle", "boat", "ship",
    "airplane", "train",
    # Furniture / objects
    "chair", "armchair", "swivel chair", "bench", "sofa", "table",
    "dining table", "coffee table", "desk", "wardrobe", "bookcase",
    "cabinet", "shelf", "buffet",
    # Outdoor objects
    "awning", "canopy", "sign", "signboard",
    "streetlight", "pole", "antenna",
    # Surfaces that look walkable but aren't
    "escalator", "stairway", "step",
    # Ceiling / overhead
    "ceiling",
}

ADE20K_WALKABLE_NAMES = {
    "floor", "floor, flooring", "floor, flooring, floor covering",
    "rug", "mat", "carpet",
    "road", "sidewalk", "path", "pavement", "crosswalk",
    "parking", "plaza",
    "field",
    "ground",
    "corridor", "lobby", "hallway",
}

# ─── COCO class sets (used by YOLO tier) ──────────────────────────────────────

# Class indices in COCO 80-class that we treat as non-walkable obstacles
NON_WALKABLE_COCO_IDS = {
    1, 2, 3, 4, 5, 6, 7, 8,          # bicycle, car, motorcycle, airplane, bus, train, truck, boat
    9, 10, 11,                        # traffic light, fire hydrant, stop sign
    13,                               # stop sign (alt index)
    56, 57, 58, 59, 60, 61, 62, 63,  # chair, couch, potted plant, bed, mirror, dining table, window, desk
    64, 65, 66, 67, 68, 69,          # laptop, mouse, remote, keyboard, cell phone, microwave
    72, 73, 74, 75,                  # tv, laptop, mouse, remote
}

NON_WALKABLE_COCO_NAMES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "train", "boat",
    "potted plant", "tree", "bench",
    "chair", "couch", "bed", "dining table",
    "traffic light", "stop sign", "parking meter", "fire hydrant",
}