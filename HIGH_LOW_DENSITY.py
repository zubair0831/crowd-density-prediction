from ultralytics import YOLO
import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# TO CLASSIFY DENSITY IN THE IMAGE FROM YOLO
# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolov8m.pt")

# -------------------------------
# PARAMETERS
# -------------------------------
CONF_THRES = 0.5
IOU_THRES = 0.5

# Thresholds
EDGE_DENSITY_HIGH = 0.20
EDGE_DENSITY_VERY_HIGH = 0.28
OCCUPANCY_MODERATE = 0.16
OCCUPANCY_HIGH = 0.20
YOLO_LOW_COUNT = 10
YOLO_VERY_LOW_COUNT = 5
YOLO_MODERATE_COUNT = 20           # NEW
SCALE_LARGE_THRESHOLD = 0.50
SCALE_MEDIUM_THRESHOLD = 0.30
SCALE_SMALL_THRESHOLD = 0.22       # NEW - very small people
LAPLACIAN_VERY_HIGH = 5000

# -------------------------------
# LOW-LEVEL FEATURE EXTRACTION
# -------------------------------
def extract_density_features(image):
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (H * W)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    color_variance = np.var(gray)
    
    return {
        "edge_density": edge_density,
        "laplacian_var": laplacian_var,
        "color_variance": color_variance
    }

# -------------------------------
# CROWD CATEGORIZATION FUNCTION
# -------------------------------
def categorize_scene(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    H, W = image.shape[:2]
    image_area = H * W

    results = model(image, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    boxes = results[0].boxes

    person_boxes = []
    heights = []

    for box in boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append([x1, y1, x2, y2])
            heights.append(y2 - y1)

    N = len(person_boxes)

    total_box_area = sum(
        (b[2] - b[0]) * (b[3] - b[1]) for b in person_boxes
    )
    occupancy = total_box_area / image_area if image_area > 0 else 0.0

    median_height_ratio = np.median(heights) / H if heights else 0.0

    features = extract_density_features(image)

    # -------------------------------
    # REVISED LOGIC
    # -------------------------------
    high_density_signals = 0
    low_density_signals = 0
    
    # === HIGH DENSITY INDICATORS ===
    
    # Signal 1: Very high edge density
    if features["edge_density"] > EDGE_DENSITY_VERY_HIGH:
        high_density_signals += 3
    elif features["edge_density"] > EDGE_DENSITY_HIGH:
        high_density_signals += 2
    
    # Signal 2: YOLO finds NOTHING or very few but image has content
    if N == 0 and features["edge_density"] > 0.15:
        high_density_signals += 4
    elif N <= 2 and features["edge_density"] > 0.08:
        high_density_signals += 3
    
    # Signal 3: YOLO severely undercounting (moderate occupancy but very low count)
    if occupancy > OCCUPANCY_MODERATE and N < YOLO_VERY_LOW_COUNT and N > 0:
        high_density_signals += 3
    elif occupancy > OCCUPANCY_MODERATE and N < YOLO_LOW_COUNT and N > 0:
        high_density_signals += 2
    
    # Signal 4: Very high detail/texture
    if features["laplacian_var"] > LAPLACIAN_VERY_HIGH:
        high_density_signals += 2
    
    # Signal 5: Small people with moderate count (distant crowd) - EXPANDED
    if median_height_ratio < SCALE_SMALL_THRESHOLD and N >= 10 and N < YOLO_MODERATE_COUNT:
        high_density_signals += 2  # Small people + moderate count = dense distant crowd
    
    # Signal 6: High occupancy with small people (NEW - catches Crowded.JPG)
    if occupancy > OCCUPANCY_HIGH and median_height_ratio < SCALE_SMALL_THRESHOLD:
        high_density_signals += 2  # Many small people filling the image = dense
    
    # === LOW DENSITY INDICATORS ===
    
    # Counter-signal 1: Very large people (close-up = sparse)
    if median_height_ratio > SCALE_LARGE_THRESHOLD:
        low_density_signals += 4
    
    # Counter-signal 2: Good YOLO count with high occupancy
    if occupancy > 0.5 and N >= 4:
        low_density_signals += 3
    
    # Counter-signal 3: Medium-large people with reasonable detection
    if median_height_ratio > SCALE_MEDIUM_THRESHOLD and N >= 8:
        low_density_signals += 2
    
    # Counter-signal 4: Low edge density (simple scene)
    if features["edge_density"] < 0.10:
        low_density_signals += 1
    
    # Counter-signal 5: Few people detected with low occupancy
    if N <= 4 and occupancy < 0.15:
        low_density_signals += 1
    
    # Counter-signal 6: Medium-sized people with very few detections
    if median_height_ratio > SCALE_MEDIUM_THRESHOLD and N <= 6:
        low_density_signals += 2
    
    # Counter-signal 7: Reasonable YOLO count with low edge density
    if N >= 7 and features["edge_density"] < 0.12:
        low_density_signals += 2
    
    # Counter-signal 8: High YOLO count (NEW - but should not trigger for Crowded.JPG)
    if N >= YOLO_MODERATE_COUNT:  # 20+
        low_density_signals += 1
    
    # -------------------------------
    # FINAL DECISION
    # -------------------------------
    net_score = high_density_signals - low_density_signals
    
    if net_score >= 1:
        crowd_level = "HIGH"
    else:
        crowd_level = "LOW"

    return {
        "image": os.path.basename(image_path),
        "crowd_level": crowd_level,
        "yolo_count": N,
        "occupancy": occupancy,
        "median_height_ratio": median_height_ratio,
        "edge_density": features["edge_density"],
        "laplacian_var": features["laplacian_var"],
        "color_variance": features["color_variance"],
        "high_signals": high_density_signals,
        "low_signals": low_density_signals,
        "net_score": net_score
    }

# -------------------------------
# RUN ON ALL IMAGES
# -------------------------------
if __name__ == "__main__":

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".avif", ".JPG", ".JPEG", ".webp", ".png")

    results_all = []

    for file in sorted(os.listdir(BASE_DIR)):
        if file.endswith(IMAGE_EXTS):
            image_path = os.path.join(BASE_DIR, file)
            result = categorize_scene(image_path)

            if result is None:
                continue

            results_all.append(result)

            print("\n--------------------------------")
            print(f"Image              : {result['image']}")
            print(f"Crowd Level        : {result['crowd_level']}")
            print(f"YOLO Count         : {result['yolo_count']}")
            print(f"Occupancy          : {result['occupancy']:.3f}")
            print(f"Median HeightRatio : {result['median_height_ratio']:.3f}")
            print(f"Edge Density       : {result['edge_density']:.3f}")
            print(f"Laplacian Var      : {result['laplacian_var']:.1f}")
            print(f"High Signals       : {result['high_signals']}")
            print(f"Low Signals        : {result['low_signals']}")
            print(f"Net Score          : {result['net_score']}")

    print("\n✅ Finished processing all images.")