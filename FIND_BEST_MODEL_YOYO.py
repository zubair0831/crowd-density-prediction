from ultralytics import YOLO
import cv2
import os
import pandas as pd
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# MODELS TO TEST
# -------------------------------
yolo_models = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
}

# -------------------------------
# GET ALL IMAGES
# -------------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".avif", ".JPG", ".JPEG", ".webp")
image_paths = []

for file in sorted(os.listdir(BASE_DIR)):
    if file.endswith(IMAGE_EXTS):
        image_paths.append(os.path.join(BASE_DIR, file))

print(f"Found {len(image_paths)} images to test")

# -------------------------------
# PARAMETERS
# -------------------------------
CONF_THRES = 0.5
IOU_THRES = 0.5
MIN_BOX_AREA = 900

# -------------------------------
# RESULTS STORAGE
# -------------------------------
results_table = []

# -------------------------------
# MAIN LOOP
# -------------------------------
for model_name, model_path in yolo_models.items():
    print(f"\n{'='*50}")
    print(f"Loading model: {model_name}")
    print(f"{'='*50}")
    
    model = YOLO(model_path)
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"⚠️  {image_path} not found, skipping...")
            continue
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  Could not read {image_path}, skipping...")
            continue
            
        H, W = image.shape[:2]
        
        # Resize for inference
        yolo_input = cv2.resize(image, (640, 360))
        
        # -------------------------------
        # TIMED INFERENCE
        # -------------------------------
        start_time = time.perf_counter()
        results = model(
            yolo_input,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000  # ms
        
        boxes = results[0].boxes
        person_count = 0
        
        scale_x = W / 640
        scale_y = H / 360
        
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0]
                # Scale back
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > MIN_BOX_AREA:
                    person_count += 1
        
        # -------------------------------
        # STORE RESULT
        # -------------------------------
        results_table.append({
            "Model": model_name,
            "Image": os.path.basename(image_path),
            "Person_Count": person_count,
            "Inference_Time_ms": round(inference_time_ms, 2)
        })
        
        print(f"  ✓ {os.path.basename(image_path):20s} - Count: {person_count:3d} - Time: {inference_time_ms:6.2f}ms")

# -------------------------------
# CREATE COMPARISON TABLE
# -------------------------------
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

df = pd.DataFrame(results_table)

# Display full table
print("\n--- Full Results ---")
print(df.to_string(index=False))

# Pivot table for easy comparison
print("\n--- Person Count Comparison (by Image) ---")
pivot_count = df.pivot(index='Image', columns='Model', values='Person_Count')
print(pivot_count)

print("\n--- Average Inference Time (ms) by Model ---")
avg_time = df.groupby('Model')['Inference_Time_ms'].mean().round(2)
print(avg_time)

# Save to CSV
output_file = os.path.join(BASE_DIR, "yolo_comparison_results.csv")
df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

# Summary statistics
print("\n--- Model Performance Summary ---")
summary = df.groupby('Model').agg({
    'Person_Count': ['mean', 'min', 'max'],
    'Inference_Time_ms': ['mean', 'min', 'max']
}).round(2)
print(summary)