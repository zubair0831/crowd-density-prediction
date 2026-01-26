from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import io
import base64
import os
import sys

# ------------------------------------------------------------------
# Paths (NO hardcoding)
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P2PNET_DIR = os.path.join(BASE_DIR, "CrowdCounting-P2PNet")
WEIGHTS_PATH = os.path.join(P2PNET_DIR, "weights", "SHTechA.pth")

sys.path.insert(0, P2PNET_DIR)

from models import build_model

# ------------------------------------------------------------------
# Globals (managed by lifespan)
# ------------------------------------------------------------------

model = None
device = None
transform = None

# ------------------------------------------------------------------
# P2PNet Args
# ------------------------------------------------------------------

class Args:
    backbone = "vgg16_bn"
    row = 2
    line = 2

# ------------------------------------------------------------------
# Lifespan (BEST PRACTICE)
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform

    print("🚀 Loading P2PNet model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()
    model = build_model(args, training=False)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print(f"✅ Model loaded on {device}")

    yield  # ---- app runs here ----

    print("🧹 Shutting down, releasing model")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "P2PNet",
        "device": str(device),
        "description": "Crowd counting with adjustable confidence threshold",
    }

# ------------------------------------------------------------------
# Frame Processing with Adjustable Confidence
# ------------------------------------------------------------------

def process_frame(frame_bytes: bytes, threshold: float = 0.5):
    """
    Process a frame and detect people with adjustable confidence threshold.
    """
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    original_w, original_h = img.size
    
    print(f"  📐 Original image: {original_w}x{original_h}")

    # Resize if too large
    max_dim = 2048
    w, h = original_w, original_h
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
        print(f"  🔽 Scaled to: {w}x{h}")

    # Make dimensions divisible by 128
    nw = max(128, (w // 128) * 128)
    nh = max(128, (h // 128) * 128)
    
    print(f"  🤖 Model input: {nw}x{nh}")

    img_resized = img.resize((nw, nh), Image.LANCZOS)
    inp = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    scores = F.softmax(out["pred_logits"], -1)[0, :, 1]
    points = out["pred_points"][0]

    mask = scores > threshold
    pts = points[mask].cpu().numpy()
    confidence_scores = scores[mask].cpu().numpy()

    # CRITICAL: P2PNet outputs PIXEL coordinates relative to MODEL INPUT size (nw x nh)
    # NOT normalized 0-1 coordinates!
    # We scale from model input size to original image size
    
    scale_x = original_w / nw
    scale_y = original_h / nh

    coords = [
        [float(x * scale_x), float(y * scale_y)]
        for x, y in pts
    ]
    
    if len(coords) > 0:
        print(f"  ✅ Detected: {len(coords)} people")
        print(f"  📍 Model output (pixels in {nw}x{nh}): ({pts[0][0]:.1f}, {pts[0][1]:.1f})")
        print(f"  📏 Scale: {scale_x:.3f}x, {scale_y:.3f}y")
        print(f"  📍 Final coord (pixels in {original_w}x{original_h}): ({coords[0][0]:.1f}, {coords[0][1]:.1f})")

    return {
        "count": len(coords),
        "coordinates": coords,
        "confidence_scores": [float(s) for s in confidence_scores],
        "avg_confidence": float(confidence_scores.mean()) if len(confidence_scores) > 0 else 0.0,
        "original_size": [original_w, original_h],
    }

# ------------------------------------------------------------------
# WebSocket Endpoint with Confidence Support
# ------------------------------------------------------------------

@app.websocket("/ws/process-frames/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"📡 Session {session_id} connected")

    processed = 0
    confidence_threshold = 0.5  # Default

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "frame":
                frame_no = data["frame_number"]
                total = data.get("total_frames", 0)
                
                # Get confidence threshold if provided
                confidence_threshold = data.get("confidence", 0.5)

                frame_bytes = base64.b64decode(
                    data["frame_data"].split(",")[1]
                )

                result = process_frame(frame_bytes, threshold=confidence_threshold)

                processed += 1
                progress = int((processed / total) * 100) if total else 0

                await websocket.send_json({
                    "type": "result",
                    "frame": frame_no,
                    "count": result["count"],
                    "coordinates": result["coordinates"],
                    "avg_confidence": result["avg_confidence"],
                    "original_size": result["original_size"],
                    "progress": progress,
                    "total_frames": total,
                })

                print(f"✅ Frame {frame_no}: {result['count']} people (avg conf: {result['avg_confidence']:.2f}) at {result['original_size'][0]}x{result['original_size'][1]}")

            elif data["type"] == "complete":
                await websocket.send_json({
                    "type": "complete",
                    "total_frames_processed": processed,
                })
                print(f"🏁 Session {session_id} complete ({processed} frames)")
                break

    except WebSocketDisconnect:
        print(f"🔌 Session {session_id} disconnected")

    except Exception as e:
        print(f"❌ Error in session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })

# ------------------------------------------------------------------
# Local Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)