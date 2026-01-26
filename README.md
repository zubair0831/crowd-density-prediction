# AI Crowd Monitoring System for Stampede Prevention

## Overview
A real-time, web-based crowd monitoring system that uses deep learning to detect and count people in video footage. The system provides early warnings when crowd density exceeds safe limits, helping prevent dangerous situations such as stampedes.

The project is built using **P2PNet**, a state-of-the-art crowd counting model, and focuses on accuracy, real-time feedback, and clear visual interpretation.

---

## Problem Statement
High crowd density in public spaces can quickly become hazardous. Manual monitoring is often slow and unreliable. This system aims to automate crowd analysis and provide early alerts so that authorities can take preventive action.

---

## Key Features
- **Real-time video analysis** with frame-by-frame processing  
- **AI-powered crowd counting** using P2PNet  
- **Live statistics** (current, average, and maximum crowd count)  
- **Automatic alerts** when safety thresholds are exceeded  
- **Visual overlays** including detection points and heatmaps  
- **Custom controls** for confidence threshold, alert limit, and processing FPS  

---

## System Architecture

### Backend
- FastAPI with WebSocket support
- PyTorch-based P2PNet integration
- Real-time inference and alert logic

### Frontend
- HTML5 Canvas-based visualization
- Vanilla JavaScript with WebSocket communication
- Responsive and lightweight UI

---

## Technologies Used

### Machine Learning
- **Model:** P2PNet (Point-to-Point Network)
- **Framework:** PyTorch
- **Backbone:** VGG16-BN
- **Dataset:** ShanghaiTech Part A (pre-trained weights)

### Backend
- Python
- FastAPI
- WebSockets
- torchvision, Pillow

### Frontend
- HTML5 Canvas
- JavaScript
- CSS (responsive layout)

---

## How It Works
1. Video is split into frames at a configurable FPS  
2. Frames are preprocessed and passed to P2PNet  
3. The model predicts point locations for each person  
4. Results are visualized in real time with statistics and alerts  

---

## Limitations
- Accuracy depends on video quality and lighting
- Extremely dense crowds may reduce precision
- Processing speed depends on available hardware

---

## Future Work
- Multi-camera support  
- Live video stream input  
- Zone-based crowd monitoring  
- Historical analytics and reporting  
- Cloud and mobile deployment  

---

## Status
🟢 **Active Development** – Core system implemented with ongoing improvements

---

## Acknowledgments
- P2PNet – Tencent Youtu Research  
- ShanghaiTech Crowd Counting Dataset  
- PyTorch and FastAPI communities
