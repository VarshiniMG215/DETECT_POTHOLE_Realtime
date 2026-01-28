# Real-Time Pothole Analytics with Depth Estimation

This project utilizes **YOLOv8 Segmentation** and **MiDaS (Monocular Depth Estimation)** to identify potholes in real-time video feeds and calculate their distance, area, and estimated depth in centimeters.

## Features
- Real-Time Detection: Uses YOLOv8-seg for high-speed road damage identification.
- Depth Estimation: Integrates MiDaS to predict a 3D depth map from a 2D camera feed.
- Metric Analytics: Estimates distance to the pothole and provides a severity rating (depth).
- Webcam Support: Optimized for live dashcam or webcam analysis.

##  Tech Stack
- Language: Python 3.x
- Computer Vision: OpenCV
- AI Models: Ultralytics (YOLOv8), PyTorch (MiDaS)
- Deep Learning Libraries: Timm, NumPy

##  Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Real-Time-Pothole-Analytics.git](https://github.com/YOUR_USERNAME/Real-Time-Pothole-Analytics.git)
