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

  ## Technical Core & Hardware Integration
Sensor Fusion (The Eyes): Integrated a Livox Avia 3D LiDAR using Triple Return mode to ensure reliable road profiling in high-dust and low-visibility environments.

Edge Computing (The Brain): Deployed an NVIDIA Jetson Orin NX to process high-density point clouds (240k points/sec) and execute deep learning models for real-time pothole geometry extraction.

Industrial Control (The Reflexes): Engineered a real-time communication bridge between the Jetson and a Beckhoff CX9240 PLC using EtherCAT/UDP, achieving sub-50ms system latency.

Feedback & Validation: Utilized MTN/2330 Tri-axial Accelerometers to monitor driver-seat G-forces and validate a 35%+ reduction in vertical vibration.

Actuation: Controlled a 4-channel Pneumo-Hydraulic system to adjust suspension damping/stiffness dynamically before tire-pothole impact.

##  Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Real-Time-Pothole-Analytics.git](https://github.com/YOUR_USERNAME/Real-Time-Pothole-Analytics.git)
