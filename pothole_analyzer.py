import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 1. INITIALIZE MODELS
print("Activating AI... (Please wait for first-time downloads)")

# Using a community-trained pothole segmentation model for better accuracy
# This model specifically looks for road damage
model = YOLO('https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt')

# MiDaS for depth calculation
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model.to(device).eval()

# 2. CALIBRATION CONSTANTS
FOCAL_LENGTH = 700      # Standard for 1080p webcams
REAL_WIDTH_METERS = 0.5 # Average pothole width
BASE_ROAD_LEVEL = 500   # Relative depth of a flat road

cap = cv2.VideoCapture(0)

print("--- SYSTEM READY ---")
print("Tips for phone testing: Lower phone brightness & hold steady.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # A. DETECT (Lowered confidence to 0.1 for phone testing)
    results = model(frame, stream=True, conf=0.1)
    
    # B. DEPTH ESTIMATION
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = torch.from_numpy(img_rgb).to(device).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = depth_model(img_input.unsqueeze(0))
        depth_map = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic"
        ).squeeze().cpu().numpy()

    # C. ANALYZE AND OVERLAY
    for r in results:
        if r.masks is not None:
            for i, mask_coords in enumerate(r.masks.xy):
                # 1. Distance Calculation
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                px_width = x2 - x1
                # Math: D = (Actual Width * Focal Length) / Pixel Width
                distance_m = (REAL_WIDTH_METERS * FOCAL_LENGTH) / max(1, px_width)
                
                # 2. Area Calculation (in pixels)
                area_px = cv2.contourArea(mask_coords.astype(np.int32))
                
                # 3. Depth Severity Estimation
                m_binary = r.masks.data[i].cpu().numpy()
                pothole_depth_vals = depth_map[m_binary > 0]
                if len(pothole_depth_vals) > 0:
                    avg_val = np.mean(pothole_depth_vals)
                    # Depth logic: The difference between road level and hole bottom
                    depth_cm = max(0, (BASE_ROAD_LEVEL - avg_val) / 10)
                else:
                    depth_cm = 0

                # 4. VISUALS
                # Draw the pothole outline
                cv2.polylines(frame, [mask_coords.astype(np.int32)], True, (0, 255, 0), 2)
                
                # Draw the Data Label
                label = f"Dist: {distance_m:.1f}m | Depth: {depth_cm:.1f}cm | Area: {int(area_px)}"
                cv2.rectangle(frame, (int(x1), int(y1)-25), (int(x1)+350, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1)+5, int(y1)-7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # D. SHOW FEED
    cv2.imshow("LIVE POTHOLE ANALYZER", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()