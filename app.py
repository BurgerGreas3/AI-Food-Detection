import json
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load fine-tuned segmentation model
model = YOLO("yolov8n-seg.pt")

# Download a sample image for testing
url = "https://ultralytics.com/images/bus.jpg"  # Replace with a plate image URL
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 2. Generate a mock 8x8 ToF depth array (in millimeters)
# Simulated top-down view: Table is at 350mm, objects are closer (280mm)
mock_tof = np.full((8, 8), 350.0, dtype=np.float32)
mock_tof[2:6, 2:6] = 280.0  # Center object raised by 70mm

# Upscale 8x8 depth grid to match image resolution
depth_map = cv2.resize(
    mock_tof, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC
)
z_table = np.max(depth_map)

# 3. Run AI Segmentation
results = model(frame)[0]
FX, FY = 530.0, 530.0  # Simulated focal length

if results.masks is not None:
  for idx, mask_tensor in enumerate(results.masks.data):
    class_name = model.names[int(results.boxes.cls[idx])]
    mask = (
        cv2.resize(mask_tensor.cpu().numpy(), (frame.shape[1], frame.shape[0]))
        > 0.5
    )

    food_depths = depth_map[mask]
    height_map = np.maximum(0, z_table - food_depths)
    mean_z = float(np.mean(food_depths))

    pixel_area_mm2 = (mean_z / FX) * (mean_z / FY)
    volume_cm3 = (np.sum(height_map) * pixel_area_mm2) / 1000.0

    print(f"Detected: {class_name} | Estimated Volume: {volume_cm3:.1f} cm³")