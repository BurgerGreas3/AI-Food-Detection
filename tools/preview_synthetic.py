"""Save a few SyntheticPlates samples as PNGs (image on the left, answer key on the right).

  python tools/preview_synthetic.py      ->  previews/sample_0.png ... sample_7.png
Answer-key colours: black = background, white = rice, orange = chicken, green = asparagus.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.dataset import SyntheticPlates  # noqa: E402

os.makedirs("previews", exist_ok=True)
ds = SyntheticPlates(8, train=False)
palette = np.array([[0, 0, 0], [255, 255, 255], [0, 128, 255], [0, 200, 0]], np.uint8)  # BGR

for i in range(len(ds)):
    x, y = ds[i]
    rgb = ((x * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    key = palette[y.numpy()]
    cv2.imwrite(f"previews/sample_{i}.png", np.hstack([bgr, key]))
print("saved 8 samples to previews/")
