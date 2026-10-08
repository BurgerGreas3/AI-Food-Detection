"""
AI Food Detection - plate analysis server (prototype)

    uvicorn app:app --host 0.0.0.0 --port 8000

POST /analyze-plate   multipart: image (image/jpeg) + depth (JSON, 64 ints in mm, or 8x8 nested)
POST /calibrate       multipart: depth  (capture of the EMPTY table/plate -> stores z_table)
GET  /health

Geometry (per spec):
    dA_i = (z_i / fx) * (z_i / fy)          physical footprint of pixel i at depth z_i
    h_i  = max(0, z_table - z_i)            height above the reference plane
    V    = sum_{i in mask} h_i * dA_i
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("food3d")


def _env(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    # Intrinsics are defined at the reference resolution and rescaled to the real frame.
    ref_w: int = 640
    ref_h: int = 480
    # OV2640 (~65 deg diagonal, 4:3) -> f ~ 628 px at 640x480. REPLACE with checkerboard calibration.
    fx: float = _env("CAM_FX", 628.0)
    fy: float = _env("CAM_FY", 628.0)
    cx: float = _env("CAM_CX", 320.0)
    cy: float = _env("CAM_CY", 240.0)
    # VL53L5CX: 45x45 deg square FoV (63 deg diagonal).
    tof_fov_deg: float = _env("TOF_FOV_DEG", 45.0)
    tof_radial: bool = os.getenv("TOF_RADIAL", "0") == "1"   # convert ray distance -> planar z
    tof_rot90: int = int(_env("TOF_ROT90", 0))                # sensor mounting vs camera axes
    tof_flip_h: bool = os.getenv("TOF_FLIP_H", "0") == "1"
    tof_flip_v: bool = os.getenv("TOF_FLIP_V", "0") == "1"
    table_z_cm: float = _env("TABLE_Z_CM", 0.0)               # >0 pins the reference plane
    min_height_cm: float = _env("MIN_HEIGHT_CM", 0.2)         # ToF noise floor
    segmenter: str = os.getenv("SEGMENTER", "torch")          # torch (your own model) | color (demo/Wokwi)
    model_path: str = os.getenv("MODEL_PATH", "weights/food_seg.pt")
    split_components: bool = os.getenv("SPLIT_COMPONENTS", "0") == "1"  # one item per blob, not per class
    min_mask_px: int = int(_env("MIN_MASK_PX", 150))
    max_upload_bytes: int = 2_000_000
    api_key: str = os.getenv("API_KEY", "")                   # set when exposing via ngrok


CFG = Config()
_state: dict = {"table_z_cm": None}


# --------------------------------------------------------------------------- segmentation
@dataclass
class Segment:
    label: str
    conf: float
    mask: np.ndarray  # bool HxW


class Segmenter(Protocol):
    def segment(self, bgr: np.ndarray) -> list[Segment]: ...


class TorchSegmenter:
    """Semantic U-Net trained from scratch with `python -m ml.train`. One item per class
    (or per connected blob with SPLIT_COMPONENTS=1)."""

    def __init__(self, path: str):
        import torch

        from ml.model import UNet

        if not os.path.exists(path):
            raise RuntimeError(f"no checkpoint at {path}; train one with `python -m ml.train` or use SEGMENTER=color")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.torch, self.classes, self.size = torch, ck["classes"], tuple(ck["size"])
        self.model = UNet(len(self.classes), ck["base"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        log.info("loaded %s (val mIoU %.3f, classes=%s)", path, ck.get("miou", float("nan")), self.classes)

    def segment(self, bgr: np.ndarray) -> list[Segment]:
        import torch.nn.functional as F

        from ml.model import to_tensor

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            logits = self.model(to_tensor(rgb)[None])
            prob = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).softmax(1)[0].numpy()
        pred = prob.argmax(0)
        out = []
        for c, name in enumerate(self.classes):
            if c == 0:  # background
                continue
            m = pred == c
            if not m.any():
                continue
            if CFG.split_components:
                n, lab = cv2.connectedComponents(m.astype(np.uint8))
                parts = [lab == i for i in range(1, n)]
            else:
                parts = [m]
            for p in parts:
                if p.sum() >= CFG.min_mask_px:
                    out.append(Segment(name, float(prob[c][p].mean()), p))
        return out


class ColorSegmenter:
    """Deterministic HSV segmenter for the synthetic Wokwi/mock frames (rice/chicken/asparagus)."""

    RULES = {
        "rice": ((0, 0, 200), (180, 40, 255)),
        "chicken": ((5, 100, 80), (25, 255, 255)),
        "asparagus": ((35, 80, 60), (85, 255, 255)),
    }

    def segment(self, bgr: np.ndarray) -> list[Segment]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = []
        for label, (lo, hi) in self.RULES.items():
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            keep = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= CFG.min_mask_px])
            if keep.any():
                out.append(Segment(label, 0.99, keep))
        return out


@lru_cache(maxsize=1)
def get_segmenter() -> Segmenter:
    if CFG.segmenter == "color":
        return ColorSegmenter()
    # Any other model (e.g. a Mask R-CNN you write): implement .segment() -> list[Segment] and register here.
    return TorchSegmenter(CFG.model_path)


def resolve_overlaps(segs: list[Segment]) -> list[Segment]:
    """Each pixel belongs to exactly one item (highest confidence wins)."""
    if not segs:
        return segs
    owner = np.full(segs[0].mask.shape, -1, np.int16)
    for i in sorted(range(len(segs)), key=lambda j: segs[j].conf):
        owner[segs[i].mask] = i
    return [Segment(s.label, s.conf, owner == i) for i, s in enumerate(segs)]


# --------------------------------------------------------------------------- depth
def parse_depth(raw: str) -> np.ndarray:
    """JSON (64 or 8x8, millimetres) -> 8x8 float32 in cm, invalid zones filled with the median."""
    try:
        arr = np.asarray(json.loads(raw), dtype=np.float32)
    except (ValueError, TypeError):
        raise HTTPException(422, "depth must be JSON numbers")
    if arr.size != 64:
        raise HTTPException(422, f"depth must have 64 zones, got {arr.size}")
    arr = arr.reshape(8, 8)
    invalid = ~np.isfinite(arr) | (arr <= 0) | (arr > 4000)
    if invalid.sum() > 16:
        raise HTTPException(422, f"too many invalid ToF zones ({int(invalid.sum())}/64)")
    arr[invalid] = np.median(arr[~invalid])
    if CFG.tof_rot90:
        arr = np.rot90(arr, CFG.tof_rot90)
    if CFG.tof_flip_h:
        arr = np.fliplr(arr)
    if CFG.tof_flip_v:
        arr = np.flipud(arr)
    arr = np.ascontiguousarray(arr) / 10.0  # mm -> cm
    if CFG.tof_radial:
        t = np.tan(np.radians(((np.arange(8) + 0.5) / 8 - 0.5) * CFG.tof_fov_deg))
        tx, ty = np.meshgrid(t, t)
        arr = arr / np.sqrt(1 + tx**2 + ty**2)
    return arr.astype(np.float32)


def upsample_depth(tof_cm: np.ndarray, w: int, h: int, k: tuple) -> np.ndarray:
    """Bicubic 8x8 -> WxH, registered so the ToF field of view lands where it does in the RGB frame.
    Pixels outside the ToF footprint replicate the nearest edge zone."""
    fx, fy, cx, cy = k
    tan_h = math.tan(math.radians(CFG.tof_fov_deg / 2))
    span_x, span_y = 2 * fx * tan_h, 2 * fy * tan_h
    x0, y0 = cx - span_x / 2, cy - span_y / 2
    mx = ((np.arange(w) + 0.5 - x0) * 8.0 / span_x - 0.5).astype(np.float32)
    my = ((np.arange(h) + 0.5 - y0) * 8.0 / span_y - 0.5).astype(np.float32)
    gx, gy = np.meshgrid(mx, my)
    up = cv2.remap(tof_cm, gx, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(up, float(tof_cm.min()), float(tof_cm.max()))  # bicubic can overshoot


def reference_plane(depth: np.ndarray, food: np.ndarray) -> tuple[float, str]:
    if CFG.table_z_cm > 0:
        return CFG.table_z_cm, "env:TABLE_Z_CM"
    if _state["table_z_cm"] is not None:
        return float(_state["table_z_cm"]), "calibration"
    bg = depth[~food]
    if bg.size >= 500:
        return float(np.percentile(bg, 90)), "background_p90"
    return float(np.percentile(depth, 90)), "scene_p90 (little visible background)"


# --------------------------------------------------------------------------- geometry
def measure(mask: np.ndarray, depth: np.ndarray, h_cm: np.ndarray, dA: np.ndarray, k: tuple) -> dict:
    fx, fy, cx, cy = k
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    pts = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy]).astype(np.float32)  # metric cm
    (_, _), (a, b), _ = cv2.minAreaRect(cv2.convexHull(pts))
    hv = h_cm[mask]
    volume = float(np.sum(hv * dA[mask]))
    return {
        "dimensions_cm": {
            "length": round(max(a, b), 2),
            "width": round(min(a, b), 2),
            "height": round(float(np.percentile(hv, 95)), 2),
        },
        "mean_height_cm": round(float(hv.mean()), 2),
        "footprint_cm2": round(float(dA[mask].sum()), 1),
        "volume_cm3": round(volume, 1),
        "volume_ml": round(volume, 1),  # 1 cm3 == 1 mL
    }


def analyze(bgr: np.ndarray, tof_cm: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    sx, sy = w / CFG.ref_w, h / CFG.ref_h
    k = (CFG.fx * sx, CFG.fy * sy, CFG.cx * sx, CFG.cy * sy)

    segs = [s for s in get_segmenter().segment(bgr) if s.mask.sum() >= CFG.min_mask_px]
    segs = resolve_overlaps(segs)

    depth = upsample_depth(tof_cm, w, h, k)
    food = np.zeros((h, w), bool)
    for s in segs:
        food |= s.mask
    z_table, z_src = reference_plane(depth, food)

    h_cm = np.maximum(0.0, z_table - depth)
    h_cm[h_cm < CFG.min_height_cm] = 0.0
    dA = (depth / k[0]) * (depth / k[1])

    items, warnings = [], []
    for s in segs:
        if not s.mask.any():
            continue
        m = measure(s.mask, depth, h_cm, dA, k)
        if m["dimensions_cm"]["height"] == 0:
            warnings.append(f"{s.label}: height below ToF noise floor; volume unreliable")
        items.append({"label": s.label, "confidence": round(s.conf, 3), "area_px": int(s.mask.sum()), **m})
    if not items:
        warnings.append("no food items detected")
    return {
        "image": {"width": w, "height": h},
        "reference_plane_cm": round(z_table, 2),
        "reference_plane_source": z_src,
        "items": items,
        "total_volume_ml": round(sum(i["volume_ml"] for i in items), 1),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- API
app = FastAPI(title="AI Food Detection - plate analyzer", version="0.1.0")


def check_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if CFG.api_key and x_api_key != CFG.api_key:
        raise HTTPException(401, "invalid or missing X-API-Key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "segmenter": CFG.segmenter, "calibrated_table_z_cm": _state["table_z_cm"]}


@app.post("/calibrate", dependencies=[Depends(check_key)])
def calibrate(depth: str = Form(...)) -> dict:
    """Point the sensor at the EMPTY table/plate and POST the 8x8 grid."""
    z = float(np.median(parse_depth(depth)))
    _state["table_z_cm"] = z
    return {"table_z_cm": round(z, 2)}


@app.post("/analyze-plate", dependencies=[Depends(check_key)])
def analyze_plate(image: UploadFile = File(...), depth: str = Form(...)) -> dict:
    t0 = time.perf_counter()
    data = image.file.read(CFG.max_upload_bytes + 1)
    if len(data) > CFG.max_upload_bytes:
        raise HTTPException(413, "image too large")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "could not decode JPEG")
    result = analyze(bgr, parse_depth(depth))
    result["timing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
