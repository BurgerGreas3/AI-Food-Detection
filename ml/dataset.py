"""Datasets for training from scratch: an on-the-fly synthetic generator and a folder loader."""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import to_tensor

CLASSES = ["background", "rice", "chicken", "asparagus"]


class SyntheticPlates(Dataset):
    """Random plates with rice/chicken/asparagus blobs, lighting, texture and noise.
    Good for proving the pipeline end to end; it will NOT transfer to real food photos."""

    def __init__(self, n: int, size=(320, 240), train: bool = True, seed: int = 1234):
        self.n, self.size, self.train, self.seed = n, size, train, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + i)
        W, H = self.size
        base = int(rng.integers(30, 150))
        img = np.empty((H, W, 3), np.uint8)
        img[:] = np.clip(base + rng.integers(-15, 15, 3), 0, 255)
        lbl = np.zeros((H, W), np.uint8)

        if rng.random() < 0.7:  # plate (label stays background)
            pc = int(rng.integers(170, 235))
            cv2.circle(img, (W // 2 + int(rng.integers(-20, 20)), H // 2), int(rng.uniform(0.36, 0.46) * H),
                       tuple(int(v) for v in np.clip(pc + rng.integers(-8, 8, 3), 0, 255)), -1)

        def blob(cls, color, c, ax, ang):
            cv2.ellipse(img, c, ax, ang, 0, 360, color, -1)
            cv2.ellipse(lbl, c, ax, ang, 0, 360, cls, -1)

        def centre():
            return (int(rng.integers(int(.2 * W), int(.8 * W))), int(rng.integers(int(.25 * H), int(.75 * H))))

        present = [c for c in (1, 2, 3) if rng.random() < 0.75] or [int(rng.integers(1, 4))]
        rng.shuffle(present)
        for cls in present:
            if cls == 1:
                v = int(rng.integers(200, 250))
                col = (v, v - int(rng.integers(0, 12)), v - int(rng.integers(5, 30)))
                blob(1, col, centre(), (int(rng.uniform(.07, .15) * W), int(rng.uniform(.06, .13) * W)),
                     int(rng.integers(0, 180)))
            elif cls == 2:
                col = (int(rng.integers(120, 190)), int(rng.integers(70, 120)), int(rng.integers(30, 80)))
                blob(2, col, centre(), (int(rng.uniform(.07, .14) * W), int(rng.uniform(.05, .10) * W)),
                     int(rng.integers(0, 180)))
            else:
                for _ in range(int(rng.integers(1, 5))):
                    col = (int(rng.integers(50, 110)), int(rng.integers(120, 180)), int(rng.integers(40, 90)))
                    blob(3, col, centre(), (int(rng.uniform(.10, .18) * W), int(rng.uniform(.015, .03) * W)),
                         int(rng.integers(0, 180)))

        f = img.astype(np.float32)
        f += np.linspace(-1, 1, W, dtype=np.float32)[None, :, None] * rng.uniform(-25, 25)  # lighting
        f *= rng.uniform(0.7, 1.2)
        f += rng.normal(0, rng.uniform(3, 12), f.shape)  # sensor / texture noise
        img = np.clip(f, 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return to_tensor(img), torch.from_numpy(lbl).long()


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-25, 25), rng.uniform(0.8, 1.25))
    M[:, 2] += rng.uniform(-0.1, 0.1, 2) * (w, h)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    f = img.astype(np.float32) * rng.uniform(0.7, 1.3) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-25, 25)
    return np.clip(f, 0, 255).astype(np.uint8), mask


class FolderSegDataset(Dataset):
    """images/*.jpg|png + masks/<same stem>.png holding class ids (0 = background).
    FoodSeg103's img_dir / ann_dir layout works as-is. classes_file: one class name per line, in id order."""

    def __init__(self, images: str, masks: str, classes_file: str, size=(320, 240), train: bool = True):
        self.classes = [l.strip() for l in open(classes_file) if l.strip()]
        self.masks, self.size, self.train = masks, size, train
        paths = sorted(p for e in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(images, "**", f"*.{e}"), recursive=True))
        self.items = [(p, self._mask(p)) for p in paths if os.path.exists(self._mask(p))]
        if not self.items:
            raise SystemExit(f"no image/mask pairs found under {images} / {masks}")

    def _mask(self, p):
        return os.path.join(self.masks, os.path.splitext(os.path.basename(p))[0] + ".png")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask[mask >= len(self.classes)] = 0
        if self.train:
            img, mask = augment(img, mask, np.random.default_rng())
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(mask)).long()
