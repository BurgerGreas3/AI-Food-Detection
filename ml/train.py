"""
Train the U-Net from scratch.

  python -m ml.train --data synthetic --epochs 20
  python -m ml.train --data folder --images data/img/train --masks data/ann/train \
        --val-images data/img/val --val-masks data/ann/val --classes data/classes.txt --epochs 100

Best checkpoint (by val mIoU) -> weights/food_seg.pt (git-ignored by *.pt).
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CLASSES, FolderSegDataset, SyntheticPlates
from .model import UNet


def dice_loss(logits, target, n):
    p = logits.softmax(1)
    oh = F.one_hot(target, n).permute(0, 3, 1, 2).float()
    inter, den = (p * oh).sum((0, 2, 3)), p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, n, dev):
    model.eval()
    cm = torch.zeros(n * n, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(dev)).argmax(1).cpu()
        cm += torch.bincount((y * n + pred).flatten(), minlength=n * n)
    cm = cm.reshape(n, n).float()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp(min=1)
    present = cm.sum(1) > 0
    return float(iou[present].mean()), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "folder"], default="synthetic")
    ap.add_argument("--images"); ap.add_argument("--masks"); ap.add_argument("--classes")
    ap.add_argument("--val-images"); ap.add_argument("--val-masks")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="weights/food_seg.pt")
    a = ap.parse_args()

    W, H = (int(v) for v in a.size.split("x"))
    assert W % 16 == 0 and H % 16 == 0, "size must be multiples of 16"
    if a.data == "synthetic":
        classes = CLASSES
        tr = SyntheticPlates(a.n_train, (W, H), train=True)
        va = SyntheticPlates(a.n_val, (W, H), train=False)
    else:
        tr = FolderSegDataset(a.images, a.masks, a.classes, (W, H), train=True)
        classes = tr.classes
        va = FolderSegDataset(a.val_images or a.images, a.val_masks or a.masks, a.classes, (W, H), train=False)
    n = len(classes)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n, a.base).to(dev)  # random init: nothing is downloaded or loaded
    print(f"{n} classes, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, device={dev}")
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.bs, num_workers=a.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y, n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        miou, iou = evaluate(model, vl, n, dev)
        print(f"ep {ep:3d} loss {tot/len(tl):.3f} val mIoU {miou:.3f} ({time.time()-t0:.0f}s) "
              + " ".join(f"{c[:6]}={v:.2f}" for c, v in zip(classes, iou.tolist())))
        if miou > best:
            best = miou
            torch.save({"state_dict": model.state_dict(), "classes": classes, "base": a.base,
                        "size": [W, H], "miou": miou}, a.out)
    print(f"best val mIoU {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
