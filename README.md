# AI Food Detection (prototype)

Edge camera + 8x8 ToF -> FastAPI server -> per-food length / width / height (cm) and volume (mL).

```
ESP32 (JPEG 640x480 + 8x8 ToF mm) --multipart POST--> /analyze-plate
   -> segment (your own U-Net | color demo) -> bicubic ToF upsample -> z_table -> h_i -> V = sum(h_i * dA_i)
   -> JSON { items[{label, dimensions_cm, volume_ml}], total_volume_ml, ... }
```

## Run the server
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
SEGMENTER=color uvicorn app:app --port 8000            # demo segmenter for mock frames, no model needed
python tools/mock_plate.py --post http://127.0.0.1:8000  # prints result + rough ground truth
```
With your own trained model (default): `uvicorn app:app --port 8000` (loads `weights/food_seg.pt`; override with `MODEL_PATH`).

## Train your own segmenter (no pretrained weights)
`ml/model.py` is a ~1.9M-parameter U-Net with random initialisation; nothing is downloaded.
```bash
python -m ml.train --data synthetic --epochs 20        # pipeline check, ~minutes on CPU
python -m ml.train --data folder --images data/img/train --masks data/ann/train \
    --val-images data/img/val --val-masks data/ann/val --classes data/classes.txt --epochs 100
```
- Folder format: `images/*.jpg` + `masks/<same stem>.png` holding class ids (0 = background); `classes.txt` has one
  class name per line in id order. FoodSeg103's `img_dir`/`ann_dir` layout works as-is.
- The best checkpoint by validation mIoU is saved to `weights/food_seg.pt` (git-ignored via `*.pt`) with its class list,
  so the server needs no config change when you add classes.
- The synthetic set only proves the code works; it will not transfer to real photos. Training from scratch needs
  thousands of labelled real images plus augmentation (included); expect to iterate on data before architecture.
- The model is semantic (one mask per class). For separate pieces of the same food use `SPLIT_COMPONENTS=1`, or write an
  instance model later: any class with `.segment() -> list[Segment]` can be registered in `get_segmenter()`.

## Wokwi + ngrok
1. `uvicorn app:app --port 8000` and `ngrok http 8000`.
2. Set `API_KEY` on the server (the tunnel is public) and the same value in `sketch.ino`.
3. Paste `firmware/wokwi_sim/sketch.ino` and `mock_data.h` into a Wokwi ESP32 project, set `SERVER_URL`, run.
   Regenerate the header with `python tools/mock_plate.py --header firmware/wokwi_sim/mock_data.h`.

## Configuration (env vars)
`CAM_FX/FY/CX/CY` (default 628/628/320/240 at 640x480, **replace with checkerboard values**), `TOF_FOV_DEG` (45),
`TOF_RADIAL=1` if your sensor reports ray distance, `TOF_ROT90/FLIP_H/FLIP_V`, `TABLE_Z_CM`, `MIN_HEIGHT_CM` (0.2),
`SEGMENTER` (`torch`|`color`), `MODEL_PATH`, `SPLIT_COMPONENTS`, `API_KEY`.

## Design notes
- **Registration:** the ToF's 45x45 deg footprint covers only part of the OV2640 frame (about 520 px wide, wider than the
  41.8 deg vertical FoV). The 8x8 grid is remapped onto that footprint, edges replicated outside it.
- **Reference plane:** p90 of depth outside food masks, or pin it with `POST /calibrate` on an empty table (preferred).
- **Dimensions:** min-area rectangle of the mask back-projected to cm via `X=(u-cx)z/fx`; height = p95 of `h_i`.
- Heights below `MIN_HEIGHT_CM` are zeroed to suppress ToF noise; this biases very flat foods low.

## Accuracy: what to expect
On the synthetic plate the server returns roughly -8% (rice), -15% (chicken), -54% (asparagus) volume error, because each
ToF zone spans ~4 cm at 40 cm range and averages away small features. Expect +-1 cm ToF noise on top, which is large against
1-3 cm food heights. Treat asparagus-scale items as unreliable. Upgrades, in order of payoff: fuse the ToF as a metric
anchor for a monocular depth model (e.g. Depth Anything V2) instead of bicubic upsampling; use a known-geometry plate;
capture from a fixed, calibrated mount; then add per-food density tables if you want grams.

## Git
```bash
git init && git remote add origin https://github.com/BurgerGreas3/AI-Food-Detection.git
git add app.py README.md requirements.txt .gitignore firmware tools
git commit -m "Prototype: plate analysis server + Wokwi sim" && git push -u origin main
```
`.gitignore` excludes `.venv/`, `*.pt`, `__pycache__/`.
