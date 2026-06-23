# ALPR on CCPD — University of Pisa Intelligent Systems Project

End-to-end Automatic License Plate Recognition pipeline on the
Chinese City Parking Dataset (CCPD). Two-stage architecture:
**YOLO11-pose** for plate localization (keypoint regression of the
four plate corners), followed by perspective rectification, then a
choice of three recognizers — a **7-head CNN**, **LPRNet** (Zherzdev
& Gruzdev, 2018), and **PaddleOCR** (zero-shot baseline).

Final project for the *Intelligent Systems* MSc course at the
University of Pisa. Authors: Luca Granucci, Lorenzo Marranini.

---

## Repository structure

```
IS_Project/
├── ccpd_exploration.ipynb           # Stage 0: dataset analysis & figures
├── yolo_pose_plate_training.ipynb   # Stage 1: train YOLO11-pose
├── plate_recognition_unified.ipynb  # Stage 2: train + evaluate the 3 recognizers
├── end_to_end_evaluation.ipynb      # Full pipeline on a fresh CCPD subset
│
├── Dataset/                          # CCPD raw images 
├── runs_plate_recog/                 # Recognition model checkpoints (warp + bbox)
├── recognition_results/              # CSV/JSON results from Stage 2
├── end_to_end_eval/                  # End-to-end evaluation artifacts
├── Papers_State_Of_Art/              # Reference papers
```

---

## Hardware

Developed and tested on:

- **GPU**: NVIDIA RTX 2060 (6 GB VRAM)
- **OS**: CachyOS 
- **CUDA driver**: 12.6 

---

## Environment setup


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and populate the venv

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate

#Check cuda version before installing, and change suffix of url accordingly

uv pip install paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

uv pip install -r requirements.txt

python -m ipykernel install --user --name is-project \
  --display-name "Python 3.13 (IS Project)"
```



`paddlepaddle-gpu` is installed separately *before* the requirements
file because its wheel URL is pinned to a specific CUDA version, and
the index is hosted on PaddlePaddle's own servers (not PyPI). Letting 
`pip` resolve `paddlepaddle-gpu` from PyPI will fetch the cpu version

---

## Dataset setup

Download CCPD from the official release:

- **Repository**: <https://github.com/detectRecog/CCPD>

The release we use is **CCPD2020** (~352k images, organized by failure
mode into named subsets: `ccpd_base`, `ccpd_rotate`, `ccpd_tilt`,
`ccpd_blur`, `ccpd_weather`, `ccpd_db`, `ccpd_fn`, `ccpd_challenge`).
Extract the archive so the structure is:

```
Dataset/
├── ccpd_base/        *.jpg
├── ccpd_rotate/      *.jpg
├── ccpd_tilt/        *.jpg
├── ccpd_blur/        *.jpg
├── ccpd_weather/     *.jpg
├── ccpd_db/          *.jpg
├── ccpd_fn/          *.jpg
└── ccpd_challenge/   *.jpg
```

The notebooks expect the path `Dataset/` relative to the project
root. If your CCPD lives elsewhere, edit the `DATASET_ROOT` constant
in cell 1 of each notebook.

**Note on labels**: CCPD encodes annotations directly in the filename
(7 hyphen-separated fields encoding area ratio, tilt, bounding box,
four plate corners, plate string, brightness, and blur). No separate
label files are needed.

---

## Running the notebooks

The four notebooks are designed to be run in this order — each one
depends on artifacts produced by the previous ones.

### 1. `ccpd_exploration.ipynb` — dataset analysis (no GPU needed)

Run this first to verify the dataset is correctly placed and to
generate the figures used in the paper (character distributions per
position, subset composition, sample plate visualizations with corner
overlays).

### 2. `yolo_pose_plate_training.ipynb` — YOLO11-pose training

Trains YOLO11-pose to predict the four plate corners. We do
4-fold cross-validation, comparing model sizes (Nano / Small /
Medium), then a final retrain on the full pool.

### 3. `plate_recognition_unified.ipynb` — recognizer training

Trains LPRNet and the 7-head CNN on rectified plate crops cached from CCPD. Also evaluates
PaddleOCR zero-shot.

### 4. `end_to_end_evaluation.ipynb` — full pipeline on a fresh subset

Loads the trained YOLO11-pose model and all three recognizers from
their checkpoints, then runs the full pipeline (detect → rectify →
recognize) on a **fresh subset of CCPD images** explicitly excluded
from anything used in the training/cache. This gives the honest
end-to-end numbers reported in the paper.

---

## Pretrained checkpoints

The `runs_plate_recog/` directory contains pretrained checkpoints for both crop types
(warped and unwarped axis-aligned bounding-box):

```
runs_plate_recog/
├── lprnet_warp/   lprnet_box/  
└── seven_warp/    seven_box/    
```

---

## Acknowledgements

- **CCPD dataset**: Xu et al., *Towards End-to-End License Plate
  Detection and Recognition: A Large Dataset and Baseline*,
  ECCV 2018.
- **LPRNet** architecture ported from
  [`sirius-ai/LPRNet_Pytorch`](https://github.com/sirius-ai/LPRNet_Pytorch).
- **YOLO11-pose** via
  [Ultralytics](https://github.com/ultralytics/ultralytics).
- **PaddleOCR** via the
  [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR) project.