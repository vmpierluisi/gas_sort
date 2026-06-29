# DeepGAS

DeepGAS extends [DeepSORT](https://arxiv.org/abs/1703.07402) multi-object
tracking with **adaptive, score-driven generalized autoregressive score (GAS)
motion filters** as a drop-in alternative to the classic Kalman filter (KF). The goal
is to study whether adaptive motion modelling improves pedestrian tracking on the
[MOT17 benchmark](https://motchallenge.net/data/MOT17/).

> Master's thesis by **Victor Medina Pierluisi** — MSc Econometrics and Data Science, VU Amsterdam (2026).

## Pipeline

The tracker is a modern, PyTorch-based reimplementation of DeepSORT:

- **Detection** — [YOLOv8n](https://github.com/ultralytics/ultralytics)
  (`weights/yolov8n.pt`).
- **Appearance embedding** — OSNet (`osnet_ain_x1_0`, MSMT17) via
  [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
  (`weights/osnet_ain_x1_0_msmt17_256x128.pth`).
- **Association** — matching cascade combining cosine distance on appearance
  embeddings with IoU, exactly as in DeepSORT.
- **Motion model** — pluggable filter selected at runtime (see below). This is
  the contribution of the thesis.

Computation runs on Apple Silicon (MPS) when available, otherwise CPU.

## Installation

This project uses [uv](https://docs.astral.sh/uv/). It pins the exact dependency
versions used to produce the thesis results (`pyproject.toml` + `uv.lock`).

```bash
git clone https://github.com/vmpierluisi/gas_sort.git
cd gas_sort
uv sync
```

This creates a `.venv` with Python 3.11–3.13 and all dependencies (PyTorch,
torchreid, ultralytics, OpenCV, motmetrics, …). Run commands with `uv run` or
activate the environment with `source .venv/bin/activate`.

### Weights

The model weights are committed to the repository under `weights/`:

- `weights/yolov8n.pt` — YOLOv8n detector.
- `weights/osnet_ain_x1_0_msmt17_256x128.pth` — OSNet appearance descriptor.

### Dataset

The MOT17 dataset is **not** included in the repository (it is several GB). A
prepared copy is hosted on Google Drive:

**[⬇️ Download MOT17 (Google Drive)](https://drive.google.com/file/d/1ISbDglNjn28AXDYi1AJvtHqWnvKdVKT_/view?usp=drive_link)**

Download `MOT17.zip` from the link above and unzip it into the repository root.
From the command line you can use [`gdown`](https://github.com/wkentaro/gdown):

```bash
uvx gdown 1ISbDglNjn28AXDYi1AJvtHqWnvKdVKT_     # or: pip install gdown && gdown 1ISbDglNjn28AXDYi1AJvtHqWnvKdVKT_
unzip MOT17.zip                                 # extracts ./MOT17 into the repo root
```

The dataset is also available from the official source at
[motchallenge.net](https://motchallenge.net/data/MOT17/).

Either way, extract it so the final layout is:

```
MOT17/
├── train/
│   ├── MOT17-02-FRCNN/
│   │   ├── img1/
│   │   ├── gt/gt.txt
│   │   └── seqinfo.ini
│   └── ...
└── test/
```

## Motion filters

The motion model is selected with `--filter`. Available choices:

| `--filter`    | Description                                              |
|---------------|----------------------------------------------------------|
| `kf-ca`       | KF, constant-acceleration model (default)                |
| `kf-cv`       | KF, constant-velocity model                              |
| `gas-f`       | Hybrid GAS filter with KF updates, constant-acceleration |
| `gas-f-cv`    | Hybrid GAS filter with KF updates, constant-velocity     |
| `gas-pred-f`  | Pure GAS-F filter                                        |
| `gas-local`   | Pure Gaussian location filter                            |

The filter implementations live in `filters/`. `deep_sort/tracker.py:build_filter`
maps each name to its class.

### Covariance update in `gas-f` / `gas-f-cv`

The hybrid GAS filters (`gas-f`, `gas-f-cv`) support two covariance update rules,
to be chosen the filter source file. By default they use the **custom filtered covariance**:

```python
new_covariance = new_F @ covariance @ new_F.T + Q
```

To use the standard **Joseph-form KF covariance update** instead, edit
`filters/gas_filter_f.py` (or `filters/gas_filter_f_cv.py`): comment out the line
above and uncomment the two Joseph-form lines directly below it:

```python
IKH = np.eye(12) - K @ H            # np.eye(8) in gas_filter_f_cv.py
new_covariance = IKH @ covariance @ IKH.T + K @ R_pred @ K.T
```

Only one of the two rules should be active at a time.

## Running the tracker on a single sequence

```bash
uv run python deep_sort_app.py \
    --sequence_dir=./MOT17/train/MOT17-09-FRCNN \
    --output_file=./results/MOT17-09-FRCNN.txt \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --filter=gas-f \
    --display=True
```

Run `uv run python deep_sort_app.py -h` for all options (detection confidence,
NMS overlap, cosine gating threshold, etc.). With `--display=True` the tracker
shows the live visualization.

## Evaluating on MOT17

`evaluate_motchallenge.py` runs the tracker over every sequence in a MOT
directory and reports the standard MOTChallenge metrics (MOTA, IDF1, etc.) via
[motmetrics](https://github.com/cheind/py-motmetrics):

```bash
uv run python evaluate_motchallenge.py \
    --mot_dir=./MOT17/train \
    --output_dir=./results \
    --filter=gas-f \
    --min_confidence=0.3 \
    --nn_budget=100
```

Per-sequence tracking outputs are written to `--output_dir` in MOTChallenge
format, and a summary table is printed at the end.

## Repository layout

```
deep_sort_app.py            Run the tracker on one sequence
evaluate_motchallenge.py    Run + score the tracker across a MOT directory
deep_sort/                  Core tracker (detection, matching, track, tracker)
  detect_yolo.py            YOLO detection + OSNet embedding extraction
  nn_matching.py            Nearest-neighbour cosine metric
  linear_assignment.py      Matching cascade / min-cost assignment
  iou_matching.py           IoU association
  track.py / tracker.py     Track lifecycle and multi-target tracker
filters/                    Pluggable motion filters (Kalman + GAS variants)
application_util/           Visualization and pre-processing helpers
weights/                    YOLO and OSNet model weights
```

Generated artifacts (`results/`, `detections/`, `cache/`) and the `MOT17/`
dataset are git-ignored.

## Acknowledgements & citing

This work builds on DeepSORT by Wojke et al. If you use this code, please cite
the original papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }

## License

See [LICENSE](LICENSE).
