# rPPG Pipeline — Heart Rate & Hemoglobin from Facial Video

Estimate **heart rate (HR)** and **hemoglobin (Hb)** from ordinary facial video, using
**remote photoplethysmography (rPPG)** — the tiny, periodic colour changes in facial skin
as blood pulses through it. A face-parsing network and a face-mesh detector isolate the
skin regions per frame; the average colour of those regions over time is the rPPG signal;
everything downstream (HR, Hb) is computed from that signal.

The project is a **staged, containerised pipeline**. Each stage is a small program that
reads files produced by the previous stage and writes files for the next, so the
expensive GPU work runs once and the lightweight analysis can be re-run freely. Every
stage runs as a one-shot Docker Compose service.

---

## Table of contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Build the images](#build-the-images)
- [Prepare your data](#prepare-your-data)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Data formats](#data-formats)
- [How timing is handled](#how-timing-is-handled)
- [Notes on accuracy](#notes-on-accuracy)

---

## How it works

The pipeline has two phases connected only by files on disk.

**Phase 1 — per-frame deep learning (GPU).** Two independent models look at every video
frame: `segmentation` produces a per-pixel skin/face parse, and `landmarks` produces
face-mesh landmarks. Their outputs are cached as `.npz` files. This is the slow part; it
runs once per dataset.

**Phase 2 — signal analysis and modelling (CPU/GPU).** Everything after reads the cached
`.npz` instead of the video models:

1. `signal_analysis` combines each video frame with its mask + landmarks to extract the
   per-region skin-colour time series (the rPPG signal), on a true time axis.
2. `prepare_dataset` cuts that signal into clean, uniform 20-second windows suitable for
   modelling and resamples them to a fixed rate.
3. `hr_model` / `hb_model` train the HR and Hb models on those windows.
4. `prediction` runs the DSP HR baselines and both trained models over the windows and
   produces a results table and plots.

```
data/videos ---+--> segmentation --> output/seg/*.npz --------+   Phase 1 (GPU)
               +--> landmarks    --> output/landmarks/*.npz ---+
                                                               v
                              signal_analysis --> output/signals/*.npz          (rPPG signal)
                                                               v
                              prepare_dataset --> output/dataset/*.npz           (20 s windows)
                    +------------------------------------------+-----------------------------+
                    v                                          v                             v
              hr_model.train                            hb_model.train                  prediction
             output/hr_model/                          output/hb_model/           output/prediction/
```

---

## Repository layout

```
hb_prediction/
|-- docker-compose.yml          # runs each stage as a one-shot service
|-- .gitignore
|
|-- docker/                     # the shared base image (rppg-base)
|   |-- Dockerfile              #   PyTorch CUDA runtime + ffmpeg
|   `-- requirements.txt
|
|-- common/                     # shared code, imported as a package (from common import config, ...)
|   |-- __init__.py
|   |-- config.py               #   ROI polygons, DSP frequency band, window/timing thresholds
|   |-- data_types.py           #   dataclasses shared across stages (tasks, records, ...)
|   |-- signal_processing.py    #   region extraction + rPPG DSP (POS / CHROM / green, bandpass)
|   |-- one_euro.py             #   One-Euro smoothing filter (landmarks / masks)
|   |-- ppg.py                  #   contact-PPG (.PW) parsing and HR labelling
|   |-- video_utils.py          #   frame reading with real per-frame timestamps
|   `-- visualize.py            #   shared plotting helpers
|
|-- segmentation/               # GPU . video -> skin/face masks     (own Docker image)
|   |-- segmentation.py
|   |-- Dockerfile
|   `-- requirements.txt
|-- landmarks/                  # GPU . video -> face landmarks       (own Docker image)
|   |-- landmarks.py
|   |-- Dockerfile
|   `-- requirements.txt
|
|-- signal_analysis/            # CPU . video + masks + landmarks -> rPPG signal
|   `-- analyze_signals.py
|
|-- prepare_dataset/            # CPU . signal -> clean, resampled 20 s windows
|   |-- __init__.py
|   |-- prepare_dataset.py      #   entry point (run as `python -m prepare_dataset.prepare_dataset`)
|   |-- window_generation.py    #   selects clean windows, skipping large gaps
|   `-- signal_interpolation.py #   resamples onto a uniform grid using real timestamps
|
|-- hr_model/                   # GPU . train the HR model (spectral CNN)
|   |-- __init__.py
|   |-- model.py                #   HRSpectralNet
|   |-- dataset.py
|   |-- train.py                #   entry point (`python -m hr_model.train`)
|   `-- visualize.py
|
|-- hb_model/                   # GPU . train the Hb model (feature MLP)
|   |-- __init__.py
|   |-- model.py                #   HbMLP
|   |-- features.py             #   engineered amplitude / colour features
|   |-- dataset.py
|   |-- train.py                #   entry point (`python -m hb_model.train`)
|   `-- visualization.py
|
`-- prediction/                 # CPU . DSP baselines + trained HR & Hb models -> table + plots
    |-- __init__.py
    |-- predict.py              #   entry point (`python -m prediction.predict`)
    |-- models.py               #   loads the trained checkpoints for inference
    |-- rppg_algorithms.py      #   POS / CHROM / green HR estimators
    `-- visualization.py
```

Two of the stages — `segmentation` and `landmarks` — carry their **own** Dockerfile and
requirements, because they load heavy vision models with their own dependencies. Every
other stage runs on the single shared **`rppg-base`** image and mounts its code in, so a
change to a script takes effect on the next run without rebuilding.

> **Note.** `docker-compose.yml` currently ships with only the `rppg_prediction` service
> active; the earlier stages are present as commented-out service blocks. Uncomment the
> stage you want to run (or run its script directly, as shown below) and adjust its
> `command:` flags as needed.

---

## Requirements

- **Docker Engine** and **Docker Compose v2** (`docker compose ...`).
- An **NVIDIA GPU** + driver + NVIDIA Container Toolkit for the GPU stages
  (`segmentation`, `landmarks`, `hr_model`, `hb_model`). The CPU stages run without a
  GPU. To run a GPU stage on CPU instead, remove its `deploy:` block from
  `docker-compose.yml` (training will be slower but works).
- Input videos, and — if you want HR/Hb *labels* — contact-PPG files and/or a
  `ground_truth.csv` (see [Prepare your data](#prepare-your-data)).

---

## Build the images

From the repository root:

```bash
# 1. shared base image used by all CPU stages, training, and prediction
docker build -t rppg-base ./docker

# 2. the two GPU vision stages (their own images)
docker build -t rppg-segmentation ./segmentation
docker build -t rppg-landmarks    ./landmarks
```

The base image is built from `docker/` so the shared `requirements.txt` there is the
single place dependencies are pinned. Stage code is **not** copied into `rppg-base`; the
Compose services mount each stage's folder and the `common/` package into the container at
run time.

> **Run everything from the repository root.** The services mount `./common` into
> `/app/common`, and every stage imports it as a package (`from common import config`,
> `from common.data_types import ...`), so the relative mount paths only resolve from the
> root.

---

## Prepare your data

Place inputs under `data/` (this folder is mounted read-only into the containers and is
not committed to the repo):

```
data/
|-- videos/                     # your facial video clips
|-- ppg/                        # contact-PPG .PW files      (needed for HR labels/scoring)
`-- ground_truth.csv            # per-subject labels          (needed for Hb, and HR ground truth)
```

**Clip naming.** The pipeline reads the subject, camera, and exercise state from each
clip's filename, in the form `<subject>_<camera>_<state>`, e.g. `1020_FullHDwebcam_after`.
`state` is `before` or `after` (exercise); `subject` is a numeric id. Keep this convention
— labelling and the per-condition analysis depend on it.

**Contact PPG (`data/ppg/`).** One text file per subject/state named `<subject>_<state>.PW`
(e.g. `1020_after.PW`). Each line is a sample: a value followed by a timestamp, e.g.
`123.4  2024-01-01 14:18:09.870`. These provide the ground-truth heart rate; without them,
HR is still *predicted* but cannot be *scored*.

**Ground truth (`data/ground_truth.csv`).** One row per subject/clip, keyed by
`patient_id`, with a `hemoglobin` column (g/dL) and optionally other biomarkers.
Hemoglobin is constant per subject. This file is required to train or score Hb.

---

## Running the pipeline

Run the stages in order, from the repository root. The prediction stage is a ready-to-run
Compose service; the other stages can be run either by uncommenting their service block in
`docker-compose.yml` or by invoking their script directly inside the `rppg-base` image.
The default flags below match the `command:` blocks in `docker-compose.yml`.

```bash
# ---- Phase 1: per-frame GPU preprocessing (run once per dataset) ----
docker compose run --rm segmentation      # videos -> output/seg/
docker compose run --rm landmarks          # videos -> output/landmarks/

# ---- Phase 2: signal analysis and modelling ----
docker compose run --rm signal_analysis    # -> output/signals/
docker compose run --rm prepare_dataset    # -> output/dataset/
docker compose run --rm hr_training         # (GPU) -> output/hr_model/
docker compose run --rm hb_training         # (GPU) -> output/hb_model/
docker compose run --rm rppg_prediction     # -> output/prediction/
```

The stage entry points and their key flags:

| Stage             | Invocation                                    | Key flags (defaults) |
| ----------------- | --------------------------------------------- | -------------------- |
| `segmentation`    | `segmentation/segmentation.py`                | `--input-dir`, `--output-dir`, `--batch-size 8`, `--scale 1.0`, `--overwrite` |
| `landmarks`       | `landmarks/landmarks.py`                       | `--input-dir`, `--output-dir`, `--device`, `--overwrite` |
| `signal_analysis` | `signal_analysis/analyze_signals.py`           | `--video-dir`, `--seg-dir`, `--landmarks-dir`, `--signals-dir`, `--workers`, `--no-plot`, `--no-video`, `--overwrite` |
| `prepare_dataset` | `python -m prepare_dataset.prepare_dataset`    | `--signals-dir`, `--ppg-dir`, `--ground-truth`, `--out-dir output/dataset`, `--workers` |
| `hr_model`        | `python -m hr_model.train`                     | `--segments-dir output/dataset`, `--manifest`, `--out-dir output/hr_model`, `--epochs 80`, `--batch-size 128` |
| `hb_model`        | `python -m hb_model.train`                     | `--segments-dir output/dataset`, `--manifest`, `--out-dir output/hb_model`, `--epochs 100`, `--batch-size 64` |
| `prediction`      | `python -m prediction.predict`                 | `--dataset-dir output/dataset`, `--ppg-dir`, `--manifest`, `--out-dir output/prediction`, `--hr-model`, `--hb-model`, `--no-plot`, `--workers` |

Stages are **resumable**: a clip whose output already exists is skipped, so you can stop
and restart Phase 1. `signal_analysis` and `prepare_dataset` reuse existing outputs unless
you pass `--overwrite`.

---

## Outputs

Everything is written under `output/`:

| Path                                       | Produced by      | Contents |
| ------------------------------------------ | ---------------- | -------- |
| `output/seg/<clip>_seg.npz`                | segmentation     | per-frame skin/face parse masks |
| `output/landmarks/<clip>_landmarks.npz`    | landmarks        | per-frame face landmarks |
| `output/signals/<clip>_signals.npz`        | signal_analysis  | rPPG signal + per-frame timestamps + pixel counts |
| `output/dataset/<clip>_<k>_signals.npz`    | prepare_dataset  | clean, resampled 20 s windows |
| `output/dataset/segments_manifest.csv`     | prepare_dataset  | each window's time span + labels |
| `output/hr_model/`                         | hr_model.train   | `best_model.pt`, `last_model.pt`, `history.csv`, `metrics.json`, `model_config.json`, training/scatter plots |
| `output/hb_model/`                         | hb_model.train   | `best_model.pt`, `last_model.pt`, `history.csv`, `metrics.json`, `model_config.json`, scatter plots |
| `output/prediction/`                       | prediction       | `hr_results.csv`, `hr_accuracy.png`, `hb_accuracy.png`, `plots/<segment>.png` |

The prediction stage's `hr_results.csv` has one row per window with these columns:
`segment`, `clip`, `t_start`, `t_end`, `hr_pos`, `conf_pos`, `hr_chrom`, `conf_chrom`,
`hr_green`, `conf_green`, `hr_label`, `hr_pred`, `hr_pred_conf`, `hb_label`, `hb_pred`.
That is: HR from each DSP method (POS / CHROM / green) with a confidence, the PPG-derived
HR label, the trained HR model's prediction (`hr_pred`) with its confidence, and the Hb
label and trained Hb model prediction (`hb_pred`). Its plots show predicted-vs-true HR and
Hb, plus a per-segment panel figure under `plots/`.

---

## Data formats

**Intermediate `.npz`**

| File                                | Key            | Shape         | Meaning |
| ----------------------------------- | -------------- | ------------- | ------- |
| `seg/<clip>_seg.npz`                | `masks`        | `(T, H, W)`   | per-pixel parse-class id per frame |
| `landmarks/<clip>_landmarks.npz`    | `landmarks`    | `(T, 478, 3)` | landmark `(x, y, z)`; NaN where no face |
| `signals/<clip>_signals.npz`        | `signals`      | `(3, T, 3)`   | per-region mean RGB (region x frame x RGB) |
|                                     | `timestamps`   | `(T,)`        | per-frame capture time in seconds |
|                                     | `pixel_counts` | `(3, T)`      | valid pixels per region per frame (quality) |
| `dataset/<clip>_<k>_signals.npz`    | `signals`      | `(3, 300, 3)` | clean 20 s window, uniform 15 Hz |
|                                     | `pixel_counts` | `(3, 300)`    | per-region quality over the window |
|                                     | `region_mask`  | `(3,)`        | which regions are valid for this window |
|                                     | `fps`          | scalar        | sample rate (15) |

The three regions are, in order, **forehead, left cheek, right cheek**.

The windowing parameters live in `common/config.py`: `WINDOW_SEC = 20.0` and
`TARGET_FPS = 15`, so each window is `20 x 15 = 300` samples. The DSP HR band is
`HR_FREQ_MIN_HZ = 0.83` to `HR_FREQ_MAX_HZ = 2.8` (~50–168 BPM).

**`segments_manifest.csv`** — one row per window with its `segment`, source `clip`,
`index`, `t_start`, `t_end` and absolute start time, plus the labels resolved from the
contact PPG and ground-truth CSV. Used to match each window to its HR/Hb labels.

**Trained-model checkpoints** — each model directory holds `best_model.pt` and
`last_model.pt` (network weights) alongside `model_config.json` and `metrics.json`, so the
prediction stage reproduces training-time preprocessing precisely. Point `prediction` at a
model directory with `--hr-model` / `--hb-model`.

---

## How timing is handled

rPPG is a *frequency* measurement, so the signal's time axis must be correct.
`signal_analysis` does **not** trust a video header's nominal frame rate; it reads each
frame's real presentation timestamp and stores the per-frame times alongside the signal.
Downstream, `prepare_dataset` resamples each window onto a uniform 15 Hz grid *using those
real timestamps* (interpolation), which fills short gaps and yields uniform windows without
stretching or compressing the pulse — a heartbeat stays at its true frequency even if the
camera's frame rate was variable or mislabelled.

Windows are only cut from clean stretches: a candidate 20 s window is rejected if it
contains a gap longer than `MAX_GAP_SEC` (~1 s) or too much missing/invalid data, so the
modelling data is free of large dropouts. If you record your own videos, capture real
per-frame timestamps (e.g. `ffmpeg -use_wallclock_as_timestamps 1`) so this stage has an
accurate time axis to work from.

---

## Notes on accuracy

**Heart rate.** The classical DSP methods (POS, CHROM, green) are reported per window as a
baseline and are prone to picking a wrong spectral peak (an octave or a 2:3 ratio off),
especially at elevated heart rates. The trained HR model (`HRSpectralNet`, a small 1-D
convolutional network over log-power spectra) learns to pick the true peak and is the more
reliable estimate. HR is scored against the contact PPG.

**Hemoglobin.** Hb from rPPG is a **much weaker signal** than HR — it is inferred from
subtle colour/absorption cues rather than an obvious pulse frequency, using engineered
amplitude/colour features fed to a small MLP (`HbMLP`). Treat a result as real only if it
clearly beats a naive mean-predicting baseline on **held-out subjects**.

Both training stages split **by subject** — a subject never appears in more than one of
train/validation/test — so reported accuracy reflects generalisation to new people, not
memorisation of the training set.