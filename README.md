# rPPG Pipeline

Estimate heart rate (and, experimentally, hemoglobin) from ordinary facial video using **remote photoplethysmography** — the tiny, periodic colour changes in facial skin as blood pulses through it.

The project is organised as a **staged, containerized pipeline**. It deliberately separates the small amount of GPU-heavy, per-frame deep-learning work from the larger amount of lightweight CPU signal analysis, so the expensive part runs once and everything downstream can be iterated on freely.

## How it works

The pipeline runs in two phases connected by files on disk:

**Phase 1 — GPU preprocessing (slow, run once per dataset).** Two independent deep models process every frame of every video and cache their output as `.npz` arrays: a face-parsing network produces per-pixel skin/face segmentation, and a face-mesh detector produces per-frame landmarks.

**Phase 2 — CPU analysis (fast, re-runnable).** Everything after that reads the cached `.npz` files instead of the video models. Signals are extracted from the segmented skin (and visualised in the same pass), heart rate and hemoglobin are estimated from those signals, and the results are compared against ground truth in diagnostic plots.

The cached `.npz` files are the **only contract between the two phases** — the analysis stages never load the deep models. Because the extracted signals depend only on the video (never on ground truth), you can change thresholds, swap algorithms, re-plot, or fix bugs and re-run *all* of Phase 2 in seconds without ever paying the GPU cost again.

```
 data/videos/ ─┬─▶ [ segmentation ] ─▶ output/seg/*.npz ───────┐
   (+ GPU)     └─▶ [ landmarks    ] ─▶ output/landmarks/*.npz ─┤
                                                               ▼
                                        [ signal_analysis ] ─▶ output/signals/*.npz
                                                               │        (+ output/plots/*.png, *.mp4)
                                     ┌─────────────────────────┼─────────────────────────┐
                                     ▼                         ▼                          │
                             [ hr_prediction ]         [ hb_prediction ]                  │
                                     │                         │                          │
                              output/hr/*.csv          output/hb/*.csv                    │
                                     └────────────┬────────────┘                          │
                                                  ▼                                        │
                                          [ visualization ] ◀────────── data/ground_truth.csv
                                                  │
                                          output/plots/*.png
```

*(The final `visualization` box is the evaluation step that compares HR/Hb results to ground truth — add it when `predict_hr.py` / `predict_hb.py` exist. Per-clip signal plots and overlay videos are now produced directly by `signal_analysis`.)*

## Project structure

```
rppg-pipeline/
├── docker-compose.yml            # the ONLY compose file — drives every stage
│
├── common/                       # shared code, copied into each CPU image
│   ├── config.py                 #   bands, thresholds, ROI/grid definitions
│   └── signal_processing.py      #   DSP: detrend, bandpass, POS/CHROM, clean-window
│
├── data/                         # inputs (mounted read-only)
│   ├── videos/                   #   source video clips
│   └── ground_truth.csv          #   true HR / Hb per clip
│
├── output/                       # all stage outputs (mounted read-write)
│   ├── seg/                      #   <clip>_seg.npz
│   ├── landmarks/                #   <clip>_landmarks.npz
│   ├── signals/                  #   <clip>_signals.npz
│   ├── hr/                       #   hr_results.csv
│   ├── hb/                       #   hb_results.csv
│   └── plots/                    #   per-clip signal PNGs + overlay MP4s, and evaluation PNGs
│
├── segmentation/                 # GPU · video → skin/parse masks
│   ├── Dockerfile
│   ├── requirements.txt
│   └── segmentation.py
├── landmarks/                    # GPU · video → face landmarks
│   ├── Dockerfile
│   ├── requirements.txt
│   └── landmarks.py
│
├── signal_analysis/              # CPU · video + seg + landmarks → rPPG signals + plots/overlay
│   ├── Dockerfile
│   ├── requirements.txt
│   └── analyze_signals.py
├── hr_prediction/                # CPU · signals → heart rate
│   ├── Dockerfile
│   ├── requirements.txt
│   └── predict_hr.py
├── hb_prediction/                # CPU · signals → hemoglobin
│   ├── Dockerfile
│   ├── requirements.txt
│   └── predict_hb.py
└── visualization/                # CPU · results + ground_truth → evaluation plots
    ├── Dockerfile
    ├── requirements.txt
    └── visualize.py
```

## Stages

Each stage is a self-contained Docker service with its own dependencies. The shared configuration and DSP live in `common/` and are copied into the CPU images at build time.

**`segmentation`** *(GPU)* — runs a SegFormer face-parsing model on every frame and writes one `<clip>_seg.npz` of per-pixel class ids. Keeping the full parse map (not just skin) means any region can be derived later without re-running the model.

**`landmarks`** *(GPU-capable, CPU by default)* — runs MediaPipe FaceLandmarker on every frame and writes one `<clip>_landmarks.npz` of per-frame facial landmarks. MediaPipe's Python API runs on CPU, so this service does not require the GPU.

**`signal_analysis`** *(CPU)* — combines each clip's video frames with its segmentation and landmarks to produce the rPPG signals (per-region mean skin-RGB time series, written as `<clip>_signals.npz`) and, in the same pass, the diagnostic visualizations: a per-region R/G/B trace plot (`<clip>_signal.png`) and an overlay video (`<clip>_overlay.mp4`) showing the tinted skin mask and the exact ROI regions that were averaged. Because extraction and visualization share one region-selection code path, what you see in the overlay is exactly what went into the signals. Pass `--no-plot` or `--no-video` to skip either visualization output.

**`hr_prediction`** *(CPU)* — turns the signals into a heart-rate estimate per clip (POS/CHROM projection, bandpass, spectral peak selection, and quality gating), writing `hr_results.csv` with the estimate and its quality metrics.

**`hb_prediction`** *(CPU)* — extracts optical biomarkers (AC/DC ratios, perfusion indices, chrominance ratios) from the signals and regresses hemoglobin, writing `hb_results.csv`.

**`visualization`** *(CPU)* — compares the HR/Hb results against `ground_truth.csv` and renders the evaluation figures (estimated-vs-true scatter, per-clip error, algorithm comparison) into `output/plots/`.

## Requirements

- Docker Engine and Docker Compose v2 (`docker compose`, not the legacy `docker-compose`).
- For the `segmentation` (and optionally `landmarks`) GPU stage: an NVIDIA GPU with a recent driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host.
- Input videos in `data/videos/` and a `data/ground_truth.csv`.

## Usage

Place your videos in `data/videos/` and your ground truth in `data/ground_truth.csv`, then:

```bash
# Build all images
docker compose build

# Phase 1 — GPU preprocessing (run once per dataset)
docker compose run --rm segmentation
docker compose run --rm landmarks

# Phase 2 — CPU analysis (cheap; re-run freely)
docker compose run --rm signal_analysis
docker compose run --rm hr_prediction
docker compose run --rm hb_prediction
docker compose run --rm visualization
```

The services are one-shot batch jobs — each runs to completion and exits — hence `run --rm` rather than `up`. Per-stage arguments can be adjusted inline in the `command:` fields of `docker-compose.yml`.

Phase 1 stages are **resumable**: a clip whose `.npz` already exists is skipped, so an interrupted run simply picks up where it left off. `signal_analysis` behaves the same way per output — existing signals/plots/overlays are reused unless `--overwrite` is passed.

### Editing code without rebuilding

The Docker images bake in a copy of the code at build time, but for fast iteration you can bind-mount your source over that copy so edits take effect on the next `run` — no rebuild. Add these volumes to the `signal_analysis` service in `docker-compose.yml`:

```yaml
    volumes:
      - ./data:/data:ro
      - ./output:/output
      - ./signal_analysis/analyze_signals.py:/app/analyze_signals.py:ro
      - ./common:/app/common:ro
```

The mount path must match where the Dockerfile places the code (`WORKDIR /app`). After editing a `.py` file, just re-run `docker compose run --rm signal_analysis`. You only need to `docker compose build` again when `requirements.txt` changes (dependencies live in the image layers, not the mount). To keep this dev-only, put the two extra volume lines in a `docker-compose.override.yml` instead — Compose merges it automatically.

## Data formats

### Inputs

`data/ground_truth.csv` — one row per clip, matched to the video by filename:

| column   | meaning                          |
| -------- | -------------------------------- |
| `video`  | clip name (matches the video file) |
| `pulse`  | ground-truth heart rate (BPM)    |
| `hb`     | ground-truth hemoglobin (g/dL)   |

### Intermediate `.npz` (the phase boundary)

| file                             | key         | shape         | dtype   | contents                                                     |
| -------------------------------- | ----------- | ------------- | ------- | ------------------------------------------------------------ |
| `seg/<clip>_seg.npz`             | `masks`     | `(T, H, W)`   | uint8   | per-pixel face-parse class id per frame (0 = background)     |
| `landmarks/<clip>_landmarks.npz` | `landmarks` | `(T, 478, 3)` | float32 | per-frame landmarks `(x_px, y_px, z)`; NaN where no face     |
| `signals/<clip>_signals.npz`     | `signals`   | `(R, T, 3)`   | float64 | per-region mean skin-RGB time series (regions × frames × RGB) |

The masks are boolean *geometry* — colour comes from the video. `signal_analysis` combines the two (frame + mask) to build the signals.

### Outputs

`hr/hr_results.csv` and `hb/hb_results.csv` hold the per-clip estimates and errors; `plots/` holds the per-clip diagnostic figures (signal traces and overlay videos, from `signal_analysis`) and the evaluation figures (estimated-vs-true scatter, per-clip error, algorithm comparison, from `visualization`).

## Status & known limitations

This is a research pipeline under active development. Two behaviours are known and worth keeping in mind when reading results:

- **Underestimation at high heart rates.** On elevated-HR clips (e.g. post-exercise) the estimator tends to lock onto stronger low-frequency content (respiration harmonics, motion) and report a rate well below the truth; the error grows with the true rate. This is a well-documented rPPG failure mode and is the main open problem.
- **Quality gating validates presence, not correctness.** The pulse-presence checks (SNR, spatial coherence, temporal consistency) confirm that a clean, coherent, stable periodic signal exists — which a strong artifact can satisfy just as well as a true pulse. A high-quality-looking estimate is therefore not guaranteed to be the correct one; treat the gate as a confidence signal, not a correctness signal.