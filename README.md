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
- [Custom dataset vs MCD dataset](#custom-dataset-vs-mcd-dataset)
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

**Phase 2 — signal analysis (CPU, fast, re-runnable).** Everything after reads the cached
`.npz` instead of the video models:

1. `signal_analysis` combines each video frame with its mask + landmarks to extract the
   per-region skin-colour time series (the rPPG signal), on a true time axis.
2. `prepare_dataset` cuts that signal into clean, uniform 20-second segments suitable for
   modelling.
3. `hr_training` / `hb_training` train the HR and Hb models on those segments.
4. `prediction` runs the DSP HR baseline and both trained models over the segments and
   produces results tables and plots.

```
 data/videos ---+--> segmentation --> output/seg/*.npz --------+   Phase 1 (GPU)
                +--> landmarks    --> output/landmarks/*.npz ---+
                                                                v
                                    signal_analysis --> output/signals/*.npz     (rPPG signal)
                                                                v
                                    prepare_dataset --> output/train_data/*.npz  (20 s segments)
                          +-------------------------------------+---------------------------+
                          v                                     v                           v
                    hr_training                           hb_training                  prediction
                output/hr_model/                      output/hb_model/           output/prediction/
```

---

## Repository layout

```
rppg-pipeline/
|-- docker-compose.yml          # runs every stage (one service each)
|
|-- docker/                     # the shared base image (rppg-base)
|   |-- Dockerfile
|   `-- requirements.txt
|
|-- common/                     # shared code, imported as a package: from common import ...
|   |-- __init__.py
|   |-- config.py               #   ROI definitions, DSP frequency band, thresholds
|   `-- signal_processing.py    #   region extraction + rPPG DSP (POS/CHROM, bandpass, ...)
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
|-- prepare_dataset/            # CPU . signal -> clean 20 s segments
|   `-- prepare_dataset.py
|-- hr_training/                # GPU . train the HR model
|   |-- model.py  dataset.py  train.py
|-- hb_training/                # GPU . train the Hb model
|   |-- model.py  features.py  dataset.py  train.py
|-- prediction/                 # CPU . DSP baseline + trained HR & Hb models -> tables + plots
|   `-- predict.py
|
|-- tools/                      # standalone utilities
|   `-- mcd_fps_sidecar.py      #   (MCD only) per-clip true fps from PPG duration
|
|-- data/                       # inputs  (mounted read-only)
|   |-- videos/
|   |-- ppg/                    #   contact-PPG .PW files (for HR labels)
|   `-- ground_truth.csv        #   per-subject HR / Hb / biomarkers
|
`-- output/                     # all stage outputs (mounted read-write)
```

Two of the stages — `segmentation` and `landmarks` — carry their **own** Dockerfile and
requirements, because they load heavy vision models with their own dependencies. Every
other stage runs on the single shared **`rppg-base`** image and mounts its code in, so a
change to a script takes effect on the next run without rebuilding.

---

## Requirements

- **Docker Engine** and **Docker Compose v2** (`docker compose ...`).
- An **NVIDIA GPU** + driver + NVIDIA Container Toolkit for the GPU stages
  (`segmentation`, `landmarks`, `hr_training`, `hb_training`). The CPU stages run without a
  GPU. To run a GPU stage on CPU instead, delete its `deploy:` block from
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
> `/app/common`, and every stage imports it as `from common import ...`, so the relative
> mount paths only resolve from the root.

---

## Prepare your data

Place inputs under `data/`:

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
`patient_id`, with a `hemoglobin` column (g/dL) and optionally other biomarkers. Hemoglobin
is constant per subject. This file is required to train or score Hb.

---

## Running the pipeline

Each stage is a Compose service; run them in order with `docker compose run --rm`, from the
repository root.

```bash
# ---- Phase 1: per-frame GPU preprocessing (run once per dataset) ----
docker compose run --rm segmentation      # videos -> output/seg/
docker compose run --rm landmarks          # videos -> output/landmarks/

# ---- (MCD dataset only) generate the fps sidecar first -- see the next section ----

# ---- Phase 2: signal analysis and modelling (CPU unless noted) ----
docker compose run --rm signal_analysis    # -> output/signals/
docker compose run --rm prepare_dataset    # -> output/train_data/
docker compose run --rm hr_training         # (GPU) -> output/hr_model/
docker compose run --rm hb_training         # (GPU) -> output/hb_model/
docker compose run --rm rppg_prediction     # -> output/prediction/
```

Stages are **resumable**: a clip whose output already exists is skipped, so you can stop
and restart Phase 1. `signal_analysis` and `prepare_dataset` reuse existing outputs unless
you pass `--overwrite` (add it to the service's `command` in `docker-compose.yml`).

The service names are defined in `docker-compose.yml`; the exact flags for each stage live
in that file's `command:` fields, so per-stage options are edited there.

---

## Custom dataset vs MCD dataset

The only real difference is **how frame timing is trusted**, which changes one flag on
`signal_analysis`. (Background in [How timing is handled](#how-timing-is-handled).)

### Custom dataset (your own recordings)

If you record your own videos with real capture timestamps (e.g.
`ffmpeg -use_wallclock_as_timestamps 1`), the pipeline trusts the video's own per-frame
timestamps. **Run `signal_analysis` without `--fps-sidecar`** — remove that flag from the
service's `command` in `docker-compose.yml`. Everything else is identical.

### MCD dataset (MCD-rPPG)

The MCD videos were re-encoded to a wrong constant frame rate, so their embedded timing
can't be trusted directly. The true rate is recovered from the contact-PPG recording
duration and supplied as a per-clip **sidecar** file. Generate it **once, before
`signal_analysis`**, using the tool in `tools/` (CPU-only; it can run alongside Phase 1):

```bash
docker compose run --rm --entrypoint python signal_analysis \
    tools/mcd_fps_sidecar.py --frames-from video \
    --video-dir /data/videos --ppg-dir /data/ppg --out /output/mcd_fps.csv
```

This writes `output/mcd_fps.csv`. Then run `signal_analysis` **with**
`--fps-sidecar /output/mcd_fps.csv` (as in the provided `docker-compose.yml`): clips listed
in the sidecar use the corrected rate; any others fall back to their embedded timing. The
sidecar must live under `output/` (writable), not `data/` (read-only).

> If you re-generate or add the sidecar after already running `signal_analysis`, re-run it
> with `--overwrite`, or the old (uncorrected) signals are kept.

---

## Outputs

Everything is written under `output/`:

| Path | Produced by | Contents |
| --- | --- | --- |
| `output/seg/<clip>_seg.npz` | segmentation | per-frame skin/face parse masks |
| `output/landmarks/<clip>_landmarks.npz` | landmarks | per-frame face landmarks |
| `output/mcd_fps.csv` | mcd_fps_sidecar (MCD only) | corrected per-clip fps |
| `output/signals/<clip>_signals.npz` | signal_analysis | rPPG signal + per-frame timestamps |
| `output/train_data/<clip>_<k>_signals.npz` | prepare_dataset | clean 20 s segments |
| `output/train_data/segments_manifest.csv` | prepare_dataset | each segment's time span |
| `output/hr_model/` | hr_training | `hr_model.pt`, `history.csv`, `metrics.json`, `training_curves.png`, `test_scatter.png` |
| `output/hb_model/` | hb_training | `hb_model.pt`, `history.csv`, `metrics.json`, `hb_test_scatter.png` |
| `output/prediction/` | prediction | `hr_results.csv`, `hr_accuracy.png`, `hr_accuracy_by_condition.png`, `hr_by_condition.csv`, `plots/` |

The prediction stage's `hr_results.csv` has one row per segment: HR from each DSP method
(POS/CHROM/green) with a confidence, the PPG-derived HR label, the trained HR model's
prediction (`hr_model`), and the trained Hb model's prediction (`hb_pred`). Its plots show
predicted-vs-true HR per method, error broken down by exercise state x camera, and a
per-segment panel figure.

---

## Data formats

**Intermediate `.npz`**

| File | Key | Shape | Meaning |
| --- | --- | --- | --- |
| `seg/<clip>_seg.npz` | `masks` | `(T, H, W)` | per-pixel parse-class id per frame |
| `landmarks/<clip>_landmarks.npz` | `landmarks` | `(T, 478, 3)` | landmark `(x, y, z)`; NaN where no face |
| `signals/<clip>_signals.npz` | `signals` | `(3, T, 3)` | per-region mean RGB (region x frame x RGB) |
| | `timestamps` | `(T,)` | per-frame capture time in seconds |
| `train_data/<clip>_<k>_signals.npz` | `signals` | `(3, 600, 3)` | clean 20 s segment, uniform 30 Hz |
| | `fps` | scalar | sample rate (30) |

The three regions are, in order, **forehead, left cheek, right cheek**.

**`segments_manifest.csv`** — `segment, clip, index, t_start, t_end, abs_start`: the real
time span each segment was cut from, used to label it from the contact PPG.

**Trained-model checkpoints** — `hr_model.pt` / `hb_model.pt` bundle the network weights
plus the exact normalisation used at training time, so the prediction stage reproduces
training-time preprocessing precisely.

---

## How timing is handled

rPPG is a *frequency* measurement, so the signal's time axis must be correct.
`signal_analysis` does **not** trust a video header's nominal frame rate; it reads each
frame's real presentation timestamp and stores the per-frame times alongside the signal.
Downstream, `prepare_dataset` resamples each segment onto a uniform 30 Hz grid *using those
real timestamps* (monotonic-cubic interpolation), which fills short gaps and yields uniform
segments without stretching or compressing the pulse — a heartbeat stays at its true
frequency even if the camera's frame rate was variable or mislabelled.

Segments are only cut from clean stretches: a candidate 20 s window is rejected if it
contains a gap longer than ~1 s, or more than ~5 s of missing/invalid frames in total, so
the modelling data is free of large dropouts.

The one dataset-specific wrinkle is the MCD re-encoding issue handled by the fps sidecar,
described above.

---

## Notes on accuracy

**Heart rate.** The classical DSP methods (POS, CHROM, green) are reported per segment as a
baseline and are prone to picking a wrong spectral peak (an octave or a 2:3 ratio off),
especially at elevated heart rates. The trained HR model learns to pick the true peak and
is the more reliable estimate. HR is scored in **MAE (BPM)** and **within-6-BPM %** against
the contact PPG; the prediction stage also breaks error down by exercise state and camera
so weaknesses are visible rather than averaged away.

**Hemoglobin.** Hb from rPPG is a **much weaker signal** than HR — it is inferred from
subtle colour/absorption cues rather than an obvious pulse frequency. The Hb stage is built
to report honestly: it always compares against a **naive baseline** (predicting the mean),
includes a **memorisation check** (train-subject vs held-out-subject error), and reports
per-subject as well as per-segment metrics. Treat a result as real only if it beats the
naive baseline on **held-out subjects**.

Both training stages split **by subject** — a subject never appears in more than one of
train/validation/test — so reported accuracy reflects generalisation to new people, not
memorisation of the training set.