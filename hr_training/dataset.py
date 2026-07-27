"""dataset.py — load train_data segments + PPG labels for training.

Reads the clean 20 s segments produced by prepare_dataset (<clip>_<k>_signals.npz,
signal (R, 600, 3) + fps), derives a per-segment HR label from the contact PPG (.PW)
over that segment's manifest time span (same logic as predict_hr.py, so labels are
consistent), and builds a model-ready tensor dataset with SUBJECT-WISE splits.

Model input channels: the 3 regions x 3 RGB = 9 channels, z-scored per channel. (POS
traces could be appended later; 9 raw channels is the honest starting point — the
model learns its own projection.)

A per-segment DSP confidence (max of POS/CHROM/green prominence, recomputed cheaply) is
carried alongside so evaluation can split metrics by high/low confidence.
"""

import glob
import os
import re
from datetime import datetime

import numpy as np

from common import config
from common import signal_processing as sp

CAMERAS = ("IriunWebcam", "FullHDwebcam", "USBVideo")
_PW_LINE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+(\d{4}-\d{2}-\d{2}[ T][\d:.]+)")
PPG_HR_LO, PPG_HR_HI = 0.6, 3.5


def parse_subject_state(clip):
    toks = clip.split("_")
    state = "after" if "after" in toks else ("before" if "before" in toks else None)
    subj = re.search(r"(\d+)", clip)
    return (subj.group(1) if subj else None), state


def load_pw(path, fallback_fs=100.0):
    vals, times = [], []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            m = _PW_LINE.match(line)
            if m:
                vals.append(float(m.group(1)))
                try:
                    times.append(datetime.fromisoformat(m.group(2).replace("T", " ")))
                except ValueError:
                    times.append(None)
            elif line.split():
                try:
                    vals.append(float(line.split()[0])); times.append(None)
                except ValueError:
                    pass
    arr = np.asarray(vals, dtype=np.float64)
    fs = fallback_fs
    if len(times) > 1 and all(t is not None for t in times):
        span = (times[-1] - times[0]).total_seconds()
        if span > 0:
            fs = (len(times) - 1) / span
    return arr, fs


def spectral_hr(sig, fps, lo=PPG_HR_LO, hi=PPG_HR_HI):
    x = np.asarray(sig, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 16 or np.std(x) < 1e-9:
        return None, 0.0
    x = x - x.mean()
    w = np.hanning(len(x))
    nfft = int(2 ** np.ceil(np.log2(len(x) * 4)))
    psd = np.abs(np.fft.rfft(x * w, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return None, 0.0
    fb, pb = freqs[band], psd[band]
    i = int(np.argmax(pb))
    return float(fb[i] * 60.0), float(pb[i] / (np.median(pb) + 1e-12))


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def segment_channels(signals):
    """(R, T, 3) -> (R*3, T) z-scored channels (regions x RGB)."""
    R, T, C = signals.shape
    chans = [_z(signals[r, :, c]) for r in range(R) for c in range(C)]
    return np.stack(chans, axis=0).astype(np.float32)


def dsp_confidence(signals, fps):
    """Max prominence across POS/CHROM/green on the face-averaged signal (cheap)."""
    face = np.nanmean(signals, axis=0)
    confs = []
    for wf in (sp.extract_pos(face, fps=fps), sp.extract_chrom(face, fps=fps), face[:, 1]):
        _, c = spectral_hr(wf, fps, config.HR_FREQ_MIN_HZ if hasattr(config, "HR_FREQ_MIN_HZ") else 0.7,
                           config.HR_FREQ_MAX_HZ if hasattr(config, "HR_FREQ_MAX_HZ") else 3.0)
        confs.append(c)
    return float(max(confs)) if confs else 0.0


def load_dataset(segments_dir, ppg_dir, manifest=None, min_bpm=40.0, max_bpm=200.0):
    """Return dict with X (N,C,T) f32, y (N,) f32, conf (N,), subject (N,), clip (N,)."""
    manifest = manifest or os.path.join(segments_dir, "segments_manifest.csv")
    spans = {}
    if os.path.exists(manifest):
        import csv
        for r in csv.DictReader(open(manifest, encoding="utf-8-sig")):
            spans[r["segment"]] = (r["clip"], float(r["t_start"]), float(r["t_end"]))

    ppg_cache = {}
    for fp in glob.glob(os.path.join(ppg_dir, "*.PW")) + glob.glob(os.path.join(ppg_dir, "*.pw")):
        m = re.search(r"(\d+)_(before|after)", os.path.basename(fp), re.IGNORECASE)
        if m:
            ppg_cache[(m.group(1), m.group(2).lower())] = load_pw(fp)

    X, y, conf, subj_list, clip_list, raw_list, fps_list = [], [], [], [], [], [], []
    skipped = 0
    for f in sorted(glob.glob(os.path.join(segments_dir, "*_signals.npz"))):
        seg_name = os.path.basename(f).replace("_signals.npz", "")
        d = np.load(f)
        signals = d["signals"]
        fps = float(d["fps"]) if "fps" in getattr(d, "files", []) else 30.0

        clip, t0, t1 = spans.get(seg_name, (None, None, None))
        if clip is None:
            clip = re.sub(r"_\d+$", "", seg_name)
        subj, state = parse_subject_state(clip)
        pw = ppg_cache.get((subj, state)) if subj and state else None
        if pw is None or t0 is None:
            skipped += 1
            continue
        ppg, ppg_fs = pw
        a, b = int(round(t0 * ppg_fs)), int(round(t1 * ppg_fs))
        seg_ppg = ppg[max(0, a):min(len(ppg), b)]
        label, _ = spectral_hr(seg_ppg, ppg_fs)
        if label is None or not (min_bpm <= label <= max_bpm):
            skipped += 1
            continue

        X.append(segment_channels(signals))
        # raw face-averaged RGB (un-normalized) so DSP baselines are computed correctly
        raw_list.append(np.nanmean(signals, axis=0).astype(np.float32))   # (T, 3)
        fps_list.append(np.float32(fps))
        y.append(np.float32(label))
        conf.append(np.float32(dsp_confidence(signals, fps)))
        subj_list.append(subj)
        clip_list.append(clip)

    return {
        "X": np.stack(X) if X else np.zeros((0, 9, 600), np.float32),
        "raw": np.stack(raw_list) if raw_list else np.zeros((0, 600, 3), np.float32),
        "fps": np.asarray(fps_list, np.float32),
        "y": np.asarray(y, np.float32),
        "conf": np.asarray(conf, np.float32),
        "subject": np.asarray(subj_list),
        "clip": np.asarray(clip_list),
        "skipped": skipped,
    }


def subject_splits(subjects, n_folds=5, test_fold=0, val_fold=1, seed=0):
    """Assign each SUBJECT to train/val/test so a subject never crosses splits."""
    uniq = sorted(set(subjects.tolist()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold = {uniq[order[i]]: (i % n_folds) for i in range(len(uniq))}
    split = np.empty(len(subjects), dtype=object)
    for i, s in enumerate(subjects):
        f = fold[s]
        split[i] = "test" if f == test_fold else ("val" if f == val_fold else "train")
    return split