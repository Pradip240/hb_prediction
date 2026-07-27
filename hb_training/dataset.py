"""dataset.py — load train_data segments + hemoglobin labels for Hb regression.

Reuses the SAME clean 20 s segments as HR (prepare_dataset output), but computes
amplitude/colour features (features.py) instead of feeding the raw signal, and labels
each segment with its subject's hemoglobin from ground_truth.csv.

Hemoglobin is a PER-SUBJECT property (constant across a subject's cameras and
before/after clips), so the label is looked up by patient_id alone, and splits are
SUBJECT-WISE — a subject never appears in two splits. This is the critical guard: with
per-segment features sharing one per-subject label, a model could otherwise score well
by memorising subject identity rather than learning hemoglobin.
"""

import csv
import glob
import os
import re

import numpy as np

from common import config
from features import extract_features, feature_names

REGION_ORDER = getattr(config, "REGION_ORDER", ("forehead", "lcheek", "rcheek"))


def load_ground_truth(path, target="hemoglobin"):
    """patient_id -> target value (float). Values are subject-constant in the CSV."""
    labels = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = str(r.get("patient_id", "")).strip()
            v = r.get(target, "")
            if pid and v not in ("", "None", None):
                try:
                    labels[pid] = float(v)
                except ValueError:
                    pass
    return labels


def subject_of(clip_or_segment):
    m = re.match(r"(\d+)", clip_or_segment)
    return m.group(1) if m else None


def load_dataset(segments_dir, ground_truth_csv, target="hemoglobin"):
    """Return dict: X (N,F) float32 features, y (N,) float32 labels, subject (N,),
    clip (N,), feat_names (list), skipped (int)."""
    labels = load_ground_truth(ground_truth_csv, target)
    fnames = feature_names(REGION_ORDER)

    X, y, subj, clip_list = [], [], [], []
    skipped = 0
    for f in sorted(glob.glob(os.path.join(segments_dir, "*_signals.npz"))):
        seg_name = os.path.basename(f).replace("_signals.npz", "")
        pid = subject_of(seg_name)
        if pid is None or pid not in labels:
            skipped += 1
            continue
        d = np.load(f)
        signals = d["signals"]
        fps = float(d["fps"]) if "fps" in getattr(d, "files", []) else 30.0
        feats = extract_features(signals, fps, REGION_ORDER)
        if not np.all(np.isfinite(feats)):
            skipped += 1
            continue
        X.append(feats)
        y.append(np.float32(labels[pid]))
        subj.append(pid)
        clip_list.append(re.sub(r"_\d+$", "", seg_name))

    return {
        "X": np.stack(X) if X else np.zeros((0, len(fnames)), np.float32),
        "y": np.asarray(y, np.float32),
        "subject": np.asarray(subj),
        "clip": np.asarray(clip_list),
        "feat_names": fnames,
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