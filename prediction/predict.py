"""predict.py — HR (DSP baseline + trained model) and Hb prediction over clean segments.

Runs over a train_data/ folder of <clip>_<k>_signals.npz segments (uniform
(R, 600, 3) rPPG signals at a fixed fps, from prepare_dataset) and, per segment:

  * estimates HR three ways — POS, CHROM, and the raw green channel — each with the
    spectral peak's prominence as a confidence (the unsupervised DSP baseline);
  * derives a per-segment ground-truth HR from the contact PPG (.PW) over that segment's
    exact time span (from the prepare_dataset manifest), so estimates are scored per
    segment even when HR drifts across a recording (post-exercise);
  * optionally runs the trained HR model (--hr-model) to fill the hr_model column + spectrum
    panel, and the trained Hb model (--hb-model) to fill the hb_pred column + plot title.
    Hb is per-subject, so its value repeats across a subject's segments.

Outputs:
  <out-dir>/hr_results.csv                per segment: HR per method + confidence, PPG
                                          label, hr_model, hb_pred.
  <out-dir>/plots/<segment>.png           stacked subplots: raw region RGB, POS, CHROM,
                                          green, PPG waveform, and the model spectrum
                                          (title shows HR label / model HR / Hb).
  <out-dir>/hr_accuracy.png               predicted-vs-true scatter per method (+ model).
  <out-dir>/hr_accuracy_by_condition.png  MAE by exercise-state x camera; + hr_by_condition.csv

Usage:
  python predict.py --segments-dir output/train_data --ppg-dir data/ppg \
      --out-dir output/prediction [--hr-model output/hr_model/hr_model.pt] \
      [--hb-model output/hb_model/hb_model.pt] [--no-plot]
"""

import argparse
import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import config
from common import signal_processing as sp

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

CAMERAS = ("IriunWebcam", "FullHDwebcam", "USBVideo")
HR_LO_HZ = getattr(config, "HR_FREQ_MIN_HZ", 0.7)
HR_HI_HZ = getattr(config, "HR_FREQ_MAX_HZ", 3.0)
PPG_HR_LO, PPG_HR_HI = 0.6, 3.5
_PW_LINE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+(\d{4}-\d{2}-\d{2}[ T][\d:.]+)")


# ----------------------------------------------------------------------
# HR from a 1-D pulse waveform (spectral peak + prominence as confidence)
# ----------------------------------------------------------------------

def spectral_hr(sig, fps, lo=HR_LO_HZ, hi=HR_HI_HZ):
    """Return (bpm, confidence). Confidence = peak power / median band power."""
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
    conf = float(pb[i] / (np.median(pb) + 1e-12))
    return float(fb[i] * 60.0), conf


def method_waveforms(signals, fps):
    """Return dict method -> (waveform, bpm, confidence) for POS, CHROM, green.

    POS/CHROM are computed on the face-averaged region RGB (mean over regions); green
    is the face-averaged green channel. Each waveform is bandpassed for display + HR.
    """
    face = np.nanmean(signals, axis=0)               # (T, 3) region-averaged RGB
    out = {}
    pos = sp.extract_pos(face, fps=fps)
    out["POS"] = (pos, *spectral_hr(pos, fps))
    chrom = sp.extract_chrom(face, fps=fps)
    out["CHROM"] = (chrom, *spectral_hr(chrom, fps))
    green = sp.bandpass_filter(sp.interpolate_nans(face[:, 1]), fps) \
        if hasattr(sp, "bandpass_filter") else sp.interpolate_nans(face[:, 1])
    out["green"] = (green, *spectral_hr(face[:, 1], fps))
    return out


# ----------------------------------------------------------------------
# PPG label for a segment's time span
# ----------------------------------------------------------------------

def load_pw(path, fallback_fs=100.0):
    """Return (values, fs, t0_abs) — samples, sample rate, absolute start time (s)."""
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
                    vals.append(float(line.split()[0]))
                    times.append(None)
                except ValueError:
                    pass
    arr = np.asarray(vals, dtype=np.float64)
    fs, t0 = fallback_fs, 0.0
    if len(times) > 1 and all(t is not None for t in times):
        span = (times[-1] - times[0]).total_seconds()
        if span > 0:
            fs = (len(times) - 1) / span
        t0 = times[0].timestamp()
    return arr, fs, t0


def ppg_segment(ppg, fs, t_start, t_end):
    """Slice the PPG to [t_start, t_end] seconds (relative to PPG start) and return it."""
    a = int(round(t_start * fs))
    b = int(round(t_end * fs))
    a, b = max(0, a), min(len(ppg), b)
    return ppg[a:b] if b > a else np.array([])


def parse_clip_state(clip):
    subj = re.search(r"(\d+)", clip)
    state = "after" if "after" in clip else ("before" if "before" in clip else None)
    return (subj.group(1) if subj else None), state


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_segment(seg_name, signals, fps, waves, ppg_wave, ppg_fs, label_hr, out_png, model_pred=None, hb_pred=None):
    T = signals.shape[1]
    t = np.arange(T) / fps
    fig, axs = plt.subplots(6, 1, figsize=(12, 13), sharex=False)

    # 1. raw region RGB (green channel per region, the most pulse-bearing)
    colors = {"forehead": "olive", "lcheek": "magenta", "rcheek": "green"}
    names = list(config.REGION_ORDER) if hasattr(config, "REGION_ORDER") else [f"r{i}" for i in range(signals.shape[0])]
    for r in range(signals.shape[0]):
        axs[0].plot(t, signals[r, :, 1], color=colors.get(names[r], None), lw=0.8, label=names[r])
    axs[0].set_ylabel("raw green\n(per region)"); axs[0].legend(fontsize=7, loc="upper right"); axs[0].grid(alpha=0.3)

    # 2-4. POS / CHROM / green pulse waveforms
    for ax, key in zip(axs[1:4], ["POS", "CHROM", "green"]):
        wf, bpm, conf = waves[key]
        wf = np.asarray(wf, dtype=np.float64)
        wf = (wf - np.nanmean(wf)) / (np.nanstd(wf) + 1e-9)
        ax.plot(t[:len(wf)], wf, lw=0.9)
        hr_txt = f"{bpm:.1f} BPM (conf {conf:.0f})" if bpm is not None else "no peak"
        ax.set_ylabel(f"{key}\n{hr_txt}"); ax.grid(alpha=0.3)

    # 5. interpolated PPG waveform (ground truth), on its own time axis
    if len(ppg_wave) > 4:
        tp = np.arange(len(ppg_wave)) / ppg_fs
        pw = (ppg_wave - np.mean(ppg_wave)) / (np.std(ppg_wave) + 1e-9)
        axs[4].plot(tp, pw, color="crimson", lw=0.8)
        axs[4].set_ylabel(f"PPG (label)\n{label_hr:.1f} BPM" if label_hr is not None else "PPG (label)")
    else:
        axs[4].text(0.5, 0.5, "no PPG for this span", ha="center", va="center", transform=axs[4].transAxes)
        axs[4].set_ylabel("PPG (label)")
    axs[4].grid(alpha=0.3)

    # 6. trained model: its predicted spectrum (pulse-likelihood over BPM), peak = its HR.
    # (The model has no time-domain waveform — it outputs a frequency distribution.)
    if model_pred is not None:
        bpm_grid, prob, model_hr = model_pred
        axs[5].plot(bpm_grid, prob, color="purple", lw=1.2)
        axs[5].axvline(model_hr, color="purple", ls="--", lw=1)
        if label_hr is not None:
            axs[5].axvline(label_hr, color="crimson", ls=":", lw=1, label="PPG label")
            axs[5].legend(fontsize=7, loc="upper right")
        axs[5].set_ylabel(f"model spectrum\n{model_hr:.1f} BPM")
        axs[5].set_xlabel("HR (BPM)")
    else:
        axs[5].text(0.5, 0.5, "Trained model (pending)", ha="center", va="center",
                    transform=axs[5].transAxes, color="gray", style="italic")
        axs[5].set_ylabel("model"); axs[5].set_xlabel("time (s)")
    axs[5].grid(alpha=0.3)

    title = f"{seg_name}   —   label {label_hr:.1f} BPM" if label_hr is not None else seg_name
    if model_pred is not None:
        title += f"   |   model {model_pred[2]:.1f} BPM"
    if hb_pred is not None:
        title += f"   |   Hb {hb_pred:.2f} g/dL"
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out_png, dpi=120)
    plt.close()


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _default_workers():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# lazy per-process model cache (avoids re-loading per segment and cross-process pickling)
_MODEL_CACHE = {}


def _load_module(mod_name, filename, search_dirs):
    """Import a module from one of search_dirs under a UNIQUE name, avoiding the
    hr_training/model.py vs hb_training/model.py name collision (both are 'model')."""
    import importlib.util
    import sys
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(mod_name, path)
            m = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = m
            spec.loader.exec_module(m)
            return m
    raise ImportError(f"{filename} not found in {search_dirs}")


def get_model(model_path):
    if not _HAS_TORCH or not model_path:
        return None
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        here = os.path.dirname(__file__)
        hr_mod = _load_module("hr_model_def", "model.py",
                              ["hr_training", "/app/hr_training", os.path.join(here, "..", "hr_training")])
        m = hr_mod.HRSpectralNet(n_channels=9)
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        m.eval()
        _MODEL_CACHE[model_path] = m
        return m
    except Exception as e:
        _MODEL_CACHE[model_path] = None
        print(f"  (HR model load failed: {type(e).__name__}: {e})", flush=True)
        return None


def model_predict(model, signals, fps):
    """Run the spectral model on a segment. Returns (bpm_grid, prob, hr) or None.

    Builds the same 9 z-scored channels the model was trained on, then reads its
    frequency distribution (the model's own band_bpm grid) and the soft-argmax HR.
    """
    if model is None:
        return None
    R, T, C = signals.shape
    chans = []
    for r in range(R):
        for c in range(C):
            x = signals[r, :, c].astype(np.float64)
            x = (x - x.mean()) / (x.std() + 1e-9)
            chans.append(x)
    X = np.stack(chans, axis=0)[None].astype(np.float32)      # (1, 9, T)
    with torch.no_grad():
        pred_bpm, logits, prob = model(torch.from_numpy(X))
    bpm_grid = model.band_bpm.cpu().numpy()
    return bpm_grid, prob[0].cpu().numpy(), float(pred_bpm.item())


# lazy per-process cache for the Hb model (checkpoint dict, not just weights)
_HB_CACHE = {}


def get_hb_model(hb_model_path):
    """Load the trained Hb model + its feature-extractor + normalisation. Returns a dict
    {model, features_mod, mu, sd, y_mean, y_std, region_order} or None. The Hb model
    consumes engineered amplitude/colour FEATURES (features.py), not the raw signal — so we
    also need the exact feature code and the train-time normalisation from the checkpoint."""
    if not _HAS_TORCH or not hb_model_path:
        return None
    if hb_model_path in _HB_CACHE:
        return _HB_CACHE[hb_model_path]
    try:
        here = os.path.dirname(__file__)
        dirs = ["hb_training", "/app/hb_training", os.path.join(here, "..", "hb_training")]
        hb_mod = _load_module("hb_model_def", "model.py", dirs)
        feat_mod = _load_module("hb_features_def", "features.py", dirs)
        ck = torch.load(hb_model_path, map_location="cpu", weights_only=False)
        mu = np.asarray(ck["mu"], dtype=np.float32)
        sd = np.asarray(ck["sd"], dtype=np.float32)
        m = hb_mod.HbMLP(len(mu))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        bundle = {"model": m, "features_mod": feat_mod, "mu": mu, "sd": sd,
                  "y_mean": float(ck["y_mean"]), "y_std": float(ck["y_std"]),
                  "region_order": list(getattr(config, "REGION_ORDER", ("forehead", "lcheek", "rcheek")))}
        _HB_CACHE[hb_model_path] = bundle
        return bundle
    except Exception as e:
        _HB_CACHE[hb_model_path] = None
        print(f"  (Hb model load failed: {type(e).__name__}: {e})", flush=True)
        return None


def hb_predict(bundle, signals, fps):
    """Predict hemoglobin for one segment. Returns a float (g/dL) or None. Uses the exact
    feature extractor + normalisation + target de-centring the model was trained with."""
    if bundle is None:
        return None
    try:
        feats = bundle["features_mod"].extract_features(signals, fps, bundle["region_order"])
        if not np.all(np.isfinite(feats)):
            return None
        x = ((feats - bundle["mu"]) / bundle["sd"]).astype(np.float32)[None]
        with torch.no_grad():
            z = bundle["model"](torch.from_numpy(x)).item()
        return float(z * bundle["y_std"] + bundle["y_mean"])   # de-center to g/dL
    except Exception as e:
        print(f"  (Hb predict failed: {type(e).__name__}: {e})", flush=True)
        return None


def process_segment(task):
    (seg_name, seg_path, span, ppg_info, out_dir, no_plot, model_path, hb_model_path) = task
    try:
        d = np.load(seg_path)
        signals = d["signals"]
        fps = float(d["fps"]) if "fps" in getattr(d, "files", []) else config.DEFAULT_FPS
        waves = method_waveforms(signals, fps)

        # label from PPG over this segment's span
        label_hr, ppg_wave, ppg_fs = None, np.array([]), 100.0
        if ppg_info is not None and span is not None:
            ppg, ppg_fs, _ = ppg_info
            ppg_wave = ppg_segment(ppg, ppg_fs, span["t_start"], span["t_end"])
            lbl, _ = spectral_hr(ppg_wave, ppg_fs, PPG_HR_LO, PPG_HR_HI)
            label_hr = lbl

        # optional trained-model predictions
        model_pred = model_predict(get_model(model_path), signals, fps) if model_path else None
        model_hr = model_pred[2] if model_pred is not None else ""
        hb_pred = hb_predict(get_hb_model(hb_model_path), signals, fps) if hb_model_path else None

        row = {
            "segment": seg_name,
            "clip": span["clip"] if span else "",
            "t_start": span["t_start"] if span else "",
            "t_end": span["t_end"] if span else "",
            "hr_pos": round(waves["POS"][1], 1) if waves["POS"][1] is not None else "",
            "conf_pos": round(waves["POS"][2], 1),
            "hr_chrom": round(waves["CHROM"][1], 1) if waves["CHROM"][1] is not None else "",
            "conf_chrom": round(waves["CHROM"][2], 1),
            "hr_green": round(waves["green"][1], 1) if waves["green"][1] is not None else "",
            "conf_green": round(waves["green"][2], 1),
            "hr_label": round(label_hr, 1) if label_hr is not None else "",
            "hr_model": round(model_hr, 1) if model_hr != "" else "",
            "hb_pred": round(hb_pred, 2) if hb_pred is not None else "",
        }

        if not no_plot:
            plots_dir = os.path.join(out_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            plot_segment(seg_name, signals, fps, waves, ppg_wave, ppg_fs, label_hr,
                         os.path.join(plots_dir, f"{seg_name}.png"), model_pred=model_pred,
                         hb_pred=hb_pred)
        return seg_name, row, None
    except Exception as exc:
        return seg_name, None, f"{type(exc).__name__}: {exc}"


def write_accuracy_plot(rows, out_dir):
    """Scatter of predicted vs PPG-label HR for POS/CHROM/green (and the model if present),
    y=x line, DSP points split by confidence. Annotates MAE/bias/within-6. A 4th panel is
    added when the hr_model column has values. Skipped if too few labeled segments."""
    def fnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    lab = np.array([fnum(r.get("hr_label")) for r in rows])
    if np.isfinite(lab).sum() < 5:
        print("  (accuracy plot skipped: fewer than 5 labeled segments)")
        return None

    # DSP methods (with confidence), plus the model if the column is populated
    panels = [("POS", "hr_pos", "conf_pos"), ("CHROM", "hr_chrom", "conf_chrom"),
              ("green", "hr_green", "conf_green")]
    model_hr = np.array([fnum(r.get("hr_model")) for r in rows])
    has_model = np.isfinite(model_hr).sum() >= 5
    if has_model:
        panels.append(("model", "hr_model", None))

    fin = lab[np.isfinite(lab)]
    lim = [max(30, np.floor(fin.min() / 10) * 10 - 5), np.ceil(fin.max() / 10) * 10 + 5]

    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(5.3 * n, 5.4))
    if n == 1:
        axs = [axs]
    for ax, (name, hc, cc) in zip(axs, panels):
        hr = np.array([fnum(r.get(hc)) for r in rows])
        ok = np.isfinite(hr) & np.isfinite(lab)
        if ok.sum() == 0:
            ax.set_title(f"{name}: no scorable segments"); ax.set_xlim(lim); ax.set_ylim(lim)
            continue
        if cc is not None:                       # DSP method: split by confidence
            cf = np.array([fnum(r.get(cc)) for r in rows])
            thr = np.nanmedian(cf[ok])
            hi = ok & (cf >= thr); lo = ok & (cf < thr)
            ax.scatter(lab[lo], hr[lo], s=10, facecolors="none", edgecolors="#bbbbbb",
                       linewidths=0.6, label=f"low conf (<{thr:.0f})")
            ax.scatter(lab[hi], hr[hi], s=12, alpha=0.5, color="#1f77b4",
                       label=f"high conf (\u2265{thr:.0f})")
        else:                                    # model: single colour (no confidence column)
            ax.scatter(lab[ok], hr[ok], s=12, alpha=0.5, color="purple", label="model")
        ax.plot(lim, lim, "k--", lw=1)
        ae = np.abs(hr - lab)
        mae, bias, w6 = np.nanmean(ae[ok]), np.nanmean((hr - lab)[ok]), np.mean(ae[ok] <= 6) * 100
        extra = ""
        if cc is not None:
            cf = np.array([fnum(r.get(cc)) for r in rows])
            hi = ok & (cf >= np.nanmedian(cf[ok]))
            if hi.any():
                extra = f"  |  hi-conf: MAE {np.nanmean(ae[hi]):.1f}, w6 {np.mean(ae[hi] <= 6)*100:.0f}%"
        ax.set_title(f"{name}\nMAE {mae:.1f}, bias {bias:+.1f}, w6 {w6:.0f}%{extra}", fontsize=10)
        ax.set_xlabel("PPG label HR (BPM)"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("predicted HR (BPM)")
    fig.suptitle(f"HR accuracy vs PPG label — {int(np.isfinite(lab).sum())} labeled segments"
                 + ("  (incl. trained model)" if has_model else ""), fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(out_dir, "hr_accuracy.png")
    plt.savefig(out_png, dpi=130)
    plt.close()
    return out_png


def write_condition_plot(rows, out_dir):
    """Grouped bar chart of MAE broken down by CONDITION (exercise state x camera) for
    each method (POS/CHROM/green, + model if present). Shows *where* each method fails:
    tall bars = high error. Also writes hr_by_condition.csv with MAE / within-6 / n per
    (method, state, camera) cell. The clip name encodes state ('before'/'after') and
    camera, so no extra inputs are needed."""
    def fnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    lab = np.array([fnum(r.get("hr_label")) for r in rows])
    if np.isfinite(lab).sum() < 5:
        return None

    cameras = list(getattr(config, "CAMERAS", ("FullHDwebcam", "IriunWebcam", "USBVideo")))

    def parse(clip):
        cam = next((c for c in cameras if c in clip), "other")
        state = "after" if "after" in clip else ("before" if "before" in clip else "?")
        return cam, state

    cams = [parse(r.get("clip", ""))[0] for r in rows]
    states = [parse(r.get("clip", ""))[1] for r in rows]

    methods = [("POS", "hr_pos"), ("CHROM", "hr_chrom"), ("green", "hr_green")]
    if np.isfinite(np.array([fnum(r.get("hr_model")) for r in rows])).sum() >= 5:
        methods.append(("model", "hr_model"))

    conds = [(st, cam) for st in ("before", "after") for cam in cameras]

    def stats(hrcol, st, cam):
        hr = np.array([fnum(r.get(hrcol)) for r in rows])
        sel = np.array([(states[i] == st and cams[i] == cam) for i in range(len(rows))])
        ok = sel & np.isfinite(hr) & np.isfinite(lab)
        if ok.sum() == 0:
            return np.nan, np.nan, 0
        ae = np.abs(hr - lab)[ok]
        return float(np.mean(ae)), float(np.mean(ae <= 6) * 100), int(ok.sum())

    table = {(mn, st, cam): stats(mc, st, cam) for mn, mc in methods for st, cam in conds}

    # ---- bar chart ----
    colors = {"POS": "#1f77b4", "CHROM": "#ff7f0e", "green": "#2ca02c", "model": "purple"}
    x = np.arange(len(conds))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(14, 6))
    for j, (mn, _) in enumerate(methods):
        vals = [table[(mn, st, cam)][0] for st, cam in conds]
        ax.bar(x + (j - (len(methods) - 1) / 2) * w, vals, w, label=mn, color=colors.get(mn))
    ax.axhline(6, ls="--", c="gray", lw=1, label="6 BPM (good)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{st}\n{cam.replace('webcam', '').replace('Video', '')}" for st, cam in conds], fontsize=9)
    ax.set_ylabel("MAE (BPM) — lower is better")
    ax.set_title("HR error by condition (exercise state \u00d7 camera) and method", fontweight="bold")
    ax.legend(ncol=len(methods) + 1, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ymax = ax.get_ylim()[1]
    for i, (st, cam) in enumerate(conds):
        ax.text(i, -0.06 * ymax, f"n={table[(methods[0][0], st, cam)][2]}", ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "hr_accuracy_by_condition.png")
    plt.savefig(out_png, dpi=130)
    plt.close()

    # ---- metrics CSV ----
    with open(os.path.join(out_dir, "hr_by_condition.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "state", "camera", "mae", "within6_pct", "n"])
        for mn, _ in methods:
            for st, cam in conds:
                mae, w6, n = table[(mn, st, cam)]
                wr.writerow([mn, st, cam, f"{mae:.2f}" if np.isfinite(mae) else "", f"{w6:.1f}" if np.isfinite(w6) else "", n])
    return out_png


def main():
    ap = argparse.ArgumentParser(description="HR (DSP + trained model) and Hb prediction over train_data segments.")
    ap.add_argument("--segments-dir", default="output/train_data")
    ap.add_argument("--ppg-dir", default="data/ppg")
    ap.add_argument("--out-dir", default="output/prediction")
    ap.add_argument("--manifest", default=None,
                    help="segments_manifest.csv (default: <segments-dir>/segments_manifest.csv)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--hr-model", default=None,
                    help="optional path to a trained HR model (hr_model.pt). If given, its "
                         "prediction fills the hr_model column and the model panel/spectrum.")
    ap.add_argument("--hb-model", default=None,
                    help="optional path to a trained Hb model (hb_model.pt). If given, its "
                         "hemoglobin prediction fills the hb_pred column and the plot title. "
                         "Hb is per-subject, so the value repeats across a subject's segments.")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seg_files = sorted(glob.glob(os.path.join(args.segments_dir, "*_signals.npz")))
    print(f"{len(seg_files)} segment file(s) in {args.segments_dir}")

    # manifest: segment -> span
    man_path = args.manifest or os.path.join(args.segments_dir, "segments_manifest.csv")
    spans = {}
    if os.path.exists(man_path):
        for r in csv.DictReader(open(man_path, encoding="utf-8-sig")):
            spans[r["segment"]] = {"clip": r["clip"], "t_start": float(r["t_start"]),
                                   "t_end": float(r["t_end"])}
        print(f"manifest: {len(spans)} segment spans from {man_path}")
    else:
        print(f"WARNING: no manifest at {man_path} — segments will be UNLABELLED "
              f"(re-run prepare_dataset to generate it).")

    # index PPG by (subject, state), loaded once
    ppg_cache = {}
    for fp in glob.glob(os.path.join(args.ppg_dir, "*.PW")) + glob.glob(os.path.join(args.ppg_dir, "*.pw")):
        m = re.search(r"(\d+)_(before|after)", os.path.basename(fp), re.IGNORECASE)
        if m:
            ppg_cache[(m.group(1), m.group(2).lower())] = load_pw(fp)
    print(f"{len(ppg_cache)} PPG file(s) indexed")

    tasks = []
    for sp_path in seg_files:
        seg_name = os.path.basename(sp_path).replace("_signals.npz", "")
        span = spans.get(seg_name)
        subj, state = parse_clip_state(span["clip"]) if span else parse_clip_state(seg_name)
        ppg_info = ppg_cache.get((subj, state)) if subj and state else None
        tasks.append((seg_name, sp_path, span, ppg_info, args.out_dir, args.no_plot, args.hr_model, args.hb_model))

    if not tasks:
        print("nothing to process.")
        return

    n_workers = args.workers if args.workers and args.workers > 0 else _default_workers()
    n_workers = max(1, min(n_workers, len(tasks)))
    rows = []
    if n_workers == 1:
        for i, task in enumerate(tasks, 1):
            nm, row, err = process_segment(task)
            print(f"[{i}/{len(tasks)}] {nm}: {'ok' if row else 'ERROR ' + err}")
            if row:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(process_segment, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                nm, row, err = fut.result()
                print(f"[{i}/{len(tasks)}] {nm}: {'ok' if row else 'ERROR ' + err}")
                if row:
                    rows.append(row)

    rows.sort(key=lambda r: r["segment"])
    csv_path = os.path.join(args.out_dir, "hr_results.csv")
    fields = ["segment", "clip", "t_start", "t_end", "hr_pos", "conf_pos", "hr_chrom",
              "conf_chrom", "hr_green", "conf_green", "hr_label", "hr_model", "hb_pred"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} segment(s) -> {csv_path}")

    # aggregate per-method standings vs the PPG label (raw FFT argmax, no best-of-three)
    def col(name):
        return np.array([float(r[name]) if r.get(name, "") not in ("", None) else np.nan for r in rows])
    lab = col("hr_label")
    if np.isfinite(lab).sum() >= 5:
        print("\n=== per-method accuracy vs PPG label (all labeled segments) ===")
        for name in ("hr_pos", "hr_chrom", "hr_green", "hr_model"):
            hr = col(name)
            ok = np.isfinite(hr) & np.isfinite(lab)
            if ok.sum() == 0:
                continue
            ae = np.abs(hr - lab)[ok]
            bias = np.mean((hr - lab)[ok])
            label = name.replace("hr_", "")
            print(f"  {label:6} MAE {np.mean(ae):6.2f}  w6 {np.mean(ae <= 6)*100:4.0f}%  bias {bias:+6.2f}  (n={ok.sum()})")

    if not args.no_plot:
        acc = write_accuracy_plot(rows, args.out_dir)
        if acc:
            print(f"accuracy plot -> {acc}")
        cond = write_condition_plot(rows, args.out_dir)
        if cond:
            print(f"by-condition plot -> {cond}")
            print(f"by-condition metrics -> {os.path.join(args.out_dir, 'hr_by_condition.csv')}")


if __name__ == "__main__":
    main()