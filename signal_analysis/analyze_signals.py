"""Extract per-region rPPG signals and (optionally) visualise them in one pass.

This stage combines what used to be two separate stages — signal extraction and
visualization — because both share the SAME region-selection logic (skin mask AND
landmark polygon, eroded inward) and both read the same inputs (video + saved
segmentation + landmarks). Doing them together guarantees that what you *see* in
the overlay is exactly what was *averaged* into the signals.

For each clip it:
  1. reads the video frames (for colour), the ``<clip>_seg.npz`` (skin mask, key
     "masks") and the ``<clip>_landmarks.npz`` (ROI polygons, key "landmarks"),
  2. applies the original tracker's region-selection logic to produce
     ``<clip>_signals.npz`` (key "signals") of shape (3, T, 3) = (region, frame,
     RGB), regions in the order forehead, lcheek, rcheek. Frames with no face —
     or where a region has fewer than the minimum skin pixels — are NaN,
  3. unless ``--no-plot`` is given, writes ``<clip>_signal.png`` (per-region
     R/G/B traces), and unless ``--no-video`` is given, writes
     ``<clip>_overlay.mp4``.

Usage:
  python analyze_signals.py --video-dir data/videos --seg-dir output/seg \
      --landmarks-dir output/landmarks --signals-dir output/signals \
      --plots-dir output/plots
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib

from signal_analysis import signal_processing as sp
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_analysis import config  # noqa: E402

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
_PLOT_COLORS = {"forehead": "olive", "lcheek": "magenta", "rcheek": "green"}


def find_video(video_dir: str, name: str) -> str | None:
    for ext in VIDEO_EXTS:
        for cand in (name + ext, name + ext.upper()):
            p = os.path.join(video_dir, cand)
            if os.path.exists(p):
                return p
    for f in os.listdir(video_dir):
        if os.path.splitext(f)[0] == name:
            return os.path.join(video_dir, f)
    return None


def video_fps(video_path: str | None) -> float:
    if video_path is None:
        return config.DEFAULT_FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    cap.release()
    return fps


# ======================================================================
# Signal extraction  (video + seg + landmarks -> per-region RGB signals)
# ======================================================================

def extract_one(video_path, seg, landmarks):
    """Return (signals (3, T, 3), fps) for one clip."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS

    n = min(len(seg), len(landmarks))
    smoothed = sp.smooth_landmarks(landmarks[:n, :, :2], fps)
    ek = sp.edge_kernel()
    signals = np.full((len(config.REGION_ORDER), n, 3), np.nan, dtype=np.float64)

    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        pts = smoothed[idx]
        if np.isfinite(pts).all():
            skin = sp.skin_mask_from_seg(seg[idx])
            hw = frame.shape[:2]
            for r, name in enumerate(config.REGION_ORDER):
                region = sp.build_region_mask(hw, pts[config.REGIONS[name]], skin, ek)
                signals[r, idx, :] = sp.region_mean_rgb(frame, region)
        idx += 1
    cap.release()
    return signals[:, :idx, :], fps


# ======================================================================
# Visualization  (plots + overlay video)
# ======================================================================

def plot_signals(signals, fps, name, out_png) -> None:
    _, T, _ = signals.shape
    t = np.arange(T) / (fps or config.DEFAULT_FPS)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    channels = ["Red", "Green", "Blue"]
    for c in range(3):
        ax = axs[c]
        for r, rname in enumerate(config.REGION_ORDER):
            ax.plot(t, signals[r, :, c], color=_PLOT_COLORS[rname], linewidth=1.0, label=rname)
        ax.set_ylabel(f"Mean {channels[c]}")
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Per-region skin RGB — {name}", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_png, dpi=130)
    plt.close()


def make_overlay(video_path, seg, landmarks, out_mp4) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = min(len(seg), len(landmarks))
    smoothed = sp.smooth_landmarks(landmarks[:n, :, :2], fps)
    ek = sp.edge_kernel()
    factor = 1 << config.SUBPIX_SHIFT

    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()

        skin = sp.skin_mask_from_seg(seg[idx])
        tint = np.zeros_like(vis)
        tint[skin > 0] = (0, 180, 0)               # green skin tint
        vis = cv2.addWeighted(vis, 1.0, tint, 0.25, 0)

        pts = smoothed[idx]
        if np.isfinite(pts).all():
            for rname in config.REGION_ORDER:
                color = config.REGION_COLORS[rname]
                poly_pts = pts[config.REGIONS[rname]]
                region = sp.build_region_mask((height, width), poly_pts, skin, ek)
                fill = np.zeros_like(vis)
                fill[region > 0] = color            # the exact averaged region
                vis = cv2.addWeighted(vis, 1.0, fill, 0.35, 0)
                hull = cv2.convexHull(poly_pts.astype(np.float32))
                hpts = np.round(hull * factor).astype(np.int32)
                cv2.polylines(vis, [hpts], True, color, 2, lineType=cv2.LINE_AA, shift=config.SUBPIX_SHIFT)
        else:
            cv2.putText(vis, "NO FACE", (15, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(vis)
        idx += 1
    cap.release()
    writer.release()
    return idx


# ======================================================================
# Driver
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract per-region rPPG signals and optionally plot + render an overlay."
    )
    ap.add_argument("--video-dir", default="data/videos")
    ap.add_argument("--seg-dir", default="output/seg")
    ap.add_argument("--landmarks-dir", default="output/landmarks")
    ap.add_argument("--signals-dir", default="output/signals")
    ap.add_argument("--plots-dir", default="output/plots")
    ap.add_argument("--no-plot", action="store_true", help="skip the per-region signal PNG")
    ap.add_argument("--no-video", action="store_true", help="skip the overlay video")
    ap.add_argument("--overwrite", action="store_true", help="redo clips whose outputs exist")
    args = ap.parse_args()

    os.makedirs(args.signals_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    seg_files = sorted(f for f in os.listdir(args.seg_dir) if f.endswith("_seg.npz"))
    print(f"{len(seg_files)} segmentation file(s) in {args.seg_dir}")

    for i, seg_file in enumerate(seg_files, 1):
        name = seg_file[: -len("_seg.npz")]
        sig_path = os.path.join(args.signals_dir, f"{name}_signals.npz")
        out_png = os.path.join(args.plots_dir, f"{name}_signal.png")
        out_mp4 = os.path.join(args.plots_dir, f"{name}_overlay.mp4")

        lmk_path = os.path.join(args.landmarks_dir, f"{name}_landmarks.npz")
        video_path = find_video(args.video_dir, name)
        if not os.path.exists(lmk_path):
            print(f"[{i}/{len(seg_files)}] {name}: no landmarks .npz, skip")
            continue
        if video_path is None:
            print(f"[{i}/{len(seg_files)}] {name}: no video, skip")
            continue

        seg = np.load(os.path.join(args.seg_dir, seg_file))["masks"]
        landmarks = np.load(lmk_path)["landmarks"]

        # --- 1. signals (reuse existing if present and not overwriting) ---
        if os.path.exists(sig_path) and not args.overwrite:
            signals = np.load(sig_path)["signals"]
            fps = video_fps(video_path)
            print(f"[{i}/{len(seg_files)}] {name}: signals exist, reuse {signals.shape}")
        else:
            signals, fps = extract_one(video_path, seg, landmarks)
            np.savez_compressed(sig_path, signals=signals)
            valid = float(np.mean(np.isfinite(signals[:, :, 1]))) if signals.size else 0.0
            print(f"[{i}/{len(seg_files)}] {name}: {signals.shape} @ {fps:.1f} fps, "
                  f"{100 * valid:.0f}% region-frames valid -> {sig_path}")

        # --- 2. per-region signal plot ---
        if not args.no_plot:
            if os.path.exists(out_png) and not args.overwrite:
                print("    plot exists, skip")
            else:
                plot_signals(signals, fps, name, out_png)
                print(f"    plot -> {out_png}")

        # --- 3. overlay video ---
        if not args.no_video:
            if os.path.exists(out_mp4) and not args.overwrite:
                print("    overlay exists, skip")
            else:
                nframes = make_overlay(video_path, seg, landmarks, out_mp4)
                print(f"    overlay ({nframes} frames) -> {out_mp4}")


if __name__ == "__main__":
    main()