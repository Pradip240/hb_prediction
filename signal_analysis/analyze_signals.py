"""Extract per-region rPPG signals and (optionally) visualise them in one pass.

This stage combines what used to be two separate stages — signal extraction and
visualization — because both share the SAME region-selection logic (skin mask AND
landmark polygon, eroded inward) and both read the same inputs (video + saved
segmentation + landmarks). Doing them together guarantees that what you *see* in
the overlay is exactly what was *averaged* into the signals.

Clips are processed in parallel, one per worker process (``--workers``, default =
all available CPU cores). Clip processing is embarrassingly parallel — each clip
reads its own inputs and writes its own outputs — so the per-clip result is
identical to running sequentially; only the outer loop is parallelised. OpenCV is
pinned to one thread per worker so N processes don't oversubscribe the cores.

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
      --plots-dir output/plots [--workers N]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

import config
import signal_processing as sp
import matplotlib.pyplot as plt

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
# Parallel worker
# ======================================================================

def _init_worker() -> None:
    """Runs once per worker process: pin OpenCV to a single thread.

    Each clip already runs in its own process, so letting OpenCV also spin up
    threads per process would oversubscribe the cores and slow everything down.
    """
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _default_workers() -> int:
    """Number of usable CPUs, respecting cgroup/taskset limits where possible."""
    try:
        return len(os.sched_getaffinity(0))  # Linux: honours --cpuset / taskset
    except AttributeError:
        return os.cpu_count() or 1


def process_clip(task: tuple) -> tuple[str, str, bool]:
    """Process one clip end-to-end. Runs in a worker process; returns (name, log, ok).

    Loads the segmentation/landmarks only when they're actually needed (extraction
    or overlay), so re-plotting a finished clip doesn't pull a large mask array
    into memory — important when many workers run at once.
    """
    (name, seg_path, lmk_path, video_path, sig_path, out_png, out_mp4,
     no_plot, no_video, overwrite) = task
    log: list[str] = []
    try:
        need_extract = overwrite or not os.path.exists(sig_path)
        need_plot = (not no_plot) and (overwrite or not os.path.exists(out_png))
        need_overlay = (not no_video) and (overwrite or not os.path.exists(out_mp4))

        seg = landmarks = None
        if need_extract or need_overlay:
            seg = np.load(seg_path)["masks"]
            landmarks = np.load(lmk_path)["landmarks"]

        # --- 1. signals (reuse existing if present and not overwriting) ---
        if need_extract:
            signals, fps = extract_one(video_path, seg, landmarks)
            np.savez_compressed(sig_path, signals=signals)
            valid = float(np.mean(np.isfinite(signals[:, :, 1]))) if signals.size else 0.0
            log.append(f"{name}: {signals.shape} @ {fps:.1f} fps, "
                       f"{100 * valid:.0f}% region-frames valid -> {sig_path}")
        else:
            signals = np.load(sig_path)["signals"]
            fps = video_fps(video_path)
            log.append(f"{name}: signals exist, reuse {signals.shape}")

        # --- 2. per-region signal plot ---
        if not no_plot:
            if need_plot:
                plot_signals(signals, fps, name, out_png)
                log.append(f"    plot -> {out_png}")
            else:
                log.append("    plot exists, skip")

        # --- 3. overlay video ---
        if not no_video:
            if need_overlay:
                nframes = make_overlay(video_path, seg, landmarks, out_mp4)
                log.append(f"    overlay ({nframes} frames) -> {out_mp4}")
            else:
                log.append("    overlay exists, skip")

        return name, "\n".join(log), True
    except Exception as exc:  # keep one bad clip from killing the whole batch
        return name, f"{name}: ERROR {type(exc).__name__}: {exc}", False


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
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes (0 = all available CPU cores). "
                         "Each worker holds one clip's mask array in memory, so lower "
                         "this if you hit memory pressure on long/high-res clips.")
    args = ap.parse_args()

    os.makedirs(args.signals_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    seg_files = sorted(f for f in os.listdir(args.seg_dir) if f.endswith("_seg.npz"))
    print(f"{len(seg_files)} segmentation file(s) in {args.seg_dir}")

    # Build the task list; cheap precondition checks (and their skip messages) happen
    # here in the parent so workers only ever get real work.
    tasks: list[tuple] = []
    for seg_file in seg_files:
        name = seg_file[: -len("_seg.npz")]
        sig_path = os.path.join(args.signals_dir, f"{name}_signals.npz")
        out_png = os.path.join(args.plots_dir, f"{name}_signal.png")
        out_mp4 = os.path.join(args.plots_dir, f"{name}_overlay.mp4")
        lmk_path = os.path.join(args.landmarks_dir, f"{name}_landmarks.npz")
        video_path = find_video(args.video_dir, name)

        if not os.path.exists(lmk_path):
            print(f"{name}: no landmarks .npz, skip")
            continue
        if video_path is None:
            print(f"{name}: no video, skip")
            continue

        seg_path = os.path.join(args.seg_dir, seg_file)
        tasks.append((name, seg_path, lmk_path, video_path, sig_path, out_png, out_mp4,
                      args.no_plot, args.no_video, args.overwrite))

    if not tasks:
        print("nothing to process.")
        return

    n_workers = args.workers if args.workers and args.workers > 0 else _default_workers()
    n_workers = max(1, min(n_workers, len(tasks)))
    total = len(tasks)
    print(f"processing {total} clip(s) on {n_workers} worker(s)")

    if n_workers == 1:
        _init_worker()
        for i, task in enumerate(tasks, 1):
            _, msg, _ok = process_clip(task)
            print(f"[{i}/{total}] {msg}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
            futures = {pool.submit(process_clip, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                _, msg, _ok = fut.result()
                print(f"[{i}/{total}] {msg}")


if __name__ == "__main__":
    main()