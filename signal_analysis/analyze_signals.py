"""Extract per-region rPPG signals on a TRUE (PTS-based) time axis, and visualise.

This stage turns each clip's video + saved segmentation + landmarks into the
per-region mean skin-RGB time series that every downstream stage (Hb prediction, HR
prediction) consumes. It combines extraction and visualization in one pass because
both share the same region-selection logic (skin mask AND landmark polygon, eroded
inward), so what you *see* in the overlay is exactly what was *averaged* into the
signals.

TIMING — the important part.
Frame rate is NOT taken from the video header (which, for re-encoded files, can be
wrong). Instead each frame's real capture time is read from the container's
presentation timestamps (PTS) via ffprobe, and that PTS is trusted as-is:

  * the per-region signal is plotted against real seconds (PTS - PTS[0]), so the
    trace is undistorted even when the true frame rate is variable (e.g. network
    cameras) or mislabelled;
  * the per-frame timestamps are stored in the signals .npz (key "timestamps", in
    absolute seconds) so downstream stages resample / align on real time instead of
    assuming a constant fps. This is the ONLY timing stored: a scalar fps is
    redundant (recover it as (n-1)/(t[-1]-t[0]) when a single number is needed) and
    is computed on the fly only for the One-Euro landmark smoother and overlay writer.

Frames are paired with segmentation and landmarks strictly by index — frame i uses
masks[i] and landmarks[i] and PTS[i] — so the correspondence holds regardless of
frame rate. Counts are reconciled with n = min(frames, masks, landmarks, pts) so a
one-frame tail difference never misaligns anything.

If ffprobe is unavailable or a file carries no usable PTS, the stage falls back to a
uniform time axis from the header fps (logged as a warning), so it degrades
gracefully rather than failing.

Clips are processed in parallel, one per worker process (``--workers``, default =
all CPU cores); each clip is independent so results are identical to running
sequentially. OpenCV is pinned to one thread per worker to avoid oversubscription.

Usage:
  python analyze_signals.py --video-dir data/videos --seg-dir output/seg \
      --landmarks-dir output/landmarks --signals-dir output/signals \
      --plots-dir output/plots [--workers N] [--no-video] [--no-plot]
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

from common import config
from common import signal_processing as sp
import matplotlib.pyplot as plt

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
_PLOT_COLORS = {"forehead": "olive", "lcheek": "magenta", "rcheek": "green"}


def load_fps_sidecar(path: str | None) -> dict[str, float]:
    """clip -> true_fps from an mcd_fps.csv sidecar (empty dict if no path/file)."""
    import csv
    out: dict[str, float] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            clip = row.get("clip")
            try:
                out[clip] = float(row.get("true_fps", ""))
            except (TypeError, ValueError):
                pass
    return out


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
    """Header fps — used ONLY as a last-resort fallback when PTS is unavailable."""
    if video_path is None:
        return config.DEFAULT_FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    cap.release()
    return fps


def read_pts_seconds(video_path: str) -> np.ndarray | None:
    """Per-frame presentation timestamps (seconds) via ffprobe, trusted as absolute.

    Tries the modern field first, then older aliases. Returns None if ffprobe is
    missing or no field yields usable (mostly-finite) timestamps, so the caller can
    fall back to header fps.
    """
    for field in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", f"frame={field}", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=1200,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            continue
        vals = []
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if ln in ("", "N/A"):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(ln))
                except ValueError:
                    vals.append(np.nan)
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size and np.isfinite(arr).mean() > 0.5:
            return arr
    return None


def _clean_pts(pts: np.ndarray) -> np.ndarray:
    """Fill occasional NaN PTS by index interpolation; enforce a non-decreasing axis.

    (For typical rPPG footage decode order == presentation order, so the sort is a
    no-op; it only guards pathological B-frame reordering so the time axis stays
    monotonic.)
    """
    p = np.asarray(pts, dtype=np.float64).copy()
    if not np.isfinite(p).all():
        idx = np.arange(len(p))
        good = np.isfinite(p)
        if good.sum() >= 2:
            p[~good] = np.interp(idx[~good], idx[good], p[good])
        else:
            p = np.arange(len(p), dtype=np.float64)  # degenerate; will be rescaled by caller
    if np.any(np.diff(p) < 0):
        p = np.sort(p)
    return p


def fps_from_timestamps(pts_seconds) -> float | None:
    """Robust average fps from per-frame PTS: (n-1) / (last - first).

    This is the trustworthy rate — the number of frames over the real elapsed time —
    NOT the video header (which for some containers reports a timebase like 30000).
    Returns None if the timestamps are missing, degenerate (zero span), or the implied
    rate is physically implausible, so the caller can warn instead of using garbage.
    """
    if pts_seconds is None:
        return None
    p = np.asarray(pts_seconds, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) < 2:
        return None
    p = np.sort(p)
    span = float(p[-1] - p[0])
    if span <= 0:
        return None
    fps = (len(p) - 1) / span
    return fps if 1.0 <= fps <= 1000.0 else None


def _sane_header_fps(raw: float) -> float | None:
    """The header fps only if it's in a plausible range (rejects the 30000 quirk)."""
    return float(raw) if raw and 1.0 <= raw <= 1000.0 else None


# ======================================================================
# Signal extraction  (video + seg + landmarks -> per-region RGB signals)
# ======================================================================

def extract_one(video_path, seg, landmarks, override_fps=None):
    """Return (signals (3, T, 3), timestamps (T,), nominal_fps, timing_source).

    If override_fps is given (MCD sidecar), timing is a uniform grid at that rate and
    the video PTS is ignored — used for re-encoded clips whose PTS is untrustworthy.
    Otherwise the real per-frame PTS is read from the video and trusted.
    """
    pts = None if override_fps is not None else read_pts_seconds(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    header_fps = _sane_header_fps(cap.get(cv2.CAP_PROP_FPS))   # None if absurd (e.g. 30000)

    n = min(len(seg), len(landmarks))
    if pts is not None:
        n = min(n, len(pts))

    if override_fps is not None:
        p = None
        nominal_fps = float(override_fps)
        timing_source = "sidecar"
    else:
        pts_fps = fps_from_timestamps(pts[:n]) if (pts is not None and n > 0) else None
        if pts_fps is not None:
            p = _clean_pts(pts[:n])                     # real per-frame times (kept as-is)
            nominal_fps = pts_fps                       # trustworthy average rate
            timing_source = "pts"
        else:
            # No usable PTS. Do NOT trust an absurd header; fall back to a sane header
            # value if there is one, else DEFAULT_FPS — and flag it loudly downstream.
            p = None
            nominal_fps = header_fps if header_fps is not None else config.DEFAULT_FPS
            timing_source = "fps_fallback"

    smoothed = sp.smooth_landmarks(landmarks[:n, :, :2], nominal_fps)
    ek = sp.edge_kernel()
    signals = np.full((len(config.REGION_ORDER), n, 3), np.nan, dtype=np.float64)

    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        pt = smoothed[idx]
        if np.isfinite(pt).all():
            skin = sp.skin_mask_from_seg(seg[idx])
            hw = frame.shape[:2]
            for r, name in enumerate(config.REGION_ORDER):
                region = sp.build_region_mask(hw, pt[config.REGIONS[name]], skin, ek)
                signals[r, idx, :] = sp.region_mean_rgb(frame, region)
        idx += 1
    cap.release()

    signals = signals[:, :idx, :]
    if timing_source == "pts":
        timestamps = p[:idx]
    else:
        timestamps = np.arange(idx, dtype=np.float64) / (nominal_fps or config.DEFAULT_FPS)
    return signals, timestamps, float(nominal_fps), timing_source


# ======================================================================
# Visualization  (plots + overlay video)
# ======================================================================

def _relative_time_axis(timestamps, T):
    """Real seconds for the x-axis; falls back to a DEFAULT_FPS grid if needed."""
    if timestamps is not None and len(timestamps) >= T and T > 0:
        t = np.asarray(timestamps[:T], dtype=np.float64)
        return t - t[0], "Time (s, from PTS)"
    return np.arange(T) / config.DEFAULT_FPS, "Time (s, assumed fps)"


def plot_signals(signals, timestamps, name, out_png) -> None:
    _, T, _ = signals.shape
    t, xlabel = _relative_time_axis(timestamps, T)
    dur = t[-1] if len(t) else 0.0
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
    axs[-1].set_xlabel(xlabel)
    fig.suptitle(f"Per-region skin RGB — {name}  ({dur:.1f}s, {T} frames)", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_png, dpi=130)
    plt.close()


def make_overlay(video_path, seg, landmarks, out_mp4, fps) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = min(len(seg), len(landmarks))
    smoothed = sp.smooth_landmarks(landmarks[:n, :, :2], fps or config.DEFAULT_FPS)
    ek = sp.edge_kernel()
    factor = 1 << config.SUBPIX_SHIFT

    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps or config.DEFAULT_FPS, (width, height))
    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()

        skin = sp.skin_mask_from_seg(seg[idx])
        tint = np.zeros_like(vis)
        tint[skin > 0] = (0, 180, 0)
        vis = cv2.addWeighted(vis, 1.0, tint, 0.25, 0)

        pts = smoothed[idx]
        if np.isfinite(pts).all():
            for rname in config.REGION_ORDER:
                color = config.REGION_COLORS[rname]
                poly_pts = pts[config.REGIONS[rname]]
                region = sp.build_region_mask((height, width), poly_pts, skin, ek)
                fill = np.zeros_like(vis)
                fill[region > 0] = color
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
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _default_workers() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def process_clip(task: tuple) -> tuple[str, str, bool]:
    (name, seg_path, lmk_path, video_path, sig_path, out_png, out_mp4,
     no_plot, no_video, overwrite, override_fps) = task
    log: list[str] = []
    try:
        need_extract = overwrite or not os.path.exists(sig_path)
        need_plot = (not no_plot) and (overwrite or not os.path.exists(out_png))
        need_overlay = (not no_video) and (overwrite or not os.path.exists(out_mp4))

        seg = landmarks = None
        if need_extract or need_overlay:
            seg = np.load(seg_path)["masks"]
            landmarks = np.load(lmk_path)["landmarks"]

        # --- 1. signals (+ per-frame timestamps) ---
        if need_extract:
            signals, timestamps, fps, source = extract_one(video_path, seg, landmarks, override_fps)
            np.savez_compressed(sig_path, signals=signals, timestamps=timestamps)
            valid = float(np.mean(np.isfinite(signals[:, :, 1]))) if signals.size else 0.0
            dur = (timestamps[-1] - timestamps[0]) if len(timestamps) else 0.0
            log.append(f"{name}: {signals.shape}  timing={source}  {fps:.2f} fps  "
                       f"({len(timestamps)} frames over {dur:.1f}s)  "
                       f"{100 * valid:.0f}% region-frames valid -> {sig_path}")
            if source == "fps_fallback":
                log.append(f"    WARNING: no usable PTS for this clip; timing fell back to "
                           f"{fps:.2f} fps (uncertain). Check that ffprobe is installed and the "
                           f"clip has valid timestamps, or supply --fps-sidecar for it.")
        else:
            d = np.load(sig_path)
            signals = d["signals"]
            timestamps = d["timestamps"] if "timestamps" in getattr(d, "files", []) else None
            # nominal fps (only for the overlay writer) recovered from the timestamps
            if timestamps is not None and len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
                fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
            else:
                fps = video_fps(video_path)
            log.append(f"{name}: signals exist, reuse {signals.shape}")

        # --- 2. per-region signal plot (real time axis) ---
        if not no_plot:
            if need_plot:
                plot_signals(signals, timestamps, name, out_png)
                log.append(f"    plot -> {out_png}")
            else:
                log.append("    plot exists, skip")

        # --- 3. overlay video ---
        if not no_video:
            if need_overlay:
                nframes = make_overlay(video_path, seg, landmarks, out_mp4, fps)
                log.append(f"    overlay ({nframes} frames) -> {out_mp4}")
            else:
                log.append("    overlay exists, skip")

        return name, "\n".join(log), True
    except Exception as exc:
        return name, f"{name}: ERROR {type(exc).__name__}: {exc}", False


# ======================================================================
# Driver
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract per-region rPPG signals (PTS-timed) and optionally plot + overlay."
    )
    ap.add_argument("--video-dir", default="data/videos")
    ap.add_argument("--seg-dir", default="output/seg")
    ap.add_argument("--landmarks-dir", default="output/landmarks")
    ap.add_argument("--signals-dir", default="output/signals")
    ap.add_argument("--plots-dir", default="output/plots")
    ap.add_argument("--no-plot", action="store_true", help="skip the per-region signal PNG")
    ap.add_argument("--no-video", action="store_true", help="skip the overlay video")
    ap.add_argument("--overwrite", action="store_true", help="redo clips whose outputs exist")
    ap.add_argument("--fps-sidecar", default=None,
                    help="optional mcd_fps.csv (clip,true_fps). For listed clips, timing is a "
                         "uniform grid at that fps and video PTS is ignored — used to correct "
                         "re-encoded datasets (e.g. MCD). Clips not listed still use their PTS.")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes (0 = all available CPU cores).")
    args = ap.parse_args()

    os.makedirs(args.signals_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    sidecar = load_fps_sidecar(args.fps_sidecar)
    if sidecar:
        print(f"fps sidecar: {len(sidecar)} clip(s) will use corrected uniform timing")

    seg_files = sorted(f for f in os.listdir(args.seg_dir) if f.endswith("_seg.npz"))
    print(f"{len(seg_files)} segmentation file(s) in {args.seg_dir}")

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
                      args.no_plot, args.no_video, args.overwrite, sidecar.get(name)))

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