"""prepare_dataset.py — clean, uniform 20 s pulse windows for HR/HB model training.

Turns each extracted <clip>_signals.npz (per-region rPPG signal + real per-frame
timestamps) into a dataset of fixed-length, evenly-sampled 20 s windows that a model
can consume directly. Two problems are solved here:

1. BROKEN REGIONS. A frame is "broken" if the region signal is NaN (face lost) or if
   the timestamps show a gap (missing frames). Both are measured in REAL SECONDS from
   the stored timestamps. A candidate window is accepted only if no single broken
   stretch exceeds MAX_GAP_SEC and the total broken time is at most
   MAX_TOTAL_BROKEN_SEC (both tunable in config.py). Windows are placed greedily:
   slide forward, emit a clean non-overlapping window where one fits, and jump past
   broken regions — packing in as many clean 20 s spans as the clip allows.

2. IRREGULAR SAMPLING. Webcam/RTSP frames arrive at uneven real times (jitter +
   dropped frames). Each accepted window is RESAMPLED onto a uniform TARGET_FPS grid
   using PCHIP (monotonic cubic) interpolation ANCHORED TO REAL TIME. Because the
   interpolation uses the true timestamps, the pulse's frequency is preserved exactly
   — the signal is neither stretched nor compressed (a peak at t=3.47 s stays at
   3.47 s). NaN samples are dropped before interpolation, so NaN frames and
   missing-frame gaps are filled by the same operation. Short gaps (< MAX_GAP_SEC)
   are bridged; long gaps never occur here because such windows are rejected first.

Output per window: <clip>_<k>_signals.npz containing the resampled signal
(R, WIN_LEN, 3) float32 and the scalar sample rate `fps`. No timestamps are stored —
after uniform resampling they are implied by `fps` (the absolute per-frame times
remain available in the source signals.npz if ever needed).

Usage:
  python prepare_dataset.py --signals-dir output/signals --out-dir output/windows
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import PchipInterpolator

from common import config

# Windowing / resampling defaults (overridable via config.py if defined there).
WINDOW_SEC = getattr(config, "WINDOW_SEC", 20.0)
TARGET_FPS = getattr(config, "TARGET_FPS", 30.0)
WIN_LEN = int(round(WINDOW_SEC * TARGET_FPS))                 # 600
STEP_SEC = getattr(config, "WINDOW_STEP_SEC", WINDOW_SEC)     # non-overlapping by default
MAX_GAP_SEC = getattr(config, "MAX_GAP_SEC", 1.0)            # no single broken stretch longer than this
MAX_TOTAL_BROKEN_SEC = getattr(config, "MAX_TOTAL_BROKEN_SEC", 5.0)
# a real inter-frame interval longer than this counts as a "missing-frame gap"
GAP_FACTOR = getattr(config, "GAP_FACTOR", 3.0)              # gap if dt > GAP_FACTOR * median dt


def _broken_runs_seconds(t_win, valid_win, median_dt):
    """Return a list of broken-run durations (seconds) within a window.

    A broken run is a maximal stretch that is either NaN (valid=False) or separated
    from the previous valid sample by a timestamp gap > GAP_FACTOR*median_dt. Both are
    accumulated in real time so the >1 s / >5 s thresholds are physical, not per-frame.
    """
    runs = []
    cur = 0.0
    n = len(t_win)
    for i in range(n):
        gap = (t_win[i] - t_win[i - 1]) if i > 0 else 0.0
        missing = gap > GAP_FACTOR * median_dt
        if not valid_win[i]:
            # NaN sample: its own share of time is ~median_dt (or the real gap to prev)
            cur += max(gap, median_dt) if i > 0 else median_dt
        elif missing:
            # valid sample but a gap precedes it: the gap itself is broken time
            cur += gap
            if cur > 0:
                runs.append(cur)
                cur = 0.0
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0.0
    if cur > 0:
        runs.append(cur)
    return runs


def _resample_window(t_valid, sig_valid, t0):
    """PCHIP-resample valid (time, value) samples onto a uniform TARGET_FPS grid.

    Grid spans [t0, t0+WINDOW_SEC) at 1/TARGET_FPS spacing (WIN_LEN points). Returns
    (WIN_LEN, C) or None if the valid samples don't fully cover the grid (would require
    extrapolation). Duplicate/non-increasing timestamps are collapsed first (network
    cameras can stamp two frames at the same wall-clock time), since PCHIP requires a
    strictly increasing x.
    """
    # collapse exact-duplicate timestamps: average the values at each unique time
    t_valid = np.asarray(t_valid, dtype=np.float64)
    if np.any(np.diff(t_valid) <= 0):
        uniq, inv = np.unique(t_valid, return_inverse=True)
        collapsed = np.empty((len(uniq), sig_valid.shape[1]), dtype=np.float64)
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        for c in range(sig_valid.shape[1]):
            collapsed[:, c] = np.bincount(inv, weights=sig_valid[:, c], minlength=len(uniq)) / counts
        t_valid, sig_valid = uniq, collapsed
    if len(t_valid) < 2:
        return None

    grid = t0 + np.arange(WIN_LEN) / TARGET_FPS
    # never extrapolate: valid data must bracket the whole grid
    if t_valid[0] > grid[0] + 1e-9 or t_valid[-1] < grid[-1] - 1e-9:
        return None
    out = np.empty((WIN_LEN, sig_valid.shape[1]), dtype=np.float64)
    for c in range(sig_valid.shape[1]):
        out[:, c] = PchipInterpolator(t_valid, sig_valid[:, c])(grid)
    return out


def process_clip(task):
    name, sig_path, out_dir, overwrite = task
    log = []
    spans = []
    try:
        d = np.load(sig_path)
        signals = d["signals"]                       # (R, T, 3)
        if "timestamps" not in getattr(d, "files", []):
            return name, f"{name}: no timestamps in npz, skip", 0, spans
        ts = np.asarray(d["timestamps"], dtype=np.float64)   # absolute seconds
        R, T, C = signals.shape
        if T < 4 or len(ts) < T:
            return name, f"{name}: too short / timestamp mismatch, skip", 0, spans

        # per-region skin-pixel counts (R, T), if present — carried through for size-aware
        # weighting downstream. Absent in older signals.npz; then no counts are stored.
        has_counts = "pixel_counts" in getattr(d, "files", [])
        pix = np.asarray(d["pixel_counts"], dtype=np.float64) if has_counts else None

        ts0 = float(ts[0])                            # absolute start of the clip
        # relative time base and per-frame validity (valid = all regions finite)
        t = ts[:T] - ts0
        valid = np.all(np.isfinite(signals[:, :, 1]), axis=0)   # face present on all regions
        dts = np.diff(t)
        median_dt = float(np.median(dts)) if len(dts) else 1.0 / TARGET_FPS
        total = t[-1] if T else 0.0

        # greedy non-overlapping placement across real time
        emitted = 0
        t0 = 0.0
        while t0 + WINDOW_SEC <= total + 1e-9:
            lo = np.searchsorted(t, t0, side="left")
            hi = np.searchsorted(t, t0 + WINDOW_SEC, side="right")
            t_win = t[lo:hi]
            valid_win = valid[lo:hi]
            if len(t_win) < 4:
                t0 += STEP_SEC
                continue

            runs = _broken_runs_seconds(t_win, valid_win, median_dt)
            worst = max(runs) if runs else 0.0
            broken_total = sum(runs)

            if worst > MAX_GAP_SEC or broken_total > MAX_TOTAL_BROKEN_SEC:
                # window unusable — jump past the FIRST offending gap, else step on
                advanced = False
                acc = 0.0
                for i in range(1, len(t_win)):
                    gap = t_win[i] - t_win[i - 1]
                    if gap > GAP_FACTOR * median_dt and gap > MAX_GAP_SEC:
                        t0 = t_win[i]           # restart just after the big gap
                        advanced = True
                        break
                if not advanced:
                    t0 += STEP_SEC
                continue

            # accepted: drop NaN samples, PCHIP-resample onto the uniform grid
            vmask = valid_win
            tv, sv = t_win[vmask], signals[:, lo:hi, :][:, vmask, :]
            # sv is (R, n_valid, C); resample each region
            resampled = np.empty((R, WIN_LEN, C), dtype=np.float32)
            ok = True
            for r in range(R):
                out = _resample_window(tv, sv[r], t0)
                if out is None:
                    ok = False
                    break
                resampled[r] = out.astype(np.float32)
            if not ok:
                t0 += STEP_SEC
                continue

            # resample per-region pixel counts onto the SAME grid (linear; counts are
            # non-negative and need no overshoot). Smooth pose ramps stay smooth ramps.
            counts_rs = None
            if pix is not None:
                grid = t0 + np.arange(WIN_LEN) / TARGET_FPS
                counts_rs = np.empty((R, WIN_LEN), dtype=np.float32)
                pv = pix[:, lo:hi][:, vmask]                 # (R, n_valid) counts at valid times
                for r in range(R):
                    counts_rs[r] = np.interp(grid, tv, pv[r]).astype(np.float32)

            emitted += 1
            seg_name = f"{name}_{emitted}"
            out_path = os.path.join(out_dir, f"{seg_name}_signals.npz")
            if overwrite or not os.path.exists(out_path):
                if counts_rs is not None:
                    np.savez_compressed(out_path, signals=resampled, fps=np.float32(TARGET_FPS),
                                        pixel_counts=counts_rs)
                else:
                    np.savez_compressed(out_path, signals=resampled, fps=np.float32(TARGET_FPS))
            # record this segment's real time span (relative to clip start) + absolute
            # start, so downstream can label it from the PPG over the exact same span.
            spans.append(dict(segment=seg_name, clip=name, index=emitted,
                              t_start=round(float(t0), 3), t_end=round(float(t0 + WINDOW_SEC), 3),
                              abs_start=round(float(ts0 + t0), 6)))
            t0 += WINDOW_SEC        # non-overlapping (STEP_SEC == WINDOW_SEC by default)

        log.append(f"{name}: {emitted} window(s) from {total:.1f}s "
                   f"({100*np.mean(valid):.0f}% frames valid, median {1/median_dt:.1f} fps)")
        return name, "\n".join(log), emitted, spans
    except Exception as exc:
        return name, f"{name}: ERROR {type(exc).__name__}: {exc}", 0, spans


def _default_workers():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser(description="Build clean uniform 20 s windows from extracted signals.")
    ap.add_argument("--signals-dir", default="output/signals")
    ap.add_argument("--out-dir", default="output/windows")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sig_files = sorted(f for f in os.listdir(args.signals_dir) if f.endswith("_signals.npz"))
    print(f"{len(sig_files)} signal file(s) in {args.signals_dir}")
    print(f"window {WINDOW_SEC:.0f}s @ {TARGET_FPS:.0f}Hz = {WIN_LEN} samples | "
          f"max gap {MAX_GAP_SEC}s, max total broken {MAX_TOTAL_BROKEN_SEC}s")

    tasks = [(f[:-len("_signals.npz")], os.path.join(args.signals_dir, f), args.out_dir, args.overwrite)
             for f in sig_files]
    if not tasks:
        print("nothing to process.")
        return

    n_workers = args.workers if args.workers and args.workers > 0 else _default_workers()
    n_workers = max(1, min(n_workers, len(tasks)))
    total_windows = 0
    all_spans = []
    if n_workers == 1:
        for i, task in enumerate(tasks, 1):
            _, msg, k, spans = process_clip(task)
            total_windows += k
            all_spans.extend(spans)
            print(f"[{i}/{len(tasks)}] {msg}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_clip, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                _, msg, k, spans = fut.result()
                total_windows += k
                all_spans.extend(spans)
                print(f"[{i}/{len(tasks)}] {msg}")

    # manifest: one row per emitted segment with its real time span (for PPG labeling)
    import csv
    man_path = os.path.join(args.out_dir, "segments_manifest.csv")
    all_spans.sort(key=lambda r: (r["clip"], r["index"]))
    with open(man_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["segment", "clip", "index", "t_start", "t_end", "abs_start"])
        w.writeheader()
        w.writerows(all_spans)

    print(f"\ntotal windows written: {total_windows} -> {args.out_dir}")
    print(f"manifest ({len(all_spans)} segments) -> {man_path}")


if __name__ == "__main__":
    main()