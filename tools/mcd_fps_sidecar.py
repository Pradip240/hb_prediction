"""mcd_fps_sidecar.py — compute correct per-clip fps for the MCD dataset from PPG.

The MCD videos were re-encoded with a wrong constant frame rate in the header, and
their per-frame PTS is self-consistent with that wrong rate — so the timing can't be
fixed by inspecting the video alone, and it can't be safely re-tagged without risking
frame loss. Instead we leave the videos untouched and emit a SIDECAR recording the
correct fps per clip:

    true_fps = frame_count / ppg_duration

The PPG (.PW) recording duration is trustworthy ground truth (a hardware clock).
analyze_signals.py reads this sidecar (via --fps-sidecar) and builds a uniform
corrected time axis for those clips, overriding the (untrustworthy) video PTS.

fps is per CLIP (per camera): the same subject/state can have different true rates on
different cameras, so the sidecar is keyed by full clip name.

FRAME-COUNT SOURCE (--frames-from):
  video      count frames by decoding the video (ffprobe -count_frames). CPU-only and
             independent of the GPU stages, so this can run in parallel with
             segmentation/landmarks. Reliable but slower (scans every frame). The
             video header's frame count is NOT trusted (often wrong on re-encodes).
  landmarks  len of the per-frame landmarks .npz (instant, but requires the GPU
             landmark stage to have finished).
  signals    frames from an already-extracted signals .npz.
  seg        len of the segmentation masks .npz.
  auto       first available of landmarks -> signals -> seg -> video (default).

Usage (video-based, runs alongside GPU preprocessing):
  python tools/mcd_fps_sidecar.py --frames-from video --video-dir data/videos \
      --ppg-dir data/ppg --out output/mcd_fps.csv
"""

import argparse
import csv
import glob
import os
import re
import subprocess
from datetime import datetime

import numpy as np

CAMERAS = ("IriunWebcam", "FullHDwebcam", "USBVideo")
VIDEO_EXTS = (".avi", ".mp4", ".mkv", ".mov")
_PW_LINE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+(\d{4}-\d{2}-\d{2}[ T][\d:.]+)")
_SUFFIX = {"landmarks": "_landmarks.npz", "signals": "_signals.npz", "seg": "_seg.npz"}
FPS_MIN, FPS_MAX = 15.0, 60.0


def parse_clip_name(name):
    toks = name.split("_")
    state = "after" if "after" in toks else ("before" if "before" in toks else None)
    camera = next((t for t in toks if t in CAMERAS), None)
    anchor = None
    for i, t in enumerate(toks):
        if t == camera or t == state:
            anchor = i
            break
    subject = None
    if anchor is not None:
        for t in reversed(toks[:anchor]):
            if t.isdigit():
                subject = t
                break
    if subject is None:
        digits = [t for t in toks if t.isdigit()]
        subject = digits[-1] if digits else None
    return subject, camera, state


def ppg_duration(path, fallback_fs):
    times, n = [], 0
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            m = _PW_LINE.match(line)
            if m:
                n += 1
                try:
                    times.append(datetime.fromisoformat(m.group(2).replace("T", " ")))
                except ValueError:
                    times.append(None)
            elif line.split():
                n += 1
    if len(times) > 1 and all(t is not None for t in times):
        span = (times[-1] - times[0]).total_seconds()
        if span > 0:
            return span
    return (n / fallback_fs) if n else None


# ---- per-source frame counters ----

def _count_npz(path, key, axis):
    return (int(np.load(path)[key].shape[axis]), None) if os.path.exists(path) else (None, None)


def _count_video(clip, d):
    for ext in VIDEO_EXTS:
        for cand in (clip + ext, clip + ext.upper()):
            vp = os.path.join(d, cand)
            if os.path.exists(vp):
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", vp],
                    capture_output=True, text=True)
                try:
                    return int(out.stdout.strip()), "video"
                except ValueError:
                    return None, None
    return None, None


def frame_count(clip, dirs, force="auto"):
    order = [force] if force != "auto" else ["landmarks", "signals", "seg", "video"]
    for src in order:
        d = dirs.get(src)
        if not d:
            continue
        if src == "video":
            n, _ = _count_video(clip, d)
        elif src == "landmarks":
            n, _ = _count_npz(os.path.join(d, f"{clip}_landmarks.npz"), "landmarks", 0)
        elif src == "signals":
            n, _ = _count_npz(os.path.join(d, f"{clip}_signals.npz"), "signals", 1)
        elif src == "seg":
            n, _ = _count_npz(os.path.join(d, f"{clip}_seg.npz"), "masks", 0)
        else:
            n = None
        if n:
            return n, src
    return None, None


def discover_clips(dirs, force="auto"):
    order = [force] if force != "auto" else ["landmarks", "signals", "seg", "video"]
    for src in order:
        d = dirs.get(src)
        if not (d and os.path.isdir(d)):
            continue
        if src == "video":
            out = [os.path.splitext(f)[0] for f in os.listdir(d)
                   if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        else:
            out = [os.path.basename(f).replace(_SUFFIX[src], "")
                   for f in glob.glob(os.path.join(d, "*" + _SUFFIX[src]))]
        if out:
            return sorted(out)
    return []


def main():
    ap = argparse.ArgumentParser(description="Compute per-clip true fps from PPG duration.")
    ap.add_argument("--frames-from", choices=["auto", "video", "landmarks", "signals", "seg"],
                    default="auto", help="where to read each clip's frame count (default auto). "
                                         "Use 'video' to stay CPU-only / independent of the GPU stages.")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--landmarks-dir", default=None)
    ap.add_argument("--signals-dir", default=None)
    ap.add_argument("--seg-dir", default=None)
    ap.add_argument("--ppg-dir", default="data/ppg")
    ap.add_argument("--out", default="output/mcd_fps.csv")
    ap.add_argument("--ppg-fs", type=float, default=100.0, help="fallback PPG rate if no timestamps")
    args = ap.parse_args()

    dirs = {"video": args.video_dir, "landmarks": args.landmarks_dir,
            "signals": args.signals_dir, "seg": args.seg_dir}
    if args.frames_from != "auto" and not dirs.get(args.frames_from):
        ap.error(f"--frames-from {args.frames_from} requires --{args.frames_from}-dir")

    ppg_index = {}
    for fp in glob.glob(os.path.join(args.ppg_dir, "*.PW")) + glob.glob(os.path.join(args.ppg_dir, "*.pw")):
        m = re.search(r"(\d+)_(before|after)", os.path.basename(fp), re.IGNORECASE)
        if m:
            ppg_index[(m.group(1), m.group(2).lower())] = fp
    print(f"{len(ppg_index)} PPG file(s) indexed")

    clips = discover_clips(dirs, args.frames_from)
    print(f"{len(clips)} clip(s) discovered  (frame source: {args.frames_from})")

    rows, skipped, flagged = [], 0, 0
    for clip in clips:
        subj, camera, state = parse_clip_name(clip)
        if subj is None or state is None:
            skipped += 1
            continue
        pw = ppg_index.get((subj, state))
        if pw is None:
            skipped += 1
            continue
        dur = ppg_duration(pw, args.ppg_fs)
        n, used = frame_count(clip, dirs, args.frames_from)
        if not dur or not n:
            skipped += 1
            continue
        true_fps = n / dur
        flag = "IMPLAUSIBLE" if not (FPS_MIN <= true_fps <= FPS_MAX) else ""
        flagged += 1 if flag else 0
        rows.append(dict(clip=clip, true_fps=round(true_fps, 4), n_frames=n,
                         ppg_duration_s=round(dur, 3), frames_from=used, flag=flag))
        print(f"  {clip:34} {n:5d} frames / {dur:6.1f}s = {true_fps:5.2f} fps {flag}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "true_fps", "n_frames",
                                           "ppg_duration_s", "frames_from", "flag"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} clip(s) -> {args.out}  (skipped {skipped}, flagged {flagged})")
    if flagged:
        print("  NOTE: flagged clips have implausible fps (<15 or >60) — likely truncated "
              "or corrupt; inspect before trusting them.")


if __name__ == "__main__":
    main()