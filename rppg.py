"""
End-to-End rPPG Extraction Pipeline — Custom ROI Drawing
----------------------------------------------------------------------
1. Rigidly tracks the face using optical flow and strict thresholds.
2. Extracts raw RGB signals from forehead and cheeks.
3. Draws the EXACT analytical skin polygons and the background bounding box
   onto the output video, rather than the default MediaPipe web.
4. Plots the RGB channels over time.

NOTE: Only the plotting and the per-region data collection were changed.
The face tracking, optical flow, rigid transform, and video drawing are
identical to the original.
"""

import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from scipy.signal import butter, filtfilt, welch  # <-- added for spectrum/bandpass

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

def ensure_model(path):
    if not os.path.exists(path):
        print(f"Downloading face_landmarker.task -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path

# --- ROI DEFINITIONS ---
ANCHOR_INDICES = [10, 151, 9, 8, 168, 6, 197, 113, 342, 227, 447]
FOREHEAD_POLY  = [67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 63, 68]
LEFT_CHEEK     = [50, 101, 119, 100, 142, 36, 205, 187, 123]
RIGHT_CHEEK    = [280, 330, 348, 329, 371, 266, 425, 411, 352]

def process_video_and_extract_signals(input_path, output_video, output_plot, model_path):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    )

    MP_RESET_THRESHOLD = 15.0
    JITTER_DEADZONE = 0.5

    base_mesh = None
    base_anchors = None
    tracked_anchors = None
    prev_gray = None
    prev_gray_anchors = None
    last_drawn_points = None

    # --- DATA COLLECTION ARRAYS (CHANGED: regions kept separate) ---
    times = []
    forehead_signals = []
    lcheek_signals = []
    rcheek_signals = []
    bg_signals = []

    # Background Box Coordinates (Top-Left corner)
    bx1, by1, bx2, by2 = 10, 10, 70, 70

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated_frame = frame.copy()
            trigger_mediapipe = False

            # 1. TRACK ANCHORS VIA OPTICAL FLOW
            if tracked_anchors is not None and prev_gray is not None:
                next_anchors, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, current_gray, tracked_anchors, None, **lk_params
                )

                if status is not None and np.sum(status) >= 8:
                    dist_from_base = np.mean(np.linalg.norm(next_anchors - base_anchors, axis=1))
                    if dist_from_base > MP_RESET_THRESHOLD:
                        trigger_mediapipe = True
                    else:
                        tracked_anchors = next_anchors
                else:
                    trigger_mediapipe = True
            else:
                trigger_mediapipe = True

            # 2. RUN MEDIAPIPE ONLY ON RE-KEYFRAME TRIGGER
            if trigger_mediapipe:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                ts = int(round((frame_idx * 1000.0) / fps))

                result = landmarker.detect_for_video(mp_image, ts)

                if result.face_landmarks:
                    face_landmarks_proto = result.face_landmarks[0]
                    mp_points = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks_proto], dtype=np.float32)

                    base_mesh = mp_points.copy()
                    base_anchors = mp_points[ANCHOR_INDICES].copy()
                    tracked_anchors = base_anchors.copy()
                    last_drawn_points = base_mesh.copy()
                else:
                    base_mesh = None
                    tracked_anchors = None

            # 3. APPLY RIGID TRANSFORM (WITH DEADZONE SHOCK ABSORPTION)
            elif base_mesh is not None and tracked_anchors is not None and prev_gray_anchors is not None:
                movement_since_last_frame = np.mean(np.linalg.norm(tracked_anchors - prev_gray_anchors, axis=1))

                if movement_since_last_frame > JITTER_DEADZONE:
                    matrix, inliers = cv2.estimateAffinePartial2D(base_anchors, tracked_anchors, method=cv2.RANSAC)
                    if matrix is not None:
                        ones_mesh = np.ones((base_mesh.shape[0], 1))
                        base_mesh_3d = np.hstack([base_mesh, ones_mesh])
                        last_drawn_points = base_mesh_3d.dot(matrix.T).astype(np.float32)

            # 4. EXPORT VISUALS AND COLOR ANALYSIS DATA
            if last_drawn_points is not None:
                # Isolate the exact analysis pixels via masks
                forehead_pts = last_drawn_points[FOREHEAD_POLY].astype(np.int32)
                l_cheek_pts = last_drawn_points[LEFT_CHEEK].astype(np.int32)
                r_cheek_pts = last_drawn_points[RIGHT_CHEEK].astype(np.int32)

                # CHANGED: mean RGB extracted per region (was one merged mask)
                def _region_mean_rgb(pts):
                    m = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(m, [pts], 255)
                    bgr = cv2.mean(frame, mask=m)[:3]
                    return np.array(bgr[::-1])  # BGR -> RGB

                fh_rgb = _region_mean_rgb(forehead_pts)
                lc_rgb = _region_mean_rgb(l_cheek_pts)
                rc_rgb = _region_mean_rgb(r_cheek_pts)

                # Mean BGR extraction of background box region
                bg_crop = frame[by1:by2, bx1:bx2]
                bg_bgr = cv2.mean(bg_crop)[:3]
                bg_rgb = np.array(bg_bgr[::-1])

                # Append data to processing arrays
                times.append(frame_idx / fps)
                forehead_signals.append(fh_rgb)
                lcheek_signals.append(lc_rgb)
                rcheek_signals.append(rc_rgb)
                bg_signals.append(bg_rgb)

                # -- DRAW ANALYSIS BOUNDING OBJECTS DIRECTLY TO VIDEO (UNCHANGED) --
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
                cv2.putText(annotated_frame, "BG REF", (bx1, by2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.polylines(annotated_frame, [forehead_pts], True, (255, 255, 0), 2)   # Cyan
                cv2.polylines(annotated_frame, [l_cheek_pts], True, (255, 0, 255), 2)   # Magenta
                cv2.polylines(annotated_frame, [r_cheek_pts], True, (0, 255, 0), 2)     # Green

                cv2.putText(annotated_frame, "ANALYSIS MODE: RIGID LOCK", (15, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if tracked_anchors is not None:
                prev_gray_anchors = tracked_anchors.copy()
            prev_gray = current_gray.copy() if current_gray is not None else None

            writer.write(annotated_frame)
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    print(f"Video saved to: {output_video}")

    # 5. GENERATE FINAL PLOT (CHANGED: per-region arrays passed in)
    plot_signals(
        np.array(times),
        np.array(forehead_signals),
        np.array(lcheek_signals),
        np.array(rcheek_signals),
        np.array(bg_signals),
        output_plot,
        fps,
    )


def _bandpass(sig, fps, low=0.7, high=4.0, order=4):
    """Zero-phase Butterworth bandpass; falls back to mean-removal if too short."""
    nyq = 0.5 * fps
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    sig = sig - np.mean(sig)
    if low_n <= 0 or high_n >= 1 or low_n >= high_n:
        return sig
    b, a = butter(order, [low_n, high_n], btype="band")
    if len(sig) <= 3 * max(len(a), len(b)):
        return sig
    return filtfilt(b, a, sig)


def pos_signal(rgb, fps):
    """
    Plane-Orthogonal-to-Skin (POS) rPPG extraction, Wang et al. 2017.
    rgb: (N, 3) array of mean R,G,B over time. Returns 1-D pulse signal.
    Uses a sliding window of ~1.6 s, temporal normalisation, the fixed POS
    projection, and overlap-add.
    """
    rgb = np.asarray(rgb, dtype=float)
    N = rgb.shape[0]
    H = np.zeros(N)
    L = int(np.round(1.6 * fps))  # window length
    if L < 2 or N < L:
        # too short: fall back to a simple version on the whole signal
        mean = np.mean(rgb, axis=0) + 1e-9
        Cn = rgb / mean
        S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn.T  # (2, N)
        h = S[0] + (np.std(S[0]) / (np.std(S[1]) + 1e-9)) * S[1]
        return h - np.mean(h)

    for n in range(0, N - L + 1):
        block = rgb[n:n + L]                       # (L, 3)
        mu = np.mean(block, axis=0) + 1e-9
        Cn = block / mu                            # temporal normalisation
        Cn = Cn.T                                  # (3, L)
        # POS projection
        S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn  # (2, L)
        s1, s2 = S[0], S[1]
        alpha = np.std(s1) / (np.std(s2) + 1e-9)
        h = s1 + alpha * s2
        h = h - np.mean(h)
        H[n:n + L] += h                            # overlap-add
    return H - np.mean(H)


def chrom_signal(rgb, fps):
    """
    CHROM rPPG extraction, de Haan & Jeanne 2013.
    rgb: (N,3) mean R,G,B over time. Returns 1-D pulse signal.
    """
    rgb = np.asarray(rgb, dtype=float)
    mean = np.mean(rgb, axis=0) + 1e-9
    Cn = rgb / mean                                # normalise per channel
    R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]
    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B
    Xf = _bandpass(Xs, fps)
    Yf = _bandpass(Ys, fps)
    alpha = np.std(Xf) / (np.std(Yf) + 1e-9)
    s = Xf - alpha * Yf
    return s - np.mean(s)


def estimate_hr(sig, fps, fmin=0.7, fmax=4.0):
    """Return (bpm, peak_freq, snr) from the dominant spectral peak in band."""
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.mean(sig)
    n = len(sig)
    if n < 8:
        return None, None, None
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None, None, None
    band_power = spec[band]
    band_freqs = freqs[band]
    peak_i = np.argmax(band_power)
    peak_f = band_freqs[peak_i]
    # crude SNR: peak power vs mean of the rest of the band
    others = np.delete(band_power, peak_i)
    snr = band_power[peak_i] / (np.mean(others) + 1e-12)
    return peak_f * 60.0, peak_f, snr



def _detect_reset_frames(ref_signal, k=6.0, pad=1):
    """
    Identify keyframe-reset frames from the frame-to-frame change of a reference
    signal (combined face green works well). A reset shows up as |diff| far above
    the robust spread (MAD). Returns a boolean mask over sample indices marking
    frames to be repaired; `pad` also flags the neighbours of each spike, because
    a step contaminates the sample on each side too.
    """
    diff = np.abs(np.diff(ref_signal))
    if len(diff) == 0:
        return np.zeros(len(ref_signal), dtype=bool)
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) + 1e-9
    # Robust z-score; threshold k*MAD above the median
    spike_at_diff = diff > (med + k * 1.4826 * mad)
    bad = np.zeros(len(ref_signal), dtype=bool)
    spike_idx = np.where(spike_at_diff)[0]
    for i in spike_idx:
        lo = max(0, i - pad)
        hi = min(len(ref_signal), i + 1 + pad + 1)  # diff[i] sits between i and i+1
        bad[lo:hi] = True
    return bad


def _despike(sig, bad_mask):
    """Replace flagged samples by linear interpolation from the good neighbours."""
    sig = np.asarray(sig, dtype=float).copy()
    good = ~bad_mask
    if good.sum() < 2 or bad_mask.sum() == 0:
        return sig
    idx = np.arange(len(sig))
    sig[bad_mask] = np.interp(idx[bad_mask], idx[good], sig[good])
    return sig



def plot_signals(times, forehead_rgb, lcheek_rgb, rcheek_rgb, bg_rgb, output_path, fps=25.0):
    """
    Six-panel diagnostic figure. NO background subtraction applied.
      1-3) Raw mean R / G / B per region vs background (background mean-shifted to overlay)
      4)   Frame-to-frame |delta G| of the combined face green
      5)   Green-channel power spectrum (log) with rPPG + breathing bands shaded
      6)   Pulsatile band (0.7-4 Hz): bandpassed face green vs bandpassed background green
    """
    if len(times) < 4:
        print("Not enough data extracted. Skipping plot.")
        return

    # Combined face green (mean of 3 regions) drives panels 4-6. No subtraction.
    face_green = np.mean(
        np.vstack([forehead_rgb[:, 1], lcheek_rgb[:, 1], rcheek_rgb[:, 1]]), axis=0
    )
    bg_green = bg_rgb[:, 1]

    # Combined face RGB (mean of 3 regions, per channel) for POS / CHROM
    face_rgb = np.mean(np.stack([forehead_rgb, lcheek_rgb, rcheek_rgb], axis=0), axis=0)
    bg_rgb_full = bg_rgb  # already (N,3)

    fig, axs = plt.subplots(8, 1, figsize=(13, 30))

    channel_names = ["R", "G", "B"]
    region_data = [
        ("Forehead",    forehead_rgb, "olive"),
        ("Left cheek",  lcheek_rgb,   "magenta"),
        ("Right cheek", rcheek_rgb,   "green"),
    ]

    # --- Panels 1-3: per-channel mean values, face regions vs background ---
    for ch in range(3):
        ax = axs[ch]
        for label, data, color in region_data:
            ax.plot(times, data[:, ch], color=color, linewidth=1.2, label=label)

        # Background mean-shifted to overlay near the face traces (shape compare only)
        bg_ch = bg_rgb[:, ch]
        target_level = (forehead_rgb[:, ch].mean() + rcheek_rgb[:, ch].mean()) / 2
        bg_plot = bg_ch - np.mean(bg_ch) + target_level
        ax.plot(times, bg_plot, color="black", linestyle="--",
                linewidth=1.0, alpha=0.6, label="Background (mean-shifted)")

        ax.set_ylabel(f"Mean {channel_names[ch]}")
        ax.set_title(f"{channel_names[ch]} channel \u2014 face regions vs background")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    # --- Panel 4: frame-to-frame change in green ---
    ax = axs[3]
    dG = np.abs(np.diff(face_green))
    ax.plot(times[1:], dG, color="green", linewidth=0.8)
    ax.set_ylabel("|\u0394G| per frame")
    ax.set_title("Frame-to-frame change in green channel")
    ax.grid(alpha=0.3)

    # --- Panel 5: green power spectrum (Welch), log scale, shaded bands ---
    ax = axs[4]
    g_detr = face_green - np.mean(face_green)
    nperseg = min(len(g_detr), 256)
    freqs, psd = welch(g_detr, fs=fps, nperseg=nperseg)
    ax.semilogy(freqs, psd, color="green", linewidth=1.5)
    ax.axvspan(0.7, 4.0, color="green", alpha=0.15, label="rPPG band (0.7-4 Hz)")
    ax.axvspan(0.1, 0.5, color="orange", alpha=0.25, label="Breathing band")
    ax.set_xlim(0, min(freqs.max(), 11))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Green channel power spectrum \u2014 peaks reveal noise sources")
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel 6: bandpassed pulsatile band, face vs background (no subtraction) ---
    ax = axs[5]
    face_bp = _bandpass(face_green, fps)
    bg_bp = _bandpass(bg_green, fps)
    ax.plot(times, face_bp, color="green", linewidth=1.2, label="Face green (bandpassed)")
    ax.plot(times, bg_bp, color="black", linewidth=1.0, alpha=0.7,
            label="Background green (bandpassed)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filtered amplitude")
    ax.set_title("Pulsatile band (0.7-4 Hz) \u2014 face vs background")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # --- Reset-frame detection (used to EXCLUDE resets from POS/CHROM spectra) ---
    # Nothing is altered or plotted here; we only compute which frames to skip.
    bad_mask = _detect_reset_frames(face_green)
    good = ~bad_mask

    # Find contiguous runs of good samples
    segments = []
    start = None
    for i, g in enumerate(good):
        if g and start is None:
            start = i
        elif not g and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(good)))

    # Keep only segments long enough for a meaningful spectrum (>= 2 s)
    min_len = max(int(2 * fps), 16)
    usable = [(s, e) for (s, e) in segments if (e - s) >= min_len]

    n_bad = int(bad_mask.sum())
    print(f"Reset frames excluded from POS/CHROM spectra: {n_bad}/{len(face_green)} "
          f"frames, {len(usable)} clean segments used")

    # --- POS & CHROM extraction (all 3 channels, robust to motion) ---
    pos = pos_signal(face_rgb, fps)
    chrom = chrom_signal(face_rgb, fps)
    pos_bg = pos_signal(bg_rgb_full, fps)  # background POS = sanity check (should be flat)

    # Bandpass the constructed pulse signals to the rPPG band
    pos_bp = _bandpass(pos, fps)
    chrom_bp = _bandpass(chrom, fps)
    pos_bg_bp = _bandpass(pos_bg, fps)

    # Heart-rate estimate from each (over reset-excluded samples for fairness)
    good_pos = pos_bp[good] if good.sum() > 8 else pos_bp
    good_chrom = chrom_bp[good] if good.sum() > 8 else chrom_bp
    hr_pos, f_pos, snr_pos = estimate_hr(good_pos, fps)
    hr_chrom, f_chrom, snr_chrom = estimate_hr(good_chrom, fps)
    hr_g, f_g, snr_g = estimate_hr(face_green[good] if good.sum() > 8 else face_green, fps)

    print("--- Heart-rate estimates (dominant peak in 0.7-4 Hz) ---")
    for name, hr, snr in [("Green", hr_g, snr_g), ("POS", hr_pos, snr_pos),
                          ("CHROM", hr_chrom, snr_chrom)]:
        if hr is not None:
            print(f"  {name:6s}: {hr:5.1f} bpm   (peak SNR {snr:5.1f})")

    # --- Panel 7: POS & CHROM power spectra (reset frames excluded) ---
    ax = axs[6]

    def _avg_spectrum(sig):
        """Welch PSD averaged over clean inter-reset segments of sig."""
        acc = None; fref = None; tot = 0
        segs = usable if usable else [(0, len(sig))]
        for (s, e) in segs:
            seg = sig[s:e] - np.mean(sig[s:e])
            if len(seg) < 16:
                continue
            nps = min(len(seg), 256)
            f_, p_ = welch(seg, fs=fps, nperseg=nps)
            if fref is None:
                fref = f_; acc = np.zeros_like(p_)
            if len(f_) != len(fref):
                p_ = np.interp(fref, f_, p_)
            acc += p_ * (e - s); tot += (e - s)
        if acc is None:
            return None, None
        return fref, acc / tot

    f_pos_s, psd_pos_s = _avg_spectrum(pos_bp)
    f_chr_s, psd_chr_s = _avg_spectrum(chrom_bp)
    f_bg_s, psd_bg_s = _avg_spectrum(pos_bg_bp)

    if psd_pos_s is not None:
        ax.semilogy(f_pos_s, psd_pos_s, color="crimson", linewidth=1.8, label="POS (face)")
    if psd_chr_s is not None:
        ax.semilogy(f_chr_s, psd_chr_s, color="navy", linewidth=1.3, alpha=0.8, label="CHROM (face)")
    if psd_bg_s is not None:
        ax.semilogy(f_bg_s, psd_bg_s, color="gray", linewidth=1.0, alpha=0.6,
                    linestyle="--", label="POS (background)")

    # Mark the POS heart-rate peak
    if f_pos is not None:
        ax.axvline(f_pos, color="crimson", linestyle=":", alpha=0.7)
        ax.text(f_pos, ax.get_ylim()[1], f" {hr_pos:.0f} bpm",
                color="crimson", fontsize=9, va="top")

    ax.axvspan(0.7, 4.0, color="green", alpha=0.12)
    ax.set_xlim(0, min(freqs.max(), 11))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("POS & CHROM power spectrum \u2014 pulse peak should rise here if present")
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel 8: POS pulse waveform in time (face vs background) ---
    ax = axs[7]
    ax.plot(times, pos_bp, color="crimson", linewidth=1.2, label="POS pulse (face)")
    ax.plot(times, pos_bg_bp, color="gray", linewidth=1.0, alpha=0.6,
            label="POS (background)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("POS amplitude")
    title = "POS pulse waveform (0.7-4 Hz)"
    if hr_pos is not None:
        title += f"  \u2014  est. {hr_pos:.0f} bpm, SNR {snr_pos:.1f}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Signal plot saved to: {output_path}")


if __name__ == "__main__":
    for video in os.listdir('video'):
        file = video.split('.mkv')[0]
        INPUT_VIDEO = f"video/{file}.mkv"
        OUTPUT_VIDEO = f"output/{file}.mp4"
        OUTPUT_PLOT = f"output/rppg_signals_{file}.png"
        MODEL_PATH = "face_landmarker.task"

        ensure_model(MODEL_PATH)
        process_video_and_extract_signals(INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_PLOT, MODEL_PATH)