"""Shared processing library for the CPU stages.

Two sections:
  1. Region extraction — turn the saved segmentation + landmarks into a per-region
     skin mask and mean RGB (the original tracker's region-selection logic).
     Used by signal_extraction and by the visualization overlay, so both agree.
  2. rPPG DSP — detrend, bandpass, POS/CHROM projection, clean-window search.
     Used by the HR / Hb stages.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from common import config


# ======================================================================
# 1. Region extraction  (segmentation + landmarks -> per-region RGB)
# ======================================================================

class OneEuroFilter:
    """Vectorized 1-Euro temporal filter (identical to the original tracker)."""

    def __init__(self, freq: float, min_cutoff: float = 0.1, beta: float = 0.005, d_cutoff: float = 1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: NDArray[np.float64] | None = None
        self.dx_prev: NDArray[np.float64] | None = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) * self.freq
        ad = self._alpha(self.d_cutoff)
        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev
        vmag = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * vmag
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


def edge_kernel():
    """Erosion kernel for shrinking the ROI inward, or None if erosion disabled."""
    k = 2 * config.ROI_EROSION_PX + 1
    return np.ones((k, k), np.uint8) if config.ROI_EROSION_PX > 0 else None


def smooth_landmarks(pts_xy: NDArray[np.float64], fps: float) -> NDArray[np.float64]:
    """One-Euro smooth a (T, N, 2) landmark sequence.

    No-face frames (any NaN) stay NaN and reset the filter, exactly as the
    original tracker did when a detection was missed.
    """
    pts_xy = np.asarray(pts_xy, dtype=np.float64)
    out = np.full_like(pts_xy, np.nan)
    f = OneEuroFilter(freq=fps, min_cutoff=config.SMOOTH_MIN_CUTOFF, beta=config.SMOOTH_BETA)
    for i in range(pts_xy.shape[0]):
        p = pts_xy[i]
        if not np.isfinite(p).all():
            f.reset()
            continue
        out[i] = f(p)
    return out


def skin_mask_from_seg(seg_frame: NDArray[np.integer]) -> NDArray[np.uint8]:
    """Binary 0/255 skin mask from a face-parse class map."""
    return np.isin(seg_frame, config.SKIN_CLASS_IDS).astype(np.uint8) * 255


def smooth_masks(masks_seq, fps: float):
    """One-Euro smooth a (T, H, W) binary skin-mask sequence over TIME, per pixel.

    The SegFormer parse flickers frame-to-frame: a pixel (typically near a skin
    boundary such as the hairline or a turned cheek edge) flips skin/not-skin between
    adjacent frames even when the face is still. This is the mask analog of the landmark
    jitter that `smooth_landmarks` already fixes — so we apply the *same* One-Euro filter
    here, per pixel, to a soft "skin-ness" value in [0, 1], then re-threshold.

    Because One-Euro is motion-adaptive (its cutoff rises with the rate of change), a
    pixel that flickers slowly gets heavy smoothing, while pixels genuinely sweeping in/out
    during real head motion are smoothed less and so don't lag/smear — exactly the
    behaviour that makes One-Euro suitable for the landmarks. The filter is applied to the
    whole frame uniformly; it makes no assumption about which region is unstable.

    Input may be 0/255 (as from `skin_mask_from_seg`) or 0/1; output is 0/255 uint8,
    one stabilised mask per frame, same shape as the input.

    Note: this materialises the whole sequence. For long clips, prefer `MaskSmoother`
    (below), which filters one frame at a time with flat memory.
    """
    masks_seq = np.asarray(masks_seq)
    sm = MaskSmoother(fps)
    out = np.empty(masks_seq.shape, dtype=np.uint8)
    for i in range(masks_seq.shape[0]):
        out[i] = sm.step(masks_seq[i])
    return out


class MaskSmoother:
    """Stateful, memory-flat One-Euro temporal smoother for per-frame skin masks.

    Feed frames in order with .step(mask); it holds only the running per-pixel filter
    state (one frame), never the whole sequence — so it is safe on full-length clips
    inside the extraction loop. Advance it on EVERY frame (including no-face frames) to
    keep the temporal spacing consistent; use the returned mask only where needed.
    """

    def __init__(self, fps: float):
        self.f = OneEuroFilter(
            freq=fps,
            min_cutoff=getattr(config, "MASK_SMOOTH_MIN_CUTOFF", config.SMOOTH_MIN_CUTOFF),
            beta=getattr(config, "MASK_SMOOTH_BETA", config.SMOOTH_BETA),
        )
        self.threshold = getattr(config, "MASK_SMOOTH_THRESHOLD", 0.5)

    def step(self, mask) -> NDArray[np.uint8]:
        soft = (np.asarray(mask) > 0).astype(np.float64)   # 0/1 skin-ness this frame
        sm = self.f(soft)                                   # temporally-smoothed
        return (sm >= self.threshold).astype(np.uint8) * 255

    def reset(self) -> None:
        self.f.reset()


def fill_polygon_subpix(shape_hw: tuple[int, int], points_xy: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Rasterise the convex hull of points at sub-pixel accuracy (no edge flicker)."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    hull = cv2.convexHull(points_xy.astype(np.float32))
    factor = 1 << config.SUBPIX_SHIFT
    pts = np.round(hull * factor).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_8, shift=config.SUBPIX_SHIFT)
    return mask


def build_region_mask(shape_hw, points_xy, skin_mask, ek) -> NDArray[np.uint8]:
    """ROI = (landmark polygon) AND (skin), eroded inward. The original logic."""
    poly = fill_polygon_subpix(shape_hw, points_xy)
    region = cv2.bitwise_and(poly, skin_mask)
    if ek is not None:
        region = cv2.erode(region, ek, iterations=1)
    return region


def region_mean_rgb(frame_bgr, region) -> NDArray[np.float64]:
    """Mean RGB over the region, or NaN if too few skin px (original threshold)."""
    if cv2.countNonZero(region) < config.MIN_SKIN_PIXELS:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    mean_bgr = cv2.mean(frame_bgr, mask=region)[:3]
    return np.array(mean_bgr[::-1], dtype=np.float64)  # BGR -> RGB


def region_mean_rgb_count(frame_bgr, region):
    """Like region_mean_rgb, but also return the skin-pixel count used for the mean.

    Returns (rgb (3,) float64, count int). Below MIN_SKIN_PIXELS the mean is NaN and the
    count is the (sub-threshold) pixel count — stored so downstream weighting can down-
    weight or drop the region. The count is what size-aware combination weights by.
    """
    count = int(cv2.countNonZero(region))
    if count < config.MIN_SKIN_PIXELS:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), count
    mean_bgr = cv2.mean(frame_bgr, mask=region)[:3]
    return np.array(mean_bgr[::-1], dtype=np.float64), count


def combine_regions(signals, pixel_counts=None, fps=None):
    """Combine per-region signals (R, T, C) into one (T, C) signal.

    If REGION_WEIGHT_ENABLED and pixel_counts (R, T) are given, each region is weighted
    PER SAMPLE by its skin-pixel count under a floor/cap scheme: weight 0 below
    MIN_SKIN_PIXELS, else min(count, REGION_WEIGHT_CAP_PX). Because the weight is per
    sample, it tracks head movement within a window — when the head turns and a cheek
    shrinks from a solid region to a sliver, that cheek's weight falls at that moment and
    the still-solid regions take over. The count series is first low-pass smoothed
    (REGION_WEIGHT_SMOOTH_SEC) so slow pose drift is kept but any residual fast flicker
    can't modulate the combined signal in the pulse band. NaN region-samples never
    contribute. Falls back to an equal-weight nanmean when weighting is disabled or counts
    are unavailable.
    """
    signals = np.asarray(signals, dtype=np.float64)     # (R, T, C)
    R, T, C = signals.shape

    if not getattr(config, "REGION_WEIGHT_ENABLED", False) or pixel_counts is None:
        return np.nanmean(signals, axis=0)              # equal-weight fallback

    counts = np.asarray(pixel_counts, dtype=np.float64)  # (R, T)

    # smooth each region's count series to keep pose drift but drop fast flicker
    smooth_sec = getattr(config, "REGION_WEIGHT_SMOOTH_SEC", 0.0)
    fps = fps or getattr(config, "TARGET_FPS", config.DEFAULT_FPS)
    win = int(round(smooth_sec * fps)) if smooth_sec and smooth_sec > 0 else 0
    if win >= 2 and T >= win:
        kernel = np.ones(win) / win
        counts = np.vstack([np.convolve(counts[r], kernel, mode="same") for r in range(R)])

    floor = config.MIN_SKIN_PIXELS
    cap = getattr(config, "REGION_WEIGHT_CAP_PX", 1000)
    w = np.where(counts >= floor, np.minimum(counts, cap), 0.0)   # (R, T)
    valid = np.isfinite(signals[:, :, 1])                # (R, T) — region can't contribute where NaN
    w = w * valid
    w3 = w[:, :, None]                                    # (R, T, 1) broadcast over C

    wsum = w.sum(axis=0)                                  # (T,)
    num = np.nansum(np.where(np.isfinite(signals), signals, 0.0) * w3, axis=0)  # (T, C)
    out = np.divide(num, wsum[:, None], out=np.full((T, C), np.nan), where=wsum[:, None] > 0)
    return out


# ======================================================================
# 2. rPPG DSP  (used by the HR / Hb stages)
# ======================================================================

def interpolate_nans(sig: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linearly interpolates NaN values in a 1D or 2D time-series array."""
    sig_clean = np.asarray(sig, dtype=np.float64).copy()
    if sig_clean.ndim == 1:
        valid = np.isfinite(sig_clean)
        if valid.sum() >= 2 and not valid.all():
            indices = np.arange(len(sig_clean))
            sig_clean[~valid] = np.interp(indices[~valid], indices[valid], sig_clean[valid])
        elif valid.sum() < 2:
            sig_clean[~valid] = 0.0
        return sig_clean
    for col in range(sig_clean.shape[1]):
        sig_clean[:, col] = interpolate_nans(sig_clean[:, col])
    return sig_clean


def smoothness_detrend(sig: NDArray[np.float64], lambda_param: float = config.DETREND_LAMBDA) -> NDArray[np.float64]:
    """Removes low-frequency baseline wander via smoothness priors (Tarvainen 2002)."""
    sig_clean = interpolate_nans(sig)
    n_samples = len(sig_clean)
    if n_samples < 5:
        return sig_clean - np.mean(sig_clean)
    identity = np.eye(n_samples)
    diff_matrix = np.diff(identity, n=2, axis=0)
    try:
        penalty = (lambda_param ** 2) * (diff_matrix.T @ diff_matrix)
        trend = np.linalg.solve(identity + penalty, sig_clean)
    except np.linalg.LinAlgError:
        return sig_clean - np.mean(sig_clean)
    return sig_clean - trend


def bandpass_filter(
    sig: NDArray[np.float64],
    fps: float = config.DEFAULT_FPS,
    low_hz: float = config.HR_FREQ_MIN_HZ,
    high_hz: float = config.HR_FREQ_MAX_HZ,
    order: int = config.BANDPASS_ORDER,
    apply_detrend: bool = True,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth bandpass to isolate the physiological pulse."""
    processed_sig = np.asarray(sig, dtype=np.float64)
    if apply_detrend:
        processed_sig = smoothness_detrend(processed_sig)
    processed_sig = processed_sig - np.mean(processed_sig)
    nyquist = 0.5 * fps
    low_norm = low_hz / nyquist
    high_norm = min(high_hz / nyquist, 0.99)
    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        return processed_sig
    b_coeff, a_coeff = butter(order, [low_norm, high_norm], btype="band")
    if len(processed_sig) <= 3 * max(len(a_coeff), len(b_coeff)):
        return processed_sig
    return filtfilt(b_coeff, a_coeff, processed_sig)


def extract_pos(rgb_signal: NDArray[np.float64], fps: float = config.DEFAULT_FPS) -> NDArray[np.float64]:
    """Extracts a pulse waveform using Plane-Orthogonal-to-Skin (POS)."""
    rgb_clean = interpolate_nans(rgb_signal)
    n_samples = rgb_clean.shape[0]
    pulse_accum = np.zeros(n_samples)
    win_len = int(np.round(1.6 * fps))
    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    if win_len < 2 or n_samples < win_len:
        mean_rgb = np.mean(rgb_clean, axis=0) + 1e-9
        norm_rgb = rgb_clean / mean_rgb
        s_coords = projection_matrix @ norm_rgb.T
        alpha = np.std(s_coords[0]) / (np.std(s_coords[1]) + 1e-9)
        pulse = s_coords[0] + alpha * s_coords[1]
        return pulse - np.mean(pulse)
    for start_idx in range(0, n_samples - win_len + 1):
        window = rgb_clean[start_idx: start_idx + win_len]
        mean_rgb = np.mean(window, axis=0) + 1e-9
        norm_window = (window / mean_rgb).T
        s_coords = projection_matrix @ norm_window
        s1, s2 = s_coords[0], s_coords[1]
        alpha = np.std(s1) / (np.std(s2) + 1e-9)
        h_signal = s1 + alpha * s2
        pulse_accum[start_idx: start_idx + win_len] += h_signal - np.mean(h_signal)
    return pulse_accum - np.mean(pulse_accum)


def extract_chrom(rgb_signal: NDArray[np.float64], fps: float = config.DEFAULT_FPS) -> NDArray[np.float64]:
    """Extracts a pulse waveform using Chrominance-based rPPG (CHROM)."""
    rgb_clean = interpolate_nans(rgb_signal)
    mean_rgb = np.mean(rgb_clean, axis=0) + 1e-9
    norm_rgb = rgb_clean / mean_rgb
    r_chan, g_chan, b_chan = norm_rgb[:, 0], norm_rgb[:, 1], norm_rgb[:, 2]
    x_stride = 3.0 * r_chan - 2.0 * g_chan
    y_stride = 1.5 * r_chan + g_chan - 1.5 * b_chan
    x_filtered = bandpass_filter(x_stride, fps=fps, low_hz=config.SPEC_FREQ_MIN_HZ, high_hz=config.SPEC_FREQ_MAX_HZ, apply_detrend=True)
    y_filtered = bandpass_filter(y_stride, fps=fps, low_hz=config.SPEC_FREQ_MIN_HZ, high_hz=config.SPEC_FREQ_MAX_HZ, apply_detrend=True)
    alpha = np.std(x_filtered) / (np.std(y_filtered) + 1e-9)
    pulse = x_filtered - alpha * y_filtered
    return pulse - np.mean(pulse)


def find_clean_window(
    ref_signal: NDArray[np.float64],
    fps: float = config.DEFAULT_FPS,
    win_sec: float = 1.0,
    mult: float = 3.0,
    min_sec: float = config.MIN_CLEAN_WINDOW_SEC,
    target_sec: float | None = None,
) -> tuple[int, int] | None:
    """Finds a clean, low-motion window in a raw reference signal."""
    ref_clean = interpolate_nans(ref_signal)
    n_samples = len(ref_clean)
    if n_samples < int(min_sec * fps):
        return None

    diff = np.abs(np.diff(ref_clean, prepend=ref_clean[0]))
    w = max(int(round(win_sec * fps)), 3)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    local_mean = np.convolve(diff, kernel, mode="same")
    local_var = np.convolve(diff * diff, kernel, mode="same") - local_mean ** 2
    local_motion = np.sqrt(np.clip(local_var, 0, None))
    baseline = np.median(local_motion) + 1e-9
    is_clean = local_motion <= (mult * baseline)

    rolling_mean = np.convolve(ref_clean, kernel, mode="same")
    before = np.roll(rolling_mean, w)
    after = np.roll(rolling_mean, -w)
    level_change = np.abs(after - before)
    level_change[: 2 * w] = 0.0
    level_change[-2 * w:] = 0.0
    lc_med = np.median(level_change)
    lc_mad = np.median(np.abs(level_change - lc_med)) + 1e-9
    step_frames = np.where(level_change > lc_med + 8.0 * 1.4826 * lc_mad)[0]
    for s in step_frames:
        lo, hi = max(0, s - w), min(n_samples, s + w + 1)
        is_clean[lo:hi] = False

    best_len = 0
    best_window = None
    start = None
    for i, c in enumerate(is_clean):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start > best_len:
                best_len = i - start
                best_window = (start, i)
            start = None
    if start is not None and n_samples - start > best_len:
        best_len = n_samples - start
        best_window = (start, n_samples)

    if best_window is None or best_len < int(min_sec * fps):
        return None

    if target_sec is not None:
        target_len = int(round(target_sec * fps))
        a, b = best_window
        if (b - a) > target_len:
            best_start, best_score = a, np.inf
            for s in range(a, b - target_len + 1):
                score = float(np.mean(local_motion[s: s + target_len]))
                if score < best_score:
                    best_score, best_start = score, s
            best_window = (best_start, best_start + target_len)

    return best_window