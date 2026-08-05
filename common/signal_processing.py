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
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt # type: ignore

from common import config
from common.one_euro import OneEuroFilter


def smooth_landmarks(pts_xy: NDArray[np.float64], fps: float) -> NDArray[np.float64]:
    """
    Apply One-Euro temporal smoothing to face landmarks.

    Each landmark coordinate is filtered independently over time. Frames
    containing invalid landmarks (NaN values) are preserved as NaN and reset
    the filter state so smoothing does not propagate across missed detections.

    Args:
        pts_xy: Landmark coordinates with shape (T, N, 2), where T is the number
            of frames and N is the number of landmarks.
        fps: Sampling rate of the landmark sequence in frames per second.

    Returns:
        Smoothed landmark coordinates with the same shape as the input.
    """
    # Convert to floating-point precision suitable for recursive filtering
    pts_xy = np.asarray(pts_xy, dtype=np.float64)

    # Initialize output with NaNs so invalid frames remain invalid
    out = np.full_like(pts_xy, np.nan)

    # One-Euro filter shared across all landmarks
    f = OneEuroFilter(freq=fps, min_cutoff=config.SMOOTH_MIN_CUTOFF, beta=config.SMOOTH_BETA)

    # Filter each frame independently
    for i in range(pts_xy.shape[0]):
        p = pts_xy[i]
        # Reset after missed detections to avoid smoothing across gaps
        if not np.isfinite(p).all():
            f.reset()
            continue
        out[i] = f(p)
    return out


def edge_kernel() -> NDArray[np.uint8] | None:
    """
    Create the erosion kernel used for region mask generation.

    The kernel is used to shrink each facial region inward before signal
    extraction, reducing contamination from unstable boundary pixels.

    Returns:
        Square erosion kernel, or None when region erosion is disabled.
    """
    k = 2 * config.ROI_EROSION_PX + 1
    return np.ones((k, k), np.uint8) if config.ROI_EROSION_PX > 0 else None


def skin_mask_from_seg(seg_frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert a face-parsing mask into a binary skin mask.

    Pixels belonging to the configured skin classes are assigned a value of
    255, while all remaining pixels are assigned 0.

    Args:
        seg_frame: Face-parsing class map for a single frame.

    Returns:
        Binary skin mask with values in {0, 255}.
    """
    return np.isin(seg_frame, config.SKIN_CLASS_IDS).astype(np.uint8) * 255


def smooth_masks(masks: NDArray[np.uint8], fps: float) -> NDArray[np.uint8]:
    """
    Apply One-Euro temporal smoothing to a sequence of skin masks.

    Each pixel is treated as an independent time series. The binary mask is converted to a soft
    skin-membership image (0 or 1), filtered over time, and thresholded back to a binary mask.

    Args:
        masks: Binary skin masks with shape (T, H, W).
        fps: Sampling rate of the mask sequence in frames per second.

    Returns:
        Smoothed binary skin masks with the same shape as the input.
    """
    masks = np.asarray(masks)
    # One-Euro filter shared across all segmentaions
    filter = OneEuroFilter(freq=fps, min_cutoff=config.SMOOTH_MIN_CUTOFF, beta=config.SMOOTH_BETA)

    out = np.empty_like(masks, dtype=np.uint8)
    for i in range(masks.shape[0]):
        # Convert the binary mask to a soft skin-membership image
        soft = (masks[i] > 0).astype(np.float64)
        # Apply temporal smoothing
        smoothed = filter(soft)
        # Convert back to a binary mask
        out[i] = (smoothed >= config.MASK_SMOOTH_THRESHOLD).astype(np.uint8) * 255
    return out


def fill_polygon_subpix(shape_hw: tuple[int, int], points_xy: NDArray[np.float64]) -> NDArray[np.uint8]:
    """
    Rasterize a facial region polygon into a binary mask.

    The polygon is first converted to its convex hull and then filled using sub-pixel coordinates
    to reduce boundary jitter between consecutive frames.

    Args:
        shape_hw: Output mask shape as (height, width).
        points_xy: Landmark coordinates defining the facial region.

    Returns:
        Binary mask of the filled facial region with values in {0, 255}.
    """
    # Allocate the output mask
    mask = np.zeros(shape_hw, dtype=np.uint8)

    # Build a convex polygon from the region landmarks
    hull = cv2.convexHull(points_xy.astype(np.float32))

    # Convert to fixed-point coordinates for sub-pixel rasterization
    factor = 1 << config.SUBPIX_SHIFT
    pts = np.round(hull * factor).astype(np.int32)

    # Fill the polygon at sub-pixel precision
    cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_8, shift=config.SUBPIX_SHIFT)
    return mask


def build_region_masks(
    shape_hw: tuple[int, int],
    landmarks_xy: NDArray[np.float64],
    skin_masks: NDArray[np.uint8],
    kernel: NDArray[np.uint8] | None
) -> NDArray[np.uint8]:
    """
    Construct skin-constrained masks for all facial regions across a frames.

    The region polygons are defined by the supplied landmarks, intersected with the skin
    segmentation mask, and eroded inward to reduce boundary contamination.

    Args:
        shape_hw: Output mask shape as (height, width).
        landmarks_xy: Smoothed landmark coordinates with shape (478, 2).
        skin_masks: Binary skin masks with shape (H, W).
        kernel: Morphological erosion kernel applied to each region mask.

    Returns:
        Binary region masks with shape (R, H, W).
    """
    n_regions = len(config.REGION_ORDER)
    region_masks = np.zeros((n_regions, *shape_hw), dtype=np.uint8)

    # Process each region
    for region_idx, region_name in enumerate(config.REGION_ORDER):
        region_points = landmarks_xy[config.REGIONS[region_name]]
        # Rasterize the landmark-defined facial region.
        polygon = fill_polygon_subpix(shape_hw, region_points)
        # Keep only pixels classified as skin.
        region = polygon & skin_masks
        # Remove unstable boundary pixels.
        if kernel is not None:
            region_masks[region_idx] = cv2.erode(region, kernel, iterations=1)
    return region_masks


def region_mean_rgb_count(
    frame_bgr: MatLike,
    region_masks: NDArray[np.uint8],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute the mean RGB value within each facial region for a frame.

    The mean is computed only when the region contains a sufficient number of
    valid skin pixels. Otherwise, NaN values are returned so downstream stages
    can ignore or down-weight the region while still retaining the pixel count.

    Args:
        frame_bgr: Video frame with shape (H, W, 3).
        region_masks: Binary region masks with shape (R, H, W).

    Returns:
        A tuple containing:
            - Mean RGB values with shape (R, 3).
            - Valid skin-pixel counts with shape (R).
    """
    n_regions = region_masks.shape[0]
    mean_rgb = np.full((n_regions, 3), np.nan, dtype=np.float64)
    pixel_counts = np.zeros((n_regions), dtype=np.int32)

    # Process each facial region.
    for region_idx in range(n_regions):
        mask = region_masks[region_idx]
        # Count the number of valid pixels in the region.
        pixel_count = int(cv2.countNonZero(mask))
        pixel_counts[region_idx] = pixel_count
        # Reject regions with insufficient skin coverage.
        if pixel_count < config.MIN_SKIN_PIXELS:
            continue
        # Compute the mean color over the masked region.
        mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
        # Convert BGR to RGB.
        mean_rgb[region_idx] = mean_bgr[::-1]
    return mean_rgb, pixel_counts


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