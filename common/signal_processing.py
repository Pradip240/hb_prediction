"""
Shared signal-processing utilities for the CPU pipeline.

The module is divided into three stages:

1. Landmark and mask processing
   * Smooth facial landmarks.
   * Convert face-parsing output into skin masks.
   * Build and smooth facial-region masks.

2. Region signal extraction
   * Extract mean RGB values and valid pixel counts from each region.
   * Combine multiple facial regions into a single RGB signal when required.

3. rPPG signal processing
   * Remove low-frequency baseline drift.
   * Apply physiological bandpass filtering.
   * Estimate heart rate and spectral confidence from a pulse waveform.

The same functions are used by signal extraction, visualization, and
downstream HR processing so that all stages operate on consistent region
definitions and signal-processing logic.
"""

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt  # type: ignore

from common import config
from common.one_euro import OneEuroFilter


# ---------------------------------------------------------------------------
# Landmark and mask processing
# ---------------------------------------------------------------------------
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
    kernel: NDArray[np.uint8] | None,
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


# ---------------------------------------------------------------------------
# Region signal extraction
# ---------------------------------------------------------------------------
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


def combine_regions(signals: np.ndarray, pixel_counts: np.ndarray, fps: float) -> np.ndarray:
    """
    Combine per-region RGB signals into one RGB signal.

    Args:
        signals: RGB signals with shape (T, R, 3).
        pixel_counts: Valid skin-pixel counts with shape (T, R).
        fps: Sampling frequency of the signals.

    Returns:
        Combined RGB signal with shape (T, 3).
    """
    signals = np.asarray(signals, dtype=np.float64)
    pixel_counts = np.asarray(pixel_counts, dtype=np.float64)

    # Fall back to equal-weight averaging when region weighting is disabled.
    if not config.REGION_WEIGHT_ENABLED:
        return np.nanmean(signals, axis=1)

    # Smooth pixel counts to reduce fast fluctuations in region weights.
    window = int(round(config.REGION_WEIGHT_SMOOTH_SEC * fps))
    n_frames = signals.shape[0]
    if (window >= 2) and (n_frames >= window):
        kernel = np.ones(window) / window
        pixel_counts = np.stack(
            [np.convolve(pixel_counts[:, region], kernel, mode="same") for region in range(pixel_counts.shape[1])],
            axis=1,
        )

    # Compute the weighted RGB mean independently for each frame.
    weighted_sum = np.nansum(np.where(np.isfinite(signals), signals, 0.0) * pixel_counts[:, :, None], axis=1)
    weight_sum = pixel_counts.sum(axis=1)
    return weighted_sum / weight_sum[:, None]


# ---------------------------------------------------------------------------
# rPPG signal processing
# ---------------------------------------------------------------------------
def smoothness_detrend(signal: np.ndarray) -> np.ndarray:
    """
    Remove low-frequency baseline drift using smoothness priors.

    Args:
        signal: Input pulse signal.

    Returns:
        Detrended signal with the estimated baseline removed.
    """
    n_samples = len(signal)

    # Very short signals cannot provide a meaningful second-order
    # smoothness estimate, so only remove their mean.
    if n_samples < 5:
        return signal - np.mean(signal)

    # Construct the second-order difference matrix used to penalize
    # rapid changes in the estimated baseline.
    identity = np.eye(n_samples)
    difference = np.diff(identity, n=2, axis=0)

    try:
        # Solve for the smooth baseline that minimizes the signal residual
        # while penalizing curvature in the baseline.
        penalty = config.DETREND_LAMBDA**2 * (difference.T @ difference)
        baseline = np.linalg.solve(identity + penalty, signal)
    except np.linalg.LinAlgError:
        # Fall back to mean removal if the linear system cannot be solved.
        return signal - np.mean(signal)

    # Remove the estimated low-frequency baseline from the original signal.
    return signal - baseline


def bandpass_filter(signal: np.ndarray, fps: float, apply_detrend: bool = True) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a pulse waveform.

    Args:
        signal: Input pulse waveform.
        fps: Sampling frequency of the signal.
        apply_detrend: Whether to remove the slow baseline trend before filtering.

    Returns:
        Bandpass-filtered waveform. The centered input signal is returned
        unchanged when the requested filter cannot be applied.
    """
    processed_signal = np.asarray(signal, dtype=np.float64)

    # Remove slow baseline changes before isolating the pulse-frequency band.
    if apply_detrend:
        processed_signal = smoothness_detrend(processed_signal)

    # Remove the remaining DC component before filtering.
    processed_signal -= np.mean(processed_signal)

    # Normalize the cutoff frequencies to the Nyquist frequency required
    # by scipy's Butterworth filter.
    nyquist = 0.5 * fps
    low_normalized = config.HR_FREQ_MIN_HZ / nyquist
    high_normalized = min(config.HR_FREQ_MAX_HZ / nyquist, 0.99)

    # Return the centered signal when the requested frequency range is invalid.
    if (low_normalized <= 0) or (high_normalized >= 1) or (low_normalized >= high_normalized):
        return processed_signal

    # Design a Butterworth bandpass filter for the physiological pulse band.
    numerator, denominator = butter(  # type: ignore
        config.BANDPASS_ORDER, [low_normalized, high_normalized], btype="band", output="ba"
    )

    # filtfilt requires enough samples for the filter's edge padding.
    minimum_samples = 3 * max(len(numerator), len(denominator))  # type: ignore
    if len(processed_signal) <= minimum_samples:
        return processed_signal

    # Apply the filter in both directions to avoid introducing phase shift.
    return filtfilt(numerator, denominator, processed_signal)


def spectral_hr(signal: np.ndarray, fps: float) -> tuple[float | None, float]:
    """
    Estimate heart rate from the dominant frequency of a pulse waveform.

    Args:
        signal: Pulse waveform.
        fps: Sampling frequency of the waveform.

    Returns:
        Tuple containing the heart rate in BPM and peak-to-median spectral power confidence.
    """
    # Require at least two cycles at the lowest detectable HR.
    minimum_samples = int(np.ceil(fps / config.HR_FREQ_MIN_HZ))
    if len(signal) < minimum_samples or np.std(signal) < 1e-9:
        return None, 0.0

    # Remove the DC component before computing the spectrum.
    signal = signal - np.mean(signal)

    # Apply a Hann window to reduce spectral leakage.
    window = np.hanning(len(signal))

    # Zero-pad the signal to improve frequency resolution.
    nfft = int(2 ** np.ceil(np.log2(len(signal) * 4)))

    spectrum = np.abs(np.fft.rfft(signal * window, n=nfft)) ** 2
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fps)

    # Restrict the spectrum to the physiological HR range.
    band = (frequencies >= config.HR_FREQ_MIN_HZ) & (frequencies <= config.HR_FREQ_MAX_HZ)
    if not band.any():
        return None, 0.0

    band_frequencies = frequencies[band]
    band_power = spectrum[band]

    # Select the frequency with the highest spectral power.
    peak_index = int(np.argmax(band_power))
    peak_power = band_power[peak_index]

    # Measure peak prominence relative to the median band power.
    confidence = float(peak_power / (np.median(band_power) + 1e-12))

    heart_rate = float(band_frequencies[peak_index] * 60.0)
    return heart_rate, confidence
