"""Feature extraction for hemoglobin estimation.

Hemoglobin estimation is treated as a regression problem using
physically motivated facial colour and pulsatile features.

For each facial region, the feature vector contains:

- RGB DC levels:
    Mean R, G, and B intensity.

- RGB AC amplitudes:
    Standard deviation of the band-passed R, G, and B signals.

- RGB AC/DC ratios:
    Pulsatile amplitude relative to the corresponding DC level.

- Cross-channel AC/DC ratios:
    Ratios between the normalized pulsatile responses of different
    colour channels.

- DC colour ratios:
    Ratios between the baseline R, G, and B intensities.

- Pixel-count quality features:
    Statistics describing how many valid skin pixels contributed to
    each region over time.

Pixel counts are useful because a region with very few valid pixels
produces a less reliable RGB measurement. These features therefore
allow the Hb model to distinguish strong colour measurements from
measurements obtained from poorly detected regions.

The resulting feature vector is intentionally small and interpretable
so that the Hb model has limited capacity to memorize subject identity.
"""


import numpy as np

from common import signal_processing as sp


CHANNELS = ("R", "G", "B")


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio while protecting against invalid denominators."""
    if not np.isfinite(numerator):
        return 0.0
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return 0.0
    return float(numerator / denominator)


def _ac_dc(x: np.ndarray, fps: float) -> tuple[float, float]:
    """Calculate DC level and AC amplitude for one colour channel.

    Args:
        x: One-dimensional RGB signal.
        fps: Sampling frequency of the signal.

    Returns:
        Tuple containing:
            - DC: Mean signal level.
            - AC: Standard deviation of the band-passed signal.
    """
    x = np.asarray(x, dtype=np.float64)

    # A completely missing region contains no valid RGB information.
    if np.all(np.isnan(x)):
        return 0.0, 0.0

    # Valid regions are guaranteed to contain only finite values.
    if not np.all(np.isfinite(x)):
        raise ValueError("RGB signal contains unexpected non-finite values.")

    dc = float(np.mean(x))
    # A constant or invalid signal cannot provide a meaningful pulsatile amplitude.
    if not np.isfinite(dc) or np.std(x) < 1e-9:
        return dc, 0.0
    try:
        band = sp.bandpass_filter(x, fps=fps)
    except Exception:
        # Fall back to the zero-mean signal if filtering fails.
        band = x - np.mean(x)

    ac = float(np.std(band))
    if not np.isfinite(ac):
        ac = 0.0
    return dc, ac


def _pixel_count_features(pixel_counts: np.ndarray) -> list[float]:
    """Calculate region-level quality features from pixel counts.

    Args:
        pixel_counts: Pixel counts with shape (T,).

    Returns:
        List containing:
            - Mean pixel count.
            - Standard deviation of pixel count.
            - Median pixel count.
            - Fraction of detected frames.
            - Mean pixel count normalized by the maximum count.
    """
    counts = np.asarray(pixel_counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(f"Expected one-dimensional pixel counts, got {counts.shape}.")

    # Pixel counts are expected to be finite and zero for a missing region.
    if not np.all(np.isfinite(counts)):
        raise ValueError("Pixel counts contain unexpected non-finite values.")

    mean_count = float(np.mean(counts))
    std_count = float(np.std(counts))
    median_count = float(np.median(counts))

    # A positive pixel count indicates that the region was detected.
    detection_fraction = float(np.mean(counts > 0.0))
    max_count = float(np.max(counts))
    normalized_mean = _safe_ratio(mean_count, max_count)
    return [mean_count, std_count, median_count, detection_fraction, normalized_mean]


def feature_names(region_order: list[str]) -> list[str]:
    """Return feature names in the same order as extract_features().

    Args:
        region_order: Names of the facial regions.

    Returns:
        List of feature names.
    """
    names: list[str] = []

    for region_name in region_order:
        # RGB features.
        for channel in CHANNELS:
            names.extend([
                f"{region_name}_{channel}_dc",
                f"{region_name}_{channel}_ac",
                f"{region_name}_{channel}_acdc",
            ])
        # Cross-channel AC/DC ratios.
        names.extend([
            f"{region_name}_acdc_G_over_R",
            f"{region_name}_acdc_R_over_B",
            f"{region_name}_acdc_G_over_B",
        ])
        # DC colour balance.
        names.extend([
            f"{region_name}_dc_R_over_G",
            f"{region_name}_dc_R_over_B",
            f"{region_name}_dc_G_over_B",
        ])
        # Pixel-count / region-quality features.
        names.extend([
            f"{region_name}_pixels_mean",
            f"{region_name}_pixels_std",
            f"{region_name}_pixels_median",
            f"{region_name}_pixels_detection_fraction",
            f"{region_name}_pixels_normalized_mean",
        ])
    return names


def extract_features(signals: np.ndarray, pixel_counts: np.ndarray, fps: float, region_order: list[str]) -> np.ndarray:
    """Extract RGB and pixel-count features from one segment.

    Args:
        signals: RGB signals with shape (R, T, 3).
        pixel_counts: Valid skin-pixel counts with shape (R, T).
        fps: Sampling frequency of the segment.
        region_order: Names of the facial regions in region order.

    Returns:
        One-dimensional feature vector containing all RGB and
        pixel-count features.

    Raises:
        ValueError: If input shapes are invalid.
    """
    signals = np.asarray(signals, dtype=np.float64)
    pixel_counts = np.asarray(pixel_counts, dtype=np.float64)

    if signals.ndim != 3:
        raise ValueError(f"Expected signals with shape (R, T, 3), got {signals.shape}.")
    if signals.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got {signals.shape[-1]}.")
    if pixel_counts.ndim != 2:
        raise ValueError(f"Expected pixel_counts with shape (R, T), got {pixel_counts.shape}.")
    if signals.shape[:2] != pixel_counts.shape:
        raise ValueError(f"signals and pixel_counts shapes do not match: {signals.shape} vs {pixel_counts.shape}.")
    if signals.shape[0] != len(region_order):
        raise ValueError(f"Expected {len(region_order)} regions from region_order, got {signals.shape[0]}.")

    features: list[float] = []
    for region_index, _region_name in enumerate(region_order):
        dcs: list[float] = []
        acs: list[float] = []
        acdcs: list[float] = []
        # Extract DC, AC, and AC/DC features for R, G, and B.
        for channel_index in range(3):
            dc, ac = _ac_dc(signals[region_index, :, channel_index], fps)
            acdc = _safe_ratio(ac, dc)
            dcs.append(dc)
            acs.append(ac)
            acdcs.append(acdc)
            features.extend([dc, ac, acdc])
        # Cross-channel AC/DC ratios.
        features.extend([
            _safe_ratio(acdcs[1], acdcs[0]),
            _safe_ratio(acdcs[0], acdcs[2]),
            _safe_ratio(acdcs[1], acdcs[2])
        ])
        # DC colour ratios.
        features.extend([
            _safe_ratio(dcs[0], dcs[1]),
            _safe_ratio(dcs[0], dcs[2]),
            _safe_ratio(dcs[1], dcs[2]),
        ])
        # Add region detection and pixel-count quality features.
        features.extend(_pixel_count_features(pixel_counts[region_index]))
    return np.asarray(features, dtype=np.float32)