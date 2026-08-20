"""
Interpolation utilities for uniformly resampling rPPG signals.

These functions operate purely on sampled data and timestamps.
They contain no dataset-specific logic such as window selection,
gap rejection, or file I/O.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator  # type: ignore


def collapse_duplicate_samples(timestamps: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse duplicate timestamps by averaging their samples.

    PCHIP interpolation requires strictly increasing timestamps. Some cameras occasionally
    generate identical timestamps, which are merged before interpolation.

    Args:
        timestamps: Sample timestamps.
        samples: Samples having shape (N, C).

    Returns:
        Unique timestamps and the corresponding averaged samples.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)

    # Nothing to merge for fewer than two samples.
    if len(timestamps) < 2:
        return timestamps, samples

    # Fast path when timestamps are already strictly increasing.
    if np.all(np.diff(timestamps) > 0):
        return timestamps, samples

    # Group samples that share the same timestamp.
    unique_times, inverse = np.unique(timestamps, return_inverse=True)
    # Allocate the averaged output array.
    collapsed = np.empty((len(unique_times), samples.shape[1]), dtype=np.float64)
    # Number of samples contributing to each unique timestamp.
    counts = np.bincount(inverse, minlength=len(unique_times)).astype(np.float64)

    # Average each channel independently across duplicate timestamps.
    for channel in range(samples.shape[1]):
        collapsed[:, channel] = (
            np.bincount(inverse, weights=samples[:, channel], minlength=len(unique_times)) / counts
        )
    return unique_times, collapsed


def pchip_resample(timestamps: np.ndarray, samples: np.ndarray, output_times: np.ndarray) -> np.ndarray | None:
    """
    Resample multichannel samples using monotonic cubic interpolation.

    Args:
        timestamps: Input timestamps.
        samples: Samples having shape (N, C).
        output_times: Desired output timestamps.

    Returns:
        Array having shape (M, C), or None if interpolation cannot be performed.
    """
    # Ensure timestamps are strictly increasing before interpolation.
    timestamps, samples = collapse_duplicate_samples(timestamps, samples)

    # Remove samples containing missing values.
    valid = np.all(np.isfinite(samples), axis=1)
    timestamps = timestamps[valid]
    samples = samples[valid]

    # PCHIP requires at least two samples.
    if len(timestamps) < 2:
        return None

    # Reject interpolation requests that extend beyond the observed data.
    # A small boundary tolerance is allowed to accommodate irregular frame timing.
    tolerance = np.median(np.diff(timestamps)) * 1.5
    if (timestamps[0] > output_times[0] + tolerance
        or timestamps[-1] < output_times[-1] - tolerance
    ):
        return None

    # Clamp the requested timeline to the valid interpolation domain.
    output_times = np.clip(output_times, timestamps[0], timestamps[-1])

    # Allocate the interpolated output.
    output = np.empty((len(output_times), samples.shape[1]), dtype=np.float64)

    # Interpolate each channel independently.
    for channel in range(samples.shape[1]):
        output[:, channel] = PchipInterpolator(timestamps, samples[:, channel])(output_times)
    return output


def resample_region_signals(
    signals: np.ndarray,
    timestamps: np.ndarray,
    output_times: np.ndarray
) -> np.ndarray:
    """
    Resample RGB signals for all facial regions.

    Args:
        signals: Signal array of shape (T, R, 3).
        timestamps: Sample timestamps.
        output_times: Desired output timeline.

    Returns:
        Resampled signals of shape (M, R, 3). Regions that cannot be interpolated are filled with NaNs.
    """
    _, regions, channels = signals.shape

    # Allocate the resampled output signal.
    output = np.empty((len(output_times), regions, channels), dtype=np.float32)

    # Interpolate each facial region independently.
    for region in range(regions):
        interpolated = pchip_resample(timestamps, signals[:, region], output_times)

        # Mark regions that cannot be interpolated as missing data.
        if interpolated is None:
            output[:, region] = np.nan
            continue

        output[:, region] = interpolated.astype(np.float32)
    return output


def resample_pixel_counts(
    pixel_counts: np.ndarray,
    timestamps: np.ndarray,
    output_times: np.ndarray,
) -> np.ndarray:
    """
    Resample per-region pixel counts onto a uniform timeline.

    Linear interpolation is sufficient since pixel counts are used only as quality metadata.

    Args:
        pixel_counts: Pixel-count array of shape (T, R).
        timestamps: Sample timestamps.
        output_times: Desired output timeline.

    Returns:
        Resampled pixel counts of shape (M, R).
    """
    _, regions = pixel_counts.shape

    # Allocate the resampled output.
    output = np.empty((len(output_times), regions), dtype=np.float32)

    # Interpolate pixel counts independently for each facial region.
    for region in range(regions):
        output[:, region] = np.interp(output_times, timestamps, pixel_counts[:, region]).astype(np.float32)
    return output
