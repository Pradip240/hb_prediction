"""
Classical rPPG algorithms for facial RGB signals.

This module provides the rPPG methods used to extract pulse waveforms and
estimate heart rate from combined facial RGB signals:

- POS pulse-waveform extraction.
- CHROM pulse-waveform extraction.
- Green-channel pulse extraction.
- A common interface for extracting waveforms, heart-rate estimates, and confidence scores.

Shared signal-processing utilities such as region combination, detrending,
bandpass filtering, and spectral heart-rate estimation are provided by
``common.signal_processing``.
"""

import numpy as np

import common.signal_processing as sp
from common import config
from common.data_types import WaveformResult


def extract_pos(rgb_signal: np.ndarray, fps: float) -> np.ndarray:
    """
    Extract a pulse waveform using the Plane-Orthogonal-to-Skin (POS) method.

    Args:
        rgb_signal: Combined RGB signal with shape (T, 3).
        fps: Sampling frequency of the RGB signal.

    Returns:
        Extracted pulse waveform with shape (T,).
    """
    n_samples: int = rgb_signal.shape[0]
    pulse = np.zeros(n_samples)

    # Use the configured temporal window for the POS calculation.
    window_size = round(config.POS_WINDOW_SEC * fps)

    # POS projection matrix for the normalized RGB signal.
    projection = np.array([
        [0.0, 1.0, -1.0],
        [-2.0, 1.0, 1.0],
    ])

    # Accumulate the POS waveform produced by each sliding window.
    if (window_size >= 2) and (n_samples >= window_size):
        for start in range(n_samples - window_size + 1):
            window = rgb_signal[start:start + window_size]
            # Normalize RGB values by the mean color of the current window.
            mean_rgb = np.mean(window, axis=0) + 1e-9
            normalized_window = (window / mean_rgb).T
            # Project the normalized RGB signal onto the POS plane.
            components = projection @ normalized_window
            first_component, second_component = components
            # Balance the two projected components before combining them.
            alpha = np.std(first_component) / (np.std(second_component) + 1e-9)
            window_pulse = first_component + alpha * second_component
            # Remove the window mean before accumulating overlapping windows.
            pulse[start:start + window_size] += (
                window_pulse - np.mean(window_pulse)
            )
    return sp.bandpass_filter(pulse, fps=fps, apply_detrend=True)


def extract_chrom(rgb_signal: np.ndarray, fps: float) -> np.ndarray:
    """
    Extract a pulse waveform using the CHROM method.

    Args:
        rgb_signal: Combined RGB signal with shape (T, 3).
        fps: Sampling frequency of the RGB signal.

    Returns:
        Extracted pulse waveform with shape (T,).
    """
    # Normalize RGB channels by their mean to reduce the effect of
    # differences in overall skin brightness.
    mean_rgb = np.mean(rgb_signal, axis=0) + 1e-9
    normalized_rgb = rgb_signal / mean_rgb
    red, green, blue = normalized_rgb.T

    # Project the normalized RGB signal onto the two CHROM chrominance axes.
    x_signal = 3.0 * red - 2.0 * green
    y_signal = 1.5 * red + green - 1.5 * blue

    # Bandpass both chrominance signals within the configured HR range.
    x_filtered = sp.bandpass_filter(x_signal, fps=fps, apply_detrend=True)
    y_filtered = sp.bandpass_filter(y_signal, fps=fps, apply_detrend=True)

    # Balance the two chrominance components according to their
    # relative variation before combining them into the pulse waveform.
    alpha = np.std(x_filtered) / (np.std(y_filtered) + 1e-9)
    return x_filtered - alpha * y_filtered


def method_waveforms(signals: np.ndarray, fps: float, pixel_counts: np.ndarray) -> dict[str, WaveformResult]:
    """
    Extract pulse waveforms and HR estimates using rPPG algorithms.

    Args:
        signals: RGB signals for each facial region.
        fps: Sampling frequency of the input signals.
        pixel_counts: Number of valid pixels for each region and frame.

    Returns:
        Pulse waveform, heart rate, and confidence for each algorithm.
    """
    # Combine valid facial regions into one RGB signal.
    face = sp.combine_regions(signals, pixel_counts, fps)

    waveforms: dict[str, WaveformResult] = {}

    # Extract POS waveform and estimate its heart rate.
    pos = extract_pos(face, fps=fps)
    pos_hr, pos_confidence = sp.spectral_hr(pos, fps)
    waveforms["POS"] = WaveformResult(waveform=pos, heart_rate=pos_hr, confidence=pos_confidence)

    # Extract CHROM waveform and estimate its heart rate.
    chrom = extract_chrom(face, fps=fps)
    chrom_hr, chrom_confidence = sp.spectral_hr(chrom, fps)
    waveforms["CHROM"] = WaveformResult(waveform=chrom, heart_rate=chrom_hr, confidence=chrom_confidence)

    # Filter the combined green channel and estimate its heart rate.
    green = sp.bandpass_filter(face[:, 1], fps)
    green_hr, green_confidence = sp.spectral_hr(green, fps)
    waveforms["green"] = WaveformResult(waveform=green,heart_rate=green_hr, confidence=green_confidence)
    return waveforms
