"""
Utilities for loading and extracting contact PPG signals.
"""

from datetime import datetime

import numpy as np

from common.data_types import PPGSignal


def load_pw(path: str, fallback_fs: float = 100.0) -> PPGSignal:
    """
    Load PPG samples and timestamps from a PW file.

    Args:
        path: Path to the PW file.
        fallback_fs: Sampling frequency used when the timestamps cannot determine it.

    Returns:
        PPG signal containing PPG samples, sampling frequency, and absolute start timestamp in seconds.
    """
    values: list[float] = []
    timestamps: list[datetime] = []

    # Read the PPG value and timestamp from each PW record.
    with open(path, encoding="latin-1") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            values.append(float(fields[0]))
            timestamps.append(datetime.fromisoformat(f"{fields[1]} {fields[2]}"))

    # Convert the collected samples to the representation used by the pipeline.
    samples = np.asarray(values, dtype=np.float64)
    sampling_frequency = fallback_fs
    start_timestamp = 0.0

    # Estimate the sampling frequency from the recorded timestamps.
    if len(timestamps) > 1:
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        if duration > 0:
            sampling_frequency = (len(timestamps) - 1) / duration
        start_timestamp = timestamps[0].timestamp()
    return PPGSignal(samples, sampling_frequency, start_timestamp)


def ppg_segment(signal: PPGSignal, t_start: float, t_end: float) -> np.ndarray:
    """
    Extract a PPG segment from a relative time interval.

    Args:
        signal: PPG signal and timing information.
        t_start: Segment start time relative to the PPG signal.
        t_end: Segment end time relative to the PPG signal.

    Returns:
        PPG samples within the requested time interval.
    """
    # Clamp the interval to the available PPG samples.
    start = max(0, int(round(t_start * signal.sampling_frequency)))
    end = min(len(signal.samples), int(round(t_end * signal.sampling_frequency)))
    if end <= start:
        return np.array([], dtype=signal.samples.dtype)
    return signal.samples[start:end]