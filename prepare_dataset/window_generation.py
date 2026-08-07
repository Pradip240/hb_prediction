"""
Window selection utilities for dataset preparation.

These functions identify fixed-duration windows suitable for model training
using frame timestamps and per-frame validity information. They perform no
signal interpolation or file I/O.
"""

import numpy as np

from common import config
from common.data_types import WindowInfo, BrokenInterval


def _measure_broken_intervals(timestamps: np.ndarray, valid_samples: np.ndarray) -> list[BrokenInterval]:
    """
    Measure consecutive broken intervals for a facial region.

    A broken interval consists of invalid samples or missing-frame gaps inferred
    from irregular timestamps. Interval boundaries are expressed in real time so
    the quality criteria remain independent of the camera frame rate.

    Args:
        timestamps: Sample timestamps having shape (T,).
        valid_samples: Boolean validity mask having shape (T,) for a single facial region.

    Returns:
        Broken intervals ordered chronologically.
    """
    if len(timestamps) < 2:
        return []

    # Estimate the nominal sampling interval from the observed timestamps.
    median_dt = float(np.median(np.diff(timestamps)))

    broken_intervals: list[BrokenInterval] = []

    interval_start: float | None = None
    for index in range(len(timestamps)):
        gap = timestamps[index] - timestamps[index - 1] if index > 0 else median_dt
        missing_frame = gap > config.GAP_FACTOR * median_dt

        # Mark the beginning of a new broken interval.
        if interval_start is None:
            # An invalid sample becomes broken at its timestamp.
            if not valid_samples[index]:
                interval_start = timestamps[index]
            # A missing-frame gap begins where the first missing sample should have occurred.
            elif missing_frame:
                interval_start = timestamps[index - 1] + median_dt

        # Keep the broken interval open while samples remain invalid.
        if not valid_samples[index]:
            continue

        # Record the completed broken interval.
        if interval_start is not None:
            broken_intervals.append(
                BrokenInterval(start=interval_start, end=timestamps[index])
            )
            interval_start = None

    # Record a broken interval extending to the end of the clip.
    if interval_start is not None:
        broken_intervals.append(
            BrokenInterval(start=interval_start, end=timestamps[-1])
        )
    return broken_intervals


def find_windows(clip_name: str, timestamps: np.ndarray, valid_regions: np.ndarray) -> list[WindowInfo]:
    """
    Generate training windows from one clip.

    Broken intervals are first measured independently for every facial region across the entire clip.
    Candidate windows are then evaluated in chronological order using those intervals. A window is
    accepted once at least ``config.MIN_VALID_REGIONS`` facial regions satisfy the configured gap
    constraints. Otherwise, the search resumes immediately after the earliest broken interval whose
    removal would make that facial region satisfy the quality constraints.

    Args:
        clip_name: Source clip name.
        timestamps: Sample timestamps having shape (T,).
        valid_regions: Boolean validity mask having shape (T, R).

    Returns:
        Metadata describing every accepted dataset window.
    """
    windows: list[WindowInfo] = []

    if len(timestamps) < 2:
        return windows

    # Express timestamps relative to the beginning of the clip.
    relative = timestamps - timestamps[0]
    absolute_start = float(timestamps[0])
    clip_duration = float(relative[-1])

    # Measure broken intervals once for every facial region.
    region_intervals = [
        _measure_broken_intervals(relative, valid_regions[:, region])
        for region in range(valid_regions.shape[1])
    ]

    required_regions = min(config.MIN_VALID_REGIONS, valid_regions.shape[1])
    window_index = 1
    window_start = 0.0
    # Evaluate candidate windows chronologically.
    while window_start + config.WINDOW_SEC <= clip_duration + 1e-9:
        valid_region_count = 0
        region_mask = np.zeros(valid_regions.shape[1], dtype=bool)
        window_end = window_start + config.WINDOW_SEC
        # Earliest restart making the current window acceptable.
        earliest_restart: float | None = None

        # Evaluate every facial region independently.
        for region_idx, broken_intervals in enumerate(region_intervals):
            broken_overlaps: list[BrokenInterval] = []

            # Clip each broken interval to the current candidate window.
            for interval in broken_intervals:
                overlap_start = max(interval.start, window_start)
                overlap_end = min(interval.end, window_end)
                # Keep only broken intervals overlapping the current window.
                if overlap_end > overlap_start:
                    broken_overlaps.append(BrokenInterval(start=overlap_start, end=overlap_end))

            # Measure broken data remaining inside the current window.
            durations = [interval.end - interval.start for interval in broken_overlaps]
            longest_broken = max(durations, default=0.0)
            total_broken = sum(durations)

            # Region satisfies the quality criteria.
            if (longest_broken <= config.MAX_GAP_SEC) and (total_broken <= config.MAX_TOTAL_BROKEN_SEC):
                region_mask[region_idx] = True
                valid_region_count += 1
                continue

            # Find the earliest interval after which the remaining broken data satisfies the quality constraints.
            for remove_idx in range(len(broken_overlaps)):
                remaining = durations[remove_idx + 1 :]
                remaining_longest = max(remaining, default=0.0)
                remaining_total = sum(remaining)
                # Advancing beyond this interval leaves the remaining broken data within the configured quality limits.
                if (remaining_longest <= config.MAX_GAP_SEC) and (remaining_total <= config.MAX_TOTAL_BROKEN_SEC):
                    restart = broken_overlaps[remove_idx].end
                    # Keep the earliest restart found across all facial regions.
                    if (earliest_restart is None) or (restart < earliest_restart):
                        earliest_restart = restart
                    break

        # Accept the window once enough facial regions satisfy the quality constraints.
        if valid_region_count >= required_regions:
            windows.append(
                WindowInfo(
                    segment=f"{clip_name}_{window_index}",
                    clip=clip_name,
                    index=window_index,
                    t_start=float(window_start),
                    t_end=float(window_end),
                    abs_start=float(absolute_start + window_start),
                    region_mask=region_mask
                )
            )
            # Advance according to the configured window overlap.
            window_start += config.WINDOW_STEP_SEC
            window_index += 1
        else:
            # Resume searching immediately after the earliest broken interval that
            # prevented the current window from being accepted.
            if earliest_restart is None:
                window_start += config.WINDOW_STEP_SEC
            else:
                window_start = earliest_restart

    return windows