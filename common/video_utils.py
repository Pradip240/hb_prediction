"""
Utility functions for reading video metadata.

Provides helpers for obtaining video frame counts, timestamps, durations, and frame rates.
"""

import subprocess


def video_frame_count(video_path: str) -> int | None:
    """
    Count the number of frames in a video.

    Uses ffprobe to decode the entire video and obtain an accurate frame
    count rather than relying on the frame count stored in the video header.

    Args:
        video_path: Path to the input video.

    Returns:
        Number of decoded frames, or None if the frame count cannot be determined.
    """
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )

    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def video_timestamps(video_path: str) -> list[float] | None:
    """
    Read the presentation timestamp of every video frame.

    Uses ffprobe to extract each decoded frame's presentation timestamp. The timestamps
    are returned in seconds and represent the true presentation time of each frame.

    Args:
        video_path: Path to the input video.

    Returns:
        List of per-frame timestamps in seconds, or None if the timestamps cannot be determined.
    """
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )

    # Fall back if ffprobe failed
    if out.returncode != 0:
        return None

    # Extract timestamps
    timestamps: list[float] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        # Skip missing timestamps
        if line in ("", "N/A"):
            continue
        try:
            timestamps.append(float(line))
        except ValueError:
            continue
    # Return timestamps if available
    return timestamps if len(timestamps) >= 2 else None


def video_duration(video_path: str) -> float | None:
    """
    Read the wall-clock duration of a video.

    Args:
        video_path: Path to the input video.

    Returns:
        Video duration in seconds, or None if unavailable.
    """
    timestamps = video_timestamps(video_path)
    if timestamps is None:
        return None
    return float(timestamps[-1] - timestamps[0])


def video_metadata_fps(video_path: str) -> float | None:
    """
    Read the frame rate stored in the video metadata.

    This value is obtained directly from the video header and may be
    inaccurate for re-encoded videos.

    Args:
        video_path: Path to the input video.

    Returns:
        Metadata frame rate, or None if unavailable.
    """
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )

    try:
        num, den = out.stdout.strip().split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None


def video_fps(video_path: str, duration_s: float | None = None) -> float | None:
    """
    Estimate the true frame rate of a video.

    The frame rate is determined using the following priority:
        1. Decoded frame count and the provided duration.
        2. Decoded frame count and the video's wall-clock duration.
        3. Frame rate stored in the video metadata.

    Args:
        video_path: Path to the input video.
        duration_s: Optional known video duration in seconds.

    Returns:
        Estimated frame rate, or None if no estimate can be obtained.
    """
    n_frames = video_frame_count(video_path)

    if n_frames is None:
        return video_metadata_fps(video_path)

    # Use the provided duration if available
    if duration_s is not None and duration_s > 0:
        return n_frames / duration_s

    # Otherwise, fall back to the wall-clock duration
    wallclock = video_duration(video_path)
    if wallclock is not None and wallclock > 0:
        return n_frames / wallclock

    # Finally, fall back to the metadata FPS
    return video_metadata_fps(video_path)