"""
Data models shared across the rPPG processing pipeline.
"""

from enum import StrEnum
from dataclasses import dataclass

import numpy as np


VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov")


class FileExtension(StrEnum):
    """
    Pipeline output file suffixes.
    """
    SEGMENTATION = "_seg.npz"
    LANDMARKS = "_landmarks.npz"
    SIGNAL_PLOT = "_signals.png"
    VIDEO_OVERLAY = "_overlay.mp4"
    SIGNALS = "_signals.npz"
    DATASET_SAMPLE = "_sample.npz"



# ============================================================================
# Signal processing
# ============================================================================
@dataclass(slots=True, frozen=True)
class ClipTask:
    """
    Input required to process a single video clip.

    Attributes:
        name: Clip name.
        video_path: Path to the input video.
        segmentation_path: Path to the segmentation archive.
        landmark_path: Path to the landmark archive.
        signal_path: Output path for the extracted signal archive.
        plot_path: Output path for the signal plot.
        overlay_path: Output path for the visualization overlay.
        no_plot: Whether signal plotting is disabled.
        no_video: Whether overlay generation is disabled.
        overwrite: Whether existing outputs should be recomputed.
    """
    name: str
    video_path: str
    segmentation_path: str
    landmark_path: str
    signal_path: str
    plot_path: str
    overlay_path: str
    no_plot: bool
    no_video: bool
    overwrite: bool


@dataclass(slots=True, frozen=True)
class ClipResult:
    """
    Result returned after processing a single video clip.

    Attributes:
        name: Name of the processed clip.
        log: Processing summary suitable for console output.
        success: True if processing completed successfully, otherwise False.
    """
    name: str
    log: str
    success: bool



# ============================================================================
# Dataset preparation
# ============================================================================
@dataclass(slots=True, frozen=True)
class DatasetTask:
    """
    Input required to process a single extracted signal archive.

    Attributes:
        name: Clip name.
        signal_path: Path to the extracted signal archive.
        output_dir: Directory where dataset windows are written.
    """
    name: str
    signal_path: str
    output_dir: str


@dataclass(slots=True, frozen=True)
class BrokenInterval:
    """
    Time span of one consecutive broken interval within a candidate window.

    Attributes:
        start: Start time of the broken interval relative to the window.
        end: End time of the broken interval relative to the window.
    """
    start: float
    end: float


@dataclass(slots=True, frozen=True)
class WindowInfo:
    """
    Metadata describing one emitted dataset window.

    Attributes:
        segment: Segment identifier.
        clip: Source clip name.
        index: Window index within the clip.
        t_start: Window start time relative to the clip.
        t_end: Window end time relative to the clip.
        abs_start: Absolute timestamp of the window start.
        region_mask: Boolean mask indicating valid facial regions.
    """
    segment: str
    clip: str
    index: int
    t_start: float
    t_end: float
    abs_start: float
    region_mask: np.ndarray


@dataclass(slots=True)
class DatasetResult:
    """
    Result returned after processing a single signal archive.

    Attributes:
        name: Clip name.
        log: Processing summary suitable for console output.
        windows: Metadata for every emitted dataset window.
        count: Number of generated windows.
        success: True if processing completed successfully.
    """
    name: str
    log: str
    windows: list[WindowInfo]
    count: int
    success: bool