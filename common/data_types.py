"""
Data models shared across the rPPG processing pipeline.
"""

from enum import StrEnum
from dataclasses import dataclass


VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov")


class FileExtension(StrEnum):
    """
    Pipeline output file suffixes.
    """
    SEGMENTATION = "_seg.npz"
    LANDMARKS = "_landmarks.npz"
    SIGNALS = "_signals.npz"
    SIGNAL_PLOT = "_signals.png"
    VIDEO_OVERLAY = "_overlay.mp4"


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