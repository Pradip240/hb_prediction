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
        region_mask: Optional Boolean mask indicating valid facial regions.
    """
    segment: str
    clip: str
    index: int
    t_start: float
    t_end: float
    abs_start: float
    region_mask: np.ndarray | None = None


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



# ============================================================================
# Prediction
# ============================================================================
@dataclass(slots=True, frozen=True)
class PPGSignal:
    """
    PPG signal and timing information.

    Attributes:
        samples: PPG samples.
        sampling_frequency: Sampling frequency in Hz.
        start_timestamp: Absolute start timestamp in seconds.
    """
    samples: np.ndarray
    sampling_frequency: float
    start_timestamp: float


@dataclass(slots=True, frozen=True)
class PredictionTask:
    """
    Input required to process a single dataset sample.

    Attributes:
        name: Dataset sample name.
        sample_path: Path to the dataset sample archive.
        segment: Source clip and time span metadata.
        ppg_info: Reference PPG data for the sample's subject and state.
        output_dir: Directory where prediction outputs are written.
        no_plot: Whether prediction plotting is disabled.
        hr_true: Ground-truth pulse value used when no PPG is available.
        hb_true: Ground-truth hemoglobin value for the subject.
        hr_model: Path to the trained HR model.
        hb_model: Path to the trained Hb model.
    """
    name: str
    sample_path: str
    segment: WindowInfo
    ppg_info: PPGSignal | None
    output_dir: str
    no_plot: bool
    hr_true: float | None
    hb_true: float | None
    hr_model: str | None
    hb_model: str | None
    


@dataclass(slots=True, frozen=True)
class PredictionRecord:
    """
    Prediction values for a single dataset sample.

    Attributes:
        segment: Dataset sample name.
        clip: Source clip name.
        t_start: Segment start time.
        t_end: Segment end time.
        hr_pos: POS-based heart rate.
        conf_pos: POS confidence.
        hr_chrom: CHROM-based heart rate.
        conf_chrom: CHROM confidence.
        hr_green: Green-channel heart rate.
        conf_green: Green-channel confidence.
        hr_label: Ground-truth heart rate.
        hr_pred: Trained HR model prediction.
        hr_pred_conf: Trained HR model confidence.
        hb_label: Ground-truth hemoglobin value.
        hb_pred: Trained hemoglobin model prediction.
    """
    segment: str
    clip: str
    t_start: float
    t_end: float
    hr_pos: float | None
    conf_pos: float | None
    hr_chrom: float | None
    conf_chrom: float | None
    hr_green: float | None
    conf_green: float | None
    hr_label: float | None
    hr_pred: float | None
    hr_pred_conf: float | None
    hb_label: float | None
    hb_pred: float | None


@dataclass(slots=True, frozen=True)
class PredictionResult:
    """
    Result returned after processing a single dataset sample.

    Attributes:
        name: Dataset sample name.
        prediction: Prediction values for the sample.
        success: True if processing completed successfully, otherwise False.
        error: Error message when processing fails.
    """
    name: str
    prediction: PredictionRecord | None
    success: bool
    error: str | None


@dataclass(slots=True, frozen=True)
class WaveformResult:
    """
    Pulse waveform and heart-rate estimate.

    Attributes:
        waveform: Extracted pulse waveform.
        heart_rate: Estimated heart rate in BPM.
        confidence: Confidence of the heart-rate estimate.
    """
    waveform: np.ndarray
    heart_rate: float | None
    confidence: float