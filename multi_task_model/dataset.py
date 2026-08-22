"""
Dataset for joint HR + Hb multi-task training.

Each segment yields four items:
    signal: Channel-first z-scored time series with shape (C, T), where
        C = R * 4.
    hr_label: Heart-rate target in BPM with shape ().
    hb_label: Hemoglobin target in g/dL with shape ().
    hb_features: Engineered colour and AC-DC feature vector with shape (F,).

The input archives contain RGB signals with shape (T, R, 3) and pixel counts
with shape (T, R). The signal preprocessing converts these into four features
per facial region:
    (T, R, 3) + (T, R, 1) -> (T, R, 4) -> (R, 4, T) -> (R * 4, T)

The signal preprocessing matches SpectralDataset.prepare_signal so that the
shared multi-task trunk receives the same representation used by the
single-task HR model.

The engineered hemoglobin features are extracted from the original signals
and pixel counts using the same feature extraction pipeline used by the Hb
model. Features are cached per segment because extraction is relatively
expensive.

Both HR and hemoglobin labels are required for a segment to be included.
The HR label prefers the PPG-derived value when available and falls back to
the pulse value otherwise.
"""

import csv
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common import config
from common.data_types import FileExtension
from hb_model.features import extract_features, feature_names


class MultiTaskDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """
    Dataset for joint heart-rate and hemoglobin training.

    Each sample contains the preprocessed facial signal, an HR target, an
    Hb target, and engineered features for the hemoglobin prediction head.

    Args:
        segments_dir: Directory containing prepared signal archives.
        manifest: Path to the CSV manifest containing segment and target
            information.
        hr_min_bpm: Minimum inclusive heart-rate value to include, in BPM.
        hr_max_bpm: Maximum inclusive heart-rate value to include, in BPM.
        hb_min: Minimum inclusive hemoglobin value to include, in g/dL.
        hb_max: Maximum inclusive hemoglobin value to include, in g/dL.
    """

    def __init__(
        self,
        segments_dir: str,
        manifest: str,
        hr_min_bpm: float,
        hr_max_bpm: float,
        hb_min: float = 0.0,
        hb_max: float = 30.0,
    ) -> None:
        """
        Initialize the multi-task dataset from a segment manifest.

        Only segments with valid files, valid HR and Hb labels, and labels
        inside the configured ranges are included.

        Args:
            segments_dir: Directory containing the dataset segment files.
            manifest: Path to the CSV manifest containing segment and target
                information.
            hr_min_bpm: Minimum inclusive heart-rate value to include, in BPM.
            hr_max_bpm: Maximum inclusive heart-rate value to include, in BPM.
            hb_min: Minimum inclusive hemoglobin value to include, in g/dL.
            hb_max: Maximum inclusive hemoglobin value to include, in g/dL.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
        """
        super().__init__()

        self.segments_dir = segments_dir
        self.manifest = manifest
        self.feature_names = feature_names(list(config.REGION_ORDER))
        self.samples: list[tuple[str, float, float]] = []  # (path, hr, hb)

        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        # Load valid segment paths and both training targets from the manifest.
        with open(manifest, encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                segment = row["segment"].strip()
                hr_raw = (row.get("ppg_hr") or row.get("pulse") or "").strip()
                hb_raw = (row.get("hemoglobin") or "").strip()

                # Both targets are required for multi-task training.
                if not segment or not hr_raw or not hb_raw:
                    continue

                try:
                    heart_rate = float(hr_raw)
                    hemoglobin = float(hb_raw)
                except ValueError:
                    continue

                # Reject non-finite labels before applying the configured ranges.
                if not (np.isfinite(heart_rate) and np.isfinite(hemoglobin)):
                    continue

                # Keep only samples inside the configured training ranges.
                if not (hr_min_bpm <= heart_rate <= hr_max_bpm):
                    continue
                if not (hb_min <= hemoglobin <= hb_max):
                    continue

                segment_path = os.path.join(
                    segments_dir,
                    f"{segment}{FileExtension.DATASET_SAMPLE}",
                )

                # Ignore manifest entries whose prepared segment is missing.
                if not os.path.isfile(segment_path):
                    continue

                self.samples.append((segment_path, heart_rate, hemoglobin))

        # Cache engineered Hb features because extraction is relatively
        # expensive and the same segment may be accessed multiple times.
        self._feature_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        """Return the number of valid multi-task training samples."""
        return len(self.samples)

    @staticmethod
    def _prepare_signal(signals: np.ndarray, pixel_counts: np.ndarray) -> np.ndarray:
        """
        Convert raw RGB signals and pixel counts into model input channels.

        The input contains RGB signals with shape (T, R, 3) and pixel counts
        with shape (T, R). Pixel counts are appended as a fourth feature for
        each region before converting to channel-first format.

        The resulting representation is:
            (T, R, 3) + (T, R, 1) -> (T, R, 4)
            -> (R, 4, T) -> (R * 4, T)

        Each resulting channel is independently z-score normalized over time.

        Args:
            signals: RGB signals with shape (T, R, 3).
            pixel_counts: Valid skin-pixel counts with shape (T, R).

        Returns:
            Preprocessed signal with shape (R * 4, T).
        """
        # Replace NaN RGB values from undetected regions with zero.
        signals = np.nan_to_num(signals, nan=0.0)

        # Add pixel count as the fourth feature for every facial region.
        features = np.concatenate([signals, pixel_counts[..., None]], axis=-1)

        # Convert from time-major region format to region-major channel format.
        features = np.transpose(features, (1, 2, 0))  # (R, 4, T)
        # Flatten region and feature dimensions for the convolutional model.
        features = features.reshape(features.shape[0] * features.shape[1], features.shape[2])  # (R * 4, T)

        # Normalize every input channel independently over time.
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        return (features - mean) / (std + 1e-6)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Load one multi-task training sample.

        The signal uses the same preprocessing as the single-task HR dataset.
        Engineered Hb features are extracted from the original unnormalized
        signals and pixel counts, then cached for subsequent accesses.

        Args:
            index: Dataset sample index.

        Returns:
            Tuple containing:
                - Input signal with shape (R * 4, T).
                - Heart-rate target with shape () in BPM.
                - Hemoglobin target with shape () in g/dL.
                - Engineered Hb features with shape (F,).
        """
        segment_path, heart_rate, hemoglobin = self.samples[index]

        # Load the prepared segment archive.
        data = np.load(segment_path)
        signals = np.asarray(data["signals"], dtype=np.float32)
        pixel_counts = np.asarray(data["pixel_counts"], dtype=np.float32)
        fps = float(data["fps"])

        # Prepare the signal using the same representation as the HR dataset.
        signal = self._prepare_signal(signals, pixel_counts)

        # Reuse cached Hb features when this segment has already been processed.
        if segment_path in self._feature_cache:
            hb_features = self._feature_cache[segment_path]
        else:
            # Feature extraction expects region-first signals and pixel counts.
            hb_features = extract_features(
                signals=np.transpose(signals, (1, 0, 2)),
                pixel_counts=np.transpose(pixel_counts, (1, 0)),
                fps=fps,
                region_order=list(config.REGION_ORDER),
            ).astype(np.float32)

            self._feature_cache[segment_path] = hb_features

        return (
            torch.from_numpy(signal),  # type: ignore
            torch.tensor(heart_rate, dtype=torch.float32),
            torch.tensor(hemoglobin, dtype=torch.float32),
            torch.from_numpy(hb_features),  # type: ignore
        )


def load_dataset(
    segments_dir: str,
    manifest: str,
) -> MultiTaskDataset:
    """
    Create the multi-task HR + Hb training dataset.

    Args:
        segments_dir: Directory containing prepared signal archives.
        manifest: Path to the segment manifest CSV file.

    Returns:
        Dataset ready to be passed to a PyTorch DataLoader.
    """
    return MultiTaskDataset(
        segments_dir=segments_dir,
        manifest=manifest,
        hr_min_bpm=config.HR_FREQ_MIN_HZ * 60.0,
        hr_max_bpm=config.HR_FREQ_MAX_HZ * 60.0,
    )
