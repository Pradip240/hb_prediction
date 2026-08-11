"""Dataset for training the spectral heart-rate model.

The prepared segment archives contain:
- signals: RGB signals with shape (T, R, 3).
- pixel_counts: Valid skin-pixel counts with shape (T, R).
- fps: Sampling frequency of the segment.

The dataset combines the three RGB channels with the pixel-count channel
to produce four features per facial region:
    (T, R, 3) + (T, R, 1) -> (T, R, 4)

For HRSpectralNet, the data is converted to channel-first format:
    (T, R, 4) -> (R * 4, T)

The heart-rate target uses the PPG-derived heart rate when available,
falling back to the ground-truth pulse value otherwise.
"""

import os
import csv

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from common import config
from common.data_types import FileExtension


class SpectralDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Dataset for HRSpectralNet training.

    Each sample is returned as:
        signal: (C, T)
        heart_rate: ()

    where C = R * 4.

    The four channels for each region are:
        R, G, B, pixel_count

    The input archives themselves contain RGB signals with shape (T, R, 3)
    and pixel counts with shape (T, R). The fourth feature is constructed
    from pixel_counts when the sample is loaded.

    Args:
        segments_dir: Directory containing prepared signal archives.
        manifest: Path to the segment manifest CSV file.
        min_bpm: Minimum accepted heart-rate label.
        max_bpm: Maximum accepted heart-rate label.
    """

    def __init__(self, segments_dir: str, manifest: str, min_bpm: float, max_bpm: float) -> None:
        super().__init__()

        self.segments_dir = segments_dir
        self.manifest = manifest
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.samples: list[tuple[str, float]] = []  # (file_path, hr)

        if not os.path.isfile(self.manifest):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest}")

        # Load segment paths and heart-rate targets from the manifest.
        with open(self.manifest, encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                segment = os.path.join(segments_dir, f'{row["segment"]}{FileExtension.DATASET_SAMPLE}')
                # Prefer the PPG-derived HR when available.
                heart_rate = row["ppg_hr"] or row["pulse"]
                if not heart_rate:
                    continue
                heart_rate = float(heart_rate)
                if not self.min_bpm <= heart_rate <= self.max_bpm:
                    continue
                if not os.path.isfile(segment):
                    continue
                self.samples.append((segment, heart_rate))


    def __len__(self) -> int:
        """
        Return the number of valid training samples.
        """
        return len(self.samples)


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Load one training sample.

        Returns:
            Tuple containing:
                - Input signal with shape (R * 4, T).
                - Heart-rate target with shape ().
        """
        path, heart_rate = self.samples[index]

        data = np.load(path)
        signals = np.asarray(data["signals"], dtype=np.float32)
        pixel_counts = np.asarray(data["pixel_counts"], dtype=np.float32)

        if signals.ndim != 3:
            raise ValueError(f"Expected signals with shape (T, R, 3), got {signals.shape} in {path}.")
        if signals.shape[-1] != 3:
            raise ValueError(f"Expected RGB signals with 3 channels, got {signals.shape[-1]} in {path}.")
        if pixel_counts.ndim != 2:
            raise ValueError(f"Expected pixel_counts with shape (T, R), got {pixel_counts.shape} in {path}.")
        if signals.shape[:2] != pixel_counts.shape:
            raise ValueError(f"signals and pixel_counts shapes do not match: {signals.shape} vs {pixel_counts.shape}.")

        # Replace NaN RGB values from undetected regions with zero.
        signals = np.nan_to_num(signals, nan=0.0)

        # Add pixel count as the fourth feature for every region.
        features = np.concatenate([signals, pixel_counts[..., None]], axis=-1)
        # Convert: (T, R, 4) to (R, 4, T)
        features = np.transpose(features, (1, 2, 0))
        # Flatten region and feature dimensions: (R, 4, T) -> (R * 4, T)
        features = features.reshape(features.shape[0] * features.shape[1], features.shape[2])

        # Normalize every input channel independently over time.
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        features = (features - mean) / (std + 1e-6)
        return torch.from_numpy(features), torch.tensor(heart_rate, dtype=torch.float32) # type: ignore


def load_dataset(segments_dir: str, manifest: str) -> SpectralDataset:
    """
    Create the spectral HR training dataset.

    Args:
        segments_dir: Directory containing prepared signal archives.
        manifest: Path to the segment manifest CSV file.

    Returns:
        Dataset ready to be passed to a PyTorch DataLoader.
    """
    return SpectralDataset(
        segments_dir=segments_dir,
        manifest=manifest,
        min_bpm=config.HR_FREQ_MIN_HZ * 60.0,
        max_bpm=config.HR_FREQ_MAX_HZ * 60.0
    )
