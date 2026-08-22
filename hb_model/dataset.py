"""
Dataset for training the hemoglobin regression model.

The prepared segment archives contain:

- signals: RGB signals with shape (T, R, 3).
- pixel_counts: Valid skin-pixel counts with shape (T, R).
- fps: Sampling frequency of the segment.

Hemoglobin is a segment-level target stored directly in the segment
manifest. Each segment therefore receives the hemoglobin value from
its own manifest row.

The dataset converts each segment into an engineered feature vector
using features.extract_features(). The resulting features describe
RGB colour, pulsatile amplitude, colour ratios, and region pixel-count
quality.
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


class HbDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Dataset for hemoglobin regression.

    Each sample returns:
        features: Engineered Hb features with shape (F,).
        hemoglobin: Hemoglobin target with shape ().

    Args:
        segments_dir: Directory containing prepared segment archives.
        manifest: CSV containing segment and hemoglobin information.
        min_hb: Minimum accepted hemoglobin value.
        max_hb: Maximum accepted hemoglobin value.
    """

    def __init__(self, segments_dir: str, manifest: str, min_hb: float, max_hb: float) -> None:
        """
        Initialize the dataset and load or create its feature cache.

        Args:
            segments_dir: Directory containing the dataset segment files.
            manifest: Path to the manifest containing segment paths and hemoglobin targets.
            min_hb: Minimum inclusive hemoglobin value to include.
            max_hb: Maximum inclusive hemoglobin value to include.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
        """
        super().__init__()

        self.segments_dir = segments_dir
        self.manifest = manifest
        self.min_hb = min_hb
        self.max_hb = max_hb
        self.features: np.ndarray
        self.targets: np.ndarray

        # Feature names are useful when inspecting or saving the model configuration later.
        self.feature_names = feature_names(list(config.REGION_ORDER))
        self.samples: list[tuple[str, float]] = []  # (path, hemoglobin)

        if not os.path.isfile(self.manifest):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest}")

        self._load_samples()
        self._load_or_create_feature_cache()

    def _load_samples(self) -> None:
        """Load valid segment paths and hemoglobin labels."""
        with open(self.manifest, encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {self.manifest}")
            if "segment" not in reader.fieldnames:
                raise ValueError("Manifest is missing required 'segment' column.")
            if "hemoglobin" not in reader.fieldnames:
                raise ValueError("Manifest is missing required 'hemoglobin' column.")

            for row in reader:
                segment_name = row["segment"].strip()
                value = row["hemoglobin"].strip()

                if not segment_name or not value:
                    continue
                try:
                    hemoglobin = float(value)
                except ValueError:
                    continue
                if not np.isfinite(hemoglobin):
                    continue

                # Apply the configured Hb range.
                if not self.min_hb <= hemoglobin <= self.max_hb:
                    continue

                segment_path = os.path.join(self.segments_dir, f"{segment_name}{FileExtension.DATASET_SAMPLE}")
                if not os.path.isfile(segment_path):
                    continue

                self.samples.append((segment_path, hemoglobin))

    def _load_or_create_feature_cache(self) -> None:
        """
        Load engineered features from disk or create the cache.

        The cache contains the raw engineered features before any
        train-set normalization. This is important because normalization
        statistics must be calculated from the training split only.
        """
        cache_path = os.path.join(self.segments_dir, "hb_features_cache.npz")

        if os.path.isfile(cache_path):
            print(f"Loading cached Hb features: {cache_path}")
            cache = np.load(cache_path)
            self.features = np.asarray(cache["features"], dtype=np.float32)
            self.targets = np.asarray(cache["targets"], dtype=np.float32)
            if self.features.shape[0] != len(self.samples):
                raise ValueError(
                    "Cached feature count does not match dataset samples: "
                    f"{self.features.shape[0]} vs {len(self.samples)}."
                )
            if self.features.shape[1] != len(self.feature_names):
                raise ValueError(
                    "Cached feature count does not match feature names: "
                    f"{self.features.shape[1]} vs {len(self.feature_names)}."
                )
            if self.targets.shape[0] != len(self.samples):
                raise ValueError(
                    "Cached target count does not match dataset samples: "
                    f"{self.targets.shape[0]} vs {len(self.samples)}."
                )
            return

        print("Creating Hb feature cache...")
        features_list: list[np.ndarray] = []
        targets_list: list[float] = []
        for sample_index, (path, hemoglobin) in enumerate(self.samples, start=1):
            print(f"    extracting features: {sample_index}/{len(self.samples)}", flush=True)
            data = np.load(path)
            signals = np.asarray(data["signals"], dtype=np.float32)
            pixel_counts = np.asarray(data["pixel_counts"], dtype=np.float32)
            fps = float(data["fps"])
            if signals.ndim != 3:
                raise ValueError(f"Expected signals with shape (T, R, 3), got {signals.shape} in {path}.")
            if signals.shape[-1] != 3:
                raise ValueError(f"Expected RGB signals with 3 channels, got {signals.shape[-1]} in {path}.")
            if pixel_counts.ndim != 2:
                raise ValueError(f"Expected pixel_counts with shape (T, R), got {pixel_counts.shape} in {path}.")
            if signals.shape[:2] != pixel_counts.shape:
                raise ValueError(
                    f"signals and pixel_counts shapes do not match: {signals.shape} vs {pixel_counts.shape}."
                )

            # The archive stores:
            #   signals:      (T, R, 3)
            #   pixel_counts: (T, R)
            #
            # Feature extraction expects:
            #   signals:      (R, T, 3)
            #   pixel_counts: (R, T)
            features = extract_features(
                signals=np.transpose(signals, (1, 0, 2)),
                pixel_counts=np.transpose(pixel_counts, (1, 0)),
                fps=fps,
                region_order=list(config.REGION_ORDER),
            )
            if not np.all(np.isfinite(features)):
                raise ValueError(f"Non-finite Hb features generated for {path}.")
            if features.shape[0] != len(self.feature_names):
                raise ValueError(
                    f"Feature count does not match feature names: {features.shape[0]} vs {len(self.feature_names)}."
                )
            features_list.append(features.astype(np.float32))
            targets_list.append(float(hemoglobin))

        print()
        self.features = np.stack(features_list, axis=0).astype(np.float32)
        self.targets = np.asarray(targets_list, dtype=np.float32)
        np.savez_compressed(cache_path, features=self.features, targets=self.targets)
        print(f"Saved Hb feature cache: {cache_path}")

    def __len__(self) -> int:
        """Return the number of valid training samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Load and convert one segment into Hb features.

        Args:
            index: Dataset sample index.

        Returns:
            Tuple containing:
                - Feature tensor with shape (F,).
                - Hemoglobin target with shape ().
        """
        features = self.features[index]
        hemoglobin = self.targets[index]
        return torch.from_numpy(features), torch.tensor(hemoglobin, dtype=torch.float32)  # type: ignore


def load_dataset(segments_dir: str, manifest: str) -> HbDataset:
    """
    Create the hemoglobin training dataset.

    Args:
        segments_dir: Directory containing prepared segment archives.
        manifest: Segment manifest containing hemoglobin labels.

    Returns:
        Dataset ready for use with a PyTorch DataLoader.
    """
    return HbDataset(segments_dir=segments_dir, manifest=manifest, min_hb=0.0, max_hb=30.0)
