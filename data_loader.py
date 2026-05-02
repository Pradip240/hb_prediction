"""
Data loading for Hb prediction.

Improvements over the previous version:
- Weighted sampling for the training set so rare Hb values are seen
  more often (fights the "predict the mean" failure mode).
- Stronger augmentation (color jitter, slight rotations/translations,
  random erasing) to break shortcut learning.
- Removed RandomVerticalFlip (faces are not vertically symmetric).
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


class HbDataset(Dataset):
    """CSV columns expected: image_path, hb, mrn"""

    def __init__(self, csv_file: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        hb_value = torch.tensor([float(row["hb"])], dtype=torch.float32)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (256, 256), (0, 0, 0))
            print(f"Warning: Failed to load {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, hb_value


def compute_inverse_frequency_weights(labels: np.ndarray, num_bins: int = 15) -> np.ndarray:
    """
    Higher weight for rare Hb values. Uses sqrt-inverse frequency to avoid
    extreme upweighting of outliers.
    """
    labels = np.asarray(labels, dtype=np.float32)
    hist, bin_edges = np.histogram(labels, bins=num_bins)
    bin_idx = np.digitize(labels, bin_edges[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    weights = 1.0 / np.sqrt(hist[bin_idx] + 1.0)
    # Normalize so that mean weight is 1.0
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def get_loaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Stronger training augmentation:
    # - Resize a bit larger then random-crop, so position varies slightly.
    # - Color jitter to handle lighting variation (also breaks color shortcuts).
    # - Small rotations/translations for robustness.
    # - Random erasing as an internal "occlusion" that prevents the model
    #   from latching onto a single trivial cue.
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_ds = HbDataset(train_csv, transform=train_transform)
    val_ds = HbDataset(val_csv, transform=val_transform)

    if use_weighted_sampler:
        train_labels = pd.read_csv(train_csv)["hb"].values
        sample_weights = compute_inverse_frequency_weights(train_labels)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import os

    print("Sanity check on data loader...")
    for f in ["train_data.csv", "val_data.csv"]:
        if not os.path.exists(f):
            print(f"Missing: {f}")
            exit(1)

    train_l, val_l = get_loaders("train_data.csv", "val_data.csv", batch_size=8)
    images, labels = next(iter(train_l))
    print(f"Image batch: {images.shape}, dtype {images.dtype}")
    print(f"Label batch: {labels.shape}")
    print(f"Pixel mean (post-norm): {images.mean():.4f}  std: {images.std():.4f}")
    print(f"Sampled Hb values: {labels.flatten().tolist()}")
