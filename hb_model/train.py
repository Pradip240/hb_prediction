"""Train HbMLP on engineered facial RGB and pixel-count features.

The Hb model uses low-dimensional, physically motivated features extracted from
the prepared RGB facial-region signals.

Each segment contains:

- RGB signals with shape (T, R, 3).
- Pixel counts with shape (T, R).
- Sampling frequency.

The feature extractor converts these measurements into a compact feature
vector containing colour, amplitude, AC/DC, colour-ratio, and region-quality
features.

The training procedure:

1. Load prepared segments and hemoglobin labels from the manifest.
2. Extract engineered Hb features through HbDataset.
3. Split samples by patient so segments from one patient cannot cross splits.
4. Standardize the feature vector using training-set statistics.
5. Train HbMLP using Smooth-L1 loss.
6. Track validation MAE and RMSE.
7. Restore the best validation checkpoint.
8. Evaluate on the held-out test set.
9. Save model weights, metrics, history, configuration, and diagnostic plots.

Usage:
python train.py \\
    --segments-dir output/dataset \\
    --manifest output/dataset/segments_manifest.csv \\
    --out-dir output/hb_model \\
    --epochs 100 \\
    --batch-size 64
"""

import os
import csv
import json
import argparse

import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from common.data_types import FileExtension
from hb_model.visualization import plot_history
from hb_model.dataset import HbDataset, load_dataset
from hb_model.model import HbMLP


def evaluate(
    model: HbMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the Hb model.

    Args:
        model: Trained HbMLP.
        loader: Validation or test DataLoader.
        device: Device used for inference.

    Returns:
        Tuple containing:
            - Mean absolute error in g/dL.
            - Root mean squared error in g/dL.
    """
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for features, hemoglobin in loader:
            features = features.to(device)
            predicted = model(features)
            predictions.append(predicted.cpu().numpy())
            targets.append(hemoglobin.numpy())

    if not predictions:
        return float("nan"), float("nan")

    predicted = np.concatenate(predictions)
    target = np.concatenate(targets)
    error = predicted - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return mae, rmse


def load_subjects(dataset: HbDataset) -> list[str]:
    """
    Load the patient ID for every dataset sample from the manifest.

    Args:
        dataset: Loaded HbDataset.

    Returns:
        Patient ID for each dataset sample.
    """
    # Map each segment name to its patient ID from the manifest.
    subjects_by_segment: dict[str, str] = {}

    with open(
        dataset.manifest, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            segment = row["segment"].strip()
            patient_id = row["patient_id"].strip()
            if patient_id:
                subjects_by_segment[segment] = patient_id

    subjects: list[str] = []
    # Keep patient IDs in the same order as dataset.samples.
    for segment_path, _ in dataset.samples:
        segment_name = os.path.basename(segment_path)[:-len(FileExtension.DATASET_SAMPLE)]
        patient_id = subjects_by_segment.get(segment_name)
        if patient_id is None:
            raise ValueError(f"No patient_id found in manifest for segment: {segment_name}")
        subjects.append(patient_id)
    return subjects


def split_subjects(
    subjects: list[str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split subjects into train, validation, and test sets.

    All samples belonging to one patient remain in the same split.

    Args:
        subjects: Patient ID for every dataset sample.
        seed: Random seed.
        train_ratio: Fraction of subjects assigned to training.
        val_ratio: Fraction of subjects assigned to validation.
        test_ratio: Fraction of subjects assigned to testing.

    Returns:
        Dataset indices for train, validation, and test.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0.")

    unique_subjects = np.array(sorted(set(subjects)))
    if len(unique_subjects) < 3:
        raise ValueError("At least 3 subjects are required for train/validation/test splitting.")

    # Shuffle subjects so all samples from a patient stay in the same split.
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    # Calculate the number of subjects assigned to validation and test sets.
    n_test = max(1, int(round(n_subjects * test_ratio)))
    n_val = max(1, int(round(n_subjects * val_ratio)))

    # Ensure at least one subject remains for training.
    if n_test + n_val >= n_subjects:
        n_test = 1
        n_val = 1

    # Assign subjects to each split.
    test_subjects = set(unique_subjects[:n_test])
    val_subjects = set(unique_subjects[n_test:n_test + n_val])
    train_subjects = set(unique_subjects[n_test + n_val:])

    subject_array = np.asarray(subjects)
    # Convert subject assignments into dataset sample indices.
    train_indices = np.where(np.isin(subject_array, list(train_subjects)))[0]
    val_indices = np.where(np.isin(subject_array, list(val_subjects)))[0]
    test_indices = np.where(np.isin(subject_array, list(test_subjects)))[0]
    return train_indices, val_indices, test_indices


def standardize_features(train_features: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Standardize features using statistics calculated from training data.

    Args:
        train_features: Training feature matrix.
        features: Feature matrix to transform.

    Returns:
        Standardized feature matrix.
    """
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std


def train(
    model: HbMLP,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[HbMLP, list[dict[str, float]]]:
    """
    Train HbMLP and restore the best validation checkpoint.

    Args:
        model: HbMLP to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Device used for training.
        output_dir: Directory for checkpoints and history.
        epochs: Maximum number of epochs.
        learning_rate: Initial learning rate.
        weight_decay: Adam weight decay.
        patience: Epochs without validation improvement before stopping.

    Returns:
        Tuple containing:
            - Model restored to its best validation checkpoint.
            - Training history.
    """
    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    best_weights_path = os.path.join(output_dir, "best_model.pt")
    last_weights_path = os.path.join(output_dir, "last_model.pt")
    history_path = os.path.join(output_dir, "history.csv")

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        sample_count = 0
        # Show batch-level training progress for the current epoch.
        progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", unit="batch", leave=False)
        for features, hemoglobin in progress:
            features = features.to(device)
            hemoglobin = hemoglobin.to(device)
            predicted = model(features)
            loss = criterion(predicted, hemoglobin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # type: ignore
            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            # Update the progress bar with the current batch loss.
            progress.set_postfix(loss=f"{loss.item():.4f}") # type: ignore

        train_loss = total_loss / max(1, sample_count)
        val_mae, val_rmse = evaluate(model, val_loader, device)
        scheduler.step(val_mae) # type: ignore
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Save the history after every epoch so progress is not lost.
        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "lr": current_lr
        })

        with open(history_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_mae", "val_rmse", "lr"])
            writer.writeheader()
            writer.writerows(history)

        torch.save(model.state_dict(), last_weights_path)

        # Save a separate checkpoint whenever validation MAE improves.
        if (np.isfinite(val_mae) and val_mae < best_val_mae):
            best_val_mae = val_mae
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, best_weights_path)
            tag = " *best"
        else:
            bad_epochs += 1
            tag = ""

        print(
            f"epoch {epoch:3d}/{epochs} "
            f"loss={train_loss:.4f} "
            f"val_mae={val_mae:.3f} g/dL "
            f"val_rmse={val_rmse:.3f} g/dL "
            f"lr={current_lr:.2e}"
            f"{tag}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping after {patience} epochs without validation improvement.")
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a valid best checkpoint.")

    # Restore the best validation model before returning.
    model.load_state_dict(best_state)
    model.to(device)
    print(f"best checkpoint: {best_weights_path}")
    print(f"last checkpoint: {last_weights_path}")
    print(f"history:         {history_path}")
    return model, history


def main() -> None:
    """
    Train and evaluate HbMLP.
    """
    parser = argparse.ArgumentParser(
        description=("Train hemoglobin regression model on facial RGB and pixel-count features.")
    )
    parser.add_argument("--segments-dir", default="output/dataset")
    parser.add_argument("--manifest", default="output/dataset/segments_manifest.csv")
    parser.add_argument("--out-dir", default="output/hb_model")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Reproducibility.
    torch.manual_seed(args.seed) # type: ignore
    np.random.seed(args.seed)
    device_name = (args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    print(f"device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset.
    print("Loading dataset...")
    dataset = load_dataset(args.segments_dir, args.manifest)
    if len(dataset) < 3:
        raise RuntimeError(f"Too few valid samples: {len(dataset)}")
    subjects = load_subjects(dataset)
    print(f"  samples:  {len(dataset)}")
    print(f"  subjects: {len(set(subjects))}")
    train_indices, val_indices, test_indices = split_subjects(
        subjects, args.seed, args.train_ratio, args.val_ratio, args.test_ratio
    )
    subject_array = np.asarray(subjects)
    print(f"  train: {len(train_indices)} samples / {len(set(subject_array[train_indices]))} subjects")
    print(f"  val:   {len(val_indices)} samples / {len(set(subject_array[val_indices]))} subjects")
    print(f"  test:  {len(test_indices)} samples / {len(set(subject_array[test_indices]))} subjects")
    if len(train_indices) == 0:
        raise RuntimeError("Training split is empty.")
    if len(val_indices) == 0:
        raise RuntimeError("Validation split is empty.")
    if len(test_indices) == 0:
        raise RuntimeError("Test split is empty.")

    # Inspect feature dimensionality.
    sample_features, _ = dataset[int(train_indices[0])]
    n_features = sample_features.shape[0]
    print(f"  feature count: {n_features}")

    # Data loaders.
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda"
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda"
    )

    # Create Model
    model = HbMLP(n_in=n_features).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters: {parameter_count / 1e3:.1f}k")

    # Train model
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.out_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

    # Final test evaluation
    test_mae, test_rmse = evaluate(model, test_loader, device)
    print()
    print("=== Test results ===")
    print(f"MAE:  {test_mae:.3f} g/dL")
    print(f"RMSE: {test_rmse:.3f} g/dL")

    # Save final metrics.
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_val_mae": min(entry["val_mae"] for entry in history if np.isfinite(entry["val_mae"])),
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "train_subjects": len(set(subject_array[train_indices])),
                "val_subjects": len(set(subject_array[val_indices])),
                "test_subjects": len(set(subject_array[test_indices])),
                "n_features": n_features,
                "seed": args.seed,
            },
            file,
            indent=2
        )

    # Save model configuration.
    config_path = os.path.join(args.out_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "n_features": n_features,
                "hidden_width": 512,
                "dropout": 0.3,
                "feature_names": dataset.feature_names,
            },
            file,
            indent=2
        )

    # Save final diagnostic plots.
    plot_history(history=history, test_mae=test_mae, test_rmse=test_rmse, output_dir=args.out_dir)

    # Print output files
    print()
    print("=== Output files ===")
    print(f"best weights: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"last weights: {os.path.join(args.out_dir, 'last_model.pt')}")
    print(f"history:      {os.path.join(args.out_dir, 'history.csv')}")
    print(f"metrics:      {metrics_path}")
    print(f"config:       {config_path}")


if __name__ == "__main__":
    main()