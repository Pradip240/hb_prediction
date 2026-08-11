"""Train HRSpectralNet on facial RGB signals and pixel-count features.

The model input is constructed from the saved per-region signals:
    signals:      (T, R, 3)
    pixel_counts: (T, R)

Each region contributes four temporal channels:
    R, G, B, pixel-count

Therefore, with three facial regions, the model receives 12 channels:
    (B, 12, T)

Pixel-count channels are log-transformed and z-score normalized before being
passed to the model. RGB channels are z-score normalized per segment.

The training procedure:
1. Load labeled segments and subject-wise train/validation/test splits.
2. Build the 12-channel model input from RGB signals and pixel counts.
3. Train HRSpectralNet using its combined Smooth-L1 + soft cross-entropy loss.
4. Early-stop using validation MAE.
5. Restore the best validation checkpoint.
6. Evaluate on held-out subjects.
7. Save metrics, training history, model weights, and diagnostic plots.

Usage:
    python train.py \
        --segments-dir output/train_data \
        --ppg-dir data/ppg \
        --out-dir output/hr_model \
        --epochs 80 \
        --batch-size 128
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
from hr_model.model import HRSpectralNet
from hr_model.dataset import SpectralDataset, load_dataset
from hr_model.visualize import plot_history


def evaluate(
    model: HRSpectralNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the model using mean absolute error.

    Args:
        model: Trained HRSpectralNet.
        loader: Validation or test DataLoader.
        device: Device used for inference.

    Returns:
        Tuple containing:
            - Mean absolute error in BPM.
            - Percentage of predictions within 6 BPM.
    """
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for signals, heart_rates in loader:
            signals = signals.to(device)
            # Run inference without computing gradients.
            predicted_bpm, _, _ = model(signals)
            predictions.append(predicted_bpm.cpu().numpy())
            targets.append(heart_rates.numpy())

    # Return NaN metrics when the loader contains no samples.
    if not predictions:
        return float("nan"), float("nan")

    predicted = np.concatenate(predictions)
    target = np.concatenate(targets)

    # Calculate absolute prediction error in BPM.
    absolute_error = np.abs(predicted - target)

    mae = float(np.mean(absolute_error))
    within_6 = float(np.mean(absolute_error <= 6.0) * 100.0)
    return mae, within_6


def load_subjects(dataset: SpectralDataset) -> list[str]:
    """
    Load the patient ID for every dataset sample from the manifest.

    Args:
        dataset: Loaded SpectralDataset.

    Returns:
        Patient ID for each dataset sample.
    """
    # Map each segment name to its patient ID from the manifest.
    subjects_by_segment: dict[str, str] = {}

    with open(dataset.manifest, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            segment = row["segment"]
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
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split subjects into train, validation, and test sets.

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


def train(
    model: HRSpectralNet,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[HRSpectralNet, list[dict[str, float]]]:
    """
    Train HRSpectralNet and restore the best validation checkpoint.

    Args:
        model: HRSpectralNet to train.
        train_loader: DataLoader containing training samples.
        val_loader: DataLoader containing validation samples.
        device: Device used for training.
        output_dir: Directory for checkpoints and training history.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: Adam weight decay.
        patience: Number of epochs without validation improvement before stopping.

    Returns:
        Tuple containing:
            - Model restored to its best validation checkpoint.
            - Training history for every completed epoch.
    """
    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

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
        for signals, heart_rates in progress:
            signals = signals.to(device)
            heart_rates = heart_rates.to(device)
            # Forward pass and combined spectral/regression loss.
            predicted_bpm, logits, _ = model(signals)
            loss, _ = model.loss(predicted_bpm, logits, heart_rates)
            optimizer.zero_grad()
            loss.backward() # type: ignore
            optimizer.step() # type: ignore
            batch_size = signals.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            # Update the progress bar with the current batch loss.
            progress.set_postfix(loss=f"{loss.item():.4f}") # type: ignore
        train_loss = total_loss / max(1, sample_count)

        # Validation BPM accuracy is used for model selection.
        val_mae, val_within_6 = evaluate(model, val_loader, device)
        scheduler.step(val_mae) # type: ignore
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Always save the most recent model state.
        torch.save(model.state_dict(), last_weights_path)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_mae": val_mae,
                "val_within_6": val_within_6,
                "lr": current_lr,
            }
        )
        # Save the history after every epoch so progress is not lost.
        with open(history_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["epoch", "train_loss", "val_mae", "val_within_6", "lr"]
            )
            writer.writeheader()
            writer.writerows(history)

        # Save a separate checkpoint whenever validation MAE improves.
        if np.isfinite(val_mae) and val_mae < best_val_mae:
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
            f"val_mae={val_mae:.2f} BPM "
            f"val_w6={val_within_6:.1f}% "
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
    Train and evaluate HRSpectralNet.
    """
    parser = argparse.ArgumentParser(
        description="Train spectral HR model on facial RGB + pixel-count segments."
    )
    parser.add_argument("--segments-dir", default="output/dataset")
    parser.add_argument("--manifest", default="output/dataset/segments_manifest.csv")
    parser.add_argument("--out-dir", default="output/hr_model")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Reproducibility.
    torch.manual_seed(args.seed) # type: ignore
    np.random.seed(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device}")

    # Load dataset.
    print("Loading dataset...")
    dataset = load_dataset(args.segments_dir, args.manifest)
    if len(dataset) < 3:
        raise RuntimeError(f"Too few valid samples: {len(dataset)}")
    subjects = load_subjects(dataset)
    print(f"  samples:  {len(dataset)}")
    print(f"  subjects: {len(set(subjects))}")

    # Subject-wise split
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

    # Data loaders
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    # Create Model
    sample_signal, _ = dataset[train_indices[0]]
    n_channels = sample_signal.shape[0]
    print(f"  input shape:    {tuple(sample_signal.shape)}")
    print(f"  input channels: {n_channels}")
    model = HRSpectralNet(n_channels=n_channels).to(device)
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
        patience=args.patience,
    )

    # Evaluate the restored best model on held-out test subjects
    test_mae, test_within_6 = evaluate(model, test_loader, device)
    print()
    print("=== Test results ===")
    print(f"MAE:      {test_mae:.2f} BPM")
    print(f"Within 6: {test_within_6:.1f}%")

    # Save final metrics
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    best_val_mae = min(entry["val_mae"] for entry in history if np.isfinite(entry["val_mae"]))
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_val_mae": best_val_mae,
                "test_mae": test_mae,
                "test_within_6": test_within_6,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "train_subjects": len(set(subject_array[train_indices])),
                "val_subjects": len(set(subject_array[val_indices])),
                "test_subjects": len(set(subject_array[test_indices])),
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "n_channels": n_channels,
                "nfft": model.nfft,
                "hr_min": float(model.band_bpm[0].item()), # type: ignore
                "hr_max": float(model.band_bpm[-1].item()), # type: ignore
                "seed": args.seed,
            },
            file,
            indent=2
        )

    # Save model configuration
    config_path = os.path.join(args.out_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "n_channels": n_channels,
                "fps": model.fps,
                "nfft": model.nfft,
                "hr_min": float(model.band_bpm[0].item()), # type: ignore
                "hr_max": float(model.band_bpm[-1].item()), # type: ignore
                "input_layout": "(B, R*4, T)",
            },
            file,
            indent=2
        )

    # Generate training and test plots
    plot_history(history=history, test_mae=test_mae, test_within_6=test_within_6, output_dir=args.out_dir)

    # Print output files
    best_weights_path = os.path.join(args.out_dir, "best_model.pt")
    last_weights_path = os.path.join(args.out_dir, "last_model.pt")
    history_path = os.path.join(args.out_dir, "history.csv")
    print()
    print("=== Output files ===")
    print(f"best weights: {best_weights_path}")
    print(f"last weights: {last_weights_path}")
    print(f"history:      {history_path}")
    print(f"metrics:      {metrics_path}")
    print(f"config:       {config_path}")


if __name__ == "__main__":
    main()