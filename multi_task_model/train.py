"""
Train MultiTaskHRHb for joint heart-rate and hemoglobin estimation.

The model learns two related tasks from each facial segment:

    1. Heart rate (HR), which provides the dominant training signal.
    2. Hemoglobin (Hb), which is treated as the auxiliary task.

HR acts as the anchor task so that the shared representation is primarily
guided by the stronger HR signal while still allowing the Hb task to learn
useful information from the shared features. The Hb loss is scaled by
``--hb-weight`` and defaults to 0.1.

Dataset samples are expected to contain:

    signal:
        Model input tensor containing the facial signal channels.
    heart_rate:
        Ground-truth HR target in BPM.
    hemoglobin:
        Ground-truth Hb target.
    hb_features:
        Engineered Hb features.

Subjects are split before training so that samples from the same subject
cannot appear in multiple splits. This prevents subject-level data leakage.

Hb feature normalization is computed using training samples only. The
resulting mean and standard deviation are saved to ``feature_norm.npz`` so
the same normalization can be reused during inference.

Training uses the combined loss implemented by ``MultiTaskHRHb.loss()``.
Validation Hb R2 is the primary checkpoint-selection metric because Hb is the
target task. HR MAE and percentage within 6 BPM are logged every epoch to
monitor whether improving Hb performance causes unacceptable HR degradation.

If validation Hb R2 is undefined because the validation targets are
degenerate, validation Hb MAE is used as a fallback checkpoint metric.

The best validation checkpoint is restored before evaluating the held-out
test subjects.

Outputs:

    best_model.pt:
        Model weights from the best validation checkpoint.

    history.csv:
        Per-epoch training loss, validation metrics, and learning rate.

    metrics.json:
        Final test metrics and training configuration.

    feature_norm.npz:
        Training-set Hb feature mean and standard deviation.

Usage:
    python train.py \
        --segments-dir output/dataset \
        --manifest output/dataset/segments_manifest.csv \
        --out-dir output/mtl_model \
        --epochs 120 \
        --batch-size 128
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

from common import config
from common.data_types import FileExtension
from multi_task_model.dataset import MultiTaskDataset, load_dataset
from multi_task_model.model import MultiTaskHRHb


def load_subjects(dataset: MultiTaskDataset) -> list[str]:
    """
    Load the subject ID corresponding to every dataset sample.

    The dataset manifest maps each segment name to a patient ID. Dataset
    samples are then matched against that manifest so the returned subject
    list has the same order as ``dataset.samples``.

    Args:
        dataset: Loaded multi-task dataset containing the manifest path and
            dataset samples.

    Returns:
        Subject ID for every dataset sample.

    Raises:
        ValueError: If a dataset segment has no patient ID in the manifest.
    """
    subjects_by_segment: dict[str, str] = {}

    with open(dataset.manifest, encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)

        for row in reader:
            segment_name = row["segment"].strip()
            subject_id = row["patient_id"].strip()

            if subject_id:
                subjects_by_segment[segment_name] = subject_id

    subjects: list[str] = []

    for sample_path, _, _ in dataset.samples:
        segment_name = os.path.basename(sample_path)[: -len(FileExtension.DATASET_SAMPLE)]
        subject_id = subjects_by_segment.get(segment_name)

        if subject_id is None:
            raise ValueError(f"No patient_id found in manifest for segment: {segment_name}")

        subjects.append(subject_id)

    return subjects


def split_subjects(
    subjects: list[str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset samples into subject-independent train/validation/test sets.

    All samples belonging to the same subject are assigned to exactly one
    split. This prevents samples from the same subject from appearing in both
    training and evaluation data.

    Args:
        subjects: Subject ID corresponding to every dataset sample.
        seed: Random seed used to shuffle unique subjects.
        train_ratio: Target fraction of subjects assigned to training.
        val_ratio: Target fraction of subjects assigned to validation.
        test_ratio: Target fraction of subjects assigned to testing.

    Returns:
        Three arrays containing dataset indices for training, validation, and
        testing, respectively.

    Raises:
        ValueError: If the split ratios do not sum to 1.0 or fewer than three
            subjects are available.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0.")

    unique_subjects = np.array(sorted(set(subjects)))

    if len(unique_subjects) < 3:
        raise ValueError("At least 3 subjects are required for train/validation/test splitting.")

    # Shuffle subjects rather than samples so subject-level separation is preserved.
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    subject_count = len(unique_subjects)
    test_subject_count = max(1, int(round(subject_count * test_ratio)))
    val_subject_count = max(1, int(round(subject_count * val_ratio)))

    # Reserve at least one subject for training.
    if test_subject_count + val_subject_count >= subject_count:
        test_subject_count = 1
        val_subject_count = 1

    test_subjects = set(unique_subjects[:test_subject_count])
    val_subjects = set(unique_subjects[test_subject_count : test_subject_count + val_subject_count])
    train_subjects = set(unique_subjects[test_subject_count + val_subject_count :])

    subject_array = np.asarray(subjects)

    train_indices = np.where(np.isin(subject_array, list(train_subjects)))[0]
    val_indices = np.where(np.isin(subject_array, list(val_subjects)))[0]
    test_indices = np.where(np.isin(subject_array, list(test_subjects)))[0]

    return train_indices, val_indices, test_indices


def compute_feature_norm(
    dataset: MultiTaskDataset,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation for Hb features.

    Statistics are calculated using the training split only. Using validation
    or test samples here would leak information from evaluation data into the
    model input preprocessing.

    Args:
        dataset: Loaded multi-task dataset.
        train_indices: Dataset indices belonging to the training split.

    Returns:
        Tuple containing:
            - Feature mean as a float32 NumPy array.
            - Feature standard deviation as a float32 NumPy array.

    Raises:
        ValueError: If the training split contains no samples.
    """
    if len(train_indices) == 0:
        raise ValueError("Cannot compute feature normalization from an empty training split.")

    training_features: list[np.ndarray] = []

    for sample_index in train_indices:
        _, _, _, hb_features = dataset[int(sample_index)]
        training_features.append(hb_features.numpy())

    feature_array = np.stack(training_features, axis=0)
    feature_mean = feature_array.mean(axis=0)
    feature_std = feature_array.std(axis=0)

    # Avoid division by zero for constant engineered features.
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    return feature_mean.astype(np.float32), feature_std.astype(np.float32)


def evaluate(
    model: MultiTaskHRHb,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate HR and Hb performance on a dataset split.

    Hb features are normalized using the training-set statistics supplied by
    ``feature_mean`` and ``feature_std``.

    Args:
        model: Trained MultiTaskHRHb model.
        loader: Validation or test DataLoader.
        device: Device used for inference.
        feature_mean: Training-set Hb feature mean.
        feature_std: Training-set Hb feature standard deviation.

    Returns:
        Dictionary containing:
            - ``hr_mae``: HR mean absolute error in BPM.
            - ``hr_w6``: Percentage of HR predictions within 6 BPM.
            - ``hb_mae``: Hb mean absolute error.
            - ``hb_r2``: Hb coefficient of determination.
            - ``hb_corr``: Pearson correlation between predicted and target Hb.
    """
    model.eval()

    hr_predictions: list[np.ndarray] = []
    hr_targets: list[np.ndarray] = []
    hb_predictions: list[np.ndarray] = []
    hb_targets: list[np.ndarray] = []

    normalized_mean = torch.tensor(feature_mean, device=device)
    normalized_std = torch.tensor(feature_std, device=device)

    with torch.no_grad():
        for signal, heart_rate, hemoglobin, hb_features in loader:
            signal = signal.to(device)
            hb_features = (hb_features.to(device) - normalized_mean) / normalized_std

            predicted_hr_bpm, _, _, predicted_hb = model(signal, hb_features)  # type: ignore

            hr_predictions.append(predicted_hr_bpm.cpu().numpy())
            hr_targets.append(heart_rate.numpy())
            hb_predictions.append(predicted_hb.cpu().numpy())
            hb_targets.append(hemoglobin.numpy())

    if not hr_predictions:
        return {
            "hr_mae": float("nan"),
            "hr_w6": float("nan"),
            "hb_mae": float("nan"),
            "hb_r2": float("nan"),
            "hb_corr": float("nan"),
        }

    predicted_hr = np.concatenate(hr_predictions)
    target_hr = np.concatenate(hr_targets)
    predicted_hb = np.concatenate(hb_predictions)
    target_hb = np.concatenate(hb_targets)

    hr_absolute_error = np.abs(predicted_hr - target_hr)
    hr_mae = float(np.mean(hr_absolute_error))
    hr_within_6 = float(np.mean(hr_absolute_error <= 6.0) * 100.0)

    hb_error = predicted_hb - target_hb
    hb_mae = float(np.mean(np.abs(hb_error)))

    residual_sum_of_squares = float(np.sum(hb_error**2))
    total_sum_of_squares = float(np.sum((target_hb - target_hb.mean()) ** 2))

    hb_r2 = 1.0 - residual_sum_of_squares / total_sum_of_squares if total_sum_of_squares > 1e-12 else float("nan")

    hb_corr = (
        float(np.corrcoef(predicted_hb, target_hb)[0, 1])
        if predicted_hb.std() > 1e-12 and target_hb.std() > 1e-12
        else float("nan")
    )

    return {
        "hr_mae": hr_mae,
        "hr_w6": hr_within_6,
        "hb_mae": hb_mae,
        "hb_r2": hb_r2,
        "hb_corr": hb_corr,
    }


def save_history(
    history: list[dict[str, float]],
    history_path: str,
) -> None:
    """
    Save the accumulated training history to CSV.

    Args:
        history: Per-epoch training and validation metrics.
        history_path: Destination CSV path.
    """
    if not history:
        return

    with open(history_path, "w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    """Train, validate, and test the joint HR + Hb multi-task model."""
    parser = argparse.ArgumentParser(description="Train MultiTaskHRHb for joint heart-rate and hemoglobin estimation.")
    parser.add_argument("--segments-dir", default="output/dataset")
    parser.add_argument("--manifest", default="output/dataset/segments_manifest.csv")
    parser.add_argument("--out-dir", default="output/mtl_model")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--hb-weight",
        type=float,
        default=0.1,
        help="Weight applied to the Hb loss. HR loss remains weighted at 1.0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Set random seeds before creating the dataset, loaders, or model.
    torch.manual_seed(args.seed)  # type: ignore
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"device: {device}")

    # Load the complete dataset before performing the subject-level split.
    print("Loading dataset...")
    dataset = load_dataset(args.segments_dir, args.manifest)

    if len(dataset) < 3:
        raise RuntimeError(f"Too few valid samples: {len(dataset)}")

    subjects = load_subjects(dataset)

    train_indices, val_indices, test_indices = split_subjects(
        subjects=subjects,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    subject_array = np.asarray(subjects)

    train_subject_count = len(set(subject_array[train_indices]))
    val_subject_count = len(set(subject_array[val_indices]))
    test_subject_count = len(set(subject_array[test_indices]))

    print(f"  samples:  {len(dataset)}")
    print(f"  subjects: {len(set(subjects))}")
    print(f"  train: {len(train_indices)} samples / {train_subject_count} subjects")
    print(f"  val:   {len(val_indices)} samples / {val_subject_count} subjects")
    print(f"  test:  {len(test_indices)} samples / {test_subject_count} subjects")

    if len(train_indices) == 0:
        raise RuntimeError("Training split is empty.")
    if len(val_indices) == 0:
        raise RuntimeError("Validation split is empty.")
    if len(test_indices) == 0:
        raise RuntimeError("Test split is empty.")

    sample_signal, _, _, sample_features = dataset[int(train_indices[0])]
    n_signal_channels = sample_signal.shape[0]
    n_hb_features = sample_features.shape[0]

    print(f"  signal channels: {n_signal_channels}")
    print(f"  Hb features:     {n_hb_features}")

    # Compute normalization from training samples only to avoid evaluation leakage.
    print("Computing Hb feature normalization from training samples...")
    feature_mean, feature_std = compute_feature_norm(dataset, train_indices)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = MultiTaskHRHb(
        n_signal_channels=n_signal_channels,
        n_hb_features=n_hb_features,
        fps=config.TARGET_FPS,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters: {parameter_count / 1e3:.1f}k")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Reduce the learning rate when validation Hb R2 stops improving.
    # ReduceLROnPlateau minimizes its monitored value, so use -R2.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    normalized_mean = torch.tensor(feature_mean, device=device)
    normalized_std = torch.tensor(feature_std, device=device)

    best_hb_r2 = float("-inf")
    best_hb_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    history: list[dict[str, float]] = []
    history_path = os.path.join(args.out_dir, "history.csv")
    best_weights_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        sample_count = 0

        progress = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{args.epochs}",
            unit="batch",
            leave=False,
        )

        for signal, heart_rate, hemoglobin, hb_features in progress:
            signal = signal.to(device)
            heart_rate = heart_rate.to(device)
            hemoglobin = hemoglobin.to(device)

            hb_features = (hb_features.to(device) - normalized_mean) / normalized_std

            predicted_hr_bpm, hr_logits, _, predicted_hb = model(  # type: ignore
                signal,
                hb_features,
            )

            loss, _ = model.loss(  # type: ignore
                predicted_hr_bpm,
                hr_logits,
                predicted_hb,
                heart_rate,
                hemoglobin,
                hb_weight=args.hb_weight,
            )

            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()  # type: ignore

            batch_size = signal.shape[0]
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size

            progress.set_postfix(loss=f"{float(loss.detach()):.3f}") # type: ignore

        train_loss = total_loss / max(1, sample_count)

        # Validation Hb R2 drives the scheduler and checkpoint selection.
        validation_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        validation_r2 = validation_metrics["hb_r2"]
        scheduler_metric = -validation_r2 if np.isfinite(validation_r2) else 0.0
        scheduler.step(scheduler_metric)  # type: ignore

        current_lr = float(optimizer.param_groups[0]["lr"])

        history_row: dict[str, int | float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "lr": current_lr,
            **{f"val_{metric_name}": metric_value for metric_name, metric_value in validation_metrics.items()},
        }
        history.append(history_row)
        save_history(history, history_path)

        # Hb R2 is the primary selection metric. Hb MAE is used only when
        # validation R2 is undefined, such as when all validation Hb targets
        # have effectively zero variance.
        if np.isfinite(validation_r2):
            improved = validation_r2 > best_hb_r2

            if improved:
                best_hb_r2 = validation_r2
        else:
            validation_mae = validation_metrics["hb_mae"]
            improved = validation_mae < best_hb_mae

            if improved:
                best_hb_mae = validation_mae

        if improved:
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, best_weights_path)
            best_marker = " *best"
        else:
            bad_epochs += 1
            best_marker = ""

        print(
            f"epoch {epoch:3d}/{args.epochs} "
            f"loss={train_loss:.3f} | "
            f"HR mae={validation_metrics['hr_mae']:.2f} BPM "
            f"w6={validation_metrics['hr_w6']:.1f}% | "
            f"Hb mae={validation_metrics['hb_mae']:.3f} "
            f"r2={validation_metrics['hb_r2']:+.3f} "
            f"corr={validation_metrics['hb_corr']:+.3f} | "
            f"lr={current_lr:.2e}"
            f"{best_marker}"
        )

        if bad_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation Hb R2 improvement.")
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a valid checkpoint.")

    # Restore the best validation checkpoint before evaluating test subjects.
    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    print()
    print("=== Test results ===")
    print(f"HR: MAE {test_metrics['hr_mae']:.2f} BPM  within 6 BPM {test_metrics['hr_w6']:.1f}%")
    print(f"Hb: MAE {test_metrics['hb_mae']:.3f}  R2 {test_metrics['hb_r2']:+.3f}  corr {test_metrics['hb_corr']:+.3f}")

    metrics_path = os.path.join(args.out_dir, "metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "best_val_hb_r2": best_hb_r2,
                "test_hr_mae": test_metrics["hr_mae"],
                "test_hr_within6": test_metrics["hr_w6"],
                "test_hb_mae": test_metrics["hb_mae"],
                "test_hb_r2": test_metrics["hb_r2"],
                "test_hb_corr": test_metrics["hb_corr"],
                "hb_weight": args.hb_weight,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "train_subjects": train_subject_count,
                "val_subjects": val_subject_count,
                "test_subjects": test_subject_count,
                "n_signal_channels": n_signal_channels,
                "n_hb_features": n_hb_features,
                "fps": config.TARGET_FPS,
                "seed": args.seed,
            },
            metrics_file,
            indent=2,
        )

    # Save the exact training-set normalization needed during inference.
    feature_norm_path = os.path.join(args.out_dir, "feature_norm.npz")
    np.savez(feature_norm_path, mean=feature_mean, std=feature_std)

    print()
    print("=== Output files ===")
    print(f"best weights:   {best_weights_path}")
    print(f"history:        {history_path}")
    print(f"metrics:        {metrics_path}")
    print(f"feature norm:   {feature_norm_path}")


if __name__ == "__main__":
    main()
