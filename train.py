"""
Training loop for HbPredictor.

Key changes vs. the previous version:
- Loss = Huber (regression) + lambda * attention-entropy penalty.
  Huber is more robust than MSE; entropy penalty pushes attention to be focused.
- Best-model selection is based on Acc-within-1.0 (the clinically relevant
  metric) rather than MAE alone.
- Logs prediction std vs. label std every epoch — this is the cleanest
  signal of "predicting the mean" behavior.
- Only optimizes parameters with requires_grad=True (so frozen ResNet
  layers don't waste optimizer state).
- Cosine LR schedule, with warmup, instead of ReduceLROnPlateau.
"""

import os
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import HbPredictor, attention_entropy_loss
from data_loader import get_loaders


# --- Config ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
ATTN_LOSS_WEIGHT = 0.1          # weight on attention-entropy penalty
ATTN_TARGET_ENTROPY = 2.5       # target entropy (uniform over 49 = ~3.89)
TOLERANCE = 1.0                 # +/- for acc metric
WARMUP_EPOCHS = 3

OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_hb_model.pth")
LATEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "latest_checkpoint.pth")


def cosine_lr(epoch: int, total_epochs: int, base_lr: float, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def within_tolerance_accuracy(outputs: torch.Tensor, labels: torch.Tensor,
                              tolerance: float = 1.0) -> float:
    return (torch.abs(outputs - labels) <= tolerance).float().mean().item()


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Log file
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr", "train_loss", "train_reg_loss", "train_attn_loss",
                "val_mse", "val_mae", f"val_acc_within_{TOLERANCE}",
                "val_pred_std", "val_label_std", "val_pred_mean", "val_label_mean",
            ])

    # Data
    train_loader, val_loader = get_loaders(
        train_csv="train_data.csv",
        val_csv="val_data.csv",
        batch_size=BATCH_SIZE,
        use_weighted_sampler=True,
    )

    # Model
    model = HbPredictor(dropout_rate=0.3, freeze_early=True).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    # Loss + optim
    huber = nn.SmoothL1Loss(beta=1.0)   # Huber with delta=1
    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0

    # --- Loop ---
    for epoch in range(EPOCHS):
        # Manual LR schedule (cosine with warmup)
        lr = cosine_lr(epoch, EPOCHS, LEARNING_RATE, WARMUP_EPOCHS)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ----- Train -----
        model.train()
        running_total = 0.0
        running_reg = 0.0
        running_attn = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (lr={lr:.2e})")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)

            reg_loss = huber(outputs, labels)
            attn = model.get_last_attention()
            attn_loss = attention_entropy_loss(attn, target_entropy=ATTN_TARGET_ENTROPY)
            loss = reg_loss + ATTN_LOSS_WEIGHT * attn_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()

            running_total += loss.item()
            running_reg += reg_loss.item()
            running_attn += attn_loss.item()
            n_batches += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "reg": f"{reg_loss.item():.3f}",
                "attn": f"{attn_loss.item():.3f}",
            })

        avg_total = running_total / n_batches
        avg_reg = running_reg / n_batches
        avg_attn = running_attn / n_batches

        # ----- Validate -----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_mse = mse_metric(all_preds, all_labels).item()
        val_mae = mae_metric(all_preds, all_labels).item()
        val_acc = within_tolerance_accuracy(all_preds, all_labels, TOLERANCE)
        pred_std = all_preds.std().item()
        label_std = all_labels.std().item()
        pred_mean = all_preds.mean().item()
        label_mean = all_labels.mean().item()

        # ----- Save -----
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "val_acc": val_acc,
        }, LATEST_CHECKPOINT_PATH)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            star = f"  New best Acc±{TOLERANCE}: {val_acc:.2%} (saved)"
        else:
            star = ""

        # ----- Log -----
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{lr:.6f}", f"{avg_total:.4f}",
                f"{avg_reg:.4f}", f"{avg_attn:.4f}",
                f"{val_mse:.4f}", f"{val_mae:.4f}", f"{val_acc:.4f}",
                f"{pred_std:.4f}", f"{label_std:.4f}",
                f"{pred_mean:.4f}", f"{label_mean:.4f}",
            ])

        print(
            f"  Epoch {epoch + 1} | "
            f"train {avg_total:.3f} (reg {avg_reg:.3f}, attn {avg_attn:.3f}) | "
            f"val MAE {val_mae:.3f} | acc±{TOLERANCE} {val_acc:.2%} | "
            f"pred σ {pred_std:.3f} vs label σ {label_std:.3f}"
            + star
        )

        # Early-warning print: if we're 5+ epochs in and pred std is < 30%
        # of label std, we're collapsing.
        if epoch >= 5 and pred_std < 0.3 * label_std:
            print("  ⚠️  Predictions are collapsing toward the mean.")

    print(f"\nTraining complete. Best Acc±{TOLERANCE}: {best_val_acc:.2%}")
    print(f"Outputs in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    train()
