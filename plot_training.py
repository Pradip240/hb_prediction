"""
Generate report-quality plots from training_log.csv.

Produces:
  1. loss_curves.png       - Train loss decomposed into reg + attention
  2. validation_metrics.png - Val MAE and Acc±1 over epochs
  3. mean_collapse_check.png - Pred std vs label std, and pred mean vs label mean
  4. learning_rate.png     - LR schedule
  5. summary_dashboard.png - All of the above in one figure
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Set a clean style suitable for reports
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "legend.frameon": False,
})


def smooth(y, window=5):
    """Simple rolling mean for smoother curves on noisy logs."""
    if len(y) < window:
        return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


def plot_loss_curves(df, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = df["epoch"]

    ax.plot(epochs, df["train_loss"], label="Total train loss", color="#1f77b4", alpha=0.85)
    ax.plot(epochs, df["train_reg_loss"], label="Regression (Huber)", color="#2ca02c", alpha=0.85)
    ax.plot(epochs, df["train_attn_loss"], label="Attention entropy penalty",
            color="#d62728", alpha=0.85, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss decomposition")
    ax.legend(loc="upper right")
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_validation_metrics(df, save_path):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    epochs = df["epoch"]

    # Val MAE on left axis
    color_mae = "#1f77b4"
    ax1.plot(epochs, df["val_mae"], label="Val MAE", color=color_mae)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (g/dL)", color=color_mae)
    ax1.tick_params(axis="y", labelcolor=color_mae)

    # Find acc column (it includes the tolerance value, e.g. val_acc_within_1.0)
    acc_col = [c for c in df.columns if c.startswith("val_acc_within_")][0]
    tolerance = acc_col.replace("val_acc_within_", "")

    # Acc on right axis
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    color_acc = "#2ca02c"
    ax2.plot(epochs, df[acc_col] * 100, label=f"Val Acc±{tolerance}",
             color=color_acc, linestyle="--")
    ax2.set_ylabel(f"Accuracy ± {tolerance} g/dL (%)", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.grid(False)

    # Mark best-acc epoch
    best_idx = df[acc_col].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_acc = df.loc[best_idx, acc_col] * 100
    best_mae = df.loc[best_idx, "val_mae"]
    ax2.axvline(best_epoch, color="grey", alpha=0.4, linestyle=":")
    ax2.annotate(
        f"Best: epoch {best_epoch}\nAcc {best_acc:.1f}%, MAE {best_mae:.2f}",
        xy=(best_epoch, best_acc),
        xytext=(10, -30), textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9),
    )

    plt.title("Validation metrics over training")
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_mean_collapse(df, save_path):
    """Visualize pred std vs label std and pred mean vs label mean over time."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = df["epoch"]

    # Std comparison
    ax = axes[0]
    ax.plot(epochs, df["val_pred_std"], label="Prediction σ", color="#1f77b4")
    ax.plot(epochs, df["val_label_std"], label="Label σ (constant)",
            color="#d62728", linestyle="--")
    # Shade the "collapsed" zone (pred std < 30% of label std)
    label_std = df["val_label_std"].iloc[-1]
    ax.axhspan(0, 0.3 * label_std, color="red", alpha=0.08, label="Mean-collapse zone")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standard deviation (g/dL)")
    ax.set_title("Prediction spread vs. label spread")
    ax.legend(loc="lower right")

    # Mean comparison
    ax = axes[1]
    ax.plot(epochs, df["val_pred_mean"], label="Prediction mean", color="#1f77b4")
    ax.plot(epochs, df["val_label_mean"], label="Label mean (constant)",
            color="#d62728", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Hb (g/dL)")
    ax.set_title("Prediction mean vs. label mean")
    ax.legend(loc="best")

    plt.suptitle("Mean-collapse diagnostics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_learning_rate(df, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(df["epoch"], df["lr"], color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule (cosine + warmup)")
    ax.set_yscale("log")
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_dashboard(df, save_path):
    """All key plots in one figure for the report."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    epochs = df["epoch"]
    acc_col = [c for c in df.columns if c.startswith("val_acc_within_")][0]
    tolerance = acc_col.replace("val_acc_within_", "")

    # 1. Loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, df["train_loss"], label="Total", color="#1f77b4")
    ax.plot(epochs, df["train_reg_loss"], label="Regression", color="#2ca02c")
    ax.plot(epochs, df["train_attn_loss"], label="Attention", color="#d62728", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training loss")
    ax.legend(fontsize=9)

    # 2. Val MAE
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, df["val_mae"], color="#1f77b4")
    best_idx = df[acc_col].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_mae = df.loc[best_idx, "val_mae"]
    ax.axvline(best_epoch, color="grey", alpha=0.4, linestyle=":")
    ax.scatter([best_epoch], [best_mae], color="red", zorder=5,
               label=f"Best: {best_mae:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (g/dL)")
    ax.set_title("(b) Validation MAE")
    ax.legend(fontsize=9)

    # 3. Val Acc±1
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, df[acc_col] * 100, color="#2ca02c")
    best_acc = df.loc[best_idx, acc_col] * 100
    ax.scatter([best_epoch], [best_acc], color="red", zorder=5,
               label=f"Best: {best_acc:.1f}%")
    ax.axvline(best_epoch, color="grey", alpha=0.4, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Accuracy ± {tolerance} g/dL (%)")
    ax.set_title(f"(c) Validation Acc±{tolerance}")
    ax.legend(fontsize=9)

    # 4. Std comparison
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, df["val_pred_std"], label="Pred σ", color="#1f77b4")
    ax.plot(epochs, df["val_label_std"], label="Label σ", color="#d62728", linestyle="--")
    label_std = df["val_label_std"].iloc[-1]
    ax.axhspan(0, 0.3 * label_std, color="red", alpha=0.08)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Std (g/dL)")
    ax.set_title("(d) Prediction spread vs. label spread")
    ax.legend(fontsize=9)

    # 5. Mean comparison
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(epochs, df["val_pred_mean"], label="Pred mean", color="#1f77b4")
    ax.plot(epochs, df["val_label_mean"], label="Label mean", color="#d62728", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Hb (g/dL)")
    ax.set_title("(e) Prediction mean vs. label mean")
    ax.legend(fontsize=9)

    # 6. LR
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(epochs, df["lr"], color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("(f) LR schedule")
    ax.set_yscale("log")

    fig.suptitle("Hb Predictor — Training Summary", fontsize=14, y=0.995)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def print_summary(df):
    """Print a clean summary table for the report."""
    acc_col = [c for c in df.columns if c.startswith("val_acc_within_")][0]
    tolerance = acc_col.replace("val_acc_within_", "")
    best_idx = df[acc_col].idxmax()
    final_idx = df.index[-1]

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY (numbers for report)")
    print("=" * 60)
    print(f"Total epochs trained:       {len(df)}")
    print(f"Final train loss:           {df.loc[final_idx, 'train_loss']:.4f}")
    print()
    print(f"Best epoch (by Acc±{tolerance}):     {int(df.loc[best_idx, 'epoch'])}")
    print(f"  Val MAE:                  {df.loc[best_idx, 'val_mae']:.4f} g/dL")
    print(f"  Val MSE:                  {df.loc[best_idx, 'val_mse']:.4f}")
    print(f"  Val Acc±{tolerance}:              {df.loc[best_idx, acc_col] * 100:.2f}%")
    print(f"  Pred σ:                   {df.loc[best_idx, 'val_pred_std']:.4f}")
    print(f"  Label σ:                  {df.loc[best_idx, 'val_label_std']:.4f}")
    print(f"  Pred mean:                {df.loc[best_idx, 'val_pred_mean']:.4f}")
    print(f"  Label mean:               {df.loc[best_idx, 'val_label_mean']:.4f}")
    print()
    print(f"Final epoch:                {int(df.loc[final_idx, 'epoch'])}")
    print(f"  Val MAE:                  {df.loc[final_idx, 'val_mae']:.4f}")
    print(f"  Val Acc±{tolerance}:              {df.loc[final_idx, acc_col] * 100:.2f}%")
    print("=" * 60)


def main(log_file="output/training_log.csv", out_dir="output/plots"):
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found. Train the model first.")
        return

    df = pd.read_csv(log_file)
    # Ensure numeric types (CSV may have read as strings)
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    if len(df) == 0:
        print("Error: log file has no usable rows.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Plotting from {log_file} ({len(df)} epochs)")

    plot_loss_curves(df, os.path.join(out_dir, "loss_curves.png"))
    plot_validation_metrics(df, os.path.join(out_dir, "validation_metrics.png"))
    plot_mean_collapse(df, os.path.join(out_dir, "mean_collapse_check.png"))
    plot_learning_rate(df, os.path.join(out_dir, "learning_rate.png"))
    plot_dashboard(df, os.path.join(out_dir, "summary_dashboard.png"))

    print_summary(df)
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
