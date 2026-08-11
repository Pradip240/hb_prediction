"""Visualization utilities for HRSpectralNet training.

This module provides utilities for visualizing model training history and
final test-set performance.

It generates plots for training loss, validation MAE, validation accuracy,
learning-rate changes, and final test metrics.

All plots are saved as PNG files in the specified output directory.
Matplotlib is configured to use a non-interactive backend so that plots can
be generated in headless training environments.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history: list[dict[str, float]], test_mae: float, test_within_6: float, output_dir: str) -> None:
    """
    Save training-history and final test-metric plots.

    Args:
        history: Training history collected after each epoch.
        test_mae: Final test mean absolute error in BPM.
        test_within_6: Final percentage of predictions within 6 BPM.
        output_dir: Directory where plots will be saved.
    """
    if not history:
        print("No training history available for plotting.")
        return

    epochs = [int(entry["epoch"]) for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_mae = [entry["val_mae"] for entry in history]
    val_within_6 = [entry["val_within_6"] for entry in history]
    learning_rate = [entry["lr"] for entry in history]

    # Training loss and validation MAE.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", markersize=3, label="Training loss")
    plt.plot(epochs, val_mae, marker="o", markersize=3, label="Validation MAE (BPM)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Validation MAE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_loss_mae.png")
    plt.savefig(loss_path, dpi=130)
    plt.close()
    print(f"training plot: {loss_path}")

    # Validation accuracy and learning rate.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_within_6, marker="o", markersize=3, label="Validation within 6 BPM (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.title("Validation HR Accuracy")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    accuracy_path = os.path.join(output_dir, "validation_accuracy.png")
    plt.savefig(accuracy_path, dpi=130)
    plt.close()
    print(f"accuracy plot: {accuracy_path}")

    # Learning rate.

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, learning_rate, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    learning_rate_path = os.path.join(output_dir, "learning_rate.png")
    plt.savefig(learning_rate_path, dpi=130)
    plt.close()
    print(f"learning-rate plot: {learning_rate_path}")

    # Final test metrics.
    plt.figure(figsize=(7, 5))
    metric_names = ["Test MAE (BPM)", "Within 6 BPM (%)"]
    metric_values = [test_mae, test_within_6]
    plt.bar(metric_names, metric_values)
    plt.ylabel("Value")
    plt.title("Final Test Metrics")
    plt.grid(axis="y", alpha=0.3)
    # Add the exact metric value above each bar.
    for index, value in enumerate(metric_values):
        plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    test_metrics_path = os.path.join(output_dir, "test_metrics.png")
    plt.savefig(test_metrics_path, dpi=130)
    plt.close()
    print(f"test metrics plot: {test_metrics_path}")