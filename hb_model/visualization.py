"""
Visualization utilities for hemoglobin model training.

This module creates diagnostic plots from the training history and final
test metrics of the HbMLP regression model.

The generated plots are:

- Training loss and validation MAE.
- Validation RMSE.
- Learning rate.
- Final test MAE and RMSE.

All plots are saved to disk using a non-interactive Matplotlib backend so
they can be generated inside a training container without a display server.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history: list[dict[str, float]], test_mae: float, test_rmse: float, output_dir: str) -> None:
    """
    Save Hb training-history and final test-metric plots.

    Args:
        history: Training history collected after each epoch.
        test_mae: Final test mean absolute error in g/dL.
        test_rmse: Final test root mean squared error in g/dL.
        output_dir: Directory where plots will be saved.
    """
    if not history:
        print("No training history available for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    loss_path = os.path.join(output_dir, "training_loss_mae.png")
    epochs = [int(entry["epoch"]) for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_mae = [entry["val_mae"] for entry in history]
    val_rmse = [entry["val_rmse"] for entry in history]
    learning_rate = [entry["lr"] for entry in history]

    # Training loss and validation MAE.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", markersize=3, label="Training loss")
    plt.plot(epochs, val_mae, marker="o", markersize=3, label="Validation MAE (g/dL)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Hb Training Loss and Validation MAE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=130)
    plt.close()
    print(f"training plot: {loss_path}")

    # Validation RMSE.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_rmse, marker="o", markersize=3, label="Validation RMSE (g/dL)")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (g/dL)")
    plt.title("Hb Validation RMSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    rmse_path = os.path.join(output_dir, "validation_rmse.png")
    plt.savefig(rmse_path, dpi=130)
    plt.close()
    print(f"validation RMSE plot: {rmse_path}")

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
    metric_names = ["Test MAE (g/dL)", "Test RMSE (g/dL)"]
    metric_values = [test_mae, test_rmse]
    plt.bar(metric_names, metric_values)
    plt.ylabel("Error (g/dL)")
    plt.title("Final Hb Test Metrics")
    plt.grid(axis="y", alpha=0.3)
    for index, value in enumerate(metric_values):
        plt.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    test_metrics_path = os.path.join(output_dir, "test_metrics.png")
    plt.savefig(test_metrics_path, dpi=130)
    plt.close()
    print(f"test metrics plot: {test_metrics_path}")
