"""
Diagnostic script for HbPredictor.

Uses the model's BUILT-IN attention map (no Grad-CAM gymnastics) so we can
see exactly where the model is looking. Falls back to Grad-CAM if the
attention isn't accessible.

Samples one image per unique MRN from test_data.csv so that the visualization
spans patients rather than near-duplicate frames.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import HbPredictor


# --- Test dataset ---
class HbTestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        hb_value = torch.tensor([float(row["hb"])], dtype=torch.float32)
        mrn = row["mrn"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (256, 256), (0, 0, 0))
            print(f"Warning: Failed to load {img_path}: {e}")
        if self.transform:
            image = self.transform(image)
        return image, hb_value, img_path, mrn


def sample_one_per_mrn(test_csv: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    print(f"Original test set: {len(df)} images, {df['mrn'].nunique()} unique MRNs")
    sampled = df.groupby("mrn", group_keys=False).sample(n=1, random_state=seed)
    sampled = sampled.reset_index(drop=True)
    print(f"After 1-per-MRN: {len(sampled)} images")
    return sampled


def get_test_loader(df: pd.DataFrame, batch_size: int = 64, num_workers: int = 4):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    ds = HbTestDataset(df, transform=test_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def denormalize_imagenet(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_heatmap(image_np: np.ndarray, attn_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = image_np.shape[:2]
    am = cv2.resize(attn_map, (w, h))
    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * am), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * image_np + alpha * heatmap, 0, 1)


def diagnose(checkpoint_path: str = "output/best_hb_model.pth",
             test_csv: str = "test_data.csv",
             num_samples: int = 8,
             output_dir: str = "gradcam_output",
             seed: int = 42):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load model
    model = HbPredictor().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # Sample 1 image per MRN
    sampled_df = sample_one_per_mrn(test_csv, seed=seed)
    test_loader = get_test_loader(sampled_df, batch_size=64)

    # Collect everything (small enough — one image per patient)
    all_images, all_labels, all_paths, all_mrns = [], [], [], []
    for images, labels, paths, mrns in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        all_paths.extend(paths)
        all_mrns.extend(mrns)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels).squeeze()

    # Pick samples spanning Hb range for the visualization grid
    sorted_idx = torch.argsort(all_labels)
    if len(sorted_idx) < num_samples:
        num_samples = len(sorted_idx)
        print(f"Note: only {num_samples} unique MRNs available")
    pick_idx = sorted_idx[torch.linspace(0, len(sorted_idx) - 1, num_samples).long()]

    # ----- Visualization (uses built-in attention) -----
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5.5))

    print(f"\n{'#':<4}{'MRN':<22}{'True Hb':<10}{'Pred':<10}{'Error':<8}")
    print("-" * 60)

    for i, idx in enumerate(pick_idx):
        img = all_images[idx:idx + 1].to(DEVICE)
        label = all_labels[idx].item()
        mrn = all_mrns[idx]

        with torch.no_grad():
            pred = model(img)
            attn = model.get_last_attention()  # [1, 1, 7, 7]

        pred_val = pred.item()
        attn_map = attn[0, 0].cpu().numpy()
        print(f"{i:<4}{mrn:<22}{label:<10.2f}{pred_val:<10.2f}{abs(pred_val - label):<8.2f}")

        img_display = denormalize_imagenet(all_images[idx])
        overlay = overlay_heatmap(img_display, attn_map)

        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"True: {label:.2f}\n{mrn[:12]}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred: {pred_val:.2f}", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Top: Original | Bottom: Learned Attention (1 per MRN, sorted by true Hb)",
                 fontsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "attention_diagnosis_test.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved heatmap visualization to {save_path}")
    plt.close()

    # ----- Full-set diagnostic numbers -----
    all_preds = []
    with torch.no_grad():
        for images, _, _, _ in test_loader:
            all_preds.append(model(images.to(DEVICE)).cpu().squeeze())
    all_preds = torch.cat(all_preds) if all_preds[0].dim() > 0 else torch.stack(all_preds)
    if all_preds.dim() == 0:
        all_preds = all_preds.unsqueeze(0)

    mae = (all_preds - all_labels).abs().mean().item()
    acc1 = ((all_preds - all_labels).abs() <= 1.0).float().mean().item()
    acc05 = ((all_preds - all_labels).abs() <= 0.5).float().mean().item()
    acc15 = ((all_preds - all_labels).abs() <= 1.5).float().mean().item()

    print(f"\n--- Per-MRN Diagnostic ({len(all_labels)} unique patients) ---")
    print(f"Label range:    [{all_labels.min():.2f}, {all_labels.max():.2f}]   std: {all_labels.std():.3f}")
    print(f"Pred range:     [{all_preds.min():.2f}, {all_preds.max():.2f}]   std: {all_preds.std():.3f}")
    print(f"Mean label:     {all_labels.mean():.3f}")
    print(f"Mean pred:      {all_preds.mean():.3f}")
    print(f"Test MAE:       {mae:.3f}")
    print(f"Acc within ±0.5:{acc05:.2%}")
    print(f"Acc within ±1.0:{acc1:.2%}")
    print(f"Acc within ±1.5:{acc15:.2%}")

    if all_preds.std() < 0.3 * all_labels.std():
        print("\n⚠️  Prediction std is < 30% of label std — model is collapsing toward the mean.")
    else:
        print("\n✅ Prediction spread looks reasonable relative to label spread.")

    # Mean baseline
    mean_pred = all_labels.mean().item()
    baseline_mae = (all_labels - mean_pred).abs().mean().item()
    baseline_acc1 = ((all_labels - mean_pred).abs() <= 1.0).float().mean().item()
    print(f"\n--- Baseline (always predict {mean_pred:.2f}) ---")
    print(f"Baseline MAE:   {baseline_mae:.3f}")
    print(f"Baseline acc±1: {baseline_acc1:.2%}")

    if baseline_mae > 0:
        improvement = (baseline_mae - mae) / baseline_mae * 100
        print(f"\nModel improvement over baseline MAE: {improvement:.1f}%")
        if mae >= baseline_mae * 0.95:
            print("⚠️  Model is barely better than predicting the mean.")
        elif improvement > 20:
            print("✅ Model is meaningfully outperforming the baseline.")

    # ----- Attention consistency check -----
    # Compute the average attention map across all test patients. If the
    # model has learned a consistent face region, this average will be
    # peaked. If attention is random per image, the average will be uniform.
    print("\n--- Attention consistency across patients ---")
    avg_attn = torch.zeros(7, 7)
    n = 0
    with torch.no_grad():
        for images, _, _, _ in test_loader:
            _ = model(images.to(DEVICE))
            attn = model.get_last_attention().cpu()  # [B, 1, 7, 7]
            avg_attn += attn[:, 0].sum(dim=0)
            n += attn.shape[0]
    avg_attn /= n
    avg_attn_np = avg_attn.numpy()
    # Entropy of the average
    flat = avg_attn_np.flatten()
    flat = flat / (flat.sum() + 1e-8)
    avg_entropy = -np.sum(flat * np.log(flat + 1e-8))
    max_entropy = np.log(49)
    print(f"Avg attention entropy: {avg_entropy:.3f} (max possible {max_entropy:.3f})")
    print(f"Lower = more consistent focus. Higher = scattered/per-image.")

    # Save the average attention map as its own image
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(avg_attn_np, cmap="hot")
    ax.set_title(f"Avg attention across {n} patients\nentropy={avg_entropy:.2f} / {max_entropy:.2f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    avg_path = os.path.join(output_dir, "avg_attention.png")
    plt.savefig(avg_path, dpi=120, bbox_inches="tight")
    print(f"Saved average-attention map to {avg_path}")
    plt.close()


if __name__ == "__main__":
    diagnose(
        checkpoint_path="output/best_hb_model.pth",
        test_csv="test_data.csv",
        num_samples=8,
        seed=42,
    )
