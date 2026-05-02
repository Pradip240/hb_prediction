import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, csv, name in zip(axes, ["train_data.csv", "val_data.csv", "test_data.csv"],
                          ["Train", "Val", "Test"]):
    df = pd.read_csv(csv)
    # Use unique MRNs to avoid skew from per-image duplicates
    unique = df.drop_duplicates(subset="mrn")
    ax.hist(unique["hb"], bins=25, alpha=0.7, edgecolor="black")
    ax.axvline(unique["hb"].mean(), color="red", linestyle="--",
               label=f"mean={unique['hb'].mean():.2f}")
    ax.set_title(f"{name} (n={len(unique)} patients, n={len(df)} images)")
    ax.set_xlabel("Hb (g/dL)")
    ax.legend()
axes[0].set_ylabel("Count")
plt.tight_layout()
plt.savefig("output/data_distribution.png", dpi=120)
plt.show()

# Also print stats
for csv, name in zip(["train_data.csv", "val_data.csv", "test_data.csv"],
                     ["Train", "Val", "Test"]):
    df = pd.read_csv(csv).drop_duplicates(subset="mrn")
    print(f"{name}: mean={df['hb'].mean():.2f}, std={df['hb'].std():.2f}, "
          f"min={df['hb'].min():.2f}, max={df['hb'].max():.2f}, n={len(df)}")