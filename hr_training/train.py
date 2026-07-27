"""train.py — train HRSpectralNet on the train_data segments and report honest metrics.

Loads segments + PPG labels (subject-wise split), trains on the train subjects,
early-stops on val MAE, and evaluates on the held-out test subjects. The test report
gives the numbers that matter, split by DSP confidence so weak-pulse segments are
visible rather than hidden:

  * MAE and within-6-BPM %, over ALL test segments and over the HIGH-confidence subset
  * bias (mean pred - true) — the DSP baseline's failure was a big negative bias
  * DSP baseline (best of POS/CHROM/green per segment) and naive baseline (predict the
    mean training HR) on the same test segments, so "did the model actually help" is
    answered directly.

Progress is shown with tqdm (per-batch bar within each epoch). Per-epoch train loss and
val MAE / within-6 are recorded to history.csv and plotted to training_curves.png.

Usage:
  python train.py --segments-dir output/train_data --ppg-dir data/ppg \
      --out-dir output/hr_model --epochs 80 --batch-size 128
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:                       # tqdm optional; fall back to a no-op
    def tqdm(x, **k):
        return x

from model import HRSpectralNet
from dataset import load_dataset, subject_splits, spectral_hr
from common import signal_processing as sp


def log(*a):
    """Print and flush immediately (containers buffer stdout otherwise)."""
    print(*a, flush=True)


def method_hr_raw(raw_seg, fps=30.0):
    """Per-method HR (POS, CHROM, green) from a segment's RAW face-averaged RGB (T,3).

    Raw FFT argmax per method — no best-of-three, no harmonic guarding: this shows each
    method's true behavior (CHROM/green will octave-lock on some segments). Computed on
    the un-normalized signal so POS/CHROM see real amplitudes.
    """
    out = {}
    out["POS"], _ = spectral_hr(sp.extract_pos(raw_seg, fps=fps), fps)
    out["CHROM"], _ = spectral_hr(sp.extract_chrom(raw_seg, fps=fps), fps)
    out["green"], _ = spectral_hr(raw_seg[:, 1], fps)
    return out


def metrics(pred, true, tol=6.0):
    ae = np.abs(pred - true)
    ok = np.isfinite(ae)
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(ae[ok])) if ok.any() else float("nan"),
        "within_tol": float(np.mean(ae[ok] <= tol) * 100) if ok.any() else float("nan"),
        "bias": float(np.mean((pred - true)[ok])) if ok.any() else float("nan"),
    }


def evaluate(model, X, y, conf, device, conf_thr):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512]).to(device)
            preds.append(model(xb)[0].cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros((0,))
    hi = conf >= conf_thr
    out = {"all": metrics(pred, y), "high_conf": metrics(pred[hi], y[hi])}
    return out, pred


def plot_curves(history, out_png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        log(f"  (curves plot skipped: {e})"); return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, [h["train_loss"] for h in history], "-o", ms=3)
    ax[0].set_title("train loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [h["val_mae"] for h in history], "-o", ms=3, color="tab:orange")
    ax[1].set_title("val MAE (BPM)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3)
    best = min((h["val_mae"] for h in history if np.isfinite(h["val_mae"])), default=None)
    if best is not None:
        ax[1].axhline(best, ls="--", c="gray", lw=1, label=f"best {best:.2f}")
        ax[1].legend(fontsize=8)
    ax[2].plot(ep, [h["val_w6"] for h in history], "-o", ms=3, color="tab:green")
    ax[2].set_title("val within-6 BPM (%)"); ax[2].set_xlabel("epoch"); ax[2].set_ylim(0, 100); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    log(f"  curves -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Train spectral HR model on segments.")
    ap.add_argument("--segments-dir", default="output/train_data")
    ap.add_argument("--ppg-dir", default="data/ppg")
    ap.add_argument("--out-dir", default="output/hr_model")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--val-fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"device: {device}")

    log("loading segments + labels ...")
    data = load_dataset(args.segments_dir, args.ppg_dir, args.manifest)
    N = len(data["y"])
    log(f"  {N} labeled segments ({data['skipped']} skipped), "
        f"{len(set(data['subject'].tolist()))} subjects, HR {data['y'].min():.0f}-{data['y'].max():.0f}")
    if N < 20:
        raise SystemExit("too few labeled segments to train")

    split = subject_splits(data["subject"], args.n_folds, args.test_fold, args.val_fold, args.seed)
    conf_thr = float(np.median(data["conf"]))
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        log(f"  {s:5}: {len(idx[s]):5d} segments / {len(set(data['subject'][idx[s]].tolist())):3d} subjects")

    Xtr = torch.from_numpy(data["X"][idx["train"]]); ytr = torch.from_numpy(data["y"][idx["train"]])
    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)

    model = HRSpectralNet(n_channels=Xtr.shape[1]).to(device)
    log(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k | spectral bins {model.n_freq}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_val, best_state, bad = float("inf"), None, 0
    weights = os.path.join(args.out_dir, "hr_model.pt")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        bar = tqdm(train_dl, desc=f"epoch {epoch:3d}/{args.epochs}", leave=False,
                   file=sys.stdout, dynamic_ncols=True)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            pred, logits, prob = model(xb)
            loss, _ = model.loss(pred, logits, prob, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss.item():.3f}")
        tot /= max(1, len(Xtr))

        val, _ = evaluate(model, data["X"][idx["val"]], data["y"][idx["val"]], data["conf"][idx["val"]], device, conf_thr)
        vmae, vw6 = val["all"]["mae"], val["all"]["within_tol"]
        sched.step(vmae if np.isfinite(vmae) else tot)
        history.append({"epoch": epoch, "train_loss": tot, "val_mae": vmae, "val_w6": vw6,
                        "lr": opt.param_groups[0]["lr"]})

        tag = ""
        if np.isfinite(vmae) and vmae < best_val - 1e-4:
            best_val = vmae; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, weights); tag = "  *best"
        else:
            bad += 1
        log(f"epoch {epoch:3d}/{args.epochs}  loss {tot:6.3f}  val MAE {vmae:6.2f}  w6 {vw6:4.0f}%{tag}")

        # persist history every epoch so a killed run still leaves a trace + curves
        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_mae", "val_w6", "lr"])
            w.writeheader(); w.writerows(history)
        if bad >= args.patience:
            log(f"early stop (no val gain {args.patience} epochs)"); break

    plot_curves(history, os.path.join(args.out_dir, "training_curves.png"))

    if best_state:
        model.load_state_dict(best_state)

    # ---- test evaluation + per-method baselines (from RAW signal) ----
    ti = idx["test"]
    Xte, yte, cte = data["X"][ti], data["y"][ti], data["conf"][ti]
    raw_te, fps_te = data["raw"][ti], data["fps"][ti]
    test, pred = evaluate(model, Xte, yte, cte, device, conf_thr)

    # per-method DSP HR on each test segment (raw FFT argmax)
    pos_hr = np.array([method_hr_raw(raw_te[i], float(fps_te[i])).get("POS") or np.nan for i in range(len(raw_te))])
    chrom_hr = np.array([method_hr_raw(raw_te[i], float(fps_te[i])).get("CHROM") or np.nan for i in range(len(raw_te))])
    green_hr = np.array([method_hr_raw(raw_te[i], float(fps_te[i])).get("green") or np.nan for i in range(len(raw_te))])
    naive = np.full(len(yte), data["y"][idx["train"]].mean())
    hi = cte >= conf_thr

    def line(name, p):
        m = metrics(p, yte)
        return f"  {name:8} MAE {m['mae']:6.2f}  w6 {m['within_tol']:4.0f}%  bias {m['bias']:+6.2f}  (n={m['n']})"

    log("\n=== TEST (held-out subjects) — each method separately ===")
    log(line("model", pred))
    log(f"  model hi-conf: MAE {test['high_conf']['mae']:.2f}  w6 {test['high_conf']['within_tol']:.0f}%  (n={test['high_conf']['n']})")
    log(line("POS", pos_hr))
    log(line("CHROM", chrom_hr))
    log(line("green", green_hr))
    log(f"  naive    MAE {metrics(naive, yte)['mae']:6.2f}  (floor: predict mean train HR)")
    best_method = min([("POS", metrics(pos_hr, yte)["mae"]), ("CHROM", metrics(chrom_hr, yte)["mae"]),
                       ("green", metrics(green_hr, yte)["mae"])], key=lambda t: t[1])
    log(f"  -> best DSP method: {best_method[0]} ({best_method[1]:.2f});  "
        f"model {'BEATS' if test['all']['mae'] < best_method[1] else 'does NOT beat'} it, "
        f"{'beats' if test['all']['mae'] < metrics(naive, yte)['mae'] else 'WORSE than'} naive")

    json.dump({"conf_thr": conf_thr, "model_test": test,
               "POS": metrics(pos_hr, yte), "CHROM": metrics(chrom_hr, yte),
               "green": metrics(green_hr, yte), "naive": metrics(naive, yte)},
              open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(yte[~hi], pred[~hi], s=10, facecolors="none", edgecolors="#bbb", linewidths=0.6, label="low conf")
        plt.scatter(yte[hi], pred[hi], s=12, alpha=0.5, color="#1f77b4", label="high conf")
        lim = [30, max(210, float(np.nanmax(yte)) + 10)]
        plt.plot(lim, lim, "k--", lw=1); plt.xlim(lim); plt.ylim(lim)
        plt.xlabel("true HR (PPG)"); plt.ylabel("predicted HR")
        plt.title(f"Test — MAE {test['all']['mae']:.1f} BPM (w6 {test['all']['within_tol']:.0f}%)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "test_scatter.png"), dpi=130)
        log(f"  scatter -> {os.path.join(args.out_dir, 'test_scatter.png')}")
    except Exception as e:
        log(f"  (plot skipped: {e})")

    log(f"\nweights -> {weights}")


if __name__ == "__main__":
    main()