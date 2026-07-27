"""train.py — train a small regressor for hemoglobin from amplitude/colour features.

HONESTY FIRST. Hemoglobin-from-rPPG is a weak, subtle signal and may not beat simply
predicting the mean. So the report leads with the NAIVE baseline (predict mean train Hb)
and includes a MEMORISATION check (train-subject vs held-out-subject error): a model that
looks good on training subjects but fails on held-out ones has learned faces, not Hb.

Because hemoglobin is per-subject, the honest final estimate for a subject is the MEAN of
its segment predictions — so we report both per-segment and per-subject test metrics.

Design: features are standardised (fit on train only); a small MLP is trained with early
stopping on val MAE. A small MLP (not a deep CNN) suits the low-dimensional feature vector
and limits memorisation. Metrics: MAE, RMSE, and Pearson r vs the labels.

Usage:
  python train.py --segments-dir output/train_data \
      --ground-truth data/ground_truth.csv --out-dir output/hb_model --epochs 200
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

from dataset import load_dataset, subject_splits
from model import HbMLP


def log(*a):
    print(*a, flush=True)


def reg_metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 2:
        return {"n": int(ok.sum()), "mae": float("nan"), "rmse": float("nan"), "r": float("nan")}
    p, t = pred[ok], true[ok]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-9 and np.std(t) > 1e-9 else float("nan")
    return {"n": int(ok.sum()), "mae": mae, "rmse": rmse, "r": r}


def per_subject(pred, true, subjects):
    """Average predictions per subject (Hb is subject-constant) -> per-subject arrays."""
    out_p, out_t = [], []
    for s in sorted(set(subjects.tolist())):
        m = subjects == s
        out_p.append(np.mean(pred[m])); out_t.append(np.mean(true[m]))
    return np.array(out_p), np.array(out_t)


def predict(model, X, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0,))


def main():
    ap = argparse.ArgumentParser(description="Train hemoglobin regressor from rPPG features.")
    ap.add_argument("--segments-dir", default="output/train_data")
    ap.add_argument("--ground-truth", default="data/ground_truth.csv")
    ap.add_argument("--target", default="hemoglobin", help="ground_truth column to predict")
    ap.add_argument("--out-dir", default="output/hb_model")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
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

    log("loading segments + hemoglobin labels ...")
    data = load_dataset(args.segments_dir, args.ground_truth, args.target)
    N = len(data["y"])
    if N < 20:
        raise SystemExit(f"too few labeled segments to train ({N}). Check ground_truth path / patient_id match.")
    log(f"  {N} segments ({data['skipped']} skipped), {len(set(data['subject'].tolist()))} subjects, "
        f"{args.target} range {data['y'].min():.1f}-{data['y'].max():.1f}, mean {data['y'].mean():.2f}")

    split = subject_splits(data["subject"], args.n_folds, args.test_fold, args.val_fold, args.seed)
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        log(f"  {s:5}: {len(idx[s]):5d} segments / {len(set(data['subject'][idx[s]].tolist())):3d} subjects")

    # standardise features on TRAIN only
    Xtr_raw = data["X"][idx["train"]]
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
    def norm(a):
        return ((a - mu) / sd).astype(np.float32)
    Xtr, ytr = norm(data["X"][idx["train"]]), data["y"][idx["train"]]
    Xva, yva = norm(data["X"][idx["val"]]), data["y"][idx["val"]]
    Xte, yte = norm(data["X"][idx["test"]]), data["y"][idx["test"]]

    # center the target: the net regresses Hb OFFSET from the train mean (outputs ~0,
    # well-conditioned), and we add y_mean back at read-out. y_std scales the loss.
    y_mean, y_std = float(ytr.mean()), float(ytr.std() + 1e-6)
    def to_t(y):
        return ((y - y_mean) / y_std).astype(np.float32)
    def from_t(z):
        return z * y_std + y_mean

    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(to_t(ytr))),
                          batch_size=args.batch_size, shuffle=True)
    model = HbMLP(Xtr.shape[1]).to(device)
    log(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k | features {Xtr.shape[1]}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    lossf = nn.SmoothL1Loss()

    best_val, best_state, bad, history = float("inf"), None, 0, []
    weights = os.path.join(args.out_dir, "hb_model.pt")
    for epoch in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        bar = tqdm(train_dl, desc=f"epoch {epoch:3d}/{args.epochs}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        tot /= max(1, len(Xtr))
        vm = reg_metrics(from_t(predict(model, Xva, device)), yva)
        sched.step(vm["mae"] if np.isfinite(vm["mae"]) else tot)
        history.append({"epoch": epoch, "train_loss": tot, "val_mae": vm["mae"], "val_r": vm["r"]})
        tag = ""
        if np.isfinite(vm["mae"]) and vm["mae"] < best_val - 1e-4:
            best_val = vm["mae"]; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "mu": mu.tolist(), "sd": sd.tolist(),
                        "y_mean": y_mean, "y_std": y_std,
                        "feat_names": data["feat_names"], "target": args.target}, weights)
            tag = "  *best"
        else:
            bad += 1
        log(f"epoch {epoch:3d}/{args.epochs}  loss {tot:6.3f}  val MAE {vm['mae']:6.3f}  r {vm['r']:+.2f}{tag}")
        with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_mae", "val_r"]); w.writeheader(); w.writerows(history)
        if bad >= args.patience:
            log(f"early stop (no val gain {args.patience} epochs)"); break

    if best_state:
        model.load_state_dict(best_state)

    # ---- test: model vs naive, per-segment AND per-subject, + memorisation check ----
    pred_te = from_t(predict(model, Xte, device))
    naive_val = float(ytr.mean())
    naive_te = np.full(len(yte), naive_val)

    seg_model = reg_metrics(pred_te, yte)
    seg_naive = reg_metrics(naive_te, yte)
    subj_p, subj_t = per_subject(pred_te, yte, data["subject"][idx["test"]])
    subj_model = reg_metrics(subj_p, subj_t)
    subj_naive = reg_metrics(np.full(len(subj_t), naive_val), subj_t)

    # memorisation: train-subject error vs test-subject error
    tr_model = reg_metrics(from_t(predict(model, Xtr, device)), ytr)

    log("\n=== TEST (held-out subjects) — hemoglobin ===")
    log(f"  per-SEGMENT  model : MAE {seg_model['mae']:.3f}  RMSE {seg_model['rmse']:.3f}  r {seg_model['r']:+.2f}  (n={seg_model['n']})")
    log(f"  per-SEGMENT  naive : MAE {seg_naive['mae']:.3f}  (predict mean train Hb = {naive_val:.2f})")
    log(f"  per-SUBJECT  model : MAE {subj_model['mae']:.3f}  RMSE {subj_model['rmse']:.3f}  r {subj_model['r']:+.2f}  (n={subj_model['n']} subjects)")
    log(f"  per-SUBJECT  naive : MAE {subj_naive['mae']:.3f}")
    log(f"\n  memorisation check: train-subject MAE {tr_model['mae']:.3f}  vs  test-subject MAE {seg_model['mae']:.3f}")
    gap = seg_model["mae"] - tr_model["mae"]
    log(f"    gap {gap:+.3f} ({'LARGE gap -> likely memorising subjects, not learning Hb' if gap > 1.0 else 'small gap -> generalising, not just memorising'})")
    beats = seg_model["mae"] < seg_naive["mae"]
    log(f"\n  -> model {'BEATS' if beats else 'does NOT beat'} naive baseline"
        + ("" if beats else "  (expected for Hb-from-rPPG; the signal may not be recoverable)"))

    json.dump({"target": args.target, "naive_value": naive_val,
               "segment": {"model": seg_model, "naive": seg_naive},
               "subject": {"model": subj_model, "naive": subj_naive},
               "train_subject": tr_model, "memorisation_gap": gap, "beats_naive": bool(beats)},
              open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)

    # scatter: predicted vs true (per-subject is the honest view)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
        for a, (p, t, ttl, mm) in zip(ax, [(pred_te, yte, "per-segment", seg_model),
                                           (subj_p, subj_t, "per-subject", subj_model)]):
            a.scatter(t, p, s=14, alpha=0.5, color="teal")
            lo = min(np.min(t), np.min(p)) - 0.5; hi = max(np.max(t), np.max(p)) + 0.5
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.axhline(naive_val, color="gray", ls=":", lw=1, label=f"naive ({naive_val:.1f})")
            a.set_xlabel(f"true {args.target}"); a.set_ylabel(f"predicted {args.target}")
            a.set_title(f"{ttl}: MAE {mm['mae']:.2f}, r {mm['r']:+.2f}"); a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.suptitle(f"Hemoglobin — model vs naive (held-out subjects)", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(args.out_dir, "hb_test_scatter.png"), dpi=130); plt.close()
        log(f"  scatter -> {os.path.join(args.out_dir, 'hb_test_scatter.png')}")
    except Exception as e:
        log(f"  (plot skipped: {e})")

    log(f"\nweights -> {weights}")


if __name__ == "__main__":
    main()