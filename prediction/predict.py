"""Run HR and hemoglobin prediction over prepared dataset segments.

Inputs
------
Dataset directory
    --dataset-dir
        Directory containing prepared dataset segment archives.

Manifest
    --manifest
        CSV manifest describing each dataset segment, including segment
        timing, ground-truth HR, hemoglobin, and optional PPG metadata.

PPG directory
    --ppg-dir
        Directory containing contact PPG (.PW) files referenced by the
        segment manifest.

Optional trained models
    --hr-model
        Path to a trained HR model. When provided, each segment receives
        a trained HR prediction and confidence.

    --hb-model
        Path to a trained hemoglobin model. When provided, each segment
        receives a trained Hb prediction.

Processing
----------
For each dataset segment:

    - Load the uniformly sampled RGB signals and pixel-count information.
    - Extract HR using POS, CHROM, and the raw green-channel methods.
    - Load the corresponding contact PPG when available.
    - Use the manifest PPG HR as the segment-level HR ground truth.
    - Optionally run the trained HR model.
    - Optionally run the trained Hb model.
    - Save a per-segment prediction plot when plotting is enabled.

Segments are processed in parallel using ProcessPoolExecutor.

Outputs
-------
<out-dir>/hr_results.csv
    Per-segment prediction results containing:
        - segment metadata
        - POS HR and confidence
        - CHROM HR and confidence
        - green-channel HR and confidence
        - HR ground truth
        - trained HR prediction and confidence
        - hemoglobin ground truth
        - trained Hb prediction and confidence

<out-dir>/plots/<segment>.png
    Visualization of the individual segment's HR and Hb predictions.

<out-dir>/hr_accuracy.png
    Aggregate predicted-versus-ground-truth HR accuracy plot for the
    available prediction methods and trained HR model.

<out-dir>/hb_accuracy.png
    Aggregate predicted-versus-ground-truth hemoglobin accuracy plot,
    including per-segment and per-subject evaluation.

Usage
-----
python predict.py \
    --dataset-dir output/dataset \
    --manifest output/dataset/segments_manifest.csv \
    --ppg-dir data/ppg \
    --out-dir output/prediction \
    [--hr-model output/hr_model/hr_model.pt] \
    [--hb-model output/hb_model/hb_model.pt] \
    [--no-plot]
"""

import os
import csv
import argparse
from dataclasses import asdict, fields
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np

from common.data_types import (
    FileExtension, WindowInfo, PredictionTask, PredictionRecord, PredictionResult, PPGSignal
)
from common.ppg import load_pw, ppg_segment
from prediction.models import predict_hr, predict_hb
from prediction.rppg_algorithms import method_waveforms
from prediction.visualization import plot_segment, write_accuracy_plot, write_hb_accuracy_plot


def _default_workers():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def process_segment(task: PredictionTask) -> PredictionResult:
    """
    Process a single dataset sample and generate HR and Hb predictions.

    Args:
        task: Processing task describing the dataset sample and prediction options.

    Returns:
        Prediction result for the dataset sample.
    """
    try:
        # Load the uniformly sampled signals and quality information.
        data = np.load(task.sample_path)
        signals = data["signals"]
        fps = float(data["fps"])
        pixel_counts = data["pixel_counts"]
        waveforms = method_waveforms(signals, fps, pixel_counts)
        ppg_wave = np.array([], dtype=np.float64)
        hr_pred: float | None = None
        hr_pred_conf: float | None = None
        hb_pred: float | None = None
        device = torch.device("cpu")
        hr_model_prediction = None

        # Generate optional trained-model predictions.
        if task.hr_model:
            hr_model_prediction = predict_hr(task.hr_model, task.sample_path, device)
            hr_pred, hr_pred_conf, _, _ = hr_model_prediction

        # Generate optional trained-model predictions.
        if task.hb_model:
            hb_pred = predict_hb(task.hb_model, task.sample_path, device)

        # Build the prediction record for this dataset sample.
        prediction = PredictionRecord(
            segment=task.name,
            clip=task.segment.clip,
            t_start=task.segment.t_start,
            t_end=task.segment.t_end,
            hr_pos=waveforms["POS"].heart_rate,
            conf_pos=waveforms["POS"].confidence,
            hr_chrom=waveforms["CHROM"].heart_rate,
            conf_chrom=waveforms["CHROM"].confidence,
            hr_green=waveforms["green"].heart_rate,
            conf_green=waveforms["green"].confidence,
            hr_label=task.hr_true,
            hr_pred=hr_pred,
            hr_pred_conf=hr_pred_conf,
            hb_label=task.hb_true,
            hb_pred=hb_pred
        )

        # Plot the prediction results
        if not task.no_plot:
            # Load ppg wave if available
            if task.ppg_info:
                ppg_wave = ppg_segment(task.ppg_info, task.segment.t_start, task.segment.t_end)
            plots_dir = os.path.join(task.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            plot_segment(
                task=task,
                signals=signals,
                fps=fps,
                waveforms=waveforms,
                ppg_wave=ppg_wave,
                output_path=os.path.join(plots_dir, f"{task.name}.png"),
                hr_model_prediction=hr_model_prediction,
                hb_pred=hb_pred
            )
        return PredictionResult(name=task.name, prediction=prediction, success=True, error=None)
    except Exception as exc:
        return PredictionResult(name=task.name, prediction=None, success=False, error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    """
    Predict HR and Hb over dataset segments.
    """
    # Parse arguments.
    ap = argparse.ArgumentParser(
        description="HR (DSP + trained model) and Hb prediction over dataset segments."
    )
    ap.add_argument("--dataset-dir", default="output/dataset")
    ap.add_argument("--manifest", default="output/dataset/segments_manifest.csv")
    ap.add_argument("--ppg-dir", default="data/ppg")
    ap.add_argument("--out-dir", default="output/prediction")
    ap.add_argument("--no-plot", action="store_true", help="Disable prediction plots.")
    ap.add_argument("--hr-model", default=None, help="Optional path to a trained HR model.")
    ap.add_argument("--hb-model", default=None, help="Optional path to a trained Hb model.")
    ap.add_argument("--workers", type=int, default=0,
        help="Number of worker processes (0 = all available cores)."
    )
    args = ap.parse_args()

    # Create the output directory.
    os.makedirs(args.out_dir, exist_ok=True)
    # Cache PPG files so each file is loaded only once.
    ppg_cache: dict[str, PPGSignal] = {}
    # Build prediction tasks.
    tasks: list[PredictionTask] = []

    # Load segment metadata from the manifest
    manifest = None
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest file not found: {args.manifest}")
    with open(args.manifest, encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    print(f"Loaded {len(manifest)} segment(s)")

    if not manifest:
        print("Nothing to process.")
        return

    for row in manifest:
        segment_name = row["segment"]
        sample_path = os.path.join(args.dataset_dir, segment_name + FileExtension.DATASET_SAMPLE)
        if not os.path.exists(sample_path):
            print(f"WARNING: sample missing: {sample_path}")
            continue

        # Load the PPG referenced by this segment
        ppg_info = None
        ppg_file = row.get("ppg_file", "").strip()
        if ppg_file:
            if ppg_file not in ppg_cache:
                ppg_path = os.path.join(args.ppg_dir, ppg_file)
                if os.path.exists(ppg_path):
                    ppg_cache[ppg_file] = load_pw(ppg_path)
                else:
                    print(f"WARNING: PPG file missing: {ppg_path}")
            ppg_info = ppg_cache.get(ppg_file)

        # Create the segment/window metadata required by PredictionTask
        window = WindowInfo(
            segment=segment_name,
            clip=row["clip"],
            index=int(row["index"]),
            t_start=float(row["t_start"]),
            t_end=float(row["t_end"]),
            abs_start=float(row["abs_start"]),
        )

        # Create the prediction task
        tasks.append(PredictionTask(
            name=segment_name,
            sample_path=sample_path,
            segment=window,
            ppg_info=ppg_info,
            output_dir=args.out_dir,
            no_plot=args.no_plot,
            hr_model=args.hr_model,
            hb_model=args.hb_model,
            hr_true=float(row["ppg_hr"]) if row.get("ppg_hr") else float(row["pulse"]),
            hb_true=float(row["hemoglobin"])
        ))

    print(f"Created {len(tasks)} prediction task(s)")
    if not tasks:
        print("Nothing to process.")
        return

    # Process prediction tasks in parallel
    predictions: list[PredictionResult] = []
    workers = min(args.workers if args.workers > 0 else _default_workers(), len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_segment, task): task.name for task in tasks}
        # Collect results as workers finish
        for index, future in enumerate(as_completed(futures), 1):
            prediction = future.result()
            if prediction.success:
                print(f"[{index}/{len(tasks)}] {prediction.name}: ok")
                predictions.append(prediction)
            else:
                print(f"[{index}/{len(tasks)}] {prediction.name}: ERROR {prediction.error}")

    # Sort prediction results
    predictions.sort(key=lambda result: (result.prediction.segment if result.prediction else result.name))

    # Save prediction results to CSV
    csv_path = os.path.join(args.out_dir, "hr_results.csv")
    field_names = [field.name for field in fields(PredictionRecord)]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(
            asdict(result.prediction) for result in predictions if result.prediction is not None
        )

    print(f"\nGenerated {len(predictions)} prediction result(s)")
    print(f"Results -> {csv_path}")

    # Build rows for accuracy reporting and visualization.
    rows = [asdict(result.prediction) for result in predictions if result.prediction is not None]

    # Convert a CSV value to float, returning NaN for missing values.
    def col(name: str) -> np.ndarray:
        return np.array([float(row[name]) if row.get(name, "") not in ("", None) else np.nan for row in rows ])

    # Print aggregate HR accuracy for each prediction method.
    lab = col("hr_label")
    if np.isfinite(lab).sum() >= 5:
        print("\n=== HR accuracy vs PPG label ===")
        for name in ("hr_pos", "hr_chrom", "hr_green", "hr_pred"):
            hr = col(name)
            ok = np.isfinite(hr) & np.isfinite(lab)
            if ok.sum() == 0:
                continue
            error = np.abs(hr - lab)[ok]
            bias = np.mean((hr - lab)[ok])
            label = name.replace("hr_", "")
            print(
                f"  {label:6} MAE {np.mean(error):6.2f}  w6 {np.mean(error <= 6) * 100:4.0f}%  "
                f"bias {bias:+6.2f}  (n={ok.sum()})"
            )

    # Print aggregate Hb accuracy against the ground truth.
    hb_pred = col("hb_pred")
    hb_true = col("hb_label")
    ok_hb = np.isfinite(hb_pred) & np.isfinite(hb_true)
    if ok_hb.sum() >= 5:
        pred = hb_pred[ok_hb]
        true = hb_true[ok_hb]
        error = np.abs(pred - true)
        bias = np.mean(pred - true)
        correlation = np.corrcoef(pred, true)[0, 1] if np.std(pred) > 0 and np.std(true) > 0 else float("nan")
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) 
        naive_mae = np.mean(np.abs(np.mean(true) - true))

        print("\n=== Hb accuracy vs ground truth ===")
        print(
            f"  Hb     MAE {np.mean(error):6.2f} g/dL  bias {bias:+6.2f}  "
            f"r {correlation:.2f}  R2 {r2:.2f}  (n={int(ok_hb.sum())})"
        )
        print(
            f"  naive  MAE {naive_mae:6.2f} g/dL  (predict mean Hb) -> model "
            f"{'BEATS' if np.mean(error) < naive_mae else 'does NOT beat'} naive"
        )

    # Generate final aggregate plots.
    if not args.no_plot:
        accuracy_plot = write_accuracy_plot(rows, args.out_dir)
        if accuracy_plot:
            print(f"accuracy plot -> {accuracy_plot}")
        hb_accuracy_plot = write_hb_accuracy_plot(rows, args.out_dir)
        if hb_accuracy_plot:
            print(f"Hb accuracy plot -> {hb_accuracy_plot}")


if __name__ == "__main__":
    main()