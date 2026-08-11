import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import config
from common.data_types import PredictionTask, WaveformResult


def plot_segment(
    task: PredictionTask,
    signals: np.ndarray,
    fps: float,
    waveforms: dict[str, WaveformResult],
    ppg_wave: np.ndarray,
    output_path: str,
    hr_model_prediction: tuple[float, float, np.ndarray, np.ndarray] | None = None,
    hb_pred: float | None = None,
) -> None:
    """
    Plot signal waveforms, HR labels, model predictions, and Hb predictions.

    Args:
        task: Processing task describing the dataset sample.
        signals: Uniformly sampled RGB signals with shape (N, R, 3).
        fps: Sampling frequency of the dataset signals.
        waveforms: Pulse waveforms and HR estimates for each extraction method.
        ppg_wave: PPG waveform corresponding to the dataset segment.
        output_path: Path where the plot is written.
        hr_model_prediction: Optional tuple containing:
            - Predicted heart rate in BPM.
            - HR prediction confidence.
            - HR-frequency bins in BPM.
            - Spectral probability distribution.
        hb_pred: Optional tuple containing:
            - Predicted hemoglobin value.
            - Hemoglobin prediction confidence.
    """
    # Build the time axis for the uniformly sampled RGB signals.
    n_frames = signals.shape[0]
    time = np.arange(n_frames) / fps # type: ignore
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=False) # type: ignore

    # Plot the extracted pulse waveforms and their HR estimates.
    for axis, method in zip(axes[0:3], ("POS", "CHROM", "green")):
        waveform = np.asarray(waveforms[method].waveform, dtype=np.float64)
        waveform = (waveform - np.nanmean(waveform)) / (np.nanstd(waveform) + 1e-9)
        axis.plot(time[:len(waveform)], waveform, color=config.RPPG_METHOD_COLORS[method], linewidth=0.9)
        hr_text = (
            f"{waveforms[method].heart_rate:.1f} BPM (conf {waveforms[method].confidence:.0f})"
            if waveforms[method].heart_rate is not None else "no peak"
        )
        axis.set_ylabel(f"{method}\n{hr_text}")
        axis.grid(alpha=0.3)

    # Plot the HR reference, using the PPG waveform when available.
    if len(ppg_wave) > 4:
        sampling_frequency = task.ppg_info.sampling_frequency if task.ppg_info else 100.0
        ppg_time = np.arange(len(ppg_wave)) / sampling_frequency
        normalized_ppg = (ppg_wave - np.mean(ppg_wave)) / (np.std(ppg_wave) + 1e-9)
        axes[3].plot(ppg_time, normalized_ppg, color="crimson", linewidth=0.8)
        axes[3].set_ylabel(f"PPG (label)\n{task.hr_true:.1f} BPM" if task.hr_true is not None else "PPG (label)")
        axes[3].grid(alpha=0.3)
    elif task.hr_true is not None:
        text = f"HR ground truth (from CSV): {task.hr_true:.0f} BPM"
        axes[3].text(
            0.5, 0.5, text, ha="center", va="center", transform=axes[3].transAxes, 
            fontsize=12, fontweight="bold", color="crimson"
        )
        axes[3].set_ylabel("HR label\n(pulse)")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
    else:
        axes[3].text(0.5, 0.5, "no PPG for this segment", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_ylabel("PPG (label)")
        axes[3].grid(alpha=0.3)

    # Plot the trained HR model's predicted probability spectrum.
    if hr_model_prediction is not None:
        model_hr, hr_conf, bpm_grid, probability = hr_model_prediction
        axes[4].plot(bpm_grid, probability, color="purple", linewidth=1.2)
        axes[4].axvline(model_hr, color="purple", linestyle="--", linewidth=1)
        if task.hr_true is not None:
            axes[4].axvline(task.hr_true, color="crimson", linestyle=":", linewidth=1, label="HR label")
            axes[4].legend(fontsize=7, loc="upper right")
        axes[4].set_ylabel(f"model spectrum\n{model_hr:.1f} BPM (conf {hr_conf:.2f})")
        axes[4].set_xlabel("HR (BPM)")
        axes[4].grid(alpha=0.3)
    else:
        axes[4].text(
            0.5, 0.5, "Trained model (pending)", ha="center", va="center", transform=axes[4].transAxes,
            color="gray", style="italic"
        )
        axes[4].set_ylabel("model")
        axes[4].set_xlabel("HR (BPM)")
        axes[4].grid(alpha=0.3)

    # Build the plot title from the available HR and Hb predictions.
    title = task.name
    if hr_model_prediction is not None:
        model_hr = hr_model_prediction[0]
        if task.hr_true is not None:
            hr_error = abs(model_hr - task.hr_true)
            title += f"   —   HR: {model_hr:.1f} pred / {task.hr_true:.1f} GT (err {hr_error:.1f})"
        else:
            title += f"   —   HR: {model_hr:.1f} BPM (no GT)"
    elif task.hr_true is not None:
        title += f"   —   HR: {task.hr_true:.1f} GT"
    if hb_pred is not None:
        if task.hb_true is not None:
            hb_error = abs(hb_pred - task.hb_true)
            title += f"   |   Hb: {hb_pred:.2f} pred / {task.hb_true:.2f} GT (err {hb_error:.2f})"
        else:
            title += f"   |   Hb: {hb_pred:.2f} g/dL (no GT)"
    elif task.hb_true is not None:
        title += f"   |   Hb: {task.hb_true:.2f} g/dL GT"

    fig.suptitle(title, fontweight="bold") # type: ignore
    plt.tight_layout(rect=[0, 0, 1, 0.99]) # type: ignore
    plt.savefig(output_path, dpi=120) # type: ignore
    plt.close()


def write_accuracy_plot(rows: list[dict[str, str]], out_dir: str) -> str | None:
    """
    Write a predicted-vs-ground-truth HR accuracy plot.

    Args:
        rows: Prediction result rows loaded from the results CSV.
        out_dir: Directory where the plot is written.

    Returns:
        Path to the generated plot, or None when there are too few
        labeled segments.
    """
    lab = np.array([float(row["hr_label"]) for row in rows ])
    if np.isfinite(lab).sum() < 5:
        print("  (accuracy plot skipped: fewer than 5 labeled segments)")
        return None

    panels = [
        ("POS", "hr_pos", "conf_pos"),
        ("CHROM", "hr_chrom", "conf_chrom"),
        ("green", "hr_green", "conf_green"),
    ]

    model_hr = np.array([float(row["hr_pred"]) for row in rows])
    has_model = np.isfinite(model_hr).sum() >= 5
    if has_model:
        panels.append(("model", "hr_pred", "hr_pred_conf"))

    finite_labels = lab[np.isfinite(lab)]
    limits = [
        max(30, np.floor(finite_labels.min() / 10) * 10 - 5),
        np.ceil(finite_labels.max() / 10) * 10 + 5
    ]
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5.4))

    if n_panels == 1:
        axes = [axes]
    for axis, (name, hr_column, confidence_column) in zip(axes, panels):
        hr = np.array([float(row.get(hr_column)) for row in rows])
        valid = np.isfinite(hr) & np.isfinite(lab)
        if valid.sum() == 0:
            axis.set_title(f"{name}: no scorable segments")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            continue

        if confidence_column is not None:
            confidence = np.array([float(row.get(confidence_column)) for row in rows])
            threshold = 10
            if confidence_column == "hr_pred_conf":
                threshold = 0.2
            high_conf = valid & (confidence >= threshold)
            low_conf = valid & (confidence < threshold)
            axis.scatter(
                lab[low_conf], hr[low_conf], s=10, facecolors="none", edgecolors="#bbbbbb",
                linewidths=0.6, label=f"low conf (<{threshold:.2f})"
            )
            axis.scatter(
                lab[high_conf], hr[high_conf], s=12, alpha=0.5,
                color="#1f77b4", label=f"high conf (≥{threshold:.2f})"
            )
        else:
            axis.scatter(lab[valid], hr[valid], s=12, alpha=0.5, color="purple", label="model")

        axis.plot(limits, limits, "k--", lw=1)
        absolute_error = np.abs(hr - lab)
        mae = np.nanmean(absolute_error[valid])
        bias = np.nanmean((hr - lab)[valid])
        within_6 = np.mean(absolute_error[valid] <= 6) * 100
        extra = ""

        if confidence_column is not None:
            confidence = np.array([float(row.get(confidence_column)) for row in rows])
            high_conf = valid & (confidence >= np.nanmedian(confidence[valid]))
            if high_conf.any():
                extra = (
                    f"  |  hi-conf: MAE {np.nanmean(absolute_error[high_conf]):.1f}, "
                    f"w6 {np.mean(absolute_error[high_conf] <= 6) * 100:.0f}%"
                )

        axis.set_title(f"{name}\nMAE {mae:.1f}, bias {bias:+.1f}, w6 {within_6:.0f}%{extra}", fontsize=10)
        axis.set_xlabel("PPG label HR (BPM)")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("predicted HR (BPM)")
    fig.suptitle(
        f"HR accuracy vs PPG label — {int(np.isfinite(lab).sum())} labeled segments"
        + "  (incl. trained model)" if has_model else "",
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = f"{out_dir}/hr_accuracy.png"
    plt.savefig(output_path, dpi=130)
    plt.close()
    return output_path


def write_hb_accuracy_plot(rows: list[dict[str, str]], out_dir: str) -> str | None:
    """
    Write Hb prediction accuracy plots.

    The output contains both per-segment and per-subject
    predicted-vs-true Hb comparisons.

    Args:
        rows: Prediction result rows loaded from the results CSV.
        out_dir: Directory where the plot is written.

    Returns:
        Path to the generated plot, or None when there are too few
        valid predictions.
    """
    predicted = np.array([float(row.get("hb_pred")) for row in rows])
    true = np.array([float(row.get("hb_label")) for row in rows])
    valid = np.isfinite(predicted) & np.isfinite(true)

    if valid.sum() < 5:
        print("  (Hb accuracy plot skipped: fewer than 5 segments with both prediction and truth)")
        return None

    # Aggregate segment predictions by subject.
    subjects = np.array([
        (   
            re.match(r"(\d+)", row.get("clip", "") or row.get("segment", "")).group(1)
            if re.match(r"(\d+)", row.get("clip", "") or row.get("segment", "")) else "?"
        )
        for row in rows
    ])

    subject_predictions: dict[str, list[float]] = {}
    subject_truth: dict[str, float] = {}

    for index in np.where(valid)[0]:
        subject_predictions.setdefault(subjects[index], []).append(predicted[index])
        subject_truth[subjects[index]] = true[index]

    subject_ids = sorted(subject_predictions)
    subject_pred = np.array([np.mean(subject_predictions[subject]) for subject in subject_ids])
    subject_true = np.array([subject_truth[subject] for subject in subject_ids])

    def stats(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
        absolute_error = np.abs(prediction - target)
        mae = float(np.mean(absolute_error))
        bias = float(np.mean(prediction - target))
        correlation = (
            float(np.corrcoef(prediction, target)[0, 1])
            if (len(prediction) > 1 and np.std(prediction) > 0 and np.std(target) > 0)
            else float("nan")
        )
        residual_sum = float(np.sum((target - prediction) ** 2))
        total_sum = float(np.sum((target - np.mean(target)) ** 2))
        r2 = 1.0 - residual_sum / total_sum if total_sum > 1e-12 else float("nan")
        return mae, bias, correlation, r2

    segment_mae, segment_bias, segment_r, segment_r2 = stats(predicted[valid], true[valid])
    subject_mae, subject_bias, subject_r, subject_r2 = stats(subject_pred, subject_true)

    # Naive baseline: predict the mean Hb for everyone.
    naive_segment_mae = float(np.mean(np.abs(np.mean(true[valid]) - true[valid])))
    naive_subject_mae = (
        float(np.mean(np.abs(np.mean(subject_true) - subject_true))) if len(subject_true) else float("nan")
    )
    mean_hb_segment = float(np.mean(true[valid]))
    mean_hb_subject = float(np.mean(subject_true)) if len(subject_true) else float("nan")
    all_values = np.concatenate([true[valid], predicted[valid]])
    limits = [np.floor(np.nanmin(all_values)) - 0.5, np.ceil(np.nanmax(all_values)) + 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))

    # Per-segment plot.
    axes[0].scatter(true[valid], predicted[valid], s=12, alpha=0.4, color="#8c564b", label="segment")
    axes[0].plot(limits, limits, "k--", lw=1)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].axhline(mean_hb_segment, color="gray", ls=":", lw=1.2, label=f"naive=mean Hb ({mean_hb_segment:.1f})")
    axes[0].set_xlabel("true Hb (g/dL)")
    axes[0].set_ylabel("predicted Hb (g/dL)")
    verdict_segment = "BEATS naive" if segment_mae < naive_segment_mae else "WORSE than naive" 
    axes[0].set_title(
        f"per-segment\n"
        f"model MAE {segment_mae:.2f}  |  "
        f"naive MAE {naive_segment_mae:.2f} "
        f"({verdict_segment})\n"
        f"bias {segment_bias:+.2f}, "
        f"r {segment_r:.2f}, "
        f"R² {segment_r2:.2f}  "
        f"(n={int(valid.sum())})",
        fontsize=9
    )
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    # Per-subject plot.
    axes[1].scatter(
        subject_true, subject_pred, s=40, alpha=0.7, color="#9467bd",
        edgecolors="k", linewidths=0.5, label="subject"
    )
    axes[1].plot(limits, limits, "k--", lw=1)
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].axhline(mean_hb_subject, color="gray", ls=":", lw=1.2, label=f"naive=mean Hb ({mean_hb_subject:.1f})")
    axes[1].set_xlabel("true Hb (g/dL)")
    axes[1].set_ylabel("mean predicted Hb (g/dL)")
    verdict_subject = "BEATS naive" if subject_mae < naive_subject_mae else "WORSE than naive"
    axes[1].set_title(
        f"per-subject (mean pred)\n"
        f"model MAE {subject_mae:.2f}  |  "
        f"naive MAE {naive_subject_mae:.2f} "
        f"({verdict_subject})\n"
        f"bias {subject_bias:+.2f}, "
        f"r {subject_r:.2f}, "
        f"R² {subject_r2:.2f}  "
        f"({len(subject_ids)} subjects)",
        fontsize=9
    )
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Hb accuracy — predicted vs true (naive = predict mean Hb)", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = f"{out_dir}/hb_accuracy.png"
    plt.savefig(output_path, dpi=130)
    plt.close()
    return output_path