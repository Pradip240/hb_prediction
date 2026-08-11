"""
Prepare uniformly sampled dataset windows from extracted rPPG signals.

Inputs
------
Input directory

    --signals-dir
        Directory containing extracted signal archives.

Command-line arguments

    --out-dir
        Directory where generated dataset windows are written.

    --workers
        Number of parallel worker processes.

    --overwrite
        Regenerate existing dataset windows.

Outputs
-------
For each accepted window <clip>_<k>, writes

    <out-dir>/<clip>_<k>_signals.npz

containing:

    signals : float32, shape (N, R, 3)
        Uniformly resampled RGB signals.

    pixel_counts : float32, shape (N, R)
        Uniformly resampled per-region pixel counts.

    fps : float32
        Sampling frequency of the resampled signals.

Additionally writes

    <out-dir>/segments_manifest.csv

containing metadata for every emitted dataset window.

Processing
----------
For each extracted signal archive:

    - Load extracted signals.
    - Identify valid signal frames.
    - Generate valid training windows.
    - Resample accepted windows onto a uniform timeline.
    - Save dataset windows.
    - Record window metadata.
"""

import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import config
from common.data_types import DatasetTask, DatasetResult, FileExtension, WindowInfo, PPGSignal
from common.ppg import load_pw, ppg_segment
from common.signal_processing import spectral_hr
from prepare_dataset.window_generation import find_windows
from prepare_dataset.signal_interpolation import resample_region_signals, resample_pixel_counts


def _default_workers() -> int:
    """
    Determine the default number of worker processes.

    Returns:
        Number of available CPU cores.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _write_manifest(windows: list[WindowInfo], output_dir: str, ppg_dir: str, ground_truth: str) -> None:
    """
    Save metadata describing every emitted dataset window.

    The manifest preserves the correspondence between generated training
    samples and their original locations within each source clip.

    Args:
        windows: Metadata for all emitted windows.
        output_dir: Dataset output directory.
        ppg_dir: Directory containing the contact PPG files. 
        ground_truth: Path to the ground-truth CSV file.
    """
    windows.sort(key=lambda window: (window.clip, window.index))

    # Store ground-truth and PPG metadata by clip name.
    ground_truth_data: dict[str, tuple[str, float, float, str, PPGSignal | None]] = {}

    with open(ground_truth, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video = row["video"].strip()
            clip = os.path.splitext(os.path.basename(video))[0]
            patient_id = row["patient_id"].strip()
            if not patient_id:
                continue
            pulse = float(row["pulse"])
            hemoglobin = float(row["hemoglobin"])

            # Default values when no matching PPG file is available.
            ppg_file = ""
            ppg_data = None

            # Add PPG data if available
            parts = clip.split("_")
            if len(parts) > 2:
                subject_id = parts[0]
                state = parts[2]
                ppg_file = os.path.join(ppg_dir, f"{subject_id}_{state}.PW")
                if os.path.isfile(ppg_file):
                    ppg_data = load_pw(ppg_file)

            # Store all metadata so it can be added to every segment generated from this clip.
            ground_truth_data[clip] = (
                patient_id,
                pulse,
                hemoglobin,
                ppg_file,
                ppg_data
            )

    manifest_path = os.path.join(output_dir, "segments_manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "segment",
            "clip",
            "index",
            "t_start",
            "t_end",
            "abs_start",
            "patient_id",
            "pulse",
            "hemoglobin",
            "ppg_file",
            "ppg_hr",
            "ppg_start",
            "ppg_end",
            "ppg_sampling_frequency"
        ])

        for index, window in enumerate(windows, 1):
            print(f"[{index}/{len(windows)}] {window.segment}")
            # Match each generated segment with its clip-level ground-truth and PPG metadata.
            data = ground_truth_data.get(window.clip)
            # Keep the row with empty metadata if ground truth is missing.
            patient_id = ""
            pulse = ""
            hemoglobin = ""
            ppg_file = ""
            ppg_hr = ""
            ppg_start = ""
            ppg_end = ""
            ppg_sampling_frequency = ""
            if not data:
                print(f"  [WARNING] No ground truth found: {window.clip}")
            else:
                patient_id, pulse, hemoglobin, ppg_file, ppg_data = data
                if not ppg_data:
                    print(f"  [WARNING] No PPG data: {window.clip}")
                else:
                    try:
                        # Convert the segment's relative time to the PPG's absolute time.
                        ppg_start = window.abs_start
                        ppg_end = window.abs_start + window.t_end - window.t_start
                        ppg_sampling_frequency = ppg_data.sampling_frequency
                        ppg_hr, _ = spectral_hr(ppg_segment(ppg_data, ppg_start, ppg_end), ppg_sampling_frequency)
                    except Exception as error:
                        print(f"  [ERROR] Failed to calculate PPG HR: {window.segment}: {error}")
            # Write window, ground-truth, and PPG metadata.
            writer.writerow([
                window.segment,
                window.clip,
                window.index,
                window.t_start,
                window.t_end,
                window.abs_start,
                patient_id,
                pulse,
                hemoglobin,
                ppg_file,
                ppg_hr,
                ppg_start,
                ppg_end,
                ppg_sampling_frequency
            ])


def process_dataset_task(task: DatasetTask) -> DatasetResult:
    """
    Process a single extracted signal archive.

    Performs:
        - window selection
        - signal interpolation
        - dataset generation

    Args:
        task: Processing task describing the input archive and output location.

    Returns:
        Processing result for the archive.
    """
    messages: list[str] = []
    windows: list[WindowInfo] = []

    try:
        # Load extracted signals.
        data = np.load(task.signal_path)
        signals = data["signals"]
        timestamps = data["timestamps"]
        pixel_counts = data["pixel_counts"]

        # Resample timestamps for 3 minute video
        timestamps = np.linspace(0, 180, len(timestamps), dtype=float)

        # A facial region is valid only when all three RGB channels are finite.
        valid_regions = np.isfinite(signals).all(axis=-1)

        # Generate candidate dataset windows.
        windows = find_windows(task.name, timestamps, valid_regions)

        # Nothing to emit.
        if not windows:
            return DatasetResult(
                name=task.name,
                log=f"{task.name}: no valid windows.",
                windows=[],
                count=0,
                success=True
            )

        messages.append(f"{task.name}: {len(windows)} window(s)")

        absolute_windows: list[WindowInfo] = []
        # Process each accepted window.
        for window in windows:
            output_path = os.path.join(task.output_dir, f"{window.segment}{FileExtension.DATASET_SAMPLE}")

            # Locate the source samples corresponding to the selected interval.
            first = np.searchsorted(timestamps, window.abs_start, side="left")
            last = np.searchsorted(timestamps, window.abs_start + config.WINDOW_SEC, side="right")

            window_times = timestamps[first:last]
            window_signals = signals[first:last]
            window_counts = pixel_counts[first:last]

            # Construct the target uniformly sampled timeline.
            output_times = (
                window.abs_start
                + np.arange(0.0, config.WINDOW_SEC, 1.0 / config.TARGET_FPS, dtype=np.float64)
            )

            # Interpolate RGB signals.
            resampled_signals = resample_region_signals(window_signals, window_times, output_times)
            # Interpolate quality metadata.
            resampled_pixel_counts = resample_pixel_counts(window_counts, window_times, output_times)

            if window.region_mask is None:
                raise ValueError(f"Region mask is missing for window '{window.segment}'.")

            # Mark rejected facial regions as unavailable.
            resampled_signals[:, ~window.region_mask] = np.nan
            resampled_pixel_counts[:, ~window.region_mask] = 0.0

            # Save the generated dataset sample.
            np.savez_compressed(
                output_path,
                signals=resampled_signals,
                pixel_counts=resampled_pixel_counts,
                region_mask=window.region_mask,
                fps=np.float32(config.TARGET_FPS)
            )
            absolute_windows.append(window)

        return DatasetResult(
            name=task.name,
            log="\n".join(messages),
            windows=absolute_windows,
            count=len(absolute_windows),
            success=True
        )

    except Exception as exc:
        return DatasetResult(
            name=task.name,
            log=f"{task.name}: ERROR {type(exc).__name__}: {exc}",
            windows=[],
            count=0,
            success=False
        )


def main() -> None:
    """
    Generate uniformly sampled dataset windows from extracted rPPG signals.
    """
    # Parse arguments.
    ap = argparse.ArgumentParser(
        description=("Generate fixed-length training windows from extracted rPPG signals.")
    )
    ap.add_argument("--signals-dir", default="output/signals")
    ap.add_argument("--ppg-dir", default="data/ppg")
    ap.add_argument("--ground-truth", default="data/ground_truth.csv")
    ap.add_argument("--out-dir", default="output/dataset")
    ap.add_argument("--workers", type=int, default=0,
        help="Number of worker processes (0 = all available cores)."
    )
    args = ap.parse_args()

    # Create the output directory.
    os.makedirs(args.out_dir, exist_ok=True)

    # List available signal archives.
    signal_files: list[str] = sorted(
        file for file in os.listdir(args.signals_dir) if file.endswith(FileExtension.SIGNALS) # type: ignore
    )
    print(f"Loaded {len(signal_files)} signal archive(s)")

    # Build one processing task per signal archive.
    tasks: list[DatasetTask] = []
    extension_length = len(FileExtension.SIGNALS)
    for signal_file in signal_files:
        name = signal_file[:-extension_length]
        tasks.append(
            DatasetTask(
                name=name,
                signal_path=os.path.join(args.signals_dir, signal_file),
                output_dir=args.out_dir
            )
        )

    # Exit if no signal archives are available.
    if not tasks:
        print("Tasks not generated. Nothing to process.")
        return

    # Launch worker processes.
    workers = min(args.workers if args.workers > 0 else _default_workers(), len(tasks))
    print(f"Processing {len(tasks)} signal archive(s) with {workers} worker(s)")

    manifest: list[WindowInfo] = []
    total_windows = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_dataset_task, task): task.name for task in tasks}

        # Report completed tasks as workers finish.
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            manifest.extend(result.windows)
            total_windows += result.count
            print(f"[{index}/{len(tasks)}] {result.log}")

    # Save dataset metadata.
    print("Generating manifest file...")
    _write_manifest(manifest, args.out_dir, args.ppg_dir, args.ground_truth)
    print(f"\nGenerated {total_windows} dataset window(s)")
    print(f"Manifest -> {os.path.join(args.out_dir, 'segments_manifest.csv')}")


if __name__ == "__main__":
    main()