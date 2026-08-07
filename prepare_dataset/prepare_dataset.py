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
from common.data_types import DatasetTask, DatasetResult, FileExtension, WindowInfo
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


def _write_manifest(windows: list[WindowInfo], output_dir: str) -> None:
    """
    Save metadata describing every emitted dataset window.

    The manifest preserves the correspondence between generated training
    samples and their original locations within each source clip.

    Args:
        windows: Metadata for all emitted windows.
        output_dir: Dataset output directory.
    """
    windows.sort(key=lambda window: (window.clip, window.index))

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
        ])

        for window in windows:
            writer.writerow([
                window.segment,
                window.clip,
                window.index,
                window.t_start,
                window.t_end,
                window.abs_start,
            ])


def process_dataset(task: DatasetTask) -> DatasetResult:
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

            # Mark rejected facial regions as unavailable.
            resampled_signals[:, ~window.region_mask] = np.nan
            resampled_pixel_counts[:, ~window.region_mask] = np.nan

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


# ============================================================================
# Command line interface
# ============================================================================
def main() -> None:
    """
    Generate uniformly sampled dataset windows from extracted rPPG signals.
    """
    # Parse arguments.
    ap = argparse.ArgumentParser(
        description=("Generate fixed-length training windows from extracted rPPG signals.")
    )
    ap.add_argument("--signals-dir", default="output/signals")
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
        futures = {pool.submit(process_dataset, task): task.name for task in tasks}

        # Report completed tasks as workers finish.
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            manifest.extend(result.windows)
            total_windows += result.count
            print(f"[{index}/{len(tasks)}] {result.log}")

    # Save dataset metadata.
    _write_manifest(manifest, args.out_dir)
    print(f"\nGenerated {total_windows} dataset window(s)")
    print(f"Manifest -> {os.path.join(args.out_dir, 'segments_manifest.csv')}")


if __name__ == "__main__":
    main()