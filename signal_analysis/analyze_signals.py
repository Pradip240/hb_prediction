"""
Extract per-region rPPG RGB signals from videos.

Inputs
------
Input directories

    --video-dir
        Directory containing input videos:
        (.mkv, .mp4, .avi, .mov).

    --seg-dir
        Directory containing saved face segmentation masks.

    --landmarks-dir
        Directory containing MediaPipe landmark archives.

Command-line arguments

    --signals-dir
        Directory where extracted signal archives are written.

    --plots-dir
        Directory where optional signal plots and overlay videos are written.

    --workers
        Number of parallel worker processes.

    --no-plot
        Disable RGB signal plots.

    --no-video
        Disable visualization overlay videos.

    --overwrite
        Recompute outputs even if they already exist.

Outputs
-------
For each processed video <video>, writes

    <signals-dir>/<video>_signals.npz

containing:

    signals : float64, shape (T, R, 3)
        Mean RGB values for each facial region.

    where:
        T: Number of processed frames.
        R: Number of regions defined in config.REGION_ORDER.
        3 (Channels): [Red, Green, Blue].

    timestamps : float64, shape (T,)
        Frame timestamps in seconds.

        When available these are extracted from video presentation timestamps (PTS).
        Otherwise a uniform FPS-based timeline is used.

    pixel_counts : int32, shape (T, R)
        Number of valid skin pixels contributing to each frame/region pair.

Processing
----------
For each frame:
    - Load segmentation mask.
    - Generate skin mask.
    - Smooth facial landmarks.
    - Intersect skin mask with landmark-defined regions.
    - Compute mean RGB values for each region.

Frames are aligned by index:

    frame[i] ↔ segmentation[i] ↔ landmarks[i] ↔ timestamp[i]

The number of processed frames is limited to the shortest available input.
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from common import config
from common import signal_processing as sp
from common.data_types import ClipResult, ClipTask, FileExtension, VIDEO_EXTENSIONS
from common.visualize import plot_signals, make_overlay_frame
from common.video_utils import video_timestamps, video_fps


def _init_worker() -> None:
    """
    Limit OpenCV threading inside worker processes.

    Prevents CPU oversubscription when multiple clips are processed in parallel.
    """
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


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


def process_clip(task: ClipTask) -> ClipResult:
    """
    Process a single video clip.

    Performs:
        - signal extraction
        - signal archive saving
        - optional signal plotting
        - optional overlay video generation

    Args:
        task:
            Processing task describing the input files, output paths,
            and processing options for a single clip.

    Returns:
        Processing result for the clip.
    """
    messages: list[str] = []
    try:
        # Determine which outputs need to be generated
        need_extract = task.overwrite or not os.path.exists(task.signal_path)
        need_plot = not task.no_plot and (task.overwrite or not os.path.exists(task.plot_path))
        need_overlay = not task.no_video and (task.overwrite or not os.path.exists(task.overlay_path))

        # Nothing needs to be generated.
        if not (need_extract or need_plot or need_overlay):
            return ClipResult(name=task.name, log=f"{task.name}: nothing to do.", success=True)

        # Load per-frame timestamps and determine the video frame rate.
        timestamps = np.asarray(video_timestamps(task.video_path), dtype=np.float32)
        fps = video_fps(task.video_path) or config.DEFAULT_FPS

        # Load inputs
        segmentation = np.load(task.segmentation_path)["masks"]
        landmarks = np.load(task.landmark_path)["landmarks"]
        n_frames = min(len(timestamps), len(segmentation), len(landmarks))
        signals = np.full((n_frames, len(config.REGION_ORDER), 3), np.nan, dtype=np.float64)
        pixel_counts = np.zeros((n_frames, len(config.REGION_ORDER)), dtype=np.int32)

        # Reduce landmark and segmentation jitter before downstream processing.
        smoothed_landmarks = sp.smooth_landmarks(landmarks[:n_frames, :, :2], fps)
        skin_masks = np.stack([sp.skin_mask_from_seg(mask) for mask in segmentation[:n_frames]])
        smoothed_masks = sp.smooth_masks(skin_masks, fps)

        # Create the erosion kernel once.
        kernel = sp.edge_kernel()

        # Open the video for sequential frame decoding.
        capture = cv2.VideoCapture(task.video_path)
        if not capture.isOpened():
            raise IOError(task.video_path)

        # Create overlay writer if needed.
        writer = None
        if need_overlay:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(task.overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) # type: ignore

        processed_frames = 0
        # Decode and process video frames.
        try:
            for frame_idx in range(n_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                processed_frames += 1

                # Skip frames with invalid landmarks.
                if not np.isfinite(smoothed_landmarks[frame_idx]).all():
                    continue

                # Build skin-constrained masks for all facial regions.
                region_masks = sp.build_region_masks(
                    frame.shape[:2], smoothed_landmarks[frame_idx], smoothed_masks[frame_idx], kernel
                )

                # Extract RGB signals.
                if need_extract:
                    signals[frame_idx], pixel_counts[frame_idx] = sp.region_mean_rgb_count(frame, region_masks)

                # Write overlay.
                if writer is not None:
                    vis = make_overlay_frame(frame, smoothed_masks[frame_idx], region_masks, smoothed_landmarks[frame_idx])
                    writer.write(vis)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        # Update processed frames
        signals = signals[:processed_frames]
        pixel_counts = pixel_counts[:processed_frames]
        timestamps = timestamps[:processed_frames]
        n_frames = processed_frames

        # Extract RGB signals
        if need_extract:
            # Save the signal
            np.savez_compressed(task.signal_path, signals=signals, timestamps=timestamps, pixel_counts=pixel_counts)

            # A region is valid if all three RGB values are finite
            valid_regions = np.isfinite(signals).all(axis=-1)   # Shape: (T, R)
            # A frame is valid if all regions are valid
            valid_frames = valid_regions.any(axis=-1)            # Shape: (T,)
            # Fraction of valid frames
            valid_fraction = valid_frames.mean() if signals.size else 0.0
            # Calculate duration
            duration = (timestamps[-1] - timestamps[0] if len(timestamps) else 0.0)

            # Add logs
            messages.append(
                f"{task.name}: {signals.shape}, fps:{fps:.2f}"
                f" ({len(timestamps)} frames, {duration:.1f}s) "
                f"{100 * valid_fraction:.0f}% valid -> {task.signal_path}"
            )

        # Load data if extraction of signal is not needed
        else:
            data = np.load(task.signal_path)
            signals = data["signals"]
            timestamps = data["timestamps"]
            messages.append(f"{task.name}: signals exist, skip extraction {signals.shape}")


        # Plot signals
        if not task.no_plot:
            if need_plot:
                plot_signals(signals, timestamps, task.name, task.plot_path)
                messages.append(f"  plot -> {task.plot_path}")
            else:
                messages.append("   plot exists, skip")


        # Send overlay video message
        if not task.no_video:
            if need_overlay:
                messages.append(f"    overlay ({n_frames} frames) -> {task.overlay_path}")
            else:
                messages.append("    overlay exists, skip")
        return ClipResult(task.name, "\n".join(messages), True)
    except Exception as exc:
        return ClipResult(task.name, f"{task.name}: ERROR {type(exc).__name__}: {exc}", False)



# ============================================================================
# Command line interface
# ============================================================================
def main() -> None:
    """
    Extract rPPG signals for all videos in a directory.
    """
    # Parse arguments
    ap = argparse.ArgumentParser(
        description=("Extract per-region rPPG RGB signals from segmentation masks and landmarks.")
    )
    ap.add_argument("--video-dir", default="data/videos")
    ap.add_argument("--seg-dir", default="output/seg")
    ap.add_argument("--landmarks-dir", default="output/landmarks")
    ap.add_argument("--signals-dir", default="output/signals")
    ap.add_argument("--plots-dir", default="output/plots")
    ap.add_argument("--workers", type=int, default=0,
        help=("Number of worker processes (0 = all available cores).")
    )
    ap.add_argument("--no-plot", action="store_true", help="Disable signal plots.")
    ap.add_argument("--no-video", action="store_true", help="Disable overlay videos.")
    ap.add_argument("--overwrite", action="store_true", help="Redo existing outputs.")
    args = ap.parse_args()

    # Create directories
    os.makedirs(args.signals_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # List available segmentation archives
    seg_files: list[str] = sorted(
        f for f in os.listdir(args.seg_dir) if f.endswith(FileExtension.SEGMENTATION) # type: ignore
    )
    print(f"Loaded {len(seg_files)} segmentation file(s)")

    # Build one processing task per clip
    tasks: list[ClipTask] = []
    ext_len = len(FileExtension.SEGMENTATION)
    for seg_file in seg_files:
        # Resolve input and output paths
        name = seg_file[:-ext_len]
        seg_path = os.path.join(args.seg_dir, seg_file)
        landmark_path = os.path.join(args.landmarks_dir, f"{name}{FileExtension.LANDMARKS}")
        video_path = None
        for extension in VIDEO_EXTENSIONS:
            for candidate in (name + extension, name + extension.upper()):
                path = os.path.join(args.video_dir, candidate)
                if os.path.exists(path):
                    video_path = path
                    break
        # Skip incomplete clips
        if video_path is None:
            print(f"{name}: video missing, skip")
            continue
        if not os.path.exists(landmark_path):
            print(f"{name}: landmarks missing, skip")
            continue
        if not os.path.exists(seg_path):
            print(f"{name}: segmentation missing, skip")
            continue

        # Create processing task
        tasks.append(ClipTask(
            name=name,
            video_path=video_path,
            segmentation_path=seg_path,
            landmark_path=landmark_path,
            signal_path=os.path.join(args.signals_dir, f"{name}{FileExtension.SIGNALS}"),
            plot_path=os.path.join(args.plots_dir, f"{name}{FileExtension.SIGNAL_PLOT}"),
            overlay_path=os.path.join(args.plots_dir, f"{name}{FileExtension.VIDEO_OVERLAY}"),
            no_plot=args.no_plot,
            no_video=args.no_video,
            overwrite=args.overwrite,
        ))

    # Exit if no clips are available
    if not tasks:
        print("Tasks not generated. Nothing to process.")
        return

    # Launch worker processes
    workers = min(args.workers if args.workers > 0 else _default_workers(), len(tasks))
    print(f"processing {len(tasks)} clip(s) with {workers} worker(s)")

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(process_clip, task): task.name for task in tasks}
        # Report completed tasks as workers finish
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{idx}/{len(tasks)}] {result.log}")


if __name__ == "__main__":
    main()