"""
Generate per-frame MediaPipe face landmarks for every video in a directory.

Inputs
------
Input directory
    Contains video files with one of the supported extensions:
    (.mkv, .mp4, .avi, .mov).

Command-line arguments
    --input-dir
        Directory containing input videos.
    --output-dir
        Directory where landmark archives are written.
    --device
        Inference device ("gpu" or "cpu"). GPU is attempted by default and
        automatically falls back to CPU if unavailable.
    --overwrite
        Recompute outputs even if they already exist.

Outputs
-------
For each input video <video>, writes

    <output-dir>/<video>_landmarks.npz

containing a single array:

    landmarks : float32, shape (T, 478, 3)

where:
    T = number of frames.

Each landmark stores (x_pixels, y_pixels, z), where x and y are image pixel
coordinates and z is MediaPipe's relative depth estimate. Frames with no
detected face are filled with NaN values.
"""

import os
import argparse

import cv2
import numpy as np
import mediapipe as mp  # type: ignore

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
MODEL_PATH = "face_landmarker.task"
N_LANDMARKS = 478
FRAME_INTERVAL_MS = 33


def make_options(delegate: mp.tasks.BaseOptions.Delegate) -> mp.tasks.vision.FaceLandmarkerOptions:  # type: ignore
    """
    Create FaceLandmarker options for video inference.
    """
    return mp.tasks.vision.FaceLandmarkerOptions(  # type: ignore
        base_options=mp.tasks.BaseOptions(  # type: ignore
            model_asset_path=MODEL_PATH,
            delegate=delegate,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,  # type: ignore
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def resolve_device(requested: str) -> tuple[mp.tasks.BaseOptions.Delegate, str]:  # type: ignore
    """
    Resolve the requested inference device.

    Attempts to initialize the GPU device when requested and falls back to
    the CPU device if initialization fails.
    """
    delegate = mp.tasks.BaseOptions.Delegate  # type: ignore
    if requested == "cpu":
        return delegate.CPU, "cpu"  # type: ignore
    try:
        probe = mp.tasks.vision.FaceLandmarker.create_from_options(make_options(delegate.GPU)) # type: ignore
        probe.detect_for_video(  # type: ignore
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)), 0
        )
        probe.close() # type: ignore
        return delegate.GPU, "gpu"  # type: ignore
    except Exception as exc:
        print(f"GPU device unavailable ({type(exc).__name__}: {exc}); falling back to CPU.")
        return delegate.CPU, "cpu"  # type: ignore


def main() -> None:
    """
    Process all input videos and save per-frame face landmarks as .npz files.
    """
    # Parse arguments
    ap = argparse.ArgumentParser(
        description="Save per-frame MediaPipe face landmarks as .npz."
    )
    ap.add_argument("--input-dir", default="videos")
    ap.add_argument("--output-dir", default="landmarks")
    ap.add_argument("--device", default="gpu", help="'gpu' or 'cpu' (auto if omitted)")
    ap.add_argument("--overwrite", action="store_true", help="redo clips whose .npz exists")
    args = ap.parse_args()

    # List videos to process
    os.makedirs(args.output_dir, exist_ok=True)
    videos: list[str] = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(VIDEO_EXTS))  # type: ignore
    print(f"{len(videos)} video(s) in {args.input_dir}")

    # Resolve inference device
    delegate_enum, active = resolve_device(args.device)  # type: ignore
    print(f"device: {active}")

    # Process each video
    for i, fname in enumerate(videos, 1):
        name = os.path.splitext(fname)[0]
        out = os.path.join(args.output_dir, f"{name}_landmarks.npz")

        # Skip already processed videos
        if os.path.exists(out) and not args.overwrite:
            print(f"[{i}/{len(videos)}] {name}: exists, skip")
            continue

        # Open video
        cap = cv2.VideoCapture(os.path.join(args.input_dir, fname))
        if not cap.isOpened():
            print(f"[{i}/{len(videos)}] {name}: cannot open, skip")
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Extract landmarks
        frames: list[np.ndarray] = []
        with mp.tasks.vision.FaceLandmarker.create_from_options(make_options(delegate_enum)) as landmarker: # type: ignore
            frame_idx = 0
            # Read frames
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = frame_idx * FRAME_INTERVAL_MS
                frame_idx += 1
                # Extract landmarks
                result = landmarker.detect_for_video(  # type: ignore
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
                )
                if result.face_landmarks: # type: ignore
                    pts = np.array(
                        [[lm.x * width, lm.y * height, lm.z] for lm in result.face_landmarks[0]], # type: ignore
                        dtype=np.float32
                    )
                else:
                    pts = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
                frames.append(pts)
        cap.release()

        # Save landmark array
        arr: np.ndarray = (np.stack(frames, axis=0) if frames else np.empty((0, N_LANDMARKS, 3), dtype=np.float32))
        np.savez_compressed(out, landmarks=arr)
        print(f"[{i}/{len(videos)}] {name}: {arr.shape} -> {out}")


if __name__ == "__main__":
    main()