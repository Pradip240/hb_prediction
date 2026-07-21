"""Run MediaPipe FaceLandmarker on every frame of every video; save one .npz per clip.

Output: <output-dir>/<video>_landmarks.npz with key "landmarks" and shape
(T, 478, 3) float32, where the last axis is (x_pixels, y_pixels, z). z is
MediaPipe's relative depth (roughly in image-width units; smaller = closer).
Frames with no detected face are NaN.

Runs on the GPU delegate by default (MediaPipe uses OpenGL/EGL, Ubuntu only) and
falls back to CPU automatically if the GPU delegate can't initialise.

Deps:  pip install mediapipe opencv-python-headless numpy

Usage: python landmarks.py --input-dir videos --output-dir lmk_out [--delegate gpu|cpu]
"""

import argparse
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"
N_LANDMARKS = 478


def make_options(delegate):
    """FaceLandmarker options for VIDEO mode using the given delegate."""
    return mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH, delegate=delegate),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def resolve_delegate(requested: str):
    """Return (delegate_enum, name).

    If GPU is requested but the delegate can't initialise (no EGL display in the
    container, or a wheel built without GPU flags), fall back to CPU so the run
    completes instead of crashing. Probes with one tiny inference.
    """
    Delegate = mp_python.BaseOptions.Delegate
    if requested == "cpu":
        return Delegate.CPU, "cpu"
    try:
        probe = mp_vision.FaceLandmarker.create_from_options(make_options(Delegate.GPU))
        probe.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)), 0
        )
        probe.close()
        return Delegate.GPU, "gpu"
    except Exception as exc:
        print(f"GPU delegate unavailable ({type(exc).__name__}: {exc}); falling back to CPU.")
        return Delegate.CPU, "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Save per-frame MediaPipe face landmarks as .npz.")
    ap.add_argument("--input-dir", default="videos")
    ap.add_argument("--output-dir", default="lmk_out")
    ap.add_argument("--delegate", choices=["gpu", "cpu"], default="gpu",
                    help="inference delegate; 'gpu' (OpenGL/EGL) falls back to CPU if unavailable")
    ap.add_argument("--overwrite", action="store_true", help="redo clips whose .npz exists")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    os.makedirs(args.output_dir, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(VIDEO_EXTS))
    print(f"{len(videos)} video(s) in {args.input_dir}")

    delegate_enum, active = resolve_delegate(args.delegate)
    print(f"delegate: {active}")

    for i, fname in enumerate(videos, 1):
        name = os.path.splitext(fname)[0]
        out = os.path.join(args.output_dir, f"{name}_landmarks.npz")
        if os.path.exists(out) and not args.overwrite:
            print(f"[{i}/{len(videos)}] {name}: exists, skip")
            continue

        cap = cv2.VideoCapture(os.path.join(args.input_dir, fname))
        if not cap.isOpened():
            print(f"[{i}/{len(videos)}] {name}: cannot open, skip")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        with mp_vision.FaceLandmarker.create_from_options(make_options(delegate_enum)) as landmarker:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = int(round(frame_idx * 1000.0 / fps))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
                )
                if result.face_landmarks:
                    pts = np.array(
                        [[lm.x * width, lm.y * height, lm.z] for lm in result.face_landmarks[0]],
                        dtype=np.float32,
                    )
                else:
                    pts = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
                frames.append(pts)
                frame_idx += 1
        cap.release()

        arr = np.stack(frames, axis=0) if frames else np.empty((0, N_LANDMARKS, 3), dtype=np.float32)
        np.savez_compressed(out, landmarks=arr)
        print(f"[{i}/{len(videos)}] {name}: {arr.shape} -> {out}")


if __name__ == "__main__":
    main()