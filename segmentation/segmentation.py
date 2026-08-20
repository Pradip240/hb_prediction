"""
Generate per-frame face-parsing masks for every video in a directory.

Inputs
------
Input directory
    Contains video files with one of the supported extensions:
    (.mkv, .mp4, .avi, .mov).

Command-line arguments
    --input-dir
        Directory containing input videos.
    --output-dir
        Directory where segmentation archives are written.
    --device
        Torch device to use ("cuda" or "cpu"). Automatically selected if omitted.
    --batch-size
        Number of frames processed together during inference.
    --scale
        Scale factor applied to each segmentation mask before saving.
    --overwrite
        Recompute outputs even if they already exist.

Outputs
-------
For each input video <video>, writes

    <output-dir>/<video>_seg.npz

containing a single array:

    masks : uint8, shape (T, H, W)

where:
    T = number of frames,
    H = output mask height,
    W = output mask width.

Each pixel stores the predicted face-parsing class ID, with 0 representing
background. The model's class ID to label mapping is printed once at startup.
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")


def main() -> None:
    """Process all input videos and save per-frame face-parsing masks as .npz files."""
    # Parse arguments
    ap = argparse.ArgumentParser(description="Save per-frame face-parse masks as .npz.")
    ap.add_argument("--input-dir", default="videos")
    ap.add_argument("--output-dir", default="seg")
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if omitted)")
    ap.add_argument(
        "--batch-size", type=int, default=8, choices=range(1, 64), help="number of frames processed per GPU batch"
    )
    ap.add_argument(
        "--scale", type=float, default=1.0, help="resize each mask by this factor (nearest); <1 shrinks the output"
    )
    ap.add_argument("--overwrite", action="store_true", help="redo clips whose .npz exists")
    args = ap.parse_args()

    # Create model and processor
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")  # type: ignore
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device).eval()  # type: ignore
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device}  class legend: {model.config.id2label}")

    # List videos to process
    os.makedirs(args.output_dir, exist_ok=True)
    videos: list[str] = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(VIDEO_EXTS))  # type: ignore
    print(f"{len(videos)} video(s) in {args.input_dir}")

    # Start procesing each video
    for i, fname in enumerate(videos, 1):
        name = os.path.splitext(fname)[0]
        out = os.path.join(args.output_dir, f"{name}_seg.npz")
        # Skip already processed videos
        if os.path.exists(out) and not args.overwrite:
            print(f"[{i}/{len(videos)}] {name}: exists, skip")
            continue

        # Run openCV to capture frames
        cap = cv2.VideoCapture(os.path.join(args.input_dir, fname))
        if not cap.isOpened():
            print(f"[{i}/{len(videos)}] {name}: cannot open, skip")
            continue

        # Extract segmentation masks
        masks: list[np.ndarray] = []
        batch_frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if ok:
                batch_frames.append(frame)
            # Process batch
            if len(batch_frames) == args.batch_size or (not ok and batch_frames):
                images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
                inputs = processor(images=images, return_tensors="pt").to(device)  # type: ignore
                with torch.inference_mode():
                    logits = model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=batch_frames[0].shape[:2], mode="bilinear", align_corners=False
                )
                class_maps = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                for class_map in class_maps:
                    masks.append(class_map)
                batch_frames.clear()
            if not ok:
                break
        cap.release()

        # Save numpy array
        arr = np.stack(masks, axis=0) if masks else np.empty((0, 0, 0), dtype=np.uint8)
        np.savez_compressed(out, masks=arr)
        print(f"[{i}/{len(videos)}] {name}: {arr.shape} -> {out}")


if __name__ == "__main__":
    main()
