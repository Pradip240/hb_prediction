"""Run SegFormer face-parsing on every frame of every video; save one .npz per clip.

Output: <output-dir>/<video>_seg.npz with key "masks" and shape (T, H, W) uint8,
where each pixel is a face-parse class id (0 = background). The class-id ->
label legend for the model is printed once at startup.

Deps:  pip install torch transformers pillow opencv-python numpy

Usage: python run_segmentation.py --input-dir videos --output-dir seg_out
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
    ap = argparse.ArgumentParser(description="Save per-frame face-parse masks as .npz.")
    ap.add_argument("--input-dir", default="videos")
    ap.add_argument("--output-dir", default="seg_out")
    ap.add_argument("--model", default="jonathandinu/face-parsing")
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if omitted)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize each mask by this factor (nearest); <1 shrinks the output")
    ap.add_argument("--overwrite", action="store_true", help="redo clips whose .npz exists")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    print(f"device={device}  class legend: {model.config.id2label}")

    os.makedirs(args.output_dir, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(VIDEO_EXTS))
    print(f"{len(videos)} video(s) in {args.input_dir}")

    for i, fname in enumerate(videos, 1):
        name = os.path.splitext(fname)[0]
        out = os.path.join(args.output_dir, f"{name}_seg.npz")
        if os.path.exists(out) and not args.overwrite:
            print(f"[{i}/{len(videos)}] {name}: exists, skip")
            continue

        cap = cv2.VideoCapture(os.path.join(args.input_dir, fname))
        if not cap.isOpened():
            print(f"[{i}/{len(videos)}] {name}: cannot open, skip")
            continue

        masks = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode="bilinear", align_corners=False
            )
            class_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            if args.scale != 1.0:
                h, w = class_map.shape
                class_map = cv2.resize(
                    class_map, (round(w * args.scale), round(h * args.scale)),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks.append(class_map)
        cap.release()

        arr = np.stack(masks, axis=0) if masks else np.empty((0, 0, 0), dtype=np.uint8)
        np.savez_compressed(out, masks=arr)
        print(f"[{i}/{len(videos)}] {name}: {arr.shape} -> {out}")


if __name__ == "__main__":
    main()