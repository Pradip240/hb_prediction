"""
Strict Threshold Face Landmarking
----------------------------------------------------------------------
Uses optical flow on rigid skull anchors. 
- Deadzone threshold: Ignores sub-pixel sensor noise.
- Reset threshold: Only triggers MediaPipe if the head moves significantly.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

def ensure_model(path):
    if not os.path.exists(path):
        print(f"Downloading face_landmarker.task -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path

# Rigid skull points (Forehead, nose bridge, cheekbones)
ANCHOR_INDICES = [10, 151, 9, 8, 168, 6, 197, 113, 342, 227, 447]

def process_video(input_path, output_path, model_path):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    )

    # --- THRESHOLDS ---
    # 1. If movement from the base keyframe exceeds this (in pixels), trigger MediaPipe.
    # Adjust this based on how much the person moves.
    MP_RESET_THRESHOLD = 15.0  
    
    # 2. If frame-to-frame movement is less than this, ignore it (locks out sensor jitter).
    JITTER_DEADZONE = 0.5      

    base_mesh = None          
    base_anchors = None       
    tracked_anchors = None    
    prev_gray = None
    
    # Keep track of the final drawn points to apply the deadzone
    last_drawn_points = None  
    face_landmarks_proto = None

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated_frame = frame.copy()
            
            trigger_mediapipe = False

            # 1. TRACK ANCHORS
            if tracked_anchors is not None and prev_gray is not None:
                next_anchors, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, current_gray, tracked_anchors, None, **lk_params
                )
                
                if status is not None and np.sum(status) >= 8:
                    # Calculate how far the head has moved from the ORIGINAL keyframe
                    dist_from_base = np.mean(np.linalg.norm(next_anchors - base_anchors, axis=1))
                    
                    if dist_from_base > MP_RESET_THRESHOLD:
                        trigger_mediapipe = True
                    else:
                        tracked_anchors = next_anchors
                else:
                    trigger_mediapipe = True # Flow lost too many points
            else:
                trigger_mediapipe = True

            # 2. RUN MEDIAPIPE ONLY IF THRESHOLD CROSSED
            if trigger_mediapipe:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                ts = int(round((frame_idx * 1000.0) / fps))
                
                result = landmarker.detect_for_video(mp_image, ts)

                if result.face_landmarks:
                    face_landmarks_proto = result.face_landmarks[0]
                    mp_points = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks_proto], dtype=np.float32)
                    
                    base_mesh = mp_points.copy()
                    base_anchors = mp_points[ANCHOR_INDICES].copy()
                    tracked_anchors = base_anchors.copy()
                    last_drawn_points = base_mesh.copy()
                    
                    print(f"Frame {frame_idx}: MediaPipe Keyframe Triggered")
                else:
                    base_mesh = None
                    tracked_anchors = None

            # 3. APPLY RIGID TRANSFORM (WITH DEADZONE)
            elif base_mesh is not None and tracked_anchors is not None:
                # Calculate frame-to-frame movement to apply the deadzone
                movement_since_last_frame = np.mean(np.linalg.norm(tracked_anchors - prev_gray_anchors, axis=1))

                if movement_since_last_frame > JITTER_DEADZONE:
                    matrix, inliers = cv2.estimateAffinePartial2D(base_anchors, tracked_anchors, method=cv2.RANSAC)
                    if matrix is not None:
                        ones_mesh = np.ones((base_mesh.shape[0], 1))
                        base_mesh_3d = np.hstack([base_mesh, ones_mesh])
                        last_drawn_points = base_mesh_3d.dot(matrix.T).astype(np.float32)
                # Else: do nothing, `last_drawn_points` remains exactly where it was last frame.

            # 4. DRAW
            if last_drawn_points is not None and face_landmarks_proto is not None:
                for i, pt in enumerate(last_drawn_points):
                    face_landmarks_proto[i].x = pt[0] / width
                    face_landmarks_proto[i].y = pt[1] / height

                drawing_utils.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
                )
                drawing_utils.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
                )

            if tracked_anchors is not None:
                prev_gray_anchors = tracked_anchors.copy()
            prev_gray = current_gray.copy() if current_gray is not None else None

            writer.write(annotated_frame)
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    print(f"Finished! Output saved to: {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO = "test.mp4"              
    OUTPUT_VIDEO = "output_rppg_threshold.mp4" 
    MODEL_PATH = "face_landmarker.task"
    
    ensure_model(MODEL_PATH)
    process_video(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH)