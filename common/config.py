"""Configuration shared across the CPU processing pipeline.

The parameters are grouped by processing stage:

    - Video loading
    - Temporal smoothing
    - Region extraction
    - Overlay visualization
"""

# ============================================================================
# Video
# ============================================================================

# Fallback frame rate used when the input video does not provide one.
DEFAULT_FPS: float = 30.0



# ============================================================================
# Temporal smoothing
# ============================================================================

# One-Euro filter parameters used for landmark smoothing.
SMOOTH_MIN_CUTOFF = 0.1
SMOOTH_BETA = 0.005

# Apply temporal smoothing to binary skin masks before region extraction.
#
# The face parser may flicker near hairlines and facial boundaries. Smoothing
# the masks improves temporal consistency while preserving the overall shape.
MASK_SMOOTH_ENABLED = True

# Threshold used to convert the smoothed mask back to a binary image.
MASK_SMOOTH_THRESHOLD = 0.5



# ============================================================================
# Facial regions
# ============================================================================

# Landmark indices defining each facial region
# (MediaPipe Face Landmarker, 478 landmarks).
FOREHEAD_POLY = [67, 109, 10, 338, 297, 299, 337, 151, 108, 69, 104, 9]

LEFT_CHEEK_POLY = [
    116, 117, 118, 119, 100, 142, 36, 205, 187, 123, 50, 101,
    147, 213, 192, 214, 135, 138,
]

RIGHT_CHEEK_POLY = [
    345, 346, 347, 348, 329, 371, 266, 425, 411, 352, 280, 330,
    376, 433, 416, 434, 364, 367,
]

# Region definitions.
REGIONS = {
    "forehead": FOREHEAD_POLY,
    "lcheek": LEFT_CHEEK_POLY,
    "rcheek": RIGHT_CHEEK_POLY,
}

# Fixed region order used throughout the pipeline.
REGION_ORDER = ("forehead", "lcheek", "rcheek")

# Face-parsing class IDs treated as skin.
#
# SegFormer classes:
#   0 background
#   1 skin
#   2 nose
#   3 eye_g
#   4 left_eye
#   5 right_eye
#   6 left_brow
#   7 right_brow
#   8 left_ear
#   9 right_ear
#   10 mouth
#   11 upper_lip
#   12 lower_lip
#   13 hair
#   14 hat
#   15 ear_ring
#   16 neck_l
#   17 neck
#   18 cloth
SKIN_CLASS_IDS = (1,)

# Region-mask construction.
ROI_EROSION_PX = 3      # Erode region boundaries to reduce edge contamination.
SUBPIX_SHIFT = 3        # Polygon rasterization precision (1 / 2^SUBPIX_SHIFT px).
MIN_SKIN_PIXELS = 80    # Regions below this size are marked invalid.



# ============================================================================
# Overlay visualization
# ============================================================================

# Region colors in BGR order.
REGION_COLORS = {
    "forehead": (0, 255, 255),   # yellow
    "lcheek": (255, 0, 255),     # magenta
    "rcheek": (0, 255, 0),       # green
}

# Overlay transparency.
OVERLAY_SKIN_TINT_ALPHA = 0.25
OVERLAY_REGION_FILL_ALPHA = 0.35

# Colors used to distinguish classical rPPG methods in plots.
RPPG_METHOD_COLORS = {
    "POS": "tab:blue",
    "CHROM": "tab:orange",
    "green": "tab:green",
}


# ============================================================================
# Dataset preparation
# ============================================================================

# Fixed window duration (seconds).
WINDOW_SEC = 20.0

# Step between consecutive windows (seconds).
#
# Set equal to WINDOW_SEC for non-overlapping windows.
WINDOW_STEP_SEC = 15

# Minimum number of facial regions that must satisfy the quality criteria for a window to be accepted.
MIN_VALID_REGIONS = 2

# Maximum duration of an individual missing-data gap allowed within a training window.
MAX_GAP_SEC = 1.0

# Maximum cumulative duration of all missing-data gaps within a training window.
MAX_TOTAL_BROKEN_SEC = 5.0

# Consecutive samples separated by more than this multiple of the
# nominal sampling interval are considered a gap.
GAP_FACTOR = 2.0

# Target sampling frequency after interpolation (Hz).
TARGET_FPS = 30



# ============================================================================
# Region weighting
# ============================================================================

# Enable skin-pixel-count weighting when combining facial regions.
REGION_WEIGHT_ENABLED = True

# Temporal smoothing duration for region pixel counts before calculating weights.
#
# Smoothing reduces rapid changes in region size caused by segmentation noise
# while preserving slower changes caused by head movement.
REGION_WEIGHT_SMOOTH_SEC = 1.0



# ============================================================================
# rPPG signal processing
# ============================================================================

# Duration of the sliding window used by the POS algorithm (seconds).
POS_WINDOW_SEC = 1.6

# Physiological heart-rate frequency range used by rPPG algorithms (Hz).
HR_FREQ_MIN_HZ = 0.9     # ~54 BPM
HR_FREQ_MAX_HZ = 2.8     # ~168 BPM

# Order of the Butterworth bandpass filter used by CHROM and other signal-processing methods.
BANDPASS_ORDER = 3

# Smoothness-prior parameter used to remove low-frequency baseline drift.
#
# Larger values produce a smoother estimated baseline and therefore remove
# slower variations more aggressively.
DETREND_LAMBDA = 100.0
