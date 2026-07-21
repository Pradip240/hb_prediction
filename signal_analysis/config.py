"""Shared configuration for the CPU analysis stages.

Two groups of settings:
  * Region-of-interest definitions and mask-formation parameters, used to turn
    the saved segmentation + landmarks into per-region signals.
  * DSP frequency bands, detrending, and quality thresholds, used by the HR / Hb
    stages.
All values match the original pipeline, so results are unchanged.
"""

# ======================================================================
# Region of interest  (signal extraction + overlay)
# ======================================================================

# ROI landmark indices (MediaPipe FaceLandmarker 478-point mesh).
FOREHEAD_POLY = [67, 109, 10, 338, 297, 299, 337, 151, 108, 69, 104, 9]
LEFT_CHEEK_POLY = [
    116, 117, 118, 119, 100, 142, 36, 205, 187, 123, 50, 101,
    147, 213, 192, 214, 135, 138,
]
RIGHT_CHEEK_POLY = [
    345, 346, 347, 348, 329, 371, 266, 425, 411, 352, 280, 330,
    376, 433, 416, 434, 364, 367,
]

# Region name -> landmark polygon, and the fixed order along axis 0 of signals.npy.
REGIONS = {
    "forehead": FOREHEAD_POLY,
    "lcheek": LEFT_CHEEK_POLY,
    "rcheek": RIGHT_CHEEK_POLY,
}
REGION_ORDER = ("forehead", "lcheek", "rcheek")

# Colour used to draw each region in the overlay video (BGR).
REGION_COLORS = {
    "forehead": (0, 255, 255),   # yellow
    "lcheek": (255, 0, 255),     # magenta
    "rcheek": (0, 255, 0),       # green
}

# Face-parse class ids treated as skin. Legend: 0 background, 1 skin, 2 nose,
# 3 eye_g, 4 l_eye, 5 r_eye, 6 l_brow, 7 r_brow, 8 l_ear, 9 r_ear, 10 mouth,
# 11 u_lip, 12 l_lip, 13 hair, 14 hat, 15 ear_r, 16 neck_l, 17 neck, 18 cloth.
# The original pipeline used skin only (add 2 for 'nose' if a cheek borders it).
SKIN_CLASS_IDS = (1,)

# Region-mask formation (unchanged from the original tracker).
ROI_EROSION_PX = 3     # erode the ROI inward this many px (drops the jittery rim)
SUBPIX_SHIFT = 3       # sub-pixel polygon rasterisation (1/8 px)
MIN_SKIN_PIXELS = 80   # a region with fewer skin px is recorded as NaN

# One-Euro landmark smoothing (unchanged from the original tracker).
SMOOTH_MIN_CUTOFF = 0.1
SMOOTH_BETA = 0.005

# ======================================================================
# Signal processing (DSP)  —  used by the HR / Hb stages
# ======================================================================
DEFAULT_FPS = 30.0

# Physiological heart-rate passband (Hz).
HR_FREQ_MIN_HZ = 0.83     # ~50 BPM
HR_FREQ_MAX_HZ = 2.8      # ~168 BPM

# Extended band for spectrum analysis / de Haan SNR (Hz).
SPEC_FREQ_MIN_HZ = 0.7
SPEC_FREQ_MAX_HZ = 4.5

# Smoothness-priors detrending regularisation (Tarvainen et al.).
DETREND_LAMBDA = 50.0
# Butterworth bandpass order.
BANDPASS_ORDER = 4

# Gating / quality-control thresholds.
AGREE_TOLERANCE_BPM = 6.0
AGREE_MIN_VIEWS = 3
MAX_CLUSTER_SPREAD_BPM = 8.0
MIN_CLEAN_WINDOW_SEC = 5.0
ANALYSIS_WINDOW_SEC = 20.0

# Pulse-presence gate (intrinsic; no ground truth).
MIN_SNR_DB = 1.5
MIN_SPATIAL_COH = 0.05
MIN_TEMPORAL_CONSISTENCY = 0.40
MAX_SPECTRAL_FLATNESS = 0.50