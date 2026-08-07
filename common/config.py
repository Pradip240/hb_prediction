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
REGION_ORDER = (
    "forehead",
    "lcheek",
    "rcheek",
)

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



# ============================================================================
# Dataset preparation
# ============================================================================

# Fixed window duration (seconds).
WINDOW_SEC = 20.0

# Step between consecutive windows (seconds).
#
# Set equal to WINDOW_SEC for non-overlapping windows.
WINDOW_STEP_SEC = 20.0

# Minimum number of facial regions that must satisfy the quality criteria for a window to be accepted.
MIN_VALID_REGIONS = 1

# Maximum duration of an individual missing-data gap allowed within a training window.
MAX_GAP_SEC = 1.0

# Maximum cumulative duration of all missing-data gaps within a training window.
MAX_TOTAL_BROKEN_SEC = 5.0

# Consecutive samples separated by more than this multiple of the
# nominal sampling interval are considered a gap.
GAP_FACTOR = 2.0

# Target sampling frequency after interpolation (Hz).
TARGET_FPS = 15











# --- size-aware region weighting --------------------------------------------------
# When the 3 regions are combined into one signal, weight each by its skin-pixel count
# so a tiny sliver (e.g. a turned-away cheek at ~20 px) doesn't contribute equally to a
# solid region (~1900 px) whose mean is far less noisy. Floor/cap scheme: a region below
# MIN_SKIN_PIXELS gets weight 0; above that, weight = pixel count clipped to
# REGION_WEIGHT_CAP_PX. Applied PER SAMPLE, so it tracks head turns (a cheek that becomes
# a sliver mid-window loses weight at that moment). Set REGION_WEIGHT_ENABLED = False for
# an equal-weight nanmean fallback.
REGION_WEIGHT_ENABLED = True
REGION_WEIGHT_CAP_PX = 1000        # counts above this are clipped (diminishing returns on size)
REGION_WEIGHT_SMOOTH_SEC = 1.0     # low-pass the count series before weighting: keeps slow pose
                                   # drift, drops fast flicker so the weight can't inject pulse-band
                                   # noise. 0 = no smoothing.



# ======================================================================
# Signal processing (DSP)  —  used by the HR / Hb stages
# ======================================================================

# --- timing / PTS sanity bounds ---------------------------------------------------
FPS_SANITY_MIN = 1.0       # reject an implied/header fps below this (Hz)
FPS_SANITY_MAX = 1000.0    # ...or above this (guards absurd headers like 30000)
PTS_MIN_FINITE_FRAC = 0.5  # a PTS stream needs at least this fraction finite to be used



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
# These four thresholds decide whether a clip is ACCEPTED. Loosening them accepts
# more clips but admits weaker/noisier signals (the gate validates *presence*, not
# *correctness* — see README). Tune against your ground truth. Direction to accept
# MORE: lower MIN_SNR_DB, lower MIN_SPATIAL_COH, lower MIN_TEMPORAL_CONSISTENCY,
# raise MAX_SPECTRAL_FLATNESS. de Haan SNR on webcam footage is often negative even
# for a real pulse, so MIN_SNR_DB is usually the binding constraint. Set to -3.0
# (tighter than a bare accept-everything) to drop the least-reliable clips now
# that harmonic-aware peak selection makes the surviving estimates more trustworthy.
MIN_SNR_DB = -6.0            # min de Haan SNR (dB) of the chosen peak
MIN_SPATIAL_COH = 0.05       # min mean cross-region pulse correlation
MIN_TEMPORAL_CONSISTENCY = 0.30   # min fraction of sub-windows agreeing on BPM
MAX_SPECTRAL_FLATNESS = 0.60      # max HR-band flatness (0=peaked, 1=broadband)


# ======================================================================
# HR prediction stage  (predict_hr.py)
# ======================================================================
# Minimum clip length to attempt HR estimation (seconds).
MIN_CLIP_SEC = 6.0

# Sliding analysis windows inside the clean segment.
HR_WINDOW_SEC = 14.0          # window length (gives the filters room to settle)
HR_WINDOW_STEP_SEC = 2.0      # hop between consecutive windows
HR_WINDOW_MAX_NAN_FRAC = 0.35 # drop a window if more than this fraction is NaN
                              # (raise toward 0.5 to tolerate more tracking loss)

# Power spectrum (compute_power_spectrum).
SPECTRUM_MIN_SAMPLES = 16     # shortest signal a spectrum is attempted on
SPECTRUM_ZERO_PAD_FACTOR = 4  # zero-pad the FFT to this multiple for finer bins

# de Haan SNR (compute_dehaan_snr).
DEHAAN_HARMONIC_WIDTH_HZ = 0.1  # half-width around f0 (2x that around 2*f0), Hz
SNR_CLIP_DB = 20.0              # SNR reported when band noise is ~0
SNR_INVALID_DB = -99.0          # sentinel when SNR cannot be computed

# Spatial coherence (compute_spatial_coherence).
COHERENCE_MIN_SEC = 2.0       # a region needs at least this many seconds to count

# Temporal consistency (compute_temporal_consistency).
TEMPORAL_WINDOW_SEC = 4.0     # sub-window length for the stability check
TEMPORAL_MIN_SEC = 2.0        # sub-window must be at least this long to be used

# Consensus selection when POS and CHROM disagree (select_consensus_hr).
CONSENSUS_TCOH_FLOOR = 0.40      # candidates below this consistency are penalised
CONSENSUS_SNR_PENALTY_DB = 10.0  # SNR penalty applied to inconsistent candidates

