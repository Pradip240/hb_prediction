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
# These four thresholds decide whether a clip is ACCEPTED. Loosening them accepts
# more clips but admits weaker/noisier signals (the gate validates *presence*, not
# *correctness* — see README). Tune against your ground truth. Direction to accept
# MORE: lower MIN_SNR_DB, lower MIN_SPATIAL_COH, lower MIN_TEMPORAL_CONSISTENCY,
# raise MAX_SPECTRAL_FLATNESS. de Haan SNR on webcam footage is often negative even
# for a real pulse, so MIN_SNR_DB is usually the binding constraint. (Original
# pipeline used -4.5; -6.0 here is a touch more permissive for webcam clips.)
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