"""features.py — physically-motivated amplitude/colour features for hemoglobin.

Unlike heart rate (a *frequency*, where amplitude is irrelevant), hemoglobin is inferred
from how blood ABSORBS light at different wavelengths — so the informative signal lives
in the amplitude and colour of the skin signal, NOT its timing. These are exactly the
quantities HR discarded by z-scoring. This module turns a (R, T, 3) segment (raw
per-region mean RGB, 0-255 scale) into a small, interpretable feature vector built from:

  * DC levels           — mean R/G/B per region (overall skin colour / baseline absorption)
  * AC amplitude         — std of the band-passed pulse per channel (pulsatile strength)
  * AC/DC ratio          — AC divided by DC per channel (the pulse-oximetry-style feature;
                           blood-volume changes modulate absorption per wavelength)
  * cross-channel ratios — AC/DC(green) / AC/DC(red), etc. — the "ratio of ratios" family
                           that pulse oximetry uses; sensitive to blood optical properties
  * DC colour ratios     — R/G, R/B, G/B of the DC levels (baseline colour balance)

Kept deliberately low-dimensional and interpretable: a small feature vector fed to a
small regressor is both more likely to capture the thin real Hb signal and more honest
than a deep net that could instead memorise subject identity (Hb is per-subject).
"""

import numpy as np

from common import signal_processing as sp

CHANNELS = ("R", "G", "B")


def _ac_dc(x, fps):
    """Return (dc, ac) for one channel: DC = mean level, AC = band-passed pulse std."""
    x = np.asarray(x, dtype=np.float64)
    x = sp.interpolate_nans(x)
    dc = float(np.mean(x))
    if not np.isfinite(dc) or abs(dc) < 1e-6 or np.std(x) < 1e-9:
        return dc, 0.0
    try:
        band = sp.bandpass_filter(x, fps=fps)
    except Exception:
        band = x - np.mean(x)
    ac = float(np.std(band))
    return dc, ac


def feature_names(region_order):
    names = []
    for rn in region_order:
        for ch in CHANNELS:
            names += [f"{rn}_{ch}_dc", f"{rn}_{ch}_ac", f"{rn}_{ch}_acdc"]
        # cross-channel AC/DC ratios (ratio of ratios)
        names += [f"{rn}_acdc_G_over_R", f"{rn}_acdc_R_over_B", f"{rn}_acdc_G_over_B"]
        # DC colour balance
        names += [f"{rn}_dc_R_over_G", f"{rn}_dc_R_over_B", f"{rn}_dc_G_over_B"]
    return names


def extract_features(signals, fps, region_order):
    """(R, T, 3) raw RGB -> 1-D feature vector (see module docstring). Order matches
    feature_names(region_order)."""
    R = signals.shape[0]
    feats = []
    for r in range(R):
        dcs, acs, acdcs = [], [], []
        for c in range(3):
            dc, ac = _ac_dc(signals[r, :, c], fps)
            acdc = ac / dc if abs(dc) > 1e-6 else 0.0
            dcs.append(dc); acs.append(ac); acdcs.append(acdc)
            feats += [dc, ac, acdc]
        # cross-channel AC/DC ratios (guard divide-by-zero)
        def ratio(a, b):
            return a / b if abs(b) > 1e-9 else 0.0
        feats += [ratio(acdcs[1], acdcs[0]), ratio(acdcs[0], acdcs[2]), ratio(acdcs[1], acdcs[2])]
        # DC colour ratios
        feats += [ratio(dcs[0], dcs[1]), ratio(dcs[0], dcs[2]), ratio(dcs[1], dcs[2])]
    return np.asarray(feats, dtype=np.float32)