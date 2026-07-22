"""predict_hr.py — Heart-rate estimation stage (Phase 2, CPU).

Reads the per-region rPPG signals produced by ``signal_analysis``
(``output/signals/<clip>_signals.npz``, key ``signals``, shape ``(3, T, 3)`` =
region × frame × RGB, region order forehead/lcheek/rcheek), reproduces the
original sliding-window POS/CHROM consensus heart-rate estimator with quality
gating, joins each clip against ``data/ground_truth.csv``, and writes:

  * ``output/hr/hr_results.csv``     — per-clip estimate, quality metrics, error
  * ``output/hr/hr_evaluation.png``  — estimated-vs-true scatter + per-clip error

The HR *algorithm* (spectral peak, de Haan SNR, spatial coherence, temporal
consistency, POS+CHROM consensus, flatness veto) is unchanged from the original
``hr_estimation.py`` and the HR half of the original ``main.py``; only the I/O is
adapted to read the cached signals ``.npz`` instead of running the face tracker.

fps handling: the signals ``.npz`` stores only the RGB array, so fps is taken
from an ``fps`` key inside the ``.npz`` if the writer stored one (recommended —
see README), otherwise from ``--fps`` (default 30). Because BPM = peak_hz × 60,
an incorrect fps scales every estimate, so store the true per-clip fps when your
clips are not all 30 fps.

Usage:
  python signal_analysis/predict_hr.py --signals-dir output/signals \\
      --ground-truth data/ground_truth.csv --output-dir output/hr
"""

import argparse
import csv
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config  # noqa: E402
import signal_processing as dsp  # noqa: E402


# ======================================================================
# HR spectral estimation + quality metrics
#   (ported verbatim from the original hr_estimation.py; only the imports and
#    hardcoded constants changed — all tunables now live in config.py.)
# ======================================================================

def compute_power_spectrum(
    sig: NDArray[np.float64],
    fps: float = config.DEFAULT_FPS,
    min_hz: float = config.SPEC_FREQ_MIN_HZ,
    max_hz: float = config.SPEC_FREQ_MAX_HZ,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Zero-padded, Hanning-windowed power spectrum of a 1D signal, band-limited."""
    sig_clean = dsp.interpolate_nans(sig)
    sig_clean = sig_clean - np.mean(sig_clean)
    n_samples = len(sig_clean)

    if n_samples < config.SPECTRUM_MIN_SAMPLES or np.std(sig_clean) < 1e-9:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    window = np.hanning(n_samples)
    n_fft = int(2 ** np.ceil(np.log2(n_samples * config.SPECTRUM_ZERO_PAD_FACTOR)))

    raw_fft = np.fft.rfft(sig_clean * window, n=n_fft)
    psd = np.abs(raw_fft) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    return freqs[band_mask], psd[band_mask]


def compute_dehaan_snr(
    sig: NDArray[np.float64],
    fps: float,
    target_hz: float,
    harmonic_width_hz: float = config.DEHAAN_HARMONIC_WIDTH_HZ,
) -> float:
    """de Haan & Jeanne (2013) pulsatile SNR in dB at the fundamental + 1st harmonic."""
    freqs, psd = compute_power_spectrum(sig, fps)
    if len(freqs) == 0:
        return config.SNR_INVALID_DB

    pulse_mask = (np.abs(freqs - target_hz) <= harmonic_width_hz) | (
        np.abs(freqs - (2.0 * target_hz)) <= (2.0 * harmonic_width_hz)
    )

    signal_energy = np.sum(psd[pulse_mask])
    noise_energy = np.sum(psd[~pulse_mask])

    if noise_energy <= 1e-12:
        return config.SNR_CLIP_DB  # cap when band noise is virtually zero

    return float(10.0 * np.log10((signal_energy + 1e-12) / noise_energy))


def compute_spectral_flatness(
    sig: NDArray[np.float64], fps: float = config.DEFAULT_FPS
) -> float:
    """Spectral flatness (Wiener entropy) in the HR band: 0 = peaked, 1 = broadband."""
    _, psd = compute_power_spectrum(
        sig, fps, min_hz=config.HR_FREQ_MIN_HZ, max_hz=config.HR_FREQ_MAX_HZ
    )
    if len(psd) == 0:
        return 1.0
    psd = psd + 1e-12
    geo_mean = float(np.exp(np.mean(np.log(psd))))
    arith_mean = float(np.mean(psd))
    return geo_mean / arith_mean


def estimate_hr_spectral(
    sig: NDArray[np.float64], fps: float = config.DEFAULT_FPS
) -> tuple[float | None, float, float]:
    """HR from the strongest spectral peak in the HR band; SNR judges, never selects."""
    freqs, psd = compute_power_spectrum(
        sig, fps, min_hz=config.HR_FREQ_MIN_HZ, max_hz=config.HR_FREQ_MAX_HZ
    )
    if len(freqs) == 0:
        return None, config.SNR_INVALID_DB, 0.0

    if len(psd) >= 3:
        is_peak = (psd[1:-1] > psd[:-2]) & (psd[1:-1] > psd[2:])
        peak_indices = np.where(is_peak)[0] + 1
    else:
        peak_indices = np.array([], dtype=int)

    if len(peak_indices) == 0:
        best_idx = int(np.argmax(psd))
    else:
        best_idx = int(peak_indices[np.argmax(psd[peak_indices])])

    peak_hz = float(freqs[best_idx])
    bpm = peak_hz * 60.0

    snr_db = compute_dehaan_snr(sig, fps, peak_hz)

    other_bins = np.delete(psd, best_idx)
    confidence = float(psd[best_idx] / (np.median(other_bins) + 1e-12))

    return bpm, snr_db, confidence


def compute_spatial_coherence(
    region_signals: list[NDArray[np.float64]], fps: float = config.DEFAULT_FPS
) -> float:
    """Mean pairwise Pearson correlation of the band-passed regional pulses."""
    filtered_sigs = []
    for sig in region_signals:
        clean = dsp.interpolate_nans(sig)
        if len(clean) < int(config.COHERENCE_MIN_SEC * fps) or np.std(clean) < 1e-9:
            continue
        filtered_sigs.append(dsp.bandpass_filter(clean, fps=fps))

    if len(filtered_sigs) < 2:
        return 0.0

    min_len = min(len(s) for s in filtered_sigs)
    trimmed = [s[:min_len] for s in filtered_sigs]

    correlations = []
    for i in range(len(trimmed)):
        for j in range(i + 1, len(trimmed)):
            std_i, std_j = np.std(trimmed[i]), np.std(trimmed[j])
            if std_i < 1e-9 or std_j < 1e-9:
                continue
            r = np.corrcoef(trimmed[i], trimmed[j])[0, 1]
            if np.isfinite(r):
                correlations.append(float(r))

    return float(np.mean(correlations)) if correlations else 0.0


def compute_temporal_consistency(
    sig: NDArray[np.float64],
    fps: float,
    target_bpm: float,
    win_sec: float = config.TEMPORAL_WINDOW_SEC,
    tol_bpm: float = config.AGREE_TOLERANCE_BPM,
) -> float:
    """Fraction of overlapping sub-windows whose spectral BPM agrees with target."""
    sig_clean = dsp.interpolate_nans(sig)
    n_samples = len(sig_clean)
    win_len = int(round(win_sec * fps))

    if n_samples < 2 * win_len or win_len < int(config.TEMPORAL_MIN_SEC * fps):
        return 1.0

    step = max(1, win_len // 2)
    agreeing_windows = 0
    total_windows = 0

    for start_idx in range(0, n_samples - win_len + 1, step):
        sub_window = sig_clean[start_idx: start_idx + win_len]
        sub_bpm, _, _ = estimate_hr_spectral(sub_window, fps)

        if sub_bpm is not None:
            total_windows += 1
            if abs(sub_bpm - target_bpm) <= tol_bpm:
                agreeing_windows += 1

    return float(agreeing_windows / total_windows) if total_windows > 0 else 0.0


def select_consensus_hr(
    pos_sig: NDArray[np.float64],
    chrom_sig: NDArray[np.float64],
    green_sig: NDArray[np.float64],
    region_greens: list[NDArray[np.float64]],
    fps: float = config.DEFAULT_FPS,
) -> dict[str, float | str | bool | None]:
    """POS/CHROM consensus decision + pulse-presence gate (SNR, coherence, consistency)."""
    bpm_pos, snr_pos, _ = estimate_hr_spectral(pos_sig, fps)
    bpm_chrom, snr_chrom, _ = estimate_hr_spectral(chrom_sig, fps)
    bpm_green, snr_green, _ = estimate_hr_spectral(green_sig, fps)

    spatial_coh = compute_spatial_coherence(region_greens, fps)

    candidates = []
    for name, bpm, snr, sig in [
        ("pos", bpm_pos, snr_pos, pos_sig),
        ("chrom", bpm_chrom, snr_chrom, chrom_sig),
        ("green", bpm_green, snr_green, green_sig),
    ]:
        if bpm is not None:
            t_coh = compute_temporal_consistency(sig, fps, bpm)
            candidates.append({"method": name, "bpm": bpm, "snr": snr, "t_coh": t_coh})

    if not candidates:
        return {
            "bpm": None, "method": "none", "snr_db": config.SNR_INVALID_DB, "accepted": False,
            "rejection_reason": "No spectral peaks detected in HR passband.",
            "spatial_coherence": spatial_coh, "temporal_consistency": 0.0,
        }

    pos_cand = next((c for c in candidates if c["method"] == "pos"), None)
    chrom_cand = next((c for c in candidates if c["method"] == "chrom"), None)

    if (
        pos_cand and chrom_cand
        and abs(pos_cand["bpm"] - chrom_cand["bpm"]) <= config.AGREE_TOLERANCE_BPM
    ):
        chosen_bpm = float((pos_cand["bpm"] + chrom_cand["bpm"]) / 2.0)
        chosen_snr = max(pos_cand["snr"], chrom_cand["snr"])
        chosen_tcoh = float((pos_cand["t_coh"] + chrom_cand["t_coh"]) / 2.0)
        method = "pos+chrom_consensus"
    else:
        best_cand = max(
            candidates,
            key=lambda c: (c["snr"] if c["t_coh"] >= config.CONSENSUS_TCOH_FLOOR
                           else c["snr"] - config.CONSENSUS_SNR_PENALTY_DB),
        )
        chosen_bpm = best_cand["bpm"]
        chosen_snr = best_cand["snr"]
        chosen_tcoh = best_cand["t_coh"]
        method = best_cand["method"]

    reasons = []
    if chosen_snr < config.MIN_SNR_DB:
        reasons.append(f"SNR {chosen_snr:.1f} dB < {config.MIN_SNR_DB:.1f}")
    if spatial_coh < config.MIN_SPATIAL_COH:
        reasons.append(f"coherence {spatial_coh:.2f} < {config.MIN_SPATIAL_COH:.2f}")
    if chosen_tcoh < config.MIN_TEMPORAL_CONSISTENCY:
        reasons.append(f"consistency {chosen_tcoh:.2f} < {config.MIN_TEMPORAL_CONSISTENCY:.2f}")

    accepted = not reasons
    rejection_reason = "" if accepted else "No measurable pulse: " + "; ".join(reasons) + "."

    return {
        "bpm": chosen_bpm, "method": method, "snr_db": chosen_snr,
        "accepted": accepted, "rejection_reason": rejection_reason,
        "spatial_coherence": spatial_coh, "temporal_consistency": chosen_tcoh,
    }


def estimate_all_methods(
    pos_sig: NDArray[np.float64],
    chrom_sig: NDArray[np.float64],
    green_sig: NDArray[np.float64],
    fps: float = config.DEFAULT_FPS,
) -> dict[str, tuple[float | None, float]]:
    """Each algorithm's own spectral estimate, for honest per-method comparison."""
    estimates: dict[str, tuple[float | None, float]] = {}
    for name, sig in [("pos", pos_sig), ("chrom", chrom_sig), ("green", green_sig)]:
        bpm, snr, _ = estimate_hr_spectral(sig, fps)
        estimates[name] = (bpm, snr)
    return estimates


# ======================================================================
# Ground truth
# ======================================================================

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")


def normalize_video_name(raw_name: str) -> str:
    """Strip a video extension without touching dots inside the stem (e.g. IPs)."""
    name = os.path.basename(raw_name).strip().lower()
    for ext in VIDEO_EXTS:
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def load_ground_truth(csv_path: str) -> dict[str, dict[str, float | None]]:
    """Load ground-truth pulse/hb per clip from CSV (tolerant of BOM + column aliases)."""
    ground_truth: dict[str, dict[str, float | None]] = {}
    if not os.path.exists(csv_path):
        print(f"WARNING: ground-truth file not found at '{csv_path}'. Errors will be blank.")
        return ground_truth

    with open(csv_path, mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return ground_truth

        vid_col = next((h for h in reader.fieldnames if h.lower().strip() in ["video", "filename", "file", "clip"]), None)
        pulse_col = next((h for h in reader.fieldnames if h.lower().strip() in ["pulse", "hr", "bpm", "heart_rate"]), None)
        hb_col = next((h for h in reader.fieldnames if h.lower().strip() in ["hb", "hemoglobin"]), None)

        if not vid_col:
            print("WARNING: ground-truth CSV has no video/clip column; errors will be blank.")
            return ground_truth

        for row in reader:
            raw_name = row.get(vid_col, "")
            if not raw_name:
                continue
            clean_key = normalize_video_name(raw_name)

            def parse_float(col_name: str | None) -> float | None:
                if not col_name:
                    return None
                try:
                    return float(row.get(col_name, "").strip())
                except (ValueError, TypeError, AttributeError):
                    return None

            ground_truth[clean_key] = {
                "pulse": parse_float(pulse_col),
                "hb": parse_float(hb_col),
            }
    return ground_truth


# ======================================================================
# Per-clip HR estimation from a signals array
#   (HR half of the original process_single_video; tracker / biomarker / Hb /
#    per-clip diagnostic-plot code removed — those belong to other stages.)
# ======================================================================

_RESULT_KEYS = (
    "video", "true_pulse", "est_pulse", "method", "est_snr", "accepted",
    "rejection_reason", "spatial_coherence", "temporal_consistency", "flatness",
    "abs_error", "correct", "pos_pulse", "pos_error", "chrom_pulse",
    "chrom_error", "green_pulse", "green_error",
)


def _blank_record(name: str, gt_pulse: float | None) -> dict[str, Any]:
    rec = {k: None for k in _RESULT_KEYS}
    rec["video"] = name
    rec["true_pulse"] = gt_pulse
    rec["accepted"] = False
    rec["rejection_reason"] = ""
    rec["correct"] = False
    return rec


def process_clip(name: str, signals: NDArray[np.float64], fps: float,
                 gt_pulse: float | None) -> dict[str, Any]:
    """Estimate HR for one clip's (3, T, 3) signals array; returns a CSV-ready record."""
    rec = _blank_record(name, gt_pulse)

    if signals.ndim != 3 or signals.shape[0] < 3 or signals.shape[2] != 3:
        rec["rejection_reason"] = f"Bad signals shape {signals.shape}; expected (3, T, 3)."
        print(f"  {name}: REJECT (bad shape {signals.shape})")
        return rec

    fh_rgb = np.asarray(signals[0], dtype=np.float64)
    lc_rgb = np.asarray(signals[1], dtype=np.float64)
    rc_rgb = np.asarray(signals[2], dtype=np.float64)
    n = fh_rgb.shape[0]

    total_duration_sec = n / fps if fps > 0 else 0.0
    if total_duration_sec < config.MIN_CLIP_SEC:
        rec["rejection_reason"] = f"Clip length ({total_duration_sec:.1f}s) under limit."
        print(f"  {name}: REJECT (too short {total_duration_sec:.1f}s < {config.MIN_CLIP_SEC:.0f}s)")
        return rec

    # Focus HR estimation on the longest clean, low-motion segment.
    face_green_full = np.mean(np.stack([fh_rgb, lc_rgb, rc_rgb], axis=0), axis=0)[:, 1]
    clean = dsp.find_clean_window(face_green_full, fps=fps, target_sec=config.ANALYSIS_WINDOW_SEC)
    if clean is not None:
        cs, ce = clean
        fh_an, lc_an, rc_an = fh_rgb[cs:ce], lc_rgb[cs:ce], rc_rgb[cs:ce]
    else:
        fh_an, lc_an, rc_an = fh_rgb, lc_rgb, rc_rgb

    n_an = fh_an.shape[0]
    window_len = int(round(config.HR_WINDOW_SEC * fps))
    window_step = int(round(config.HR_WINDOW_STEP_SEC * fps))
    if n_an <= window_len:
        window_len = n_an
        window_step = max(1, n_an)

    window_bpms: list[float] = []
    window_snrs: list[float] = []
    window_methods: list[str] = []
    window_metrics: list[tuple[float, float, float]] = []

    for start_idx in range(0, n_an - window_len + 1, window_step):
        end_idx = start_idx + window_len
        w_fh, w_lc, w_rc = fh_an[start_idx:end_idx], lc_an[start_idx:end_idx], rc_an[start_idx:end_idx]
        w_face = np.mean(np.stack([w_fh, w_lc, w_rc], axis=0), axis=0)

        if np.isnan(w_face[:, 1]).sum() > (config.HR_WINDOW_MAX_NAN_FRAC * window_len):
            continue

        w_green = w_face[:, 1]
        w_pos = dsp.extract_pos(w_face, fps=fps)
        w_chrom = dsp.extract_chrom(w_face, fps=fps)
        w_region_greens = [w_fh[:, 1], w_lc[:, 1], w_rc[:, 1]]

        w_res = select_consensus_hr(w_pos, w_chrom, w_green, w_region_greens, fps=fps)
        window_metrics.append(
            (w_res["snr_db"], w_res["spatial_coherence"], w_res["temporal_consistency"])
        )
        if w_res["bpm"] is not None and w_res["accepted"]:
            window_bpms.append(w_res["bpm"])
            window_snrs.append(w_res["snr_db"])
            window_methods.append(w_res["method"])

    if not window_bpms:
        rec["rejection_reason"] = "No windows passed structural quality gates."
        if window_metrics:
            snrs = [m[0] for m in window_metrics]
            cohs = [m[1] for m in window_metrics]
            tcs = [m[2] for m in window_metrics]
            k = len(window_metrics)
            n_snr = sum(v >= config.MIN_SNR_DB for v in snrs)
            n_coh = sum(v >= config.MIN_SPATIAL_COH for v in cohs)
            n_tc = sum(v >= config.MIN_TEMPORAL_CONSISTENCY for v in tcs)
            print(
                f"  {name}: REJECT (no window passed) | {k} window(s) | "
                f"SNR>={config.MIN_SNR_DB:+.1f}: {n_snr}/{k} (med {np.median(snrs):+.1f}) | "
                f"coh>={config.MIN_SPATIAL_COH:.2f}: {n_coh}/{k} (med {np.median(cohs):.2f}) | "
                f"consist>={config.MIN_TEMPORAL_CONSISTENCY:.0%}: {n_tc}/{k} (med {np.median(tcs):.0%})"
            )
        else:
            print(f"  {name}: REJECT (no usable windows; every window dropped for tracking loss)")
        return rec

    # Median pooling suppresses transient outliers from movement chunks.
    est_pulse = float(np.median(window_bpms))
    avg_snr = float(np.mean(window_snrs))
    winning_method = max(set(window_methods), key=window_methods.count)

    face_rgb = np.mean(np.stack([fh_an, lc_an, rc_an], axis=0), axis=0)
    green_sig = face_rgb[:, 1]
    pos_sig = dsp.extract_pos(face_rgb, fps=fps)
    chrom_sig = dsp.extract_chrom(face_rgb, fps=fps)
    region_greens = [fh_an[:, 1], lc_an[:, 1], rc_an[:, 1]]

    spatial_coh = compute_spatial_coherence(region_greens, fps)
    temporal_consistency = compute_temporal_consistency(pos_sig, fps, est_pulse)
    flatness = compute_spectral_flatness(pos_sig, fps)

    # Clip-level peakedness veto (judged on the full clean segment, robust).
    if flatness > config.MAX_SPECTRAL_FLATNESS:
        reason = (
            f"Spectrum too broadband (flatness {flatness:.3f} > "
            f"{config.MAX_SPECTRAL_FLATNESS:.2f}): no dominant cardiac peak."
        )
        rec["rejection_reason"] = reason
        print(
            f"  {name}: REJECT (broadband) | flatness {flatness:.3f} > "
            f"{config.MAX_SPECTRAL_FLATNESS:.2f} | would have reported {est_pulse:.1f} BPM"
        )
        return rec

    abs_error = abs(est_pulse - gt_pulse) if gt_pulse is not None else None
    correct = abs_error <= config.AGREE_TOLERANCE_BPM if abs_error is not None else False

    method_est = estimate_all_methods(pos_sig, chrom_sig, green_sig, fps=fps)
    pos_bpm, chrom_bpm, green_bpm = (method_est[m][0] for m in ("pos", "chrom", "green"))

    def _method_err(bpm: float | None) -> float | None:
        if bpm is None or gt_pulse is None:
            return None
        return round(abs(bpm - gt_pulse), 1)

    gt_str = f"{gt_pulse:.1f}" if gt_pulse is not None else "N/A"
    err_str = f"{abs_error:.1f}" if abs_error is not None else "N/A"
    print(
        f"  {name}: ACCEPT | Est {est_pulse:>5.1f} BPM | True {gt_str:>5} BPM | "
        f"windows {len(window_bpms)} | Err {err_str:>4} | "
        f"SNR {avg_snr:+.1f} dB | coh {spatial_coh:.2f} | "
        f"consist {temporal_consistency:.0%} | flatness {flatness:.3f}"
    )

    rec.update({
        "est_pulse": round(est_pulse, 1),
        "method": f"sliding_{winning_method}",
        "est_snr": round(avg_snr, 2),
        "accepted": True,
        "rejection_reason": "",
        "spatial_coherence": round(spatial_coh, 2),
        "temporal_consistency": round(temporal_consistency, 2),
        "flatness": round(flatness, 3),
        "abs_error": round(abs_error, 1) if abs_error is not None else None,
        "correct": correct,
        "pos_pulse": round(pos_bpm, 1) if pos_bpm is not None else None,
        "pos_error": _method_err(pos_bpm),
        "chrom_pulse": round(chrom_bpm, 1) if chrom_bpm is not None else None,
        "chrom_error": _method_err(chrom_bpm),
        "green_pulse": round(green_bpm, 1) if green_bpm is not None else None,
        "green_error": _method_err(green_bpm),
    })
    return rec


# ======================================================================
# Outputs — CSV, console summary, evaluation plot
# ======================================================================

def save_csv(results: list[dict[str, Any]], output_path: str) -> None:
    """Write per-clip HR records to CSV with a stable column order."""
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(_RESULT_KEYS))
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in _RESULT_KEYS})


def print_summary(results: list[dict[str, Any]]) -> None:
    """Console table of per-clip outcomes plus accepted-clip MAE."""
    print("\n" + "=" * 84)
    print("HEART-RATE PREDICTION SUMMARY")
    print("=" * 84)
    print(f"{'Clip':<28} {'True':>6} {'Est':>7} {'Err':>6} {'SNR':>7} {'TC%':>5}  Status")
    print("-" * 84)
    errs: list[float] = []
    for r in results:
        t, e = r.get("true_pulse"), r.get("est_pulse")
        err, snr, tc = r.get("abs_error"), r.get("est_snr"), r.get("temporal_consistency")
        t_str = f"{t:.0f}" if t is not None else "-"
        e_str = f"{e:.1f}" if e is not None else "-"
        err_str = f"{err:.1f}" if err is not None else "-"
        snr_str = f"{snr:+.1f}" if snr is not None else "-"
        tc_str = f"{tc * 100:.0f}" if tc is not None else "-"
        if r.get("accepted"):
            status = "ACCEPT -> OK" if r.get("correct") else "ACCEPT -> MISS"
        else:
            status = "REJECT"
        if err is not None:
            errs.append(float(err))
        print(f"{r.get('video', '?'):<28} {t_str:>6} {e_str:>7} {err_str:>6} {snr_str:>7} {tc_str:>5}  {status}")
    print("-" * 84)
    n_acc = sum(1 for r in results if r.get("accepted"))
    n_ok = sum(1 for r in results if r.get("accepted") and r.get("correct"))
    if errs:
        print(f"Clips: {len(results)} | accepted: {n_acc} | within tol: {n_ok} | "
              f"MAE (clips with GT): {np.mean(errs):.1f} BPM")
    else:
        print(f"Clips: {len(results)} | accepted: {n_acc} | (no ground truth for error stats)")
    print("=" * 84 + "\n")


def plot_evaluation(results: list[dict[str, Any]], tol_bpm: float, out_png: str) -> None:
    """Estimated-vs-true scatter + per-clip absolute-error bars, in one figure."""
    scored = [r for r in results if r.get("est_pulse") is not None and r.get("true_pulse") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Panel 1: estimated vs true ----
    if scored:
        true = np.array([r["true_pulse"] for r in scored], dtype=float)
        est = np.array([r["est_pulse"] for r in scored], dtype=float)
        ok = np.array([bool(r["correct"]) for r in scored])

        lo = float(min(true.min(), est.min())) - 5.0
        hi = float(max(true.max(), est.max())) + 5.0
        xs = np.array([lo, hi])
        ax1.plot(xs, xs, "--", color="gray", linewidth=1.2, label="ideal (y = x)", zorder=1)
        ax1.fill_between(xs, xs - tol_bpm, xs + tol_bpm, color="tab:green", alpha=0.08,
                         label=f"±{tol_bpm:.0f} BPM tolerance", zorder=0)
        if ok.any():
            ax1.scatter(true[ok], est[ok], s=55, color="tab:green", edgecolor="black",
                        linewidth=0.5, label="within tolerance", zorder=3)
        if (~ok).any():
            ax1.scatter(true[~ok], est[~ok], s=55, color="tab:red", edgecolor="black",
                        linewidth=0.5, label="miss", zorder=3)
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_aspect("equal", adjustable="box")
        mae = float(np.mean(np.abs(est - true)))
        ax1.set_title(f"Estimated vs True HR   (MAE {mae:.1f} BPM, n={len(scored)})", fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, "No accepted clips with ground truth", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=11, color="gray")
        ax1.set_title("Estimated vs True HR", fontweight="bold")
    ax1.set_xlabel("True HR (BPM)")
    ax1.set_ylabel("Estimated HR (BPM)")
    ax1.grid(alpha=0.3)

    # ---- Panel 2: per-clip absolute error ----
    err_rows = [r for r in scored if r.get("abs_error") is not None]
    if err_rows:
        names = [r["video"] for r in err_rows]
        errs = np.array([r["abs_error"] for r in err_rows], dtype=float)
        colors = ["tab:green" if r["correct"] else "tab:red" for r in err_rows]
        x = np.arange(len(err_rows))
        ax2.bar(x, errs, color=colors, edgecolor="black", linewidth=0.4)
        ax2.axhline(tol_bpm, color="tab:blue", linestyle="--", linewidth=1.2,
                    label=f"tolerance {tol_bpm:.0f} BPM")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Absolute error (BPM)")
        ax2.set_title("Per-clip absolute error", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No per-clip errors to show", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11, color="gray")
        ax2.set_title("Per-clip absolute error", fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Heart-rate evaluation vs ground truth", fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=130)
    plt.close()


# ======================================================================
# Driver
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Estimate heart rate from cached rPPG signals and evaluate vs ground truth."
    )
    ap.add_argument("--signals-dir", default="output/signals")
    ap.add_argument("--ground-truth", default="data/ground_truth.csv")
    ap.add_argument("--output-dir", default="output/hr")
    ap.add_argument("--fps", type=float, default=config.DEFAULT_FPS,
                    help="fallback fps when the signals .npz has no 'fps' key")
    ap.add_argument("--no-plot", action="store_true", help="write only the CSV, skip the evaluation PNG")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ground_truth = load_ground_truth(args.ground_truth)

    sig_files = sorted(f for f in os.listdir(args.signals_dir) if f.endswith("_signals.npz"))
    print(f"{len(sig_files)} signal file(s) in {args.signals_dir}")

    results: list[dict[str, Any]] = []
    for i, sig_file in enumerate(sig_files, 1):
        name = sig_file[: -len("_signals.npz")]
        print(f"[{i}/{len(sig_files)}] {name}")

        data = np.load(os.path.join(args.signals_dir, sig_file))
        signals = data["signals"]
        fps = float(data["fps"]) if "fps" in getattr(data, "files", []) else float(args.fps)

        gt = ground_truth.get(normalize_video_name(name), {})
        gt_pulse = gt.get("pulse")

        results.append(process_clip(name, signals, fps, gt_pulse))

    if not results:
        print("No signals to process — nothing written.")
        return

    csv_path = os.path.join(args.output_dir, "hr_results.csv")
    save_csv(results, csv_path)
    print(f"\nwrote {csv_path}")

    print_summary(results)

    if not args.no_plot:
        png_path = os.path.join(args.output_dir, "hr_evaluation.png")
        plot_evaluation(results, tol_bpm=config.AGREE_TOLERANCE_BPM, out_png=png_path)
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()