"""
Two-branch CNN for heart-rate estimation from facial RGB signals.

HRSpectralNet estimates heart rate from per-region RGB signals using two
complementary views of the same input:

1. A spectral branch that operates on the band-limited log-power spectrum and
   learns which spectral peaks correspond to cardiac activity.
2. A temporal branch that operates directly on the raw time-domain waveform and
   learns pulse morphology, such as the systolic upstroke and beat-to-beat
   spacing, which helps distinguish a true fundamental from a sub-harmonic even
   when the spectral peak is weak.

The two branches are fused and projected onto the HR-frequency grid. Prediction
uses a differentiable local soft-argmax around the dominant peak so the readout
commits to a single spectral mode rather than averaging across competing peaks.

The training objective combines:
- Smooth-L1 regression loss on the predicted heart rate.
- Soft cross-entropy against a Gaussian distribution centered on the
  ground-truth heart rate.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class HRSpectralNet(nn.Module):
    """Estimate heart rate from multi-channel facial RGB signals."""

    def __init__(
        self,
        n_channels: int = 9,
        fps: float = 30.0,
        nfft: int = 2048,
        hr_min: float = 40.0,
        hr_max: float = 200.0,
        width: int = 96,
        readout_halfwidth_bpm: float = 12.0,
        dropout: float = 0.3,
        augment: bool = True,
    ) -> None:
        """
        Initialize the spectral HR model.

        Args:
            n_channels: Number of input time-series channels.
            fps: Sampling frequency of the input signals in Hz.
            nfft: FFT size used for spectral conversion.
            hr_min: Lower HR limit represented by the output spectrum in BPM.
            hr_max: Upper HR limit represented by the output spectrum in BPM.
            width: Number of feature channels used by the convolutional network.
            readout_halfwidth_bpm: Half-width in BPM of the window used by the
                local soft-argmax readout around the dominant spectral peak.
            dropout: Dropout probability applied within both branches and the
                fusion head to reduce overfitting to subject-specific detail.
            augment: Whether to apply light waveform augmentation during
                training only. Augmentation is disabled automatically in eval.
        """
        super().__init__()

        if n_channels < 1:
            raise ValueError("n_channels must be positive.")
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if nfft < 2:
            raise ValueError("nfft must be at least 2.")
        if hr_min >= hr_max:
            raise ValueError("hr_min must be smaller than hr_max.")
        if readout_halfwidth_bpm <= 0:
            raise ValueError("readout_halfwidth_bpm must be positive.")

        self.n_channels = n_channels
        self.fps = float(fps)
        self.nfft = int(nfft)
        self.readout_halfwidth_bpm = float(readout_halfwidth_bpm)
        self.augment = bool(augment)

        # Build the FFT frequency grid and keep only the configured HR band.
        frequencies = np.fft.rfftfreq(self.nfft, d=1.0 / self.fps)
        low_hz = hr_min / 60.0
        high_hz = hr_max / 60.0

        band_indices = np.where((frequencies >= low_hz) & (frequencies <= high_hz))[0]

        if len(band_indices) == 0:
            raise ValueError("The requested HR range contains no FFT bins.")

        self.register_buffer("band_idx", torch.as_tensor(band_indices, dtype=torch.long))
        self.register_buffer("band_bpm", torch.as_tensor(frequencies[band_indices] * 60.0, dtype=torch.float32))

        self.n_freq = len(band_indices)

        # Spectral branch: 1-D convolutions along the frequency axis.
        self.spectral_net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )

        # Temporal branch: strided 1-D convolutions along the time axis.
        self.temporal_net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Fusion head: combine the temporal descriptor with the spectral features
        # at every frequency bin, then score each bin for pulse likelihood.
        self.fusion = nn.Sequential(
            nn.Conv1d(width * 2, width, kernel_size=1),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def _augment_waveform(self, signal: Tensor) -> Tensor:
        """
        Apply light label-preserving augmentation to the raw waveform.

        Augmentation runs only in training mode. None of the transforms change
        the underlying heart rate, so the regression and distribution targets
        remain valid. The transforms discourage the temporal branch from
        memorizing subject-specific amplitude, phase, and per-region detail.

        Args:
            signal: Input signals with shape (B, C, T).

        Returns:
            Augmented signals with shape (B, C, T).
        """
        if not (self.training and self.augment):
            return signal

        batch, channels, _ = signal.shape

        # Per-channel amplitude jitter.
        signal = signal * (1.0 + 0.1 * torch.randn(batch, channels, 1, device=signal.device, dtype=signal.dtype))

        # Additive Gaussian noise.
        signal = signal + 0.05 * torch.randn_like(signal)

        # Circular time shift preserves periodicity and therefore the heart rate.
        shift = int(torch.randint(-15, 16, (1,)).item())
        signal = torch.roll(signal, shifts=shift, dims=-1)

        # Occasionally drop one facial region's four channels to encourage the
        # model to rely on multiple regions rather than a single one.
        if channels >= 4 and torch.rand(1).item() < 0.3:
            region = int(torch.randint(0, channels // 4, (1,)).item())
            signal = signal.clone()
            signal[:, region * 4 : (region + 1) * 4, :] = 0.0
        return signal

    def spectral_input(self, signal: Tensor) -> Tensor:
        """
        Convert time-domain signals into normalized band-limited log-power spectra.

        Args:
            signal: Input signals with shape (B, C, T).

        Returns:
            Spectral representation with shape (B, C, F), where F is the
            number of FFT bins inside the configured HR range.
        """
        if signal.ndim != 3:
            raise ValueError(f"Expected input with shape (B, C, T), got {signal.shape}.")
        if signal.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} input channels, got {signal.shape[1]}.")

        # Apply a Hann window to reduce spectral leakage at the segment boundaries.
        window = torch.hann_window(signal.shape[-1], device=signal.device, dtype=signal.dtype)
        windowed_signal = signal * window
        # Convert each channel from time domain to frequency domain.
        spectrum = torch.fft.rfft(windowed_signal, n=self.nfft, dim=-1)  # type: ignore

        # Compute power spectrum.
        power = spectrum.real.square() + spectrum.imag.square()  # type: ignore

        # Keep only frequencies within the configured HR band.
        power = power.index_select(-1, self.band_idx)  # type: ignore

        # Compress the dynamic range of spectral power.
        power = torch.log1p(power)  # type: ignore

        # Normalize each channel independently across frequency.
        power = power - power.mean(dim=-1, keepdim=True)
        power = power / (power.std(dim=-1, keepdim=True) + 1e-6)
        return power

    def local_soft_argmax(self, probability: Tensor) -> Tensor:
        """
        Estimate a continuous HR by soft-argmax around the dominant peak.

        Restricting the expectation to a narrow window around the highest
        probability bin lets the readout commit to a single spectral mode rather
        than averaging across competing peaks, which would otherwise place the
        estimate in the empty valley between a sub-harmonic and the true pulse.

        Args:
            probability: Spectral probabilities with shape (B, F).

        Returns:
            Predicted HR in BPM with shape (B,).
        """
        # Locate the dominant HR bin for every sample.
        peak_index = torch.argmax(probability, dim=-1)
        peak_bpm = self.band_bpm[peak_index]  # type: ignore

        # Build a soft window around each peak so distant bins do not contribute.
        distance = (self.band_bpm[None, :] - peak_bpm[:, None]).abs()  # type: ignore
        window = (distance <= self.readout_halfwidth_bpm).to(probability.dtype) # type: ignore

        # Re-weight the probabilities by the window and renormalize.
        windowed = probability * window # type: ignore
        windowed = windowed / (windowed.sum(dim=-1, keepdim=True) + 1e-9) # type: ignore

        # Differentiable expectation restricted to the local window.
        predicted_bpm = (windowed * self.band_bpm).sum(dim=-1)  # type: ignore
        return predicted_bpm # type: ignore

    def forward(self, signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Predict heart rate and its spectral probability distribution.

        Args:
            signal: Input signals with shape (B, C, T).

        Returns:
            Tuple containing:
                - Predicted HR in BPM with shape (B,).
                - Spectral logits with shape (B, F).
                - Spectral probabilities with shape (B, F).
        """
        # Apply training-only waveform augmentation before both branches.
        signal = self._augment_waveform(signal)

        # Spectral branch over the band-limited power spectrum.
        spectrum = self.spectral_input(signal)
        spectral_features = self.spectral_net(spectrum)  # (B, width, F)

        # Temporal branch over the raw waveform, broadcast across frequency bins.
        temporal_descriptor = self.temporal_net(signal)  # (B, width, 1)
        temporal_features = temporal_descriptor.expand(-1, -1, self.n_freq)  # (B, width, F)

        # Fuse the two views and score every HR frequency bin.
        fused = torch.cat([spectral_features, temporal_features], dim=1)  # (B, 2*width, F)
        logits = self.fusion(fused).squeeze(1)  # (B, F)

        # Convert spectral scores into a probability distribution.
        probability = torch.softmax(logits, dim=-1)

        # Differentiable local soft-argmax around the dominant spectral peak.
        predicted_bpm = self.local_soft_argmax(probability)
        return predicted_bpm, logits, probability  # type: ignore

    def gaussian_target(self, heart_rate: Tensor, sigma_bpm: float = 4.0) -> Tensor:
        """
        Create Gaussian spectral targets centered on the true HR.

        Args:
            heart_rate: Ground-truth HR values with shape (B,).
            sigma_bpm: Standard deviation of the Gaussian target in BPM.

        Returns:
            Normalized target distributions with shape (B, F).
        """
        if sigma_bpm <= 0:
            raise ValueError("sigma_bpm must be positive.")

        distance = (self.band_bpm[None, :] - heart_rate[:, None]) / sigma_bpm  # type: ignore
        target = torch.exp(-0.5 * distance.square())  # type: ignore
        return target / (target.sum(dim=-1, keepdim=True) + 1e-9)

    def loss(
        self,
        predicted_bpm: Tensor,
        logits: Tensor,
        true_bpm: Tensor,
        l1_weight: float = 0.2,
        ce_weight: float = 1.0,
        sigma_bpm: float = 4.0,
        robust_k: float = 3.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the robust HR regression and spectral-distribution loss.

        The soft cross-entropy term shapes the spectral distribution into a
        single sharp peak at the true HR and carries the majority of the
        gradient. Because the scalar HR label is derived from a finite PPG
        window, a subset of labels are unreliable when the heart rate drifts
        within that window. Both loss terms are therefore made robust to such
        label outliers: each sample's contribution is capped at a per-batch
        ceiling defined by the median plus ``robust_k`` times the median
        absolute deviation. This limits how much any single mislabeled sample
        can move the parameters while leaving well-labeled samples unchanged.

        The Smooth-L1 term refines the continuous readout and is weighted
        lightly so it does not pull probability mass into the region between
        competing peaks in order to move the distribution mean.

        Args:
            predicted_bpm: Model HR predictions with shape (B,).
            logits: Spectral logits with shape (B, F).
            true_bpm: Ground-truth HR values with shape (B,).
            l1_weight: Weight applied to the Smooth-L1 regression loss.
            ce_weight: Weight applied to the soft cross-entropy loss.
            sigma_bpm: Width of the Gaussian spectral target in BPM.
            robust_k: Multiple of the median absolute deviation used to cap
                per-sample loss contributions. Larger values apply weaker
                clipping; a very large value recovers the non-robust loss.

        Returns:
            Tuple containing:
                - Total training loss.
                - Dictionary containing individual loss values.
        """
        # Per-sample soft cross-entropy against the Gaussian HR target.
        target = self.gaussian_target(true_bpm, sigma_bpm=sigma_bpm)
        log_probability = torch.log_softmax(logits, dim=-1)
        soft_ce_per_sample = -(target * log_probability).sum(dim=-1)

        # Per-sample Smooth-L1 on the continuous readout.
        smooth_l1_per_sample = nn.functional.smooth_l1_loss(predicted_bpm, true_bpm, reduction="none")

        # Robust ceiling from median and median absolute deviation. Detached so
        # the clipping threshold itself does not receive gradient.
        def robust_cap(values: Tensor) -> Tensor:
            detached = values.detach()
            median = detached.median()
            mad = (detached - median).abs().median() + 1e-6
            return median + robust_k * mad

        ce_capped = torch.minimum(soft_ce_per_sample, robust_cap(soft_ce_per_sample))
        l1_capped = torch.minimum(smooth_l1_per_sample, robust_cap(smooth_l1_per_sample))

        soft_ce = ce_capped.mean()
        smooth_l1 = l1_capped.mean()
        total_loss = l1_weight * smooth_l1 + ce_weight * soft_ce

        return total_loss, {
            "smooth_l1": float(smooth_l1.detach().item()),
            "soft_ce": float(soft_ce.detach().item()),
        }
