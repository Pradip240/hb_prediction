"""Spectral CNN for heart-rate estimation from facial RGB signals.

HRSpectralNet estimates heart rate directly from the frequency-domain
representation of per-region RGB signals.

Processing pipeline:

1. Convert each input channel from the time domain to log-power spectra.
2. Restrict the spectra to the configured physiological HR range.
3. Normalize each channel independently across frequency.
4. Use a 1-D CNN across frequency to learn which spectral peaks are
   associated with cardiac activity.
5. Apply a softmax over the HR-frequency bins.
6. Compute a differentiable soft-argmax to obtain a continuous HR estimate.

The training objective combines:
- Smooth-L1 regression loss on the predicted heart rate.
- Soft cross-entropy against a Gaussian distribution centered on the
  ground-truth heart rate.

The Gaussian target encourages the predicted spectral distribution to
concentrate around the true heart rate while the soft-argmax provides a
continuous BPM prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class HRSpectralNet(nn.Module):
    """
    Estimate heart rate from multi-channel facial RGB signals.
    """

    def __init__(
        self,
        n_channels: int = 9,
        fps: float = 30.0,
        nfft: int = 2048,
        hr_min: float = 40.0,
        hr_max: float = 200.0,
        width: int = 64
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

        self.n_channels = n_channels
        self.fps = float(fps)
        self.nfft = int(nfft)

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

        # Convolutional network operating along the frequency dimension.
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),

            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),

            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),

            nn.Conv1d(width, 1, kernel_size=1),
        )


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
        spectrum = torch.fft.rfft(windowed_signal, n=self.nfft, dim=-1) # type: ignore

        # Compute power spectrum.
        power = spectrum.real.square() + spectrum.imag.square() # type: ignore

        # Keep only frequencies within the configured HR band.
        power = power.index_select(-1, self.band_idx) # type: ignore

        # Compress the dynamic range of spectral power.
        power = torch.log1p(power) # type: ignore

        # Normalize each channel independently across frequency.
        power = power - power.mean(dim=-1, keepdim=True)
        power = power / (power.std(dim=-1, keepdim=True) + 1e-6)
        return power


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
        spectrum = self.spectral_input(signal)

        # Learn a pulse-likelihood score for every HR frequency bin.
        logits = self.net(spectrum).squeeze(1)

        # Convert spectral scores into a probability distribution.
        probability = torch.softmax(logits, dim=-1)

        # Differentiable soft-argmax over the HR spectrum.
        predicted_bpm = (probability * self.band_bpm).sum(dim=-1) # type: ignore
        return predicted_bpm, logits, probability # type: ignore


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

        distance = (self.band_bpm[None, :] - heart_rate[:, None]) / sigma_bpm # type: ignore
        target = torch.exp(-0.5 * distance.square()) # type: ignore
        return target / (target.sum(dim=-1, keepdim=True) + 1e-9)


    def loss(
        self,
        predicted_bpm: Tensor,
        logits: Tensor,
        true_bpm: Tensor,
        l1_weight: float = 1.0,
        ce_weight: float = 0.3,
        sigma_bpm: float = 4.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the combined HR regression and spectral-distribution loss.

        Args:
            predicted_bpm: Model HR predictions with shape (B,).
            logits: Spectral logits with shape (B, F).
            true_bpm: Ground-truth HR values with shape (B,).
            l1_weight: Weight applied to the Smooth-L1 regression loss.
            ce_weight: Weight applied to the soft cross-entropy loss.
            sigma_bpm: Width of the Gaussian spectral target in BPM.

        Returns:
            Tuple containing:
                - Total training loss.
                - Dictionary containing individual loss values.
        """
        smooth_l1 = nn.functional.smooth_l1_loss(predicted_bpm, true_bpm)
        target = self.gaussian_target(true_bpm, sigma_bpm=sigma_bpm)
        log_probability = torch.log_softmax(logits, dim=-1)
        soft_ce = -(target * log_probability).sum(dim=-1).mean()
        total_loss = (l1_weight * smooth_l1 + ce_weight * soft_ce)

        return total_loss, {
            "smooth_l1": float(smooth_l1.detach().item()),
            "soft_ce": float(soft_ce.detach().item()),
        }