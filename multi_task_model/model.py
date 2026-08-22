"""
Multi-task model for joint heart-rate and hemoglobin estimation.

A shared two-branch trunk (spectral + temporal) processes the facial RGB
signal into an embedding. Two heads read from it:

- HR head: spectral logits over the HR-frequency grid, read out with a local
  soft-argmax, trained with a Gaussian-distribution + Smooth-L1 objective.
- Hb head: a small MLP that consumes the shared embedding concatenated with
  engineered colour/AC-DC features, producing a scalar hemoglobin estimate.

Training jointly lets the strong HR signal shape a representation the weak Hb
task can borrow, which improves Hb generalization across subjects. The Hb loss
is weighted lightly so the HR task remains the primary anchor.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class MultiTaskHRHb(nn.Module):
    """Estimate heart rate and hemoglobin from facial RGB signals."""

    def __init__(
        self,
        n_signal_channels: int = 12,
        n_hb_features: int = 60,
        fps: float = 30.0,
        nfft: int = 2048,
        hr_min: float = 40.0,
        hr_max: float = 200.0,
        width: int = 96,
        readout_halfwidth_bpm: float = 12.0,
        dropout: float = 0.15,
        hb_dropout: float = 0.15,
    ) -> None:
        """
        Initialize the multi-task HR and hemoglobin model.

        Args:
            n_signal_channels: Number of input facial RGB signal channels.
            n_hb_features: Number of engineered colour and AC-DC features
                provided to the hemoglobin head.
            fps: Sampling frequency of the input signals in Hz.
            nfft: FFT size used for spectral conversion.
            hr_min: Lower HR limit represented by the output spectrum in BPM.
            hr_max: Upper HR limit represented by the output spectrum in BPM.
            width: Number of feature channels used by the shared convolutional
                branches and HR fusion head.
            readout_halfwidth_bpm: Half-width in BPM of the window used by the
                local soft-argmax readout around the dominant spectral peak.
            dropout: Dropout probability applied within the shared spectral and
                temporal branches.
            hb_dropout: Dropout probability applied within the hemoglobin head.
        """
        super().__init__()

        self.n_signal_channels = n_signal_channels
        self.n_hb_features = n_hb_features
        self.fps = float(fps)
        self.nfft = int(nfft)
        self.readout_halfwidth_bpm = float(readout_halfwidth_bpm)

        # Build the FFT frequency grid and keep only the configured HR band.
        frequencies = np.fft.rfftfreq(self.nfft, d=1.0 / self.fps)
        band_indices = np.where((frequencies >= hr_min / 60.0) & (frequencies <= hr_max / 60.0))[0]

        self.register_buffer("band_idx", torch.as_tensor(band_indices, dtype=torch.long))
        self.register_buffer("band_bpm", torch.as_tensor(frequencies[band_indices] * 60.0, dtype=torch.float32))
        self.n_freq = len(band_indices)

        # Shared spectral branch: 1-D convolutions along the frequency axis.
        self.spectral_net = nn.Sequential(
            nn.Conv1d(n_signal_channels, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )

        # Shared temporal branch: strided 1-D convolutions along time followed
        # by global average pooling to produce a compact temporal descriptor.
        self.temporal_net = nn.Sequential(
            nn.Conv1d(n_signal_channels, width, 7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 7, stride=2, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # HR head: combine spectral and temporal features at every frequency
        # bin and produce a pulse-likelihood logit for each bin.
        self.hr_fusion = nn.Sequential(
            nn.Conv1d(width * 2, width, 1),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, 1, 1),
        )

        # Hb head: combine the pooled temporal embedding with engineered
        # colour and AC-DC features to predict a scalar hemoglobin value.
        self.hb_head = nn.Sequential(
            nn.Linear(width + n_hb_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(hb_dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(hb_dropout),
            nn.Linear(32, 1),
        )

    def _spectral_input(self, signal: Tensor) -> Tensor:
        """
        Convert time-domain signals into normalized band-limited log-power spectra.

        Args:
            signal: Input signals with shape (B, C, T).

        Returns:
            Spectral representation with shape (B, C, F), where F is the
            number of FFT bins inside the configured HR range.
        """
        # Apply a Hann window to reduce spectral leakage at the segment
        # boundaries.
        window = torch.hann_window(signal.shape[-1], device=signal.device, dtype=signal.dtype)
        windowed_signal = signal * window

        # Convert each channel from time domain to frequency domain.
        spectrum = torch.fft.rfft(windowed_signal, n=self.nfft, dim=-1)  # type: ignore

        # Compute the power spectrum.
        power = spectrum.real.square() + spectrum.imag.square()  # type: ignore

        # Keep only frequencies within the configured HR band.
        power = power.index_select(-1, self.band_idx)  # type: ignore

        # Compress the dynamic range of spectral power.
        power = torch.log1p(power)  # type: ignore

        # Normalize each channel independently across frequency.
        power = power - power.mean(-1, keepdim=True)
        power = power / (power.std(-1, keepdim=True) + 1e-6)
        return power

    def _local_soft_argmax(self, probability: Tensor) -> Tensor:
        """
        Estimate a continuous HR by soft-argmax around the dominant peak.

        Restricting the expectation to a narrow window around the highest
        probability bin prevents distant competing peaks from pulling the
        estimate toward the region between spectral modes.

        Args:
            probability: Spectral probabilities with shape (B, F).

        Returns:
            Predicted HR in BPM with shape (B,).
        """
        # Locate the dominant HR bin for every sample.
        peak_index = torch.argmax(probability, dim=-1)
        peak_bpm = self.band_bpm[peak_index]  # type: ignore

        # Build a local window around each dominant spectral peak.
        distance = (self.band_bpm[None, :] - peak_bpm[:, None]).abs()  # type: ignore
        window = (distance <= self.readout_halfwidth_bpm).to(probability.dtype)  # type: ignore

        # Re-weight the probabilities by the local window and renormalize.
        windowed_probability = probability * window  # type: ignore
        windowed_probability = windowed_probability / (windowed_probability.sum(-1, keepdim=True) + 1e-9)  # type: ignore

        # Compute the differentiable expectation within the local window.
        predicted_bpm = (windowed_probability * self.band_bpm).sum(-1)  # type: ignore
        return predicted_bpm  # type: ignore

    def forward(self, signal: Tensor, hb_features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Predict heart rate and hemoglobin from facial RGB signals.

        The HR prediction uses the shared spectral and temporal branches,
        while the hemoglobin prediction uses the pooled temporal embedding
        together with engineered colour and AC-DC features.

        Args:
            signal: Input facial RGB signals with shape (B, C, T).
            hb_features: Engineered hemoglobin features with shape
                (B, n_hb_features).

        Returns:
            Tuple containing:
                - Predicted HR in BPM with shape (B,).
                - HR spectral logits with shape (B, F).
                - HR spectral probabilities with shape (B, F).
                - Predicted hemoglobin values with shape (B,).
        """
        # Shared branches.
        spectral_features = self.spectral_net(self._spectral_input(signal))  # (B, width, F)

        temporal_features = self.temporal_net(signal)  # (B, width, 1)
        temporal_embedding = temporal_features.squeeze(-1)  # (B, width)

        # HR head: broadcast the temporal descriptor across frequency bins
        # so it can be fused with the spectral representation.
        temporal_broadcast = temporal_embedding[:, :, None].expand(-1, -1, self.n_freq)  # (B, width, F)

        hr_logits = self.hr_fusion(torch.cat([spectral_features, temporal_broadcast], dim=1)).squeeze(1)  # (B, F)

        hr_probability = torch.softmax(hr_logits, dim=-1)
        predicted_hr_bpm = self._local_soft_argmax(hr_probability)

        # Hb head: combine the shared temporal embedding with engineered
        # colour and AC-DC features.
        hb_input = torch.cat([temporal_embedding, hb_features], dim=1)  # (B, width + n_hb_features)

        predicted_hb = self.hb_head(hb_input).squeeze(-1)  # (B,)
        return (predicted_hr_bpm, hr_logits, hr_probability, predicted_hb)

    def gaussian_target(self, heart_rate: Tensor, sigma_bpm: float = 4.0) -> Tensor:
        """
        Create Gaussian spectral targets centered on the true HR.

        Args:
            heart_rate: Ground-truth HR values with shape (B,).
            sigma_bpm: Standard deviation of the Gaussian target in BPM.

        Returns:
            Normalized target distributions with shape (B, F).
        """
        # Measure the distance from every HR-frequency bin to the true HR.
        distance = (self.band_bpm[None, :] - heart_rate[:, None]) / sigma_bpm  # type: ignore

        # Convert the distances into Gaussian probability targets.
        target = torch.exp(-0.5 * distance.square())  # type: ignore
        return target / (target.sum(-1, keepdim=True) + 1e-9)

    def loss(
        self,
        predicted_hr_bpm: Tensor,
        hr_logits: Tensor,
        predicted_hb: Tensor,
        true_hr_bpm: Tensor,
        true_hb: Tensor,
        hr_l1_weight: float = 0.2,
        hr_ce_weight: float = 1.0,
        hb_weight: float = 0.1,
        sigma_bpm: float = 4.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the combined HR and hemoglobin training loss.

        The HR objective combines Smooth-L1 regression with soft
        cross-entropy against a Gaussian distribution centered on the true
        heart rate. The hemoglobin loss is weighted lightly so the HR task
        remains the primary training signal.

        Args:
            predicted_hr_bpm: Model HR predictions with shape (B,).
            hr_logits: HR spectral logits with shape (B, F).
            predicted_hb: Model hemoglobin predictions with shape (B,).
            true_hr_bpm: Ground-truth HR values with shape (B,).
            true_hb: Ground-truth hemoglobin values with shape (B,).
            hr_l1_weight: Weight applied to the HR Smooth-L1 regression loss.
            hr_ce_weight: Weight applied to the HR soft cross-entropy loss.
            hb_weight: Weight applied to the hemoglobin regression loss.
            sigma_bpm: Width of the Gaussian HR target in BPM.

        Returns:
            Tuple containing:
                - Total multi-task training loss.
                - Dictionary containing the HR and hemoglobin loss values.
        """
        # HR distributional + Smooth-L1 loss.
        hr_smooth_l1 = nn.functional.smooth_l1_loss(predicted_hr_bpm, true_hr_bpm)
        hr_target = self.gaussian_target(true_hr_bpm, sigma_bpm)
        hr_soft_ce = -(hr_target * torch.log_softmax(hr_logits, dim=-1)).sum(-1).mean()
        hr_loss = hr_l1_weight * hr_smooth_l1 + hr_ce_weight * hr_soft_ce

        # Hb Smooth-L1 loss.
        hb_loss = nn.functional.smooth_l1_loss(predicted_hb, true_hb)

        # Keep the HR task as the primary training objective.
        total_loss = hr_loss + hb_weight * hb_loss
        return total_loss, {"hr_loss": float(hr_loss.detach()), "hb_loss": float(hb_loss.detach())}
