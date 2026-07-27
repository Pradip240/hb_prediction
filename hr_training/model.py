"""model.py — HRSpectralNet: a spectral CNN that regresses heart rate from a segment.

Design rationale (tied to why the DSP baseline fails on this data): heart rate is a
*frequency*, and POS/CHROM/green fail by locking onto the wrong spectral peak and
systematically underestimating (the strong negative bias in the scorecard). So this
model works in the frequency domain:

  1. FFT each of the input channels (per-region RGB, optionally + POS traces), take
     band-limited log-power -> a (C, F) spectral image.
  2. A small 1-D CNN across frequency mixes the channels into a single "pulse
     likelihood" curve over the HR band — it learns which region/channel to trust and
     how to suppress non-cardiac peaks, the thing a fixed DSP projection cannot do.
  3. Softmax over frequency + a soft-argmax expected value gives a smooth,
     differentiable HR in BPM.

Loss = Smooth-L1 on the predicted BPM + a soft cross-entropy toward a Gaussian centred
on the true HR (keeps the predicted spectrum unimodal at the truth rather than
averaging two peaks). The soft-argmax lets it regress a continuous HR while the CE term
keeps it from hedging across octave-apart peaks.
"""

import numpy as np
import torch
import torch.nn as nn


class HRSpectralNet(nn.Module):
    def __init__(self, n_channels=9, fps=30.0, nfft=2048,
                 hr_min=40.0, hr_max=200.0, width=64):
        super().__init__()
        self.fps = float(fps)
        self.nfft = int(nfft)

        freqs = np.fft.rfftfreq(self.nfft, d=1.0 / self.fps)
        lo, hi = hr_min / 60.0, hr_max / 60.0
        band = np.where((freqs >= lo) & (freqs <= hi))[0]
        self.register_buffer("band_idx", torch.as_tensor(band, dtype=torch.long))
        self.register_buffer("band_bpm", torch.as_tensor(freqs[band] * 60.0, dtype=torch.float32))
        self.n_freq = int(len(band))

        c = width
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, c, 5, padding=2), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, 1, 1),
        )

    def spectral_input(self, x):
        """(B, C, L) time -> (B, C, F) per-channel normalized band log-power."""
        X = torch.fft.rfft(x, n=self.nfft, dim=-1)
        power = X.real ** 2 + X.imag ** 2
        power = power.index_select(-1, self.band_idx)
        power = torch.log1p(power)
        power = power - power.mean(dim=-1, keepdim=True)
        power = power / (power.std(dim=-1, keepdim=True) + 1e-6)
        return power

    def forward(self, x):
        spec = self.spectral_input(x)
        logits = self.net(spec).squeeze(1)          # (B, F)
        prob = torch.softmax(logits, dim=-1)
        pred_bpm = (prob * self.band_bpm).sum(dim=-1)
        return pred_bpm, logits, prob

    def gaussian_target(self, hr_bpm, sigma_bpm=4.0):
        d = (self.band_bpm[None, :] - hr_bpm[:, None]) / sigma_bpm
        t = torch.exp(-0.5 * d * d)
        return t / (t.sum(dim=-1, keepdim=True) + 1e-9)

    def loss(self, pred_bpm, logits, prob, true_bpm, l1_weight=1.0, ce_weight=0.3, sigma_bpm=4.0):
        sl1 = nn.functional.smooth_l1_loss(pred_bpm, true_bpm)
        target = self.gaussian_target(true_bpm, sigma_bpm)
        logp = torch.log_softmax(logits, dim=-1)
        ce = -(target * logp).sum(dim=-1).mean()
        return l1_weight * sl1 + ce_weight * ce, {"smooth_l1": sl1.item(), "soft_ce": ce.item()}