"""model.py — HbMLP: a small regressor for hemoglobin from amplitude/colour features.

Unlike the HR model (a custom spectral CNN over the raw signal), the Hb model is
deliberately small and generic: the informative Hb signal is low-dimensional and lives
in the engineered features (see features.py), so a shallow MLP on that feature vector is
the right tool. A small model also limits the capacity to memorise subject identity
(hemoglobin is per-subject), which — with per-segment features sharing one per-subject
label — is the main failure mode to guard against.

The network regresses a *standardised, mean-centred* target (train.py handles the
de-centring at read-out), so the output layer only needs to produce values near zero.
"""

import torch.nn as nn


class HbMLP(nn.Module):
    def __init__(self, n_in, width=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(p),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.GELU(), nn.Dropout(p),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)