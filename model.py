"""
Hb Predictor with spatial attention and multi-scale feature fusion.

Key design choices:
- Spatial attention pool replaces global avg pool, so the model can learn
  WHICH face regions matter for Hb prediction.
- Attention entropy can be regularized in the training loop to encourage
  focused (rather than uniform) attention.
- ResNet50 stem/layers 1-3 are frozen by default; only layer4 + attention
  + regressor train. This reduces overfitting on small medical datasets.
"""

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class SpatialAttention(nn.Module):
    """Learns a [B, 1, H, W] attention map and uses it to weight-pool features."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        attn_logits = self.attention(x)                              # [B, 1, H, W]
        B, _, H, W = attn_logits.shape
        attn = torch.softmax(attn_logits.flatten(2), dim=2)          # [B, 1, H*W]
        attn = attn.view(B, 1, H, W)                                 # [B, 1, H, W]
        pooled = (x * attn).sum(dim=(2, 3))                          # [B, C]
        return pooled, attn


class HbPredictor(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, freeze_early: bool = True) -> None:
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        backbone = resnet50(weights=weights)

        # Use the backbone but stop before avgpool/fc — we need spatial features.
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # [B, 256, 56, 56]
        self.layer2 = backbone.layer2   # [B, 512, 28, 28]
        self.layer3 = backbone.layer3   # [B, 1024, 14, 14]
        self.layer4 = backbone.layer4   # [B, 2048, 7, 7]

        if freeze_early:
            # Freeze stem, layer1, layer2, layer3. Train only layer4.
            for module in [self.stem, self.layer1, self.layer2, self.layer3]:
                for p in module.parameters():
                    p.requires_grad = False

        # Spatial attention on the deepest features (semantic level)
        self.attention_pool = SpatialAttention(in_channels=2048)

        # We also keep a global-avg-pool of layer1 (low-level color/texture)
        # because Hb signal lives heavily in skin/lip color statistics.
        self.low_level_pool = nn.AdaptiveAvgPool2d(1)
        fused_dim = 2048 + 256

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

        # Last attention map cached so the training loop can regularize it
        # and the diagnostic script can visualize it.
        self._last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        f1 = self.layer1(x)              # low-level color/texture
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)             # semantic, used for attention

        attn_pooled, attn = self.attention_pool(f4)        # [B, 2048]
        self._last_attention = attn

        low_pooled = self.low_level_pool(f1).flatten(1)    # [B, 256]
        fused = torch.cat([attn_pooled, low_pooled], dim=1)
        return self.regressor(fused)

    def get_last_attention(self) -> torch.Tensor | None:
        """Returns the most recent attention map [B, 1, 7, 7]."""
        return self._last_attention


def attention_entropy_loss(attn: torch.Tensor, target_entropy: float = 2.5) -> torch.Tensor:
    """
    Penalize attention maps that are too uniform.

    Uniform attention over 49 (=7*7) locations has entropy log(49) ~= 3.89.
    Setting target_entropy below that pushes the model toward concentrated
    attention. 2.5 is a reasonable middle ground - focused but not collapsed
    to a single pixel.
    """
    attn_flat = attn.flatten(2)  # [B, 1, H*W]
    eps = 1e-8
    entropy = -(attn_flat * torch.log(attn_flat + eps)).sum(dim=2).mean()
    # Only penalize when entropy is above target (i.e., too diffuse)
    return torch.relu(entropy - target_entropy)


if __name__ == "__main__":
    # Sanity check
    model = HbPredictor()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")              # [2, 1]
    print(f"Attention shape: {model.get_last_attention().shape}")  # [2, 1, 7, 7]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")
