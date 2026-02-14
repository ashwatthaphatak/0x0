import torch
import torch.nn as nn
import torch.nn.functional as F


class PerturbationEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.attention_fusion = nn.Sequential(nn.Conv2d(129, 64, kernel_size=1), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, texture_features: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(texture_features)
        attention_downsampled = F.interpolate(
            attention_map,
            size=encoded.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = torch.cat([encoded, attention_downsampled], dim=1)
        fused = self.attention_fusion(fused)
        perturbation = self.decoder(fused)
        return perturbation
