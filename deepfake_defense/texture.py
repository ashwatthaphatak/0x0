import cv2
import numpy as np
import torch
import torch.nn as nn


class TextureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(2, 2)

    def bilateral_filter(self, img_np, d=31, sigma_color=75, sigma_space=15):
        return cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)

    def compute_lbp(self, img_np, radius=1, n_points=8):
        height, width = img_np.shape
        lbp = np.zeros((height, width), dtype=np.uint8)

        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = img_np[i, j]
                pattern = 0
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j - radius * np.sin(angle))
                    x = max(0, min(x, height - 1))
                    y = max(0, min(y, width - 1))
                    if img_np[x, y] >= center:
                        pattern += 2 ** p
                lbp[i, j] = pattern
        return lbp

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = img_tensor.shape[0]
        device = img_tensor.device
        texture_features = []

        for b in range(batch_size):
            img = img_tensor[b].detach().cpu().numpy()
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.116 * img[2]
            gray = (gray * 255).astype(np.uint8)
            filtered = self.bilateral_filter(gray)
            lbp = self.compute_lbp(filtered)
            lbp = lbp.astype(np.float32) / 255.0
            lbp_tensor = torch.from_numpy(lbp).unsqueeze(0).to(device)
            texture_features.append(lbp_tensor)

        lbp_batch = torch.stack(texture_features, dim=0)
        x = self.conv1(lbp_batch)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x
