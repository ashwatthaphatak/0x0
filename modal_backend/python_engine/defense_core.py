"""
python_engine/defense_core.py
────────────────────────────────────────────────────────────────────────────
Self-contained ML backend extracted from DeepFake_Defense_COMPLETE.ipynb.

Exports:
  - DualAttentionModule
  - DeepfakeDefenseFramework
  - StarGANGenerator
  - load_stargan_weights()
  - save_image()
  - compute_protection_score()
  - tensor_to_numpy()
  - calculate_metrics()
"""

from __future__ import annotations

import io
import os
import warnings
import zipfile
from pathlib import Path

import cv2
import lpips
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)


# ── Utility functions ─────────────────────────────────────────────────────────

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    return tensor.cpu().permute(1, 2, 0).numpy()


def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> dict:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    if torch.is_tensor(img1):
        img1 = tensor_to_numpy(img1)
    if torch.is_tensor(img2):
        img2 = tensor_to_numpy(img2)

    return {
        "PSNR": psnr(img1, img2, data_range=1.0),
        "SSIM": ssim(img1, img2, channel_axis=2, data_range=1.0),
        "L2":   float(np.linalg.norm(img1 - img2)),
    }


def compute_protection_score(original: torch.Tensor, vaccinated: torch.Tensor) -> float:
    """Higher = more protected. Returns L2 distance between original and vaccinated."""
    return calculate_metrics(original, vaccinated)["L2"]


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a [0,1] float tensor to disk as PNG."""
    arr = tensor_to_numpy(tensor)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = True,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx=None) -> torch.Tensor:
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.unsqueeze(1)


# ── Texture Extractor (LBP-based) ─────────────────────────────────────────────

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

    @staticmethod
    def bilateral_filter(img_np: np.ndarray, d: int = 31,
                         sigma_color: float = 75, sigma_space: float = 15) -> np.ndarray:
        return cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)

    @staticmethod
    def compute_lbp(img_np: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
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
        device = img_tensor.device
        texture_features = []
        for b in range(img_tensor.shape[0]):
            img = img_tensor[b].cpu().numpy()
            gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])
            gray = (gray * 255).astype(np.uint8)
            filtered = self.bilateral_filter(gray)
            lbp = self.compute_lbp(filtered).astype(np.float32) / 255.0
            texture_features.append(torch.from_numpy(lbp).unsqueeze(0).to(device))
        lbp_batch = torch.stack(texture_features, dim=0)
        x = self.conv1(lbp_batch)
        x = self.conv2(x)
        return self.maxpool(x)


# ── Dual Attention Module (ResNet50 + GradCAM) ────────────────────────────────

class DualAttentionModule:
    def __init__(self, device: torch.device):
        self.device = device
        self.resnet = models.resnet50(pretrained=True).to(device).eval()
        self.gradcam = GradCAM(
            model=self.resnet,
            target_layer=self.resnet.layer4[-1],
        )
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def get_attention_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_normalized = self._normalize(img_tensor.squeeze(0)).unsqueeze(0)
        img_resized    = F.interpolate(img_normalized, size=(224, 224),
                                       mode="bilinear", align_corners=False)
        with torch.enable_grad():
            cam = self.gradcam(img_resized)
        return F.interpolate(cam, size=img_tensor.shape[2:],
                             mode="bilinear", align_corners=False)


# ── Perturbation Enhancement ──────────────────────────────────────────────────

class PerturbationEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, texture_features: torch.Tensor,
                attention_map: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(texture_features)
        attn_ds = F.interpolate(attention_map, size=encoded.shape[2:],
                                mode="bilinear", align_corners=False)
        fused = self.attention_fusion(torch.cat([encoded, attn_ds], dim=1))
        return self.decoder(fused)


# ── Defense Framework ─────────────────────────────────────────────────────────

class DeepfakeDefenseFramework(nn.Module):
    def __init__(self, epsilon: float = 0.05, device: torch.device | None = None):
        super().__init__()
        self.epsilon = epsilon
        self._device  = device or torch.device("cpu")
        self.texture_extractor = TextureExtractor()
        self.perturbation_gen  = PerturbationEnhancement()
        self.lpips_loss        = lpips.LPIPS(net="alex").to(self._device)

    def generate_perturbation(self, img_tensor: torch.Tensor,
                              attention_map: torch.Tensor) -> torch.Tensor:
        texture_features = self.texture_extractor(img_tensor)
        perturbation     = self.perturbation_gen(texture_features, attention_map)
        return self.epsilon * perturbation

    def vaccinate_image(self, img_tensor: torch.Tensor,
                        attention_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            perturbation = self.generate_perturbation(img_tensor, attention_map)
            vaccinated   = torch.clamp(img_tensor + perturbation, 0, 1)
        return vaccinated, perturbation


# ── StarGAN ───────────────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x)


class StarGANGenerator(nn.Module):
    def __init__(self, conv_dim: int = 64, c_dim: int = 5, repeat_num: int = 6):
        super().__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        curr = conv_dim
        for _ in range(2):
            layers += [nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(curr * 2, affine=True),
                       nn.ReLU(inplace=True)]
            curr *= 2
        for _ in range(repeat_num):
            layers.append(_ResidualBlock(curr))
        for _ in range(2):
            layers += [nn.ConvTranspose2d(curr, curr // 2, 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(curr // 2, affine=True),
                       nn.ReLU(inplace=True)]
            curr //= 2
        layers += [nn.Conv2d(curr, 3, 7, 1, 3, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = c.view(c.size(0), c.size(1), 1, 1).expand(
            c.size(0), c.size(1), x.size(2), x.size(3))
        return self.main(torch.cat([x, c], dim=1))


def load_stargan_weights(
    device: torch.device,
    ckpt_dir: str = "stargan_celeba_128/models",
) -> StarGANGenerator:
    """
    Downloads weights if missing, strips legacy InstanceNorm stats,
    and returns a ready-to-use StarGANGenerator.
    """
    ckpt_dir  = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "200000-G.ckpt"
    zip_path  = ckpt_dir / "celeba-128x128-5attrs.zip"

    if not ckpt_path.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        url = "https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1"
        print("📥 Downloading StarGAN weights …")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ckpt_dir)
        zip_path.unlink()
        print("✅ StarGAN weights downloaded.")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    cleaned    = {k: v for k, v in state_dict.items()
                  if not (k.endswith(".running_mean") or k.endswith(".running_var"))}

    G = StarGANGenerator().to(device)
    G.load_state_dict(cleaned, strict=False)
    G.eval()
    print("✅ StarGAN Generator ready.")
    return G
