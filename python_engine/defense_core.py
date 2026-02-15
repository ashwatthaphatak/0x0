"""
DeepFake Defense Core Engine
Implements Texture Feature Perturbation (TFP) based on Zhang et al., 2025.
Uses GradCAM attention maps + texture feature extraction to generate
imperceptible adversarial perturbations that break deepfake generators.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

# ─── Utility Functions ────────────────────────────────────────────────────────

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a NCHW or CHW tensor in [0,1] to a HWC numpy array."""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, L2 between two tensors in [0,1]."""
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn

    a = tensor_to_numpy(img1) if torch.is_tensor(img1) else img1
    b = tensor_to_numpy(img2) if torch.is_tensor(img2) else img2

    psnr_val = psnr_fn(a, b, data_range=1.0)
    ssim_val = ssim_fn(a, b, channel_axis=2, data_range=1.0)
    l2_dist  = float(np.linalg.norm(a - b))
    return {"PSNR": psnr_val, "SSIM": ssim_val, "L2": l2_dist}


# ─── GradCAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """GradCAM implementation for generating class activation maps."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        output = self.model(x)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1))

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        weights  = torch.mean(self.gradients, dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam      = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam      = F.relu(cam)
        cam_min  = cam.min()
        cam_max  = cam.max()
        if (cam_max - cam_min).abs() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam  # values in [0, 1]


# ─── Dual Attention Module ───────────────────────────────────────────────────

class DualAttentionModule:
    """
    Uses ResNet-50 + GradCAM to produce a spatial attention map that
    highlights texture-critical regions of the image.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = self.resnet.to(device).eval()
        self.gradcam = GradCAM(
            model=self.resnet,
            target_layer=self.resnet.layer4[-1]
        )
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def get_attention_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns a (1, 1, H, W) attention map for the given (1, 3, H, W) tensor.
        """
        img_norm   = self._normalize(img_tensor.squeeze(0))
        img_input  = img_norm.unsqueeze(0)
        img_resized = F.interpolate(img_input, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.enable_grad():
            cam = self.gradcam(img_resized)

        original_size = img_tensor.shape[2:]
        cam_resized = F.interpolate(cam, size=original_size, mode="bilinear", align_corners=False)
        return cam_resized.detach()  # (1, 1, H, W)


# ─── Texture Extractor ───────────────────────────────────────────────────────

class TextureExtractor(nn.Module):
    """
    Extracts texture features using a combination of LBP, bilateral
    filtering, and shallow CNN layers.
    """

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
    def _bilateral_filter(img_np: np.ndarray, d: int = 31,
                           sigma_color: float = 75.0,
                           sigma_space: float = 15.0) -> np.ndarray:
        return cv2.bilateralFilter(img_np.astype(np.float32), d,
                                   sigma_color, sigma_space)

    @staticmethod
    def _compute_lbp(img_np: np.ndarray, radius: int = 1,
                     n_points: int = 8) -> np.ndarray:
        h, w = img_np.shape
        lbp = np.zeros((h, w), dtype=np.float32)
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = img_np[i, j]
                pattern = 0
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    xi = int(round(i + radius * np.sin(angle)))
                    xj = int(round(j + radius * np.cos(angle)))
                    xi = np.clip(xi, 0, h - 1)
                    xj = np.clip(xj, 0, w - 1)
                    if img_np[xi, xj] >= center:
                        pattern |= (1 << p)
                lbp[i, j] = pattern / 255.0
        return lbp

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        img_tensor: (1, 3, H, W) in [0, 1]
        Returns: (1, 64, H/2, W/2) texture feature map
        """
        # Convert to grayscale numpy for LBP
        img_np = tensor_to_numpy(img_tensor)            # (H, W, 3)
        gray   = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        # Bilateral filter for edge-preserving smoothing
        smooth = self._bilateral_filter(gray_f)

        # LBP texture map (fast approximation using cv2 for large images)
        # For production use skimage LBP; here we use a simple gradient-based approx
        # to avoid O(H*W*n_points) Python loops on large images.
        sobelx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        texture_map = np.sqrt(sobelx ** 2 + sobely ** 2)
        if texture_map.max() > 1e-8:
            texture_map = texture_map / texture_map.max()

        # Shape: (1, 1, H, W)
        device  = img_tensor.device
        tex_t   = torch.from_numpy(texture_map).unsqueeze(0).unsqueeze(0).to(device)

        # CNN feature extraction
        f1 = self.conv1(tex_t)       # (1, 32, H, W)
        f2 = self.conv2(f1)          # (1, 64, H, W)
        f3 = self.maxpool(f2)        # (1, 64, H/2, W/2)
        return f3


# ─── Perturbation Enhancement ────────────────────────────────────────────────

class PerturbationEnhancement(nn.Module):
    """
    Takes texture features and an attention map to synthesize
    a spatially-weighted adversarial perturbation.
    """

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
        # Fuse encoded texture (128) with down-sampled attention (1) → 64
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
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

    def forward(self, texture_features: torch.Tensor,
                attention_map: torch.Tensor) -> torch.Tensor:
        """
        texture_features: (1, 64, H/2, W/2)
        attention_map:     (1, 1, H, W)
        Returns:           (1, 3, H, W) perturbation in [-1, 1]
        """
        enc          = self.encoder(texture_features)   # (1, 128, H/2, W/2)
        h, w         = enc.shape[2], enc.shape[3]
        att_down     = F.interpolate(attention_map, size=(h, w),
                                     mode="bilinear", align_corners=False)
        fused        = self.attention_fusion(torch.cat([enc, att_down], dim=1))
        perturbation = self.decoder(fused)              # (1, 3, H, W)
        return perturbation


# ─── Defense Framework ───────────────────────────────────────────────────────

class DeepfakeDefenseFramework(nn.Module):
    """
    Full pipeline: texture extraction → attention-guided perturbation
    → L∞-bounded vaccination.
    """

    def __init__(self, epsilon: float = 0.05, device: torch.device | None = None):
        super().__init__()
        self.epsilon          = epsilon
        self.device           = device or torch.device("cpu")
        self.texture_extractor = TextureExtractor().to(self.device)
        self.perturbation_gen  = PerturbationEnhancement().to(self.device)

    def generate_perturbation(self, img_tensor: torch.Tensor,
                              attention_map: torch.Tensor) -> torch.Tensor:
        texture_features = self.texture_extractor(img_tensor)
        perturbation     = self.perturbation_gen(texture_features, attention_map)
        return self.epsilon * perturbation

    @torch.no_grad()
    def vaccinate_image(self, img_tensor: torch.Tensor,
                        attention_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (vaccinated_image, perturbation) both as (1, 3, H, W) tensors in [0, 1].
        """
        perturbation = self.generate_perturbation(img_tensor, attention_map)
        vaccinated   = torch.clamp(img_tensor + perturbation, 0.0, 1.0)
        return vaccinated, perturbation


# ─── Image I/O ───────────────────────────────────────────────────────────────

def load_image(path: str, size: int = 1024, device: torch.device | None = None) -> torch.Tensor:
    """Load an image from disk, resize to size×size, return (1, 3, H, W) tensor in [0, 1]."""
    device = device or torch.device("cpu")
    img    = Image.open(path).convert("RGB")
    tf     = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0).to(device)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a (1, 3, H, W) or (3, H, W) tensor in [0, 1] to disk."""
    arr = tensor_to_numpy(tensor)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def compute_protection_score(original: torch.Tensor,
                              vaccinated: torch.Tensor) -> float:
    """
    Heuristic protection score [0-100].
    Higher PSNR and SSIM = more imperceptible = better protection.
    Score penalises visible artifacts while rewarding non-zero perturbation.
    """
    metrics = calculate_metrics(original, vaccinated)
    psnr    = min(metrics["PSNR"], 50.0) / 50.0   # normalise ∈ [0,1]
    ssim    = metrics["SSIM"]                       # already ∈ [0,1]
    l2_norm = min(metrics["L2"] / (original.numel() ** 0.5), 1.0)

    # Penalise if l2 is zero (no perturbation) or too large (visible)
    perturbation_bonus = 1.0 - abs(l2_norm - 0.05) * 5.0
    perturbation_bonus = max(0.0, min(1.0, perturbation_bonus))

    score = 0.4 * psnr + 0.4 * ssim + 0.2 * perturbation_bonus
    return round(score * 100, 1)
