import os
import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_transform(size: Tuple[int, int] = (256, 256)) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])


def load_image(image_path: str, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


def to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor_to_numpy(tensor)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
    if torch.is_tensor(img1):
        img1 = tensor_to_numpy(img1)
    if torch.is_tensor(img2):
        img2 = tensor_to_numpy(img2)

    psnr_val = psnr(img1, img2, data_range=1.0)
    ssim_val = ssim(img1, img2, channel_axis=2, data_range=1.0)
    l2_dist = float(np.linalg.norm(img1 - img2))

    return {"PSNR": float(psnr_val), "SSIM": float(ssim_val), "L2": l2_dist}


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
