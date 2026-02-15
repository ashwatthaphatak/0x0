import os
import random
import ssl
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

try:
    import certifi
except Exception:
    certifi = None


def configure_runtime() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if certifi is not None:
        ca_bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)

        def _certifi_https_context(*args, **kwargs):
            return ssl.create_default_context(cafile=ca_bundle)

        ssl._create_default_https_context = _certifi_https_context


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_image_transform(size: Tuple[int, int] = (256, 256)) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])


def load_image(image_path: str, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img, _ = ensure_square_image(img)
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


def ensure_square_image(pil_img: Image.Image):
    if pil_img.width == pil_img.height:
        return pil_img, False
    side = min(pil_img.width, pil_img.height)
    left = (pil_img.width - side) // 2
    top = (pil_img.height - side) // 2
    square = pil_img.crop((left, top, left + side, top + side))
    return square, True


FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_bbox(
    pil_img: Image.Image, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (60, 60)
):
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def expand_bbox(bbox, img_w: int, img_h: int, margin: float = 0.2):
    x, y, w, h = bbox
    pad_w = int(w * margin)
    pad_h = int(h * margin)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(img_w, x + w + pad_w)
    y1 = min(img_h, y + h + pad_h)
    return (x0, y0, x1 - x0, y1 - y0)


def crop_face_region(pil_img: Image.Image, margin: float = 0.2, min_size: int = 60):
    bbox = detect_face_bbox(pil_img, min_size=(min_size, min_size))
    if bbox is None:
        return pil_img.copy(), (0, 0, pil_img.width, pil_img.height), False
    bbox = expand_bbox(bbox, pil_img.width, pil_img.height, margin)
    x, y, w, h = bbox
    crop = pil_img.crop((x, y, x + w, y + h))
    return crop, bbox, True


def composite_on_full(full_pil: Image.Image, crop_tensor: torch.Tensor, bbox, feather_ratio: float = 0.08):
    if full_pil is None or bbox is None:
        return None
    x, y, w, h = bbox
    crop_pil = to_pil(crop_tensor).resize((w, h), Image.BICUBIC)

    # Feathered mask avoids visible ROI borders after compositing.
    edge = int(max(1, min(w, h) * max(0.0, feather_ratio)))
    if edge <= 1:
        mask = Image.new("L", (w, h), 255)
    else:
        inner_w = max(1, w - 2 * edge)
        inner_h = max(1, h - 2 * edge)
        mask = Image.new("L", (w, h), 0)
        mask.paste(Image.new("L", (inner_w, inner_h), 255), (edge, edge))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=edge * 0.6))

    out = full_pil.copy()
    out.paste(crop_pil, (x, y), mask)
    return out


def blend_face_effect(base_pil: Image.Image, deepfake_full_pil: Image.Image, bbox, feather_ratio: float = 0.08):
    if base_pil is None or deepfake_full_pil is None:
        return None
    if bbox is None:
        return deepfake_full_pil.copy()

    x, y, w, h = bbox
    x = max(0, min(x, base_pil.width - 1))
    y = max(0, min(y, base_pil.height - 1))
    w = max(1, min(w, base_pil.width - x))
    h = max(1, min(h, base_pil.height - y))

    patch = deepfake_full_pil.crop((x, y, x + w, y + h))

    edge = int(max(1, min(w, h) * max(0.0, feather_ratio)))
    if edge <= 1:
        mask = Image.new("L", (w, h), 255)
    else:
        inner_w = max(1, w - 2 * edge)
        inner_h = max(1, h - 2 * edge)
        mask = Image.new("L", (w, h), 0)
        mask.paste(Image.new("L", (inner_w, inner_h), 255), (edge, edge))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=edge * 0.6))

    out = base_pil.copy()
    out.paste(patch, (x, y), mask)
    return out
