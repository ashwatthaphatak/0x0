"""
StarGAN attack utilities used by the local sidecar.

This module runs a lightweight attribute-edit attack on both the original and
sanitized images so the desktop UI can compare attack effectiveness.
"""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Callable

import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from defense_core import save_image

CHECKPOINT_URL = "https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1"
CHECKPOINT_REL_PATH = Path("stargan_celeba_128/models/200000-G.ckpt")
CHECKPOINT_ZIP_NAME = "celeba-128x128-5attrs.zip"
# Divergence is reported as RMSE in [0,1], so thresholds are resolution-independent.
BLOCKED_RMSE_THRESHOLD = 0.03
PARTIAL_RMSE_THRESHOLD = 0.015
DEFAULT_ATTACK_SIZE = 256
ENV_STARGAN_CKPT = "DEEPFAKE_DEFENSE_STARGAN_CKPT"
ENV_STARGAN_ZIP = "DEEPFAKE_DEFENSE_STARGAN_ZIP"

# CelebA checkpoint attribute order: [Black_Hair, Blond_Hair, Brown_Hair, Male, Young]
_CANONICAL_PRESETS: dict[str, tuple[str, list[float]]] = {
    "black_hair_female_old": ("Black Hair + Female + Old", [1.0, 0.0, 0.0, 0.0, 0.0]),
    "black_hair_female_young": ("Black Hair + Female + Young", [1.0, 0.0, 0.0, 0.0, 1.0]),
    "black_hair_male_old": ("Black Hair + Male + Old", [1.0, 0.0, 0.0, 1.0, 0.0]),
    "black_hair_male_young": ("Black Hair + Male + Young", [1.0, 0.0, 0.0, 1.0, 1.0]),
    "blonde_hair_female_old": ("Blonde Hair + Female + Old", [0.0, 1.0, 0.0, 0.0, 0.0]),
    "blonde_hair_female_young": ("Blonde Hair + Female + Young", [0.0, 1.0, 0.0, 0.0, 1.0]),
    "blonde_hair_male_old": ("Blonde Hair + Male + Old", [0.0, 1.0, 0.0, 1.0, 0.0]),
    "blonde_hair_male_young": ("Blonde Hair + Male + Young", [0.0, 1.0, 0.0, 1.0, 1.0]),
    "brown_hair_female_old": ("Brown Hair + Female + Old", [0.0, 0.0, 1.0, 0.0, 0.0]),
    "brown_hair_female_young": ("Brown Hair + Female + Young", [0.0, 0.0, 1.0, 0.0, 1.0]),
    "brown_hair_male_old": ("Brown Hair + Male + Old", [0.0, 0.0, 1.0, 1.0, 0.0]),
    "brown_hair_male_young": ("Brown Hair + Male + Young", [0.0, 0.0, 1.0, 1.0, 1.0]),
}

ATTRIBUTES: dict[str, list[float]] = {
    key: value for key, (_label, value) in _CANONICAL_PRESETS.items()
}

ATTACK_LABELS: dict[str, str] = {
    key: label for key, (label, _value) in _CANONICAL_PRESETS.items()
}

ALIAS_TO_CANONICAL: dict[str, str] = {
    # Backward-compatible aliases used by existing UI/CLI defaults.
    "blonde": "blonde_hair_female_old",
    "blond_hair": "blonde_hair_female_old",
    "blonde_hair": "blonde_hair_female_old",
    "black_hair": "black_hair_female_old",
    "brown_hair": "brown_hair_female_old",
    "old": "brown_hair_female_old",
    "old_age": "brown_hair_female_old",
    "young": "brown_hair_female_young",
    "young_age": "brown_hair_female_young",
    "female": "brown_hair_female_young",
    "male": "brown_hair_male_young",
}

_MODEL_CACHE: dict[str, object] = {
    "model": None,
    "device": None,
    "checkpoint": None,
}


class ResidualBlock(nn.Module):
    """Residual block used by StarGAN generator."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x)


class StarGANGenerator(nn.Module):
    """StarGAN v1 generator for CelebA 5 attributes."""

    def __init__(self, conv_dim: int = 64, c_dim: int = 5, repeat_num: int = 6) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]

        curr_dim = conv_dim
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(curr_dim * 2, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_dim *= 2

        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(curr_dim // 2, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_dim //= 2

        layers.extend([nn.Conv2d(curr_dim, 3, 7, 1, 3, bias=False), nn.Tanh()])
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_map = c.view(c.size(0), c.size(1), 1, 1)
        c_map = c_map.repeat(1, 1, x.size(2), x.size(3))
        return self.main(torch.cat([x, c_map], dim=1))


def normalize_attack_type(attack_type: str | None) -> str:
    raw = (attack_type or "").strip().lower()
    raw = raw.replace(" ", "_").replace("+", "_").replace("-", "_")
    while "__" in raw:
        raw = raw.replace("__", "_")

    canonical = ALIAS_TO_CANONICAL.get(raw, raw)
    if canonical not in ATTRIBUTES:
        canonical_keys = sorted(ATTRIBUTES)
        alias_keys = sorted(ALIAS_TO_CANONICAL)
        supported = ", ".join(canonical_keys + alias_keys)
        raise ValueError(
            f"Unsupported attack type: {attack_type}. Supported: {supported}"
        )
    return canonical


def resolve_model_root(explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        return Path(explicit_dir).expanduser()

    env_dir = os.getenv("DEEPFAKE_DEFENSE_MODEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()

    cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
    if cache_home:
        return Path(cache_home) / "deepfake-defense"

    # Keep default consistent with download_stargan_weights.py and Tauri command setup.
    return Path.home() / ".deepfake-defense-models"


def ensure_checkpoint(
    model_root: Path,
    progress: Callable[[str], None] | None = None,
) -> Path:
    ckpt_path = model_root / CHECKPOINT_REL_PATH
    if ckpt_path.exists():
        return ckpt_path

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = ckpt_path.parent / CHECKPOINT_ZIP_NAME

    local_ckpt = os.getenv(ENV_STARGAN_CKPT, "").strip()
    local_zip = os.getenv(ENV_STARGAN_ZIP, "").strip()

    def _normalize_extracted_checkpoint(source_zip: Path) -> Path:
        if ckpt_path.exists():
            return ckpt_path

        candidates = list(model_root.rglob(ckpt_path.name))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find {ckpt_path.name} after extracting {source_zip}."
            )

        shutil.copy2(candidates[0], ckpt_path)
        return ckpt_path

    if local_ckpt:
        source = Path(local_ckpt).expanduser()
        if not source.exists():
            raise FileNotFoundError(
                f"{ENV_STARGAN_CKPT} points to a missing file: {source}"
            )

        if source.suffix.lower() == ".zip":
            if progress:
                progress(f"Extracting StarGAN zip from {source}...")
            with zipfile.ZipFile(source, "r") as archive:
                archive.extractall(model_root)
            return _normalize_extracted_checkpoint(source)

        if progress:
            progress(f"Using StarGAN checkpoint from {source}...")
        shutil.copy2(source, ckpt_path)
        return ckpt_path

    if local_zip:
        source_zip = Path(local_zip).expanduser()
        if not source_zip.exists():
            raise FileNotFoundError(
                f"{ENV_STARGAN_ZIP} points to a missing file: {source_zip}"
            )
        if progress:
            progress(f"Extracting StarGAN zip from {source_zip}...")
        with zipfile.ZipFile(source_zip, "r") as archive:
            archive.extractall(model_root)
        return _normalize_extracted_checkpoint(source_zip)

    if progress:
        progress("Downloading StarGAN checkpoint (one-time setup)...")

    response = requests.get(CHECKPOINT_URL, stream=True, timeout=180)
    response.raise_for_status()

    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                out_file.write(chunk)

    if progress:
        progress("Extracting StarGAN checkpoint...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(model_root)

    try:
        zip_path.unlink(missing_ok=True)
    except Exception:
        pass

    return _normalize_extracted_checkpoint(zip_path)


def _load_generator(
    device: torch.device,
    model_root: Path,
    progress: Callable[[str], None] | None = None,
) -> StarGANGenerator:
    ckpt_path = ensure_checkpoint(model_root, progress)

    cached_model = _MODEL_CACHE.get("model")
    cached_device = _MODEL_CACHE.get("device")
    cached_checkpoint = _MODEL_CACHE.get("checkpoint")
    if (
        isinstance(cached_model, StarGANGenerator)
        and cached_device == str(device)
        and cached_checkpoint == str(ckpt_path)
    ):
        return cached_model

    if progress:
        progress("Loading StarGAN generator...")

    state_dict = torch.load(ckpt_path, map_location=device)
    cleaned = {
        k: v
        for k, v in state_dict.items()
        if not (k.endswith(".running_mean") or k.endswith(".running_var"))
    }

    model = StarGANGenerator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    load_status = model.load_state_dict(cleaned, strict=False)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(cleaned.keys())
    matched = len(model_keys & ckpt_keys)
    match_ratio = matched / max(1, len(model_keys))

    if match_ratio < 0.9:
        raise RuntimeError(
            "StarGAN checkpoint appears incompatible with expected CelebA-128 generator "
            f"(key match ratio={match_ratio:.2f}). "
            "Use checkpoint 200000-G.ckpt from celeba-128x128-5attrs."
        )

    if len(load_status.unexpected_keys) > 0 and match_ratio < 0.98:
        raise RuntimeError(
            "Unexpected keys detected while loading StarGAN checkpoint. "
            "This likely indicates the wrong pretrained file."
        )

    model.eval()

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = str(device)
    _MODEL_CACHE["checkpoint"] = str(ckpt_path)

    return model


def _center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _load_image(path: str, device: torch.device, size: int) -> torch.Tensor:
    if size <= 0:
        raise ValueError(f"attack_size must be > 0 (got {size})")
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    image = Image.open(path).convert("RGB")
    image = _center_crop_square(image)
    return transform(image).unsqueeze(0).to(device)


@torch.no_grad()
def _run_stargan_attack(
    generator: StarGANGenerator,
    image_tensor: torch.Tensor,
    attribute: torch.Tensor,
) -> torch.Tensor:
    normalized = image_tensor * 2 - 1
    attacked = generator(normalized, attribute)
    attacked = (attacked + 1) / 2
    return torch.clamp(attacked, 0.0, 1.0)


def run_attack_comparison(
    original_path: str,
    sanitized_path: str,
    attack_type: str,
    output_dir: str,
    progress: Callable[[int, str], None] | None = None,
    model_dir: str | None = None,
    attack_size: int = DEFAULT_ATTACK_SIZE,
) -> dict[str, object]:
    attack_key = normalize_attack_type(attack_type)

    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"Original image not found: {original_path}")
    if not os.path.isfile(sanitized_path):
        raise FileNotFoundError(f"Sanitized image not found: {sanitized_path}")

    def _update(pct: int, msg: str) -> None:
        if progress:
            progress(pct, msg)

    _update(10, "Preparing StarGAN attack")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = resolve_model_root(model_dir)

    generator = _load_generator(
        device,
        model_root,
        progress=lambda msg: _update(25, msg),
    )

    _update(40, "Loading input images")
    original_tensor = _load_image(original_path, device, size=attack_size)
    sanitized_tensor = _load_image(sanitized_path, device, size=attack_size)

    _update(60, f"Running attack: {ATTACK_LABELS[attack_key]}")
    attribute = torch.tensor([ATTRIBUTES[attack_key]], dtype=torch.float32, device=device)
    original_fake = _run_stargan_attack(generator, original_tensor, attribute)
    sanitized_fake = _run_stargan_attack(generator, sanitized_tensor, attribute)

    _update(80, "Saving attacked images")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = uuid.uuid4().hex[:8]
    original_fake_path = out_dir / f"deepfake-original-{suffix}.png"
    sanitized_fake_path = out_dir / f"deepfake-sanitized-{suffix}.png"

    save_image(original_fake, str(original_fake_path))
    save_image(sanitized_fake, str(sanitized_fake_path))

    diff = original_fake - sanitized_fake
    divergence = float(torch.sqrt(torch.mean(diff.pow(2))).item())
    l2_distance = float(torch.linalg.vector_norm(diff).item())

    if divergence > BLOCKED_RMSE_THRESHOLD:
        verdict = "blocked"
    elif divergence > PARTIAL_RMSE_THRESHOLD:
        verdict = "partial"
    else:
        verdict = "not_blocked"

    _update(100, "Deepfake test complete")

    return {
        "attack_type": attack_key,
        "attack_label": ATTACK_LABELS[attack_key],
        "original_fake_path": str(original_fake_path),
        "sanitized_fake_path": str(sanitized_fake_path),
        "divergence": divergence,
        "l2_distance": l2_distance,
        "verdict": verdict,
    }
