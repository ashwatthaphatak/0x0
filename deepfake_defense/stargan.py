import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
import torch
import torch.nn as nn


DROPBOX_URL = "https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1"
CELEBA_DOMAIN_ORDER = ("Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young")
RAFD_DEFAULT_EXPRESSIONS = (
    "Angry",
    "Contemptuous",
    "Disgusted",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised",
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x):
        return x + self.main(x)


class StarGANGenerator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        curr = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(curr * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr))
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr, curr // 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(curr // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr //= 2
        layers += [nn.Conv2d(curr, 3, 7, 1, 3, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1).expand(c.size(0), c.size(1), x.size(2), x.size(3))
        return self.main(torch.cat([x, c], dim=1))


def ensure_checkpoint(ckpt_dir: str = "stargan_celeba_128/models") -> Path:
    ckpt_dir_path = Path(ckpt_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    zip_path = ckpt_dir_path / "celeba-128x128-5attrs.zip"
    ckpt_path = ckpt_dir_path / "200000-G.ckpt"

    if not ckpt_path.exists():
        print("Downloading StarGAN CelebA-128 weights from Dropbox...")
        r = requests.get(DROPBOX_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ckpt_dir_path)
        zip_path.unlink(missing_ok=True)
    return ckpt_path


def _load_state_dict(ckpt_path: Path):
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    return state_dict


def _infer_checkpoint_c_dim(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state_dict.get("main.0.weight")
    if w is None or w.ndim != 4:
        return None
    in_channels = int(w.shape[1])  # 3 + c_dim
    return in_channels - 3


def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Optional[object]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = _load_state_dict(ckpt_path)
    inferred_c_dim = _infer_checkpoint_c_dim(state_dict)
    inferred_domain: Optional[str]
    if inferred_c_dim == 5:
        inferred_domain = "celeba"
    elif inferred_c_dim == 8:
        inferred_domain = "rafd"
    elif inferred_c_dim is None:
        inferred_domain = None
    else:
        inferred_domain = f"unknown(c_dim={inferred_c_dim})"

    return {
        "path": str(ckpt_path),
        "c_dim": inferred_c_dim,
        "domain": inferred_domain,
    }


def load_generator(
    device: torch.device,
    domain: str = "celeba",
    ckpt_dir: str = "stargan_celeba_128/models",
    checkpoint_path: Optional[str] = None,
) -> StarGANGenerator:
    domain = domain.lower()
    if domain == "celeba":
        ckpt_path = ensure_checkpoint(ckpt_dir)
        c_dim = 5
    elif domain == "rafd":
        if checkpoint_path is None:
            raise ValueError("RaFD mode requires --expression-checkpoint pointing to a trained StarGAN RaFD generator checkpoint.")
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise ValueError(f"Expression checkpoint not found: {checkpoint_path}")
        c_dim = 8
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    state_dict = _load_state_dict(ckpt_path)
    inferred_c_dim = _infer_checkpoint_c_dim(state_dict)
    if inferred_c_dim is not None and inferred_c_dim != c_dim:
        inferred_domain = "celeba" if inferred_c_dim == 5 else ("rafd" if inferred_c_dim == 8 else f"c_dim={inferred_c_dim}")
        raise ValueError(
            f"Checkpoint/domain mismatch: requested domain='{domain}' (c_dim={c_dim}) "
            f"but checkpoint appears to be {inferred_domain}. "
            f"Use --domain celeba with this checkpoint, or provide a true RaFD checkpoint for --domain rafd."
        )

    generator = StarGANGenerator(conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)
    cleaned = {
        k: v
        for k, v in state_dict.items()
        if not (k.endswith(".running_mean") or k.endswith(".running_var"))
    }
    generator.load_state_dict(cleaned, strict=False)
    generator.eval()
    return generator


def deepfake_attack(generator: StarGANGenerator, image_tensor: torch.Tensor, target_attribute: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        img_normalized = image_tensor * 2 - 1
        fake_img = generator(img_normalized, target_attribute)
        fake_img = (fake_img + 1) / 2
        return torch.clamp(fake_img, 0, 1)


def _make_attr_tensor(vec: Iterable[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([list(vec)], dtype=torch.float32).to(device)


def celeba_attribute_catalog() -> List[str]:
    hair_colors = ("Black", "Blond", "Brown")
    genders = ("Female", "Male")
    ages = ("Old", "Young")
    names = []
    for hair in hair_colors:
        for gender in genders:
            for age in ages:
                names.append(f"{hair} Hair + {gender} + {age}")
    return names


def rafd_attribute_catalog(expressions: Optional[List[str]] = None) -> List[str]:
    return list(expressions or RAFD_DEFAULT_EXPRESSIONS)


def attribute_catalog(domain: str = "celeba", expressions: Optional[List[str]] = None) -> List[str]:
    domain = domain.lower()
    if domain == "celeba":
        return celeba_attribute_catalog()
    if domain == "rafd":
        return rafd_attribute_catalog(expressions=expressions)
    raise ValueError(f"Unsupported domain: {domain}")


def default_attributes(
    device: torch.device,
    domain: str = "celeba",
    expressions: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    domain = domain.lower()
    if domain == "rafd":
        expr_names = list(expressions or RAFD_DEFAULT_EXPRESSIONS)
        attrs: Dict[str, torch.Tensor] = {}
        for i, expr in enumerate(expr_names):
            one_hot = [0] * len(expr_names)
            one_hot[i] = 1
            attrs[expr] = _make_attr_tensor(one_hot, device)
        return attrs

    if domain != "celeba":
        raise ValueError(f"Unsupported domain: {domain}")

    # This checkpoint is trained for 5 CelebA domains in this order:
    # [Black_Hair, Blond_Hair, Brown_Hair, Male, Young].
    # We generate a full valid grid of 3 hair colors x 2 genders x 2 ages = 12 attack vectors.
    hair_vectors = {
        "Black": (1, 0, 0),
        "Blond": (0, 1, 0),
        "Brown": (0, 0, 1),
    }
    gender_values = {"Female": 0, "Male": 1}
    age_values = {"Old": 0, "Young": 1}

    attrs: Dict[str, torch.Tensor] = {}
    for hair_name, hair_vec in hair_vectors.items():
        for gender_name, gender_val in gender_values.items():
            for age_name, age_val in age_values.items():
                name = f"{hair_name} Hair + {gender_name} + {age_name}"
                vec = (*hair_vec, gender_val, age_val)
                attrs[name] = _make_attr_tensor(vec, device)

    # Backward-compatible aliases used in existing notebook/scripts.
    attrs["Blonde Hair"] = attrs["Blond Hair + Female + Old"]
    attrs["Male"] = attrs["Brown Hair + Male + Young"]
    attrs["Old Age"] = attrs["Brown Hair + Female + Old"]
    return attrs


def resolve_attribute_selection(
    selection: str,
    attributes: Dict[str, torch.Tensor],
    domain: str = "celeba",
    expressions: Optional[List[str]] = None,
) -> List[str]:
    if not selection:
        raise ValueError("Attribute selection is empty.")

    by_lower = {k.lower(): k for k in attributes.keys()}
    raw_parts = [p.strip() for p in selection.split(",") if p.strip()]
    if len(raw_parts) == 1 and raw_parts[0].lower() == "all":
        return sorted(attribute_catalog(domain=domain, expressions=expressions))

    selected: List[str] = []
    for part in raw_parts:
        key = by_lower.get(part.lower())
        if key is None:
            valid = ", ".join(sorted(attribute_catalog(domain=domain, expressions=expressions)))
            raise ValueError(f"Unknown attribute '{part}'. Valid examples: {valid}")
        if key not in selected:
            selected.append(key)
    return selected
