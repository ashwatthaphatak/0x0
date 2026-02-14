import zipfile
from pathlib import Path

import requests
import torch
import torch.nn as nn


DROPBOX_URL = "https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1"


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


def load_generator(device: torch.device, ckpt_dir: str = "stargan_celeba_128/models") -> StarGANGenerator:
    ckpt_path = ensure_checkpoint(ckpt_dir)
    generator = StarGANGenerator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
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


def default_attributes(device: torch.device):
    return {
        "Blonde Hair": torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32).to(device),
        "Old Age": torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32).to(device),
        "Male": torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.float32).to(device),
    }
