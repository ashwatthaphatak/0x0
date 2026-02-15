import torch
import torch.nn as nn

from .perturbation import PerturbationEnhancement
from .texture import TextureExtractor

try:
    import lpips
except Exception:
    lpips = None


class DeepfakeDefenseFramework(nn.Module):
    def __init__(self, device: torch.device, epsilon: float = 0.05, enable_lpips: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.texture_extractor = TextureExtractor()
        self.perturbation_gen = PerturbationEnhancement()
        self.lpips_loss = None
        self.lpips_device = None

        # LPIPS is optional and not required by the demo defense path.
        if enable_lpips and lpips is not None:
            try:
                self.lpips_loss = lpips.LPIPS(net="alex").to(device)
                self.lpips_device = device
            except Exception:
                try:
                    self.lpips_loss = lpips.LPIPS(net="alex").to("cpu")
                    self.lpips_device = torch.device("cpu")
                except Exception:
                    self.lpips_loss = None
                    self.lpips_device = None

    def generate_perturbation(self, img_tensor: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        texture_features = self.texture_extractor(img_tensor)
        perturbation = self.perturbation_gen(texture_features, attention_map)
        return self.epsilon * perturbation

    def vaccinate_image(self, img_tensor: torch.Tensor, attention_map: torch.Tensor):
        with torch.no_grad():
            perturbation = self.generate_perturbation(img_tensor, attention_map)
            vaccinated = img_tensor + perturbation
            vaccinated = torch.clamp(vaccinated, 0, 1)
        return vaccinated, perturbation
