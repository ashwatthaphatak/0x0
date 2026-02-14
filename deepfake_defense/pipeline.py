from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch

from .attention import DualAttentionModule
from .framework import DeepfakeDefenseFramework
from .stargan import deepfake_attack, default_attributes, load_generator
from .utils import calculate_metrics, ensure_dirs, get_image_transform, load_image, show_cam_on_image, tensor_to_numpy


class DefensePipeline:
    def __init__(self, device: torch.device, epsilon: float = 0.05):
        self.device = device
        self.transform = get_image_transform((256, 256))

        self.attention_module = DualAttentionModule(device)
        self.defense_framework = DeepfakeDefenseFramework(device=device, epsilon=epsilon).to(device)
        self.stargan_generator = load_generator(device)
        self.attributes = default_attributes(device)

    def _resolve_input_image(self, image_path: Optional[str]) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            return load_image(image_path, self.transform, self.device)

        sample_candidates = [
            "sample_images/sample_0.jpg",
            "sample_images/sample_1.jpg",
            "sample_images/sample_2.jpg",
        ]
        for path in sample_candidates:
            if Path(path).exists():
                return load_image(path, self.transform, self.device)

        return torch.rand(1, 3, 256, 256).to(self.device)

    def run_demo(self, image_path: Optional[str], target_attr: str, results_dir: str = "results") -> Dict[str, float]:
        ensure_dirs(results_dir)

        original_image = self._resolve_input_image(image_path)
        target_attr_tensor = self.attributes[target_attr]

        attention_map = self.attention_module.get_attention_map(original_image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(tensor_to_numpy(original_image))
        axes[0].set_title("Original", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        attention_np = attention_map[0, 0].detach().cpu().numpy()
        axes[1].imshow(attention_np, cmap="jet")
        axes[1].set_title("Attention Heatmap", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        img_np = tensor_to_numpy(original_image)
        overlay = show_cam_on_image(img_np, attention_np)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{results_dir}/step1_attention.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        vaccinated_image, perturbation = self.defense_framework.vaccinate_image(original_image, attention_map)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(tensor_to_numpy(original_image))
        axes[0].set_title("Original", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        pert_amplified = (perturbation * 10).clamp(0, 1)
        axes[1].imshow(tensor_to_numpy(pert_amplified))
        axes[1].set_title("Perturbation (10x)", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(tensor_to_numpy(vaccinated_image))
        axes[2].set_title("Vaccinated", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        diff = torch.abs(vaccinated_image - original_image) * 20
        axes[3].imshow(tensor_to_numpy(diff))
        axes[3].set_title("Difference (20x)", fontsize=14, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(f"{results_dir}/step2_perturbation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        clean_deepfaked = deepfake_attack(self.stargan_generator, original_image, target_attr_tensor)
        vaccinated_deepfaked = deepfake_attack(self.stargan_generator, vaccinated_image, target_attr_tensor)
        defense_metrics = calculate_metrics(clean_deepfaked, vaccinated_deepfaked)

        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(tensor_to_numpy(original_image))
        ax1.set_title("1. Original", fontsize=16, fontweight="bold", color="blue")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, "→\nStarGAN\n(No Protection)", ha="center", va="center", fontsize=14, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(tensor_to_numpy(clean_deepfaked))
        ax3.set_title(f"2. Deepfaked\n({target_attr})", fontsize=16, fontweight="bold", color="red")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[0, 3])
        clean_diff = torch.abs(clean_deepfaked - original_image)
        ax4.imshow(tensor_to_numpy(clean_diff))
        ax4.set_title("Difference", fontsize=16, fontweight="bold")
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[0, 4])
        clean_metrics = calculate_metrics(original_image, clean_deepfaked)
        ax5.text(
            0.1,
            0.5,
            f"Clean Attack:\n\nPSNR: {clean_metrics['PSNR']:.2f} dB\nSSIM: {clean_metrics['SSIM']:.4f}\nL2: {clean_metrics['L2']:.4f}",
            fontsize=12,
            family="monospace",
            va="center",
        )
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[1, 0])
        ax6.imshow(tensor_to_numpy(vaccinated_image))
        ax6.set_title("1. Vaccinated", fontsize=16, fontweight="bold", color="green")
        ax6.axis("off")

        ax7 = fig.add_subplot(gs[1, 1])
        ax7.text(0.5, 0.5, "→\nStarGAN\n(Protected)", ha="center", va="center", fontsize=14, fontweight="bold")
        ax7.axis("off")

        ax8 = fig.add_subplot(gs[1, 2])
        ax8.imshow(tensor_to_numpy(vaccinated_deepfaked))
        ax8.set_title("2. Attempt", fontsize=16, fontweight="bold", color="orange")
        ax8.axis("off")

        ax9 = fig.add_subplot(gs[1, 3])
        vac_diff = torch.abs(vaccinated_deepfaked - vaccinated_image)
        ax9.imshow(tensor_to_numpy(vac_diff))
        ax9.set_title("Difference", fontsize=16, fontweight="bold")
        ax9.axis("off")

        ax10 = fig.add_subplot(gs[1, 4])
        vac_metrics = calculate_metrics(vaccinated_image, vaccinated_deepfaked)
        ax10.text(
            0.1,
            0.5,
            f"Vaccinated Attack:\n\nPSNR: {vac_metrics['PSNR']:.2f} dB\nSSIM: {vac_metrics['SSIM']:.4f}\nL2: {vac_metrics['L2']:.4f}\nDefense L2: {defense_metrics['L2']:.4f}",
            fontsize=12,
            family="monospace",
            va="center",
        )
        ax10.axis("off")

        plt.savefig(f"{results_dir}/final_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return defense_metrics
