import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.unsqueeze(1)


class DualAttentionModule:
    def __init__(self, device: torch.device):
        self.device = device
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
        except Exception:
            self.resnet = models.resnet50(pretrained=True).to(device).eval()
        self.gradcam_resnet = GradCAM(model=self.resnet, target_layer=self.resnet.layer4[-1])

    def get_attention_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_normalized = normalize(img_tensor.squeeze(0))
        img_input = img_normalized.unsqueeze(0)
        img_resized = F.interpolate(img_input, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.enable_grad():
            cam_tensor = self.gradcam_resnet(img_resized)

        original_size = img_tensor.shape[2:]
        cam_resized = F.interpolate(cam_tensor, size=original_size, mode="bilinear", align_corners=False)
        return cam_resized
