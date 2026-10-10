"""H3 SwiGLU entry point, before NVFP4 pre-scaling and paired rotation."""

from comfy.ldm.minimax.model import MiniMaxH3Model
from ..methods import weak_method


def _mlp_forward(self, x):
    return self.fc2.forward_swiglu(self.fc1(x))


def install_nvfp4_mlp_fusions(model):
    root = model.model.diffusion_model
    if not isinstance(root, MiniMaxH3Model):
        return 0
    count = 0
    for name, module in root.named_modules():
        if not name.endswith(".mlp"):
            continue
        if getattr(module.fc2, "quant_format", None) != "nvfp4":
            continue
        model.add_object_patch(
            f"diffusion_model.{name}.forward", weak_method(_mlp_forward, module)
        )
        count += 1
    return count
