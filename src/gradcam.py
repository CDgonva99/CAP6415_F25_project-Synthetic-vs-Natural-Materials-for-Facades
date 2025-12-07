# src/gradcam.py
# Grad-CAM for ResNet-18 on folder of images.
# Saves overlays to --out. Works with checkpoints trained in this project.

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms, models

# Reuse your project helpers if available
try:
    from train import build_model, load_cfg
    HAS_PROJECT = True
except Exception:
    HAS_PROJECT = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_preproc(img_size: int):
    """Standard ImageNet preprocessing used in training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GradCAM:
    """
    Minimal Grad-CAM helper:
    - Registers forward & backward hooks on a target conv layer
    - After forward+backward, computes CAM for a class index
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h_fwd = target_layer.register_forward_hook(self._fwd_hook)
        self.h_bwd = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple; take [0], which is gradient wrt layer output
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.h_fwd.remove()
        self.h_bwd.remove()

    def __call__(self, class_idx: int):
        """
        Build CAM from saved activations/gradients.
        activations: (B, C, H, W)
        gradients:   (B, C, H, W)
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM missing activations/gradients (did forward+backward run?)")

        acts = self.activations       # (B, C, H, W)
        grads = self.gradients        # (B, C, H, W)
        # Global-average-pool the gradients over HxW to get per-channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1)                # (B, H, W)
        cam = F.relu(cam)                                # keep positive contributions
        # Normalize to [0,1]
        b = cam.shape[0]
        for i in range(b):
            m, M = cam[i].min(), cam[i].max()
            if (M - m) > 1e-6:
                cam[i] = (cam[i] - m) / (M - m)
            else:
                cam[i].zero_()
        return cam


def overlay_cam(rgb_pil: Image.Image, cam_2d: np.ndarray, alpha: float = 0.35):
    """
    Overlay a [0..1] CAM onto an RGB PIL image, return a matplotlib Figure.
    """
    cam_color = plt.get_cmap("jet")(cam_2d)[:, :, :3]  # (H,W,3), float 0..1
    cam_uint8 = (cam_color * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_uint8).resize(rgb_pil.size, Image.BILINEAR)

    # Blend: alpha * CAM + (1-alpha) * image
    cam_arr = np.array(cam_img).astype(np.float32)
    rgb_arr = np.array(rgb_pil).astype(np.float32)
    mix = (alpha * cam_arr + (1 - alpha) * rgb_arr) / 255.0
    mix = np.clip(mix, 0, 1)

    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(mix)
    return fig


def load_model_and_layer(ckpt_path: Path, n_classes: int, device: torch.device):
    """
    Build ResNet-18 model and return (model, target_conv_layer).
    - If project helpers exist, use them; otherwise create a vanilla resnet18.
    - Target conv layer for Grad-CAM is last conv: model.layer4[-1].conv2
    """
    if HAS_PROJECT:
        model = build_model(n_classes=n_classes, pretrained=False).to(device)
        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    else:
        model = models.resnet18(weights=None)
        # replace the FC to match classes (assume training did the same)
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, n_classes)
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state, strict=True)
        model = model.to(device)

    model.eval()
    # Choose the last conv layer for Grad-CAM
    target_layer = model.layer4[-1].conv2
    return model, target_layer


def iter_images(dir_path: Path, limit: int):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    count = 0
    for p in dir_path.rglob("*"):
        if p.suffix.lower() in exts:
            yield p
            count += 1
            if limit and count >= limit:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml", help="YAML with data.classes and eval.checkpoint")
    ap.add_argument("--images_dir", required=True, help="Folder with images to visualize (e.g., .data\\nat\\test)")
    ap.add_argument("--out", required=True, help="Output folder for overlays")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    # Load config if available (for class list and checkpoint)
    if HAS_PROJECT:
        cfg = load_cfg(args.config)
        classes = cfg.get("data", {}).get("classes", ["brick", "glass", "concrete", "metal", "vegetation"])
        ckpt = Path(cfg.get("eval", {}).get("checkpoint", "checkpoints/best_nat.ckpt"))
    else:
        classes = ["brick", "glass", "concrete", "metal", "vegetation"]
        ckpt = Path("checkpoints/best_nat.ckpt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre = build_preproc(args.img_size)

    # Load model and set Grad-CAM on last conv
    model, target_layer = load_model_and_layer(ckpt, n_classes=len(classes), device=device)
    cam = GradCAM(model, target_layer)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs_dir = Path(args.images_dir)
    wrote = 0

    # IMPORTANT: do NOT wrap in torch.no_grad(); we need gradients for Grad-CAM.
    for img_path in iter_images(imgs_dir, args.limit):
        try:
            rgb = Image.open(img_path).convert("RGB")
            x = pre(rgb).unsqueeze(0).to(device)  # (1,C,H,W)

            # Forward pass WITH grads so hooks fill activations
            logits = model(x)                      # (1, num_classes)
            probs = F.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())

            # Backward on the predicted class score to get gradients
            model.zero_grad(set_to_none=True)
            score = logits[0, pred]
            score.backward()

            # Build CAM for the predicted class (index 'pred')
            cam_map = cam(pred)[0].cpu().numpy()   # (H', W') in [0,1]

            # Overlay and save
            fig = overlay_cam(rgb, cam_map, alpha=0.40)
            out_png = out_dir / f"{img_path.stem}_cam.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            wrote += 1

        except Exception as e:
            print(f"[GRADCAM] ERROR on {img_path}: {type(e).__name__}: {e}")

    cam.remove()
    print(f"[GRADCAM] wrote {wrote} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
