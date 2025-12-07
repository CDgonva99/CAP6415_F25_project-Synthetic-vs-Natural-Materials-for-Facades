# src/inference_server.py
# -----------------------------------------------------------------------------
# OSC inference server for TouchDesigner integration.
# - Listens:  /infer_path  "<absolute_or_relative_image_path>"
# - Replies:  /cv        [class_id, max_conf, p1..pK]
#             /cv_err    "message"
#             /cv_depth  [depth_exr_path_or_empty, depth_png16_path, cloud_ply_path]
#
# Modes:
#   dummy -> deterministic fake probabilities (no model)
#   model -> loads ResNet18 head + checkpoint from configs/eval.yaml (or --checkpoint)
#
# Depth (optional in model mode):
#   MiDaS DPT_Hybrid (torch.hub), outputs a PNG16 depth and a PLY cloud.
#   EXR saved only if OpenCV build supports it (we fall back silently).
#
# Run (Windows suggested):
#   python -u -m src.inference_server --mode model --listen_port 8000 --td_port 9000 --config .\configs\eval.yaml
# -----------------------------------------------------------------------------

import argparse
import math
from pathlib import Path
from typing import Optional

from pythonosc import dispatcher, osc_server, udp_client
import numpy as np

# Classification deps (only needed in model mode)
try:
    import torch
    import torch.nn.functional as F
    import PIL.Image as PILImage
    from torchvision import transforms
    from src.train import build_model, load_cfg
    HAS_PROJECT_MODEL = True
except Exception:
    HAS_PROJECT_MODEL = False

# Depth deps (optional)
try:
    import cv2
    from plyfile import PlyData, PlyElement
    HAS_DEPTH_DEPS = True
except Exception:
    HAS_DEPTH_DEPS = False

CLASSES = ["brick", "glass", "concrete", "metal", "vegetation"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ------------------------- Utils: model & transforms ------------------------- #
def build_preproc(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_state_dict_safely(model: "torch.nn.Module", ckpt_path: str, device: "torch.device"):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[server] loaded checkpoint: {ckpt_path}", flush=True)
    if missing or unexpected:
        print(f"[server] load_state_dict warnings -> missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)


# ------------------------- Depth: MiDaS + helpers --------------------------- #
def _load_midas(device: "torch.device"):
    import torch as _torch
    midas = _torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device).eval()
    transforms_hub = _torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_tfm = transforms_hub.dpt_transform
    return midas, midas_tfm


def estimate_depth_midas(pil_img: "PILImage.Image", device: "torch.device",
                         midas_model, midas_tfm) -> np.ndarray:
    """
    Returns depth aligned to original HxW, float32 in [0,1] (1=near).
    We avoid any PIL arithmetic: convert to NumPy first. If the official
    MiDaS transform still fails, we fall back to a manual preprocessing.
    """
    import torch as _torch

    H, W = pil_img.height, pil_img.width

    # --- Always convert PIL -> NumPy (RGB) BEFORE transform to avoid PIL math ---
    img_np = np.array(pil_img, dtype=np.uint8)  # HxWx3 RGB
    try:
        # Some MiDaS transforms expect np.ndarray and will do img/255.0 internally.
        x = midas_tfm(img_np).to(device)  # (1,3,h,w)
    except Exception as te:
        # Fallback: manual preprocessing (normalize with ImageNet stats)
        # Resize keeping aspect by padding to square, or simple resize:
        target = 384  # typical size for DPT_Hybrid
        img_res = cv2.resize(img_np, (target, target), interpolation=cv2.INTER_CUBIC)
        img_res = img_res.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)[None, None, :]
        std  = np.array(IMAGENET_STD,  dtype=np.float32)[None, None, :]
        img_norm = (img_res - mean) / std
        x = np.transpose(img_norm, (2, 0, 1))[None, ...]  # 1x3xHxW
        x = _torch.from_numpy(x).to(device)

    with _torch.no_grad():
        pred = midas_model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        # Upsample to original size:
        pred = _torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze(1).squeeze(0)  # (H, W)

    depth = pred.detach().cpu().numpy().astype("float32")
    dmin, dmax = float(depth.min()), float(depth.max())
    depth_n = (depth - dmin) / (dmax - dmin + 1e-8)
    depth_n = 1.0 - depth_n  # bright = near
    return depth_n


def write_exr(path: Path, depth_float: np.ndarray) -> bool:
    try:
        ok = cv2.imwrite(str(path), depth_float)
        return bool(ok)
    except Exception:
        return False


def write_png16(path: Path, depth_float: np.ndarray) -> None:
    d = (np.clip(depth_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), d)


def depth_to_pointcloud(depth_float: np.ndarray, rgb_np: np.ndarray,
                        fx: float = None, fy: float = None,
                        cx: float = None, cy: float = None,
                        z_scale: float = 3.0, step: int = 4) -> np.ndarray:
    """
    Back-project normalized depth to 3D points (NumPy only).
    Returns structured array with x,y,z,red,green,blue.
    """
    assert isinstance(depth_float, np.ndarray) and depth_float.ndim == 2
    assert isinstance(rgb_np, np.ndarray) and rgb_np.ndim == 3 and rgb_np.shape[2] == 3

    H, W = depth_float.shape
    if fx is None or fy is None or cx is None or cy is None:
        fov_deg = 60.0
        f = (W / (2.0 * math.tan(math.radians(fov_deg) / 2.0)))
        fx = fy = float(f)
        cx, cy = W / 2.0, H / 2.0

    pts = []
    for v in range(0, H, step):
        for u in range(0, W, step):
            z = float(depth_float[v, u]) * z_scale
            if z <= 0.0:
                continue
            x = (float(u) - cx) * z / fx
            y = (float(v) - cy) * z / fy
            r, g, b = rgb_np[v, u].tolist()
            pts.append((x, -y, -z, r, g, b))

    return np.array(
        pts,
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )


def save_ply(path: Path, structured_points: np.ndarray) -> None:
    el = PlyElement.describe(structured_points, 'vertex')
    PlyData([el], text=False).write(str(path))


# ------------------------------ Server class -------------------------------- #
class InferenceServer:
    def __init__(self,
                 td_ip: str,
                 td_port: int,
                 mode: str,
                 cfg_path: Optional[str],
                 ckpt: Optional[str],
                 img_size: int,
                 device_str: Optional[str] = None):

        self.client = udp_client.SimpleUDPClient(td_ip, td_port)
        self.mode = mode
        self.classes = CLASSES[:]
        self.img_size = int(img_size)

        self.device = None
        self.model  = None
        self.pre    = None

        if self.mode == "model":
            if not HAS_PROJECT_MODEL:
                raise RuntimeError("Model mode requires torch/torchvision/PIL and src.train.build_model/load_cfg.")

            if device_str:
                self.device = torch.device(device_str)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            cfg = load_cfg(cfg_path) if cfg_path else {}
            try:
                yaml_classes = cfg.get("data", {}).get("classes", None)
                if yaml_classes:
                    self.classes = list(yaml_classes)
            except Exception:
                pass

            self.img_size = int(cfg.get("train", {}).get("img_size", self.img_size))
            self.pre = build_preproc(self.img_size)

            self.model = build_model(n_classes=len(self.classes), pretrained=False).to(self.device).eval()

            if ckpt is None:
                ckpt = cfg.get("eval", {}).get("checkpoint", None)
            if ckpt is None:
                raise ValueError("No checkpoint provided. Use --checkpoint or set eval.checkpoint in YAML.")

            load_state_dict_safely(self.model, ckpt, self.device)

        # Depth:
        self.depth_enabled = False
        if self.mode == "model" and HAS_DEPTH_DEPS:
            try:
                self.midas_model, self.midas_tfm = _load_midas(self.device)
                self.depth_enabled = True
                print("[server] MiDaS depth enabled.", flush=True)
            except Exception as e:
                self.depth_enabled = False
                print(f"[server] depth disabled (MiDaS load failed): {e}", flush=True)
        elif self.mode == "model":
            print("[server] depth disabled (missing OpenCV/plyfile).", flush=True)

    # ------------------------- OSC helpers ------------------------- #
    def _send_cv(self, probs: np.ndarray):
        class_id = int(probs.argmax())
        max_conf = float(probs[class_id])
        payload = [class_id, max_conf] + [float(x) for x in probs.tolist()]
        self.client.send_message("/cv", payload)

    def _send_depth_paths(self, depth_exr: str, depth_png16: str, ply_path: str):
        self.client.send_message("/cv_depth", [depth_exr, depth_png16, ply_path])

    # ------------------------- Inference paths --------------------- #
    def infer_dummy(self, img_path: str):
        base = (abs(hash(Path(img_path).name)) % len(self.classes))
        probs = np.ones(len(self.classes), dtype=np.float32) * 0.05
        probs[base] = 0.80
        probs /= probs.sum()
        self._send_cv(probs)

    def infer_model(self, img_path: str):
        pil = PILImage.open(img_path).convert("RGB")
        x = self.pre(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0].astype("float32")
        self._send_cv(probs)

    # ------------------------- Depth & cloud ----------------------- #
    def build_depth_and_cloud(self, img_path: str, out_dir: Path):
        """
        Writes: PNG16 (always), EXR (if supported), PLY cloud.
        Returns: (depth_exr_or_empty, depth_png16, ply_path)
        """
        print("[server][depth] start", flush=True)

        # Load RGB strictly as NumPy (no arithmetic with PIL objects)
        pil = PILImage.open(img_path).convert("RGB")
        rgb_np = np.array(pil, dtype=np.uint8)  # HxWx3
        print(f"[server][depth] rgb_np: {rgb_np.shape} {rgb_np.dtype}", flush=True)

        # MiDaS inference -> float32 HxW in [0,1]
        depth_f = estimate_depth_midas(pil, self.device, self.midas_model, self.midas_tfm)
        if not isinstance(depth_f, np.ndarray):
            raise TypeError("depth_f is not a NumPy array")
        print(f"[server][depth] depth_f: {depth_f.shape} {depth_f.dtype} [{depth_f.min():.4f},{depth_f.max():.4f}]",
              flush=True)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem

        depth_exr = out_dir / f"{stem}_depth.exr"
        depth_png = out_dir / f"{stem}_depth16.png"
        ply_path  = out_dir / f"{stem}_cloud.ply"

        # Write EXR (optional) + PNG16 (always)
        exr_ok = write_exr(depth_exr, depth_f)
        print(f"[server][depth] write_exr ok={exr_ok}", flush=True)
        write_png16(depth_png, depth_f)
        print(f"[server][depth] wrote PNG16 -> {depth_png}", flush=True)

        # Point cloud
        pts = depth_to_pointcloud(depth_f, rgb_np, z_scale=3.0, step=3)
        save_ply(ply_path, pts)
        print(f"[server][depth] wrote PLY -> {ply_path} (N={len(pts)})", flush=True)

        return (str(depth_exr) if exr_ok else ""), str(depth_png), str(ply_path)

    # ------------------------- OSC callback ------------------------ #
    def on_infer_path(self, addr, _args, img_path: str):
        print(f"[server] received /infer_path: {img_path}", flush=True)
        p = Path(img_path)
        if not p.exists():
            self.client.send_message("/cv_err", f"not_found:{str(p)}")
            self.client.send_message("/cv", [-1, 0.0] + [0.0] * len(self.classes))
            print(f"[server] file not found: {p}", flush=True)
            return
        try:
            # Classification
            if self.mode == "model":
                self.infer_model(str(p))
            else:
                self.infer_dummy(str(p))
            print(f"[server] sent /cv for {p.name}", flush=True)

            # Depth (non-fatal)
            if self.mode == "model" and self.depth_enabled:
                try:
                    out_depth_dir = p.parent / "depth"
                    depth_exr, depth_png, ply_path = self.build_depth_and_cloud(str(p), out_depth_dir)
                    self._send_depth_paths(depth_exr, depth_png, ply_path)
                except Exception as de:
                    self.client.send_message("/cv_err", f"depth_fail:{type(de).__name__}:{str(de)}")
                    print(f"[server] depth generation failed: {type(de).__name__}: {de}", flush=True)

        except Exception as e:
            self.client.send_message("/cv_err", f"exception:{type(e).__name__}:{str(e)}")
            self.client.send_message("/cv", [-2, 0.0] + [0.0] * len(self.classes))
            print(f"[server] EXCEPTION: {type(e).__name__}: {e}", flush=True)


# --------------------------------- main ------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="OSC inference server for TouchDesigner")
    ap.add_argument("--mode", choices=["dummy", "model"], default="model")
    ap.add_argument("--listen_port", type=int, default=8000)
    ap.add_argument("--td_ip", default="127.0.0.1")
    ap.add_argument("--td_port", type=int, default=9000)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    server_obj = InferenceServer(
        td_ip=args.td_ip,
        td_port=args.td_port,
        mode=args.mode,
        cfg_path=args.config,
        ckpt=args.checkpoint,
        img_size=args.img_size,
        device_str=args.device,
    )

    disp = dispatcher.Dispatcher()
    disp.set_default_handler(lambda addr, *osc_args: None)
    disp.map("/infer_path", server_obj.on_infer_path, None)

    srv = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.listen_port), disp)
    print(f"[server] listening /infer_path on {args.listen_port} | replying /cv to {args.td_ip}:{args.td_port}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] bye.", flush=True)


if __name__ == "__main__":
    main()
