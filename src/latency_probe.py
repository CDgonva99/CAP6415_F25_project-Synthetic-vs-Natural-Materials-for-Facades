# src/latency_probe.py
# Purpose: measure per-image inference latency on GPU (batch=1).
# - Backward-compatible: still supports --img (single image and --trials)
# - New: --images_dir (iterate images), --warmup, --count, CSV export

import argparse, os, time, csv, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# We reuse helpers consistent with your repo (build_model, load_cfg)
# If not found, we fall back to a default ResNet18 head shape inferred from classes.
try:
    from train import build_model as _build_model, load_cfg as _load_cfg, _load_state_dict_safely
    HAS_PROJECT_API = True
except Exception:
    HAS_PROJECT_API = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_cfg(cfg_path: str):
    if HAS_PROJECT_API:
        return _load_cfg(cfg_path)
    import yaml
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f) or {}

def build_model(n_classes: int, pretrained: bool = False):
    if HAS_PROJECT_API:
        return _build_model(n_classes, pretrained=pretrained)
    # Minimal fallback: ResNet18 with replaced FC
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = torch.nn.Linear(in_f, n_classes)
    return m

def build_preproc(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if HAS_PROJECT_API:
        _load_state_dict_safely(model, ckpt_path, device)
    else:
        state = torch.load(ckpt_path, map_location=device)
        # Try flexible keys
        if isinstance(state, dict):
            if "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)

def run_single(model, preproc, device, img_path: Path, img_size: int):
    """Return elapsed milliseconds for a single forward pass (batch=1)."""
    img = Image.open(img_path).convert("RGB")
    x = preproc(img).unsqueeze(0).to(device, non_blocking=True)
    # Use inference_mode for best speed and correctness
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
    return (t1 - t0) * 1000.0

def summarize(times_ms):
    if not times_ms:
        return {}
    s = sorted(times_ms)
    n = len(s)
    def pct(p):
        i = int(p/100.0 * (n-1))
        return s[i]
    mean = sum(s)/n
    return {
        "n": n,
        "mean": mean,
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to eval.yaml (must contain data.classes and eval.checkpoint)")
    # Old mode (kept for compatibility):
    ap.add_argument("--img", help="Single image path (old mode). Use with --trials.")
    ap.add_argument("--trials", type=int, default=50, help="Number of trials for --img mode.")
    # New mode:
    ap.add_argument("--images_dir", help="Directory of images to iterate (supports jpg/png).")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup forwards ignored from stats (dir mode).")
    ap.add_argument("--count", type=int, default=200, help="Number of measured forwards (dir mode).")
    ap.add_argument("--csv", help="Optional CSV output of per-sample times (dir mode).")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    classes = cfg.get("data",{}).get("classes", ["brick","glass","concrete","metal","vegetation"])
    n_classes = len(classes)
    ckpt = cfg.get("eval",{}).get("checkpoint", None)
    if not ckpt:
        print("[ERROR] eval.checkpoint missing in config.", file=sys.stderr); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = build_model(n_classes=n_classes, pretrained=False).to(device).eval()
    load_checkpoint(model, ckpt, device)
    preproc = build_preproc(args.img_size)

    # --- Old single-image mode ------------------------------------------------
    if args.img and not args.images_dir:
        img_path = Path(args.img)
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}", file=sys.stderr); sys.exit(1)

        # Warmup a few times
        for _ in range(10):
            _ = run_single(model, preproc, device, img_path, args.img_size)

        times = [run_single(model, preproc, device, img_path, args.img_size) for _ in range(args.trials)]
        stats = summarize(times)
        dev = (torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
        print(f"[Latency] img={img_path.name} trials={stats.get('n',0)} | "
              f"mean={stats.get('mean',0):.2f} ms | p50={stats.get('p50',0):.2f} | "
              f"p90={stats.get('p90',0):.2f} | p95={stats.get('p95',0):.2f} | p99={stats.get('p99',0):.2f} "
              f"(Device: {dev})")
        return

    # --- Directory mode -------------------------------------------------------
    if not args.images_dir:
        print("[ERROR] Provide either --img or --images_dir.", file=sys.stderr); sys.exit(1)

    root = Path(args.images_dir)
    if not root.exists():
        print(f"[ERROR] Directory not found: {root}", file=sys.stderr); sys.exit(1)

    exts = {".jpg",".jpeg",".png",".bmp"}
    pool = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if not pool:
        print(f"[ERROR] No images found under {root}", file=sys.stderr); sys.exit(1)

    # Warmup on a small round-robin of images
    for i in range(max(1, args.warmup)):
        _ = run_single(model, preproc, device, pool[i % len(pool)], args.img_size)

    # Measured passes
    times = []
    for i in range(max(1, args.count)):
        t = run_single(model, preproc, device, pool[i % len(pool)], args.img_size)
        times.append(t)

    stats = summarize(times)
    dev = (torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
    print(f"[Latency] n={stats.get('n',0)} | mean={stats.get('mean',0):.2f} ms | "
          f"p50={stats.get('p50',0):.2f} | p90={stats.get('p90',0):.2f} | "
          f"p95={stats.get('p95',0):.2f} | p99={stats.get('p99',0):.2f} "
          f"(Device: {dev})")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_ms"])
            for t in times:
                w.writerow([f"{t:.4f}"])
        print(f"[Latency] Wrote CSV -> {args.csv}")

if __name__ == "__main__":
    main()

