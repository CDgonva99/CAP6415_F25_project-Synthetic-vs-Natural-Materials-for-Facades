import argparse, time, torch
from pathlib import Path
from torchvision import models
from datasets import build_test_loader
from train import build_model, load_cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--n_warmup', type=int, default=10)
    p.add_argument('--n_iters', type=int, default=50)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = cfg['data']['classes']
    img_size = cfg.get('train',{}).get('img_size', 224)

    # Build a 1-sample DataLoader from test set (just for shape)
    dl_test, _ = build_test_loader(cfg['data']['test_dir'], img_size, batch_size=1, num_workers=0)
    x,_ = next(iter(dl_test))
    x = x.to(device)

    # Load model + checkpoint
    ckpt_path = Path(cfg['eval']['checkpoint'])
    model = build_model(n_classes=len(classes), pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Warmup
    for _ in range(args.n_warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed iters
    times = []
    for _ in range(args.n_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)*1000)

    print(f'Latency (b=1): mean={sum(times)/len(times):.2f} ms | min={min(times):.2f} ms | max={max(times):.2f} ms')

if __name__ == '__main__':
    main()
