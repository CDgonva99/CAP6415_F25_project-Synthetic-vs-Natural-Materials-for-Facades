# src/train.py
# -----------------------------------------------------------
# CAP6415 F25 — Facades project
# Unified training/eval script with:
# - NAT/SYN training (single train/val dirs)
# - MIX training (natural + synthetic sampling ratio)
# - Final evaluation on a held-out NAT test set
# Windows-safe (no locally defined classes in workers)
# AMP uses torch.amp (modern API)
# -----------------------------------------------------------

import os, json, argparse, time, random
from pathlib import Path

import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------------
# Reproducibility and device
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Good for fixed image size; enables some cuDNN autotuning
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------
# Config helpers
# ------------------------------
def load_cfg(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def write_cfg(path: str, cfg: dict):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


# ------------------------------
# Model
# ------------------------------
def build_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 head-swapped classifier.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, n_classes)
    return m


def _load_state_dict_safely(model: nn.Module, ckpt_path: str, device: torch.device):
    """
    Loads a state dict that might be raw dict or wrapped into {'state_dict': ...}.
    """
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


# ------------------------------
# Transforms & Data loaders
# ------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def build_loaders(
    train_dir: str,
    val_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    """
    NAT or SYN training: single train/val folders.
    """
    t_train = build_transforms(img_size, train=True)
    t_eval  = build_transforms(img_size, train=False)

    ds_train = datasets.ImageFolder(train_dir, transform=t_train)
    ds_val   = datasets.ImageFolder(val_dir, transform=t_eval)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    classes = ds_train.classes
    return dl_train, dl_val, classes


def build_test_loader(
    test_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    """
    Test loader. Returns (dl_test, classes)
    """
    t_eval = build_transforms(img_size, train=False)
    ds_test = datasets.ImageFolder(test_dir, transform=t_eval)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    classes = ds_test.classes
    return dl_test, classes


class MixedDataset(Dataset):
    """
    MIX dataset that samples from natural and synthetic datasets.
    With probability `ratio` we draw from NAT; otherwise from SYN.
    Length is max(len(NAT), len(SYN)); indices wrap via modulo.

    Defined at module scope to be picklable on Windows (spawn).
    """
    def __init__(self, ds_nat, ds_syn, ratio: float = 0.5, seed: int = 1234):
        self.ds_nat = ds_nat
        self.ds_syn = ds_syn
        self.ratio  = float(ratio)
        self.len    = max(len(ds_nat), len(ds_syn))
        self._rng   = random.Random(seed)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        use_nat = (self._rng.random() < self.ratio)
        ds = self.ds_nat if use_nat else self.ds_syn
        return ds[idx % len(ds)]


def build_mix_loaders(
    nat_dir: str,
    syn_dir: str,
    val_dir: str,
    ratio: float,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    t_train = build_transforms(img_size, train=True)
    t_eval  = build_transforms(img_size, train=False)

    ds_nat = datasets.ImageFolder(nat_dir, transform=t_train)
    ds_syn = datasets.ImageFolder(syn_dir, transform=t_train)

    mix_ds   = MixedDataset(ds_nat, ds_syn, ratio=ratio)
    dl_train = DataLoader(mix_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

    ds_val = datasets.ImageFolder(val_dir, transform=t_eval)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Classes must match across NAT/SYN by folder name.
    classes = ds_nat.classes
    return dl_train, dl_val, classes


# ------------------------------
# Train / Eval loops
# ------------------------------
def train_one_epoch(model, dl, crit, opt, device, scaler: amp.GradScaler):
    model.train()
    running = 0.0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * x.size(0)
    return running / len(dl.dataset)


@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1m, y_true, y_pred


def plot_confusion(cm, classes, out_png: Path):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to YAML config')
    p.add_argument('--eval', action='store_true', help='Only evaluate on test_dir (no training)')
    args = p.parse_args()

    cfg = load_cfg(args.config)
    set_seed(42)
    device = get_device()

    exp        = cfg.get('experiment', 'exp')
    data_cfg   = cfg.get('data', {})
    train_cfg  = cfg.get('train', {})
    optim_cfg  = cfg.get('optim', {})
    model_cfg  = cfg.get('model', {})
    save_root  = Path(cfg.get('save', {}).get('dir', f'results/{exp}')); save_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir   = Path('checkpoints'); ckpt_dir.mkdir(exist_ok=True)

    img_size   = int(train_cfg.get('img_size', 224))
    batch_size = int(train_cfg.get('batch_size', 64))
    num_workers= int(train_cfg.get('num_workers', 0))  # 0 is safest on Windows; increase to 2-4 if you like
    epochs     = int(train_cfg.get('epochs', 10))
    patience   = int(train_cfg.get('early_stop_patience', 3))

    # ----- EVAL ONLY -----
    if args.eval:
        assert 'test_dir' in data_cfg, "data.test_dir is required for --eval"
        dl_test, classes = build_test_loader(
            data_cfg['test_dir'], img_size, batch_size, num_workers
        )
        model = build_model(n_classes=len(classes), pretrained=False).to(device)

        ckpt = cfg.get('eval', {}).get('checkpoint', None)
        assert ckpt is not None, "eval.checkpoint must be set in the YAML for --eval"
        _load_state_dict_safely(model, ckpt, device)

        acc, f1m, y_true, y_pred = eval_epoch(model, dl_test, device)
        cm = confusion_matrix(y_true, y_pred)

        out_dir = save_root / 'eval'
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({'accuracy': acc, 'f1_macro': f1m}, f, indent=2)
        np.savetxt(out_dir / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
        plot_confusion(cm, classes, out_dir / 'confusion_matrix.png')
        print(f'[EVAL] acc={acc:.4f} f1_macro={f1m:.4f} | saved -> {out_dir}')
        return

    # ----- TRAIN -----
    # Choose training mode by presence of keys in data_cfg:
    # NAT/SYN mode:           data.train_dir + data.val_dir
    # MIX mode:               data.train_dir_nat + data.train_dir_syn + data.val_dir (+ mix_ratio)
    is_mix = all(k in data_cfg for k in ['train_dir_nat', 'train_dir_syn', 'val_dir'])
    is_single = all(k in data_cfg for k in ['train_dir', 'val_dir'])

    if is_mix:
        dl_train, dl_val, classes = build_mix_loaders(
            data_cfg['train_dir_nat'],
            data_cfg['train_dir_syn'],
            data_cfg['val_dir'],
            float(data_cfg.get('mix_ratio', 0.5)),
            img_size, batch_size, num_workers
        )
    elif is_single:
        dl_train, dl_val, classes = build_loaders(
            data_cfg['train_dir'],
            data_cfg['val_dir'],
            img_size, batch_size, num_workers
        )
    else:
        raise KeyError("Bad 'data' section. Provide either "
                       "(train_dir, val_dir) or (train_dir_nat, train_dir_syn, val_dir).")

    model = build_model(n_classes=len(classes), pretrained=bool(model_cfg.get('pretrained', True))).to(device)
    crit  = nn.CrossEntropyLoss()

    opt_name = str(optim_cfg.get('name', 'adam')).lower()
    lr       = float(optim_cfg.get('lr', 1e-3))
    wd       = float(optim_cfg.get('weight_decay', 1e-4))

    if opt_name == 'sgd':
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scaler = amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_f1, wait = -1.0, 0
    best_path = Path('checkpoints') / f'best_{exp}.ckpt'

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, dl_train, crit, opt, device, scaler)
        acc, f1m, y_true, y_pred = eval_epoch(model, dl_val, device)
        dt = time.time() - t0
        print(f'[EP {ep:02d}] loss={tr_loss:.4f} val_acc={acc:.4f} val_f1M={f1m:.4f} ({dt:.1f}s)')

        if f1m > best_f1:
            best_f1 = f1m
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print('[EarlyStop] patience reached.')
                break

    # Load best and write validation artifacts
    _load_state_dict_safely(model, str(best_path), device)
    acc, f1m, y_true, y_pred = eval_epoch(model, dl_val, device)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=classes,
                                zero_division=0, output_dict=True)

    with open(save_root / 'metrics.json', 'w') as f:
        json.dump({'val_accuracy': acc, 'val_f1_macro': f1m, 'per_class': rep}, f, indent=2)
    np.savetxt(save_root / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    plot_confusion(cm, classes, save_root / 'confusion_matrix.png')
    print(f'[DONE] best_f1M={best_f1:.4f} | metrics+CM -> {save_root}')

    # Convenience: update configs/eval.yaml with the best checkpoint
    eval_cfg_path = Path('configs/eval.yaml')
    if eval_cfg_path.exists():
        ev = load_cfg(str(eval_cfg_path))
        if 'eval' not in ev:
            ev['eval'] = {}
        ev['eval']['checkpoint'] = str(best_path)
        write_cfg(str(eval_cfg_path), ev)
        print(f'[INFO] Updated configs/eval.yaml with checkpoint: {best_path}')


# Windows-safe entrypoint for DataLoader workers
if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()

