import os, json, argparse, time, random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import build_loaders, build_test_loader

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # good for fixed image size

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_model(n_classes: int, pretrained: bool=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, n_classes)
    return m

def train_one_epoch(model, dl, crit, opt, device, scaler: GradScaler):
    model.train()
    running = 0.0
    for x,y in dl:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=device.type=='cuda'):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * x.size(0)
    return running / len(dl.dataset)

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    for x,y in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.append(pred); y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1m, y_true, y_pred

def plot_confusion(cm, classes, out_png):
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix'); plt.xlabel('Pred'); plt.ylabel('True')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--eval', action='store_true', help='only evaluate (no training)')
    args = p.parse_args()

    cfg = load_cfg(args.config)
    exp = cfg.get('experiment','exp')
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_cfg = cfg['data']
    classes = data_cfg['classes']
    img_size = cfg.get('train',{}).get('img_size', 224)
    batch_size = cfg.get('train',{}).get('batch_size', 64)
    num_workers = cfg.get('train',{}).get('num_workers', 2)
    save_dir = Path(cfg.get('save',{}).get('dir', f'results/{exp}')); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path('checkpoints'); ckpt_dir.mkdir(exist_ok=True)

    # EVAL ONLY
    if args.eval:
        dl_test, _ = build_test_loader(data_cfg['test_dir'], img_size, batch_size, num_workers)
        model = build_model(n_classes=len(classes), pretrained=False).to(device)
        ckpt = Path(cfg['eval']['checkpoint'])
        model.load_state_dict(torch.load(ckpt, map_location=device))
        acc, f1m, y_true, y_pred = eval_epoch(model, dl_test, device)
        cm = confusion_matrix(y_true, y_pred)

        out_dir = save_dir / 'eval'; out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir/'metrics.json','w') as f:
            json.dump({'accuracy':acc, 'f1_macro':f1m}, f, indent=2)
        np.savetxt(out_dir/'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
        plot_confusion(cm, classes, out_dir/'confusion_matrix.png')
        print(f'[EVAL] acc={acc:.4f} f1_macro={f1m:.4f} | saved -> {out_dir}')
        return

    # TRAIN
    dl_train, dl_val, _ = build_loaders(
        data_cfg['train_dir'], data_cfg['val_dir'],
        img_size, batch_size, num_workers
    )
    model = build_model(n_classes=len(classes), pretrained=cfg.get('model',{}).get('pretrained', True)).to(device)
    crit = nn.CrossEntropyLoss()

    opt_name = cfg.get('optim',{}).get('name','adam').lower()
    if opt_name == 'sgd':
        opt = optim.SGD(model.parameters(), lr=cfg['optim'].get('lr',0.01),
                        momentum=0.9, weight_decay=cfg['optim'].get('weight_decay',1e-4))
    else:
        opt = optim.Adam(model.parameters(), lr=cfg['optim'].get('lr',1e-3),
                         weight_decay=cfg['optim'].get('weight_decay',1e-4))

    epochs   = cfg.get('train',{}).get('epochs', 8)
    patience = cfg.get('train',{}).get('early_stop_patience', 3)
    scaler = GradScaler(enabled=device.type=='cuda')

    best_f1, wait = -1.0, 0
    best_path = ckpt_dir / f'best_{exp}.ckpt'

    for ep in range(1, epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, dl_train, crit, opt, device, scaler)
        acc, f1m, y_true, y_pred = eval_epoch(model, dl_val, device)
        dt = time.time()-t0
        print(f'[EP {ep:02d}] loss={tr_loss:.4f} val_acc={acc:.4f} val_f1M={f1m:.4f} ({dt:.1f}s)')

        if f1m > best_f1:
            best_f1 = f1m; wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print('[EarlyStop] patience reached.')
                break

    # Final val metrics + CM
    model.load_state_dict(torch.load(best_path, map_location=device))
    acc, f1m, y_true, y_pred = eval_epoch(model, dl_val, device)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)

    with open(save_dir/'metrics.json','w') as f:
        json.dump({'val_accuracy':acc, 'val_f1_macro':f1m, 'per_class':rep}, f, indent=2)
    np.savetxt(save_dir/'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    plot_confusion(cm, classes, save_dir/'confusion_matrix.png')
    print(f'[DONE] best_f1M={best_f1:.4f} | metrics+CM -> {save_dir}')

    # Write checkpoint into eval.yaml for convenience
    eval_cfg_path = Path('configs/eval.yaml')
    if eval_cfg_path.exists():
        with open(eval_cfg_path, 'r') as f: ev = yaml.safe_load(f) or {}
        ev.setdefault('eval', {})['checkpoint'] = str(best_path)
        with open(eval_cfg_path, 'w') as f: yaml.safe_dump(ev, f)
        print(f'[INFO] Updated configs/eval.yaml with checkpoint: {best_path}')

if __name__ == '__main__':
    main()
