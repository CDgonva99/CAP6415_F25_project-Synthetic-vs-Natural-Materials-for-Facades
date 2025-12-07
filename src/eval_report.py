"""
Evaluation report utility:
- Loads eval.checkpoint
- Evaluates on data.test_dir
- Saves metrics.json, confusion_matrix.(csv|png)
Comments in English.
"""
import argparse, json
from pathlib import Path
import numpy as np
import yaml
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import build_model, _load_state_dict_safely
from datasets import build_test_loader

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def plot_cm(cm, classes, out_png):
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/eval.yaml')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    classes = cfg['data']['classes']
    ckpt = cfg['eval']['checkpoint']
    exp  = cfg.get('experiment','exp')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_test, _ = build_test_loader(cfg['data']['test_dir'], img_size=224, batch_size=64, num_workers=2)
    model = build_model(n_classes=len(classes), pretrained=False).to(device).eval()
    _load_state_dict_safely(model, ckpt, device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in dl_test:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(1).cpu().numpy()
            y_pred.append(pred); y_true.append(y.numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    out_dir = Path(f'results/{exp}/eval'); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'metrics.json','w') as f:
        json.dump({'accuracy':acc, 'macro_f1':f1m, 'checkpoint':ckpt}, f, indent=2)
    np.savetxt(out_dir/'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    plot_cm(cm, classes, out_dir/'confusion_matrix.png')
    print(f'[EVAL-REPORT] acc={acc:.4f} macro_f1={f1m:.4f} | saved -> {out_dir}')

if __name__ == '__main__':
    main()
