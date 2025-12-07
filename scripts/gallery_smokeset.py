"""
Simple contact-sheet gallery for a dataset folder (ImageFolder-like).
Writes PNG grids into results/figures/gallery/.
Comments in English.
"""
from pathlib import Path
from PIL import Image
import math

def make_grid(img_paths, cols=10, thumb=96):
    rows = math.ceil(len(img_paths)/cols)
    W,H = cols*thumb, rows*thumb
    canvas = Image.new('RGB',(W,H),(30,30,30))
    for i,p in enumerate(img_paths):
        try:
            im = Image.open(p).convert('RGB')
            im = im.resize((thumb,thumb))
            x = (i % cols) * thumb
            y = (i // cols) * thumb
            canvas.paste(im,(x,y))
        except Exception:
            pass
    return canvas

def build_gallery(root, split='train', out='results/figures/gallery'):
    root = Path(root)
    out  = Path(out); out.mkdir(parents=True, exist_ok=True)
    for cls_dir in sorted((root/split).iterdir()):
        if not cls_dir.is_dir(): continue
        imgs = list(cls_dir.glob('*.png'))[:200]
        if not imgs: continue
        grid = make_grid(imgs)
        grid.save(out / f'{split}_{cls_dir.name}.png')
    print(f'[GALLERY] wrote -> {out}')

if __name__ == '__main__':
    build_gallery('data/smoke', 'train')
