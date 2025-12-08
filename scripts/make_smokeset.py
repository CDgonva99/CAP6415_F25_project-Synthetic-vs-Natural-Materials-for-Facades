"""
Procedural smoke dataset generator (tiny, for plumbing only).
Creates N train/val/test images per class using simple graphics cues.
"""
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

CLASSES = ['brick','glass','concrete','metal','vegetation']

def draw_brick(draw, W, H, rng):
    # horizontal bands + vertical seams
    for y in range(0, H, 24):
        color = (rng.randint(120,180), rng.randint(60,80), rng.randint(50,70))
        draw.rectangle([0,y,W,y+22], fill=color)
        for x in range(0, W, 48):
            draw.line([(x,y),(x,y+22)], fill=(50,30,25), width=2)

def draw_glass(draw, W, H, rng):
    base = (rng.randint(120,170), rng.randint(160,210), rng.randint(200,255))
    draw.rectangle([0,0,W,H], fill=base)
    # faint mullions
    for x in range(0, W, 64):
        draw.line([(x,0),(x,H)], fill=(220,230,235), width=2)
    for y in range(0, H, 64):
        draw.line([(0,y),(W,y)], fill=(220,230,235), width=2)

def draw_concrete(draw, W, H, rng):
    gray = rng.randint(140, 190)
    draw.rectangle([0,0,W,H], fill=(gray,gray,gray))
    # pores
    for _ in range(400):
        x = rng.randint(0,W-1); y=rng.randint(0,H-1)
        r = rng.randint(1,2)
        draw.ellipse([x-r,y-r,x+r,y+r], fill=(rng.randint(80,120),)*3)

def draw_metal(draw, W, H, rng):
    # simple brushed bands
    base = rng.randint(160,200)
    draw.rectangle([0,0,W,H], fill=(base,base,base))
    for y in range(0,H,8):
        c = base + rng.randint(-10,10)
        draw.rectangle([0,y,W,y+3], fill=(c,c,c))

def draw_vegetation(draw, W, H, rng):
    draw.rectangle([0,0,W,H], fill=(rng.randint(20,60), rng.randint(60,120), rng.randint(20,60)))
    # leaves as green blobs
    for _ in range(250):
        x = rng.randint(0,W-1); y=rng.randint(0,H-1)
        r = rng.randint(2,6)
        draw.ellipse([x-r,y-r,x+r,y+r], fill=(rng.randint(40,90), rng.randint(100,160), rng.randint(30,80)))

DRAWERS = {
    'brick': draw_brick,
    'glass': draw_glass,
    'concrete': draw_concrete,
    'metal': draw_metal,
    'vegetation': draw_vegetation
}

def gen_split(root:Path, split:str, per_class:int, W=256, H=256, seed=123):
    rng = random.Random(seed + hash(split) % 1000)
    for c in CLASSES:
        out_dir = root / split / c
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = Image.new('RGB', (W,H), (0,0,0))
            draw = ImageDraw.Draw(img)
            DRAWERS[c](draw, W, H, rng)
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.randint(0,1)))
            img.save(out_dir / f'{c}_{i:04d}.png')

def main():
    root = Path('data/smoke')
    gen_split(root, 'train', per_class=40)
    gen_split(root, 'val',   per_class=10)
    gen_split(root, 'test',  per_class=10)
    print(f'[SMOKESET] wrote -> {root}')

if __name__ == '__main__':
    main()

