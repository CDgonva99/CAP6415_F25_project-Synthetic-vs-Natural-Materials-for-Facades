"""
Simple synthetic facade generator (faster than Blender/Unreal).
Creates synthetic/train|val folders with randomized cues per class.
"""
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter, ImageOps

CLASSES = ['brick','glass','concrete','metal','vegetation']

def jitter_color(c, j=15, rng=None):
    rng = rng or random
    r,g,b = c
    return (max(0,min(255,r+rng.randint(-j,j))),
            max(0,min(255,g+rng.randint(-j,j))),
            max(0,min(255,b+rng.randint(-j,j))))

def synth_brick(img, rng):
    W,H = img.size; d = ImageDraw.Draw(img)
    for y in range(0,H, rng.randint(18,28)):
        color = jitter_color((170,90,70), j=30, rng=rng)
        d.rectangle([0,y,W,y+rng.randint(18,26)], fill=color)
        for x in range(0,W, rng.randint(38,54)):
            d.line([(x,y),(x,y+24)], fill=(60,35,30), width=rng.randint(1,3))

def synth_glass(img, rng):
    W,H = img.size; d = ImageDraw.Draw(img)
    base = jitter_color((140,190,240), j=25, rng=rng)
    d.rectangle([0,0,W,H], fill=base)
    for x in range(0, W, rng.randint(56,76)):
        d.line([(x,0),(x,H)], fill=(230,235,240), width=2)
    for y in range(0, H, rng.randint(56,76)):
        d.line([(0,y),(W,y)], fill=(230,235,240), width=2)
    # specular streak
    d.rectangle([rng.randint(0,W//2),0, rng.randint(W//2,W),H], fill=(255,255,255,20))

def synth_concrete(img, rng):
    W,H = img.size; d = ImageDraw.Draw(img)
    g = rng.randint(130,190)
    d.rectangle([0,0,W,H], fill=(g,g,g))
    for _ in range(700):
        x=rng.randint(0,W-1); y=rng.randint(0,H-1); r=rng.randint(1,3)
        c=rng.randint(90,120)
        d.ellipse([x-r,y-r,x+r,y+r], fill=(c,c,c))

def synth_metal(img, rng):
    W,H = img.size; d = ImageDraw.Draw(img)
    base=rng.randint(150,200)
    d.rectangle([0,0,W,H], fill=(base,base,base))
    for y in range(0,H, rng.randint(6,10)):
        c=base + rng.randint(-12,12)
        d.rectangle([0,y,W,y+2], fill=(max(0,min(255,c)),)*3)

def synth_vegetation(img, rng):
    W,H = img.size; d = ImageDraw.Draw(img)
    d.rectangle([0,0,W,H], fill=(rng.randint(20,60), rng.randint(70,130), rng.randint(20,70)))
    for _ in range(500):
        x=rng.randint(0,W-1); y=rng.randint(0,H-1); r=rng.randint(2,7)
        d.ellipse([x-r,y-r,x+r,y+r], fill=(rng.randint(40,80), rng.randint(110,170), rng.randint(35,90)))

GENS = {
    'brick': synth_brick, 'glass': synth_glass, 'concrete': synth_concrete,
    'metal': synth_metal, 'vegetation': synth_vegetation
}

def gen(root:Path, split:str, per_cls:int, size=256, seed=2025):
    rng = random.Random(seed + hash(split) % 1000)
    for c in CLASSES:
        out = root / split / c
        out.mkdir(parents=True, exist_ok=True)
        for i in range(per_cls):
            img = Image.new('RGB',(size,size),(0,0,0))
            GENS[c](img, rng)
            if rng.random()<0.3:
                img = ImageOps.autocontrast(img)
            if rng.random()<0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img.save(out / f'{c}_{i:04d}.png')

def main():
    root = Path('data/synthetic')
    gen(root, 'train', per_cls=300)
    gen(root, 'val',   per_cls=60)
    print(f'[SYNTH] wrote -> {root}')

if __name__ == '__main__':
    main()

