import sys, yaml
cfg_path, ckpt = sys.argv[1], sys.argv[2]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("eval", {})["checkpoint"] = ckpt
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"[OK] eval.checkpoint -> {ckpt}")
