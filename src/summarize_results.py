# summarize_results.py — reads per-model metrics.json and prints CSV + Markdown
import json, pathlib, sys

pairs = [
    ("NAT", pathlib.Path("results/nat/eval/metrics.json")),
    ("SYN", pathlib.Path("results/syn/eval/metrics.json")),
    ("MIX", pathlib.Path("results/mix/eval/metrics.json")),
]

rows = []
for name, p in pairs:
    if not p.exists():
        rows.append((name, None, None))
        continue
    m = json.loads(p.read_text())
    acc = m.get("accuracy") or m.get("val_accuracy")
    f1  = m.get("macro_f1") or m.get("val_f1_macro")
    rows.append((name, acc, f1))

out_dir = pathlib.Path("results/summary"); out_dir.mkdir(parents=True, exist_ok=True)
csv = "model,accuracy,macro_f1\n" + "\n".join(f"{n},{a if a is not None else ''},{f if f is not None else ''}" for n,a,f in rows)
(out_dir/"summary.csv").write_text(csv)

md = ["| Model | Accuracy | Macro-F1 |","|---:|---:|---:|"]
for n,a,f in rows:
    acc = f"{a:.4f}" if isinstance(a,(int,float)) else "-"
    f1s = f"{f:.4f}" if isinstance(f,(int,float)) else "-"
    md.append(f"| {n} | {acc} | {f1s} |")
(out_dir/"summary.md").write_text("\n".join(md))

print("Wrote results/summary/summary.csv and results/summary/summary.md")
