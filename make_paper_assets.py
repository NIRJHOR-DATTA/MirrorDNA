
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--ours", default="results_scale_tiny_RC-equivariant_caduceus10B")
p.add_argument("--aug", default="results_scale_tiny_duplex_baseline_RCaug_caduceus10B")
p.add_argument("--public-long", nargs="+", default=["results_public_long", "results_public_1k_4090", "results_public_1k"])
p.add_argument("--skip-models", action="store_true", help="skip the two figures that run the models")
p.add_argument("--out", default="paper_assets")
args = p.parse_args()
OUT = Path(args.out)
OUT.mkdir(exist_ok=True)
PY = sys.executable


def run(*cmd):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")       # Windows: minus signs etc. in redirected output
    r = subprocess.run([PY, *map(str, cmd)], env=env)
    if r.returncode:
        print(f"    (exit {r.returncode}; continuing)")
    return r.returncode == 0


def move(src, dst=None):
    src = Path(src)
    if src.exists():
        shutil.copy(src, OUT / (dst or src.name))
        print(f"    -> {OUT / (dst or src.name)}")


# ------------------------------------------------------------------ statistics and data-driven figures
run("stats_tables.py")
for f in Path("stats").glob("*"):
    if f.suffix != ".tex":
        move(f)
run("lowlabel_stats.py")
move("lowlabel.pdf", "fig_lowlabel.pdf"); move("lowlabel.png", "fig_lowlabel.png"); move("lowlabel_stats.md")
move("lowlabel_stats.csv"); move("lowlabel_per_task.csv")
move("results_deepstarr/deepstarr.csv"); move("results_deepstarr/report.md", "deepstarr_report.md")
run("compare_long.py", "--public", *[d for d in args.public_long if Path(d).exists()])
move("compare_long.md"); move("compare_long.csv")
run("figs/fig_strand.py", "--out", OUT / "fig_strand")

# ------------------------------------------------------------------ figures that run the models
if not args.skip_models:
    run("figs/fig_motifs.py", args.ours, args.aug, "--only-strand", "--out", OUT / "fig_motif_strand")
    run("figs/fig_chunk_mirror.py", args.ours, args.aug, "--motif-summary", OUT / "fig_motif_strand_summary.csv",
        "--out", OUT / "fig_tokens")
for f in ["splice_tokens_report.md", "splice_tokens_report.csv", "bio_boundaries_report.md",
          *Path(".").glob("bio_boundaries_*_profiles.png")]:
    move(f)
if Path("splice_tokens_report.csv").exists():
    run("figs/fig_splice.py", "--ours", args.ours, "--aug", args.aug, "--out", OUT / "fig_splice")

# ------------------------------------------------------------------ Table 1: main results (CSV)
summ = pd.read_csv(OUT / "summary.csv")
strand = pd.read_csv(OUT / "strand.csv").set_index("model")
ROWS = ["NT-v2", "DNABERT-2", "HyenaDNA-small", "Caduceus-PS-1k", "Caduceus-PH-1k", "Chunker + RC aug.",
        "MirrorDNA (ours)"]
PARAMS = {"NT-v2": "98M", "DNABERT-2": "117M", "HyenaDNA-small": "3.3M", "Caduceus-PS-1k": "0.47M",
          "Caduceus-PH-1k": "0.47M", "Chunker + RC aug.": "2.5M", "MirrorDNA (ours)": "2.3M"}
rows = []
for m in ROWS:
    r = dict(model=m, params=PARAMS[m])
    for suite in ["NT-18", "NT 12 sym.", "NT 6 dir.", "GB-9", "Gene strand"]:
        x = summ[(summ.model == m) & (summ.suite == suite)]
        r[suite + " mean"] = round(x["mean"].iloc[0], 3) if len(x) else None
        r[suite + " sd"] = round(x["sd"].iloc[0], 3) if len(x) else None
    for c, scale in [("fwd", 1), ("rc", 1), ("both", 1), ("flips", 100)]:
        name = {"fwd": "sym forward", "rc": "sym reverse complement", "both": "sym both strands", "flips": "flips %"}[c]
        r[name + " mean"] = round(scale * strand.loc[m, c], 3) if m in strand.index else None
        r[name + " sd"] = round(scale * strand.loc[m, c + "_sd"], 3) if m in strand.index else None
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "table_main.csv", index=False)
print(f"    -> {OUT / 'table_main.csv'}")

# ------------------------------------------------------------------ Table 2: longer inputs (CSV)
if (OUT / "compare_long.csv").exists():
    lc = pd.read_csv(OUT / "compare_long.csv")
    lc = lc[lc.model.isin(["MirrorDNA (ours)", "Caduceus-PH-1k"]) & lc.site.isin(["TSS", "donor"])]
    t = lc.pivot_table(index=["site", "model"], columns="window", values=["mean", "sd"])
    t.columns = [f"{w} bp {k}" for k, w in t.columns]
    t = t[sorted(t.columns, key=lambda c: (int(c.split()[0]), c.endswith("sd")))].round(3)
    t.to_csv(OUT / "table_length.csv")
    print(f"    -> {OUT / 'table_length.csv'}")

# ------------------------------------------------------------------ supplement: every detailed table as LaTeX
run("make_supplement.py")

print(f"\nDONE. Everything is in {OUT}/:")
for f in sorted(OUT.iterdir()):
    print("  ", f.name)
