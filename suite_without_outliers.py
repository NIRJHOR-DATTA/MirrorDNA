import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NT18 = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
        "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors",
        "splice_sites_all", "splice_sites_donors"]
GB9 = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "drosophila_enhancers_stark",
       "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
       "human_nontata_promoters", "human_ocr_ensembl"]
DEFAULT_OUT = ["splice_sites_acceptors", "splice_sites_all", "splice_sites_donors", "human_ensembl_regulatory"]
ORDER = ["NT-v2", "DNABERT-2", "HyenaDNA-small", "Caduceus-PS-1k", "Caduceus-PH-1k", "Chunker + RC aug.",
         "MirrorDNA duplex", "MirrorDNA (ours)"]

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="?", default="stats/all_runs.csv")
p.add_argument("--exclude", nargs="+", default=DEFAULT_OUT, help="tasks left out of the reduced averages")
p.add_argument("--ours", default="MirrorDNA (ours)")
args = p.parse_args()

f = Path(args.runs)
df = pd.read_csv(f)
df = df[df.task.isin(NT18 + GB9)]
out_set = set(args.exclude)
SETS = {"NT-18": NT18, f"NT-{len([t for t in NT18 if t not in out_set])}": [t for t in NT18 if t not in out_set],
        "GB-9": GB9, f"GB-{len([t for t in GB9 if t not in out_set])}": [t for t in GB9 if t not in out_set],
        "All 27": NT18 + GB9, f"All {len([t for t in NT18 + GB9 if t not in out_set])}": [t for t in NT18 + GB9 if t not in out_set]}
models = [m for m in ORDER if m in set(df.model)] + sorted(set(df.model) - set(ORDER))

rows = []
for m in models:
    d = df[df.model == m]
    r = dict(model=m)
    for name, tasks in SETS.items():
        dd = d[d.task.isin(tasks)]
        seeds = [s for s in sorted(dd.seed.unique()) if dd[dd.seed == s].task.nunique() == len(tasks)]
        if not seeds:
            r[name] = r[name + "_sd"] = np.nan
            continue
        ps = [dd[dd.seed == s].fwd.mean() for s in seeds]
        r[name], r[name + "_sd"] = np.mean(ps), (np.std(ps, ddof=1) if len(ps) > 1 else np.nan)
    rows.append(r)
summ = pd.DataFrame(rows)

per_task = df.groupby(["model", "task"]).fwd.mean().unstack()
tests = []
for m in models:
    if m == args.ours or args.ours not in per_task.index:
        continue
    t = dict(model=m)
    for name, tasks in SETS.items():
        a, b = per_task.loc[args.ours, tasks], per_task.loc[m, tasks]
        ok = a.notna() & b.notna()
        diff = a[ok] - b[ok]
        t[name + "_wins"] = f"{int((diff > 0).sum())}/{int(ok.sum())}"
        t[name + "_p"] = wilcoxon(a[ok], b[ok]).pvalue if ok.sum() >= 5 and (diff != 0).any() else np.nan
    tests.append(t)
tests = pd.DataFrame(tests)

names = list(SETS)
fmt = lambda v, s: "–" if pd.isna(v) else (f"{v:.3f} ± {s:.3f}" if not pd.isna(s) else f"{v:.3f}")
lines = ["# Suite averages with and without the outlier tasks (forward MCC, mean ± SD over seeds)", "",
         f"Left out of the reduced sets: {', '.join(args.exclude)}. The full averages are shown next to them.", "",
         "| model | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for _, r in summ.iterrows():
    lines.append(f"| {r.model} | " + " | ".join(fmt(r[n], r[n + '_sd']) for n in names) + " |")
if len(tests):
    lines += ["", f"{args.ours} against each model: tasks won / tasks, and two-sided Wilcoxon p over tasks", "",
              "| model | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for _, t in tests.iterrows():
        lines.append(f"| {t.model} | " + " | ".join(
            f"{t[n + '_wins']} (p={t[n + '_p']:.2g})" if not pd.isna(t[n + "_p"]) else t[n + "_wins"] for n in names) + " |")
text = "\n".join(lines) + "\n"
summ.round(4).to_csv(f.with_name("suite_without_outliers.csv"), index=False)
f.with_name("suite_without_outliers.md").write_text(text, encoding="utf-8")
print(text.replace("±", "+-"))