import sys
from pathlib import Path

import numpy as np
import pandas as pd

NT_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
          "H4K20me1", "enhancers", "enhancers_types"]
GB_SYM = ["demo_human_or_worm", "drosophila_enhancers_stark", "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn",
          "human_enhancers_ensembl", "human_ensembl_regulatory", "human_ocr_ensembl"]
f = Path(sys.argv[1] if len(sys.argv) > 1 else "all_runs.csv")
df = pd.read_csv(f).dropna(subset=["rc"])
rows = []
for group, tasks in {"GB 7 sym.": GB_SYM, "All 19 sym.": NT_SYM + GB_SYM}.items():
    for m in df.model.unique():
        d = df[(df.model == m) & df.task.isin(tasks)]
        seeds = [s for s in sorted(d.seed.unique()) if d[d.seed == s].task.nunique() == len(tasks)]
        if not seeds:
            missing = sorted(set(tasks) - set(d.task))
            print(f"  skipped {m} ({group}): missing {missing[:3]}")
            continue
        r = dict(group=group, model=m, seeds=len(seeds))
        for c in ["fwd", "rc", "both", "flips"]:
            ps = [d[d.seed == s][c].mean() for s in seeds]
            r[c], r[c + "_sd"] = np.mean(ps), (np.std(ps, ddof=1) if len(ps) > 1 else np.nan)
        rows.append(r)
out = pd.DataFrame(rows)
out.round(4).to_csv(f.with_name("strand_gb.csv"), index=False)
for g, d in out.groupby("group", sort=False):
    print(f"\n## {g}\n")
    print("| model | seeds | fwd | RC | both | flips |\n|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        print(f"| {r.model} | {r.seeds} | {r.fwd:.3f} +- {r.fwd_sd:.3f} | {r.rc:.3f} +- {r.rc_sd:.3f} | "
              f"{r.both:.3f} +- {r.both_sd:.3f} | {100 * r.flips:.1f}% +- {100 * r.flips_sd:.1f} |")