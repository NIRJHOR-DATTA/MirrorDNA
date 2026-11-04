import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NT_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
          "H4K20me1", "enhancers", "enhancers_types"]
NT_DIR = ["promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors", "splice_sites_all",
          "splice_sites_donors"]
GB9 = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "drosophila_enhancers_stark",
       "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
       "human_nontata_promoters", "human_ocr_ensembl"]
GB_DIR = ["demo_coding_vs_intergenomic_seqs", "human_nontata_promoters"]
GB_SYM = [t for t in GB9 if t not in GB_DIR]
GENE = ["bio_gene_strand_tss", "bio_gene_strand_donor"]
SPLICE = ["splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]
REG = ["human_ensembl_regulatory"]
ORDER = ["NT-v2", "DNABERT-2", "HyenaDNA-small", "Caduceus-PS-1k", "Caduceus-PH-1k", "Chunker + RC aug.",
         "MirrorDNA duplex", "MirrorDNA (ours)"]

p = argparse.ArgumentParser()
p.add_argument("--runs", default="stats/all_runs.csv")
p.add_argument("--center", default="results_scale_tiny_RC-equivariant_center")
p.add_argument("--ours", default="MirrorDNA (ours)")
args = p.parse_args()
CTR = Path(args.center)

runs = pd.read_csv(args.runs)
sel = pd.read_csv(CTR / "head_selection.csv")
chosen = dict(zip(sel.task, sel.chosen))

rows = []
for line in (CTR / "results_suite.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    extra = r["key"].split("|")[4:]
    if r.get("variant") != "RC-equivariant" or "bf" not in extra or "inv" in extra or "len" in extra \
            or any(e.startswith("n") and e[1:].isdigit() for e in extra) or "ep10" not in extra:
        continue
    d = r["task"] in NT_DIR + GB_DIR + GENE
    rows.append(dict(model=args.ours, task=r["task"], seed=r["seed"], fwd=r["mcc"],
                     rc=np.nan if d else r.get("mcc_rc_input"), both=np.nan if d else r.get("consistent_mcc"),
                     flips=np.nan if d else r.get("flip_rate"), avg=np.nan if d else r.get("mcc_rc_avg")))
ctr = pd.DataFrame(rows).drop_duplicates(["task", "seed"], keep="last")

ours_std = runs[runs.model == args.ours]
if not len(ours_std):
    raise SystemExit(f"no rows for '{args.ours}' in {args.runs}")
keep_std = ours_std[ours_std.task.map(lambda t: chosen.get(t, "standard") == "standard")]
take_ctr = ctr[ctr.task.map(lambda t: chosen.get(t) == "centre")]
missing = [t for t, c in chosen.items() if c == "centre" and t not in set(take_ctr.task)]
if missing:
    raise SystemExit(f"centre readout chosen but no centre results for: {missing}")
merged = pd.concat([runs[runs.model != args.ours], keep_std, take_ctr[runs.columns.intersection(take_ctr.columns)]],
                   ignore_index=True)
merged.to_csv(CTR / "all_runs_selected.csv", index=False)

models = [m for m in ORDER if m in set(merged.model)] + sorted(set(merged.model) - set(ORDER))
SETS = {"NT-18": NT_SYM + NT_DIR, "NT-15": [t for t in NT_SYM + NT_DIR if t not in SPLICE], "Dir. (6)": NT_DIR,
        "GB-9": GB9, "GB-8": [t for t in GB9 if t not in REG], "Gene (2)": GENE,
        "All 27": NT_SYM + NT_DIR + GB9, "All 29": NT_SYM + NT_DIR + GB9 + GENE}


def suite(m, tasks, col="fwd"):
    d = merged[(merged.model == m) & merged.task.isin(tasks)].dropna(subset=[col])
    seeds = [s for s in sorted(d.seed.unique()) if d[d.seed == s].task.nunique() == len(tasks)]
    if not seeds:
        return np.nan, np.nan
    ps = [d[d.seed == s][col].mean() for s in seeds]
    return np.mean(ps), (np.std(ps, ddof=1) if len(ps) > 1 else np.nan)


f3 = lambda v, s: "–" if pd.isna(v) else (f"{v:.3f} ± {s:.3f}" if not pd.isna(s) else f"{v:.3f}")
L = ["# Final results: MirrorDNA with the head chosen per task on validation", "",
     "Centre readout chosen for: " + ", ".join(t for t, c in chosen.items() if c == "centre") + ".", "",
     "## Accuracy (forward MCC, mean ± SD over seeds of the suite average)", "",
     "| model | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
for m in models:
    L.append(f"| {m} | " + " | ".join(f3(*suite(m, ts)) for ts in SETS.values()) + " |")

per_task = merged.groupby(["model", "task"]).fwd.mean().unstack()
L += ["", f"## {args.ours} against each model: tasks won / tasks (two-sided Wilcoxon p over tasks)", "",
      "| model | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
for m in models:
    if m == args.ours:
        continue
    cells = []
    for ts in SETS.values():
        a, b = per_task.loc[args.ours].reindex(ts), per_task.loc[m].reindex(ts)
        ok = a.notna() & b.notna()
        diff = a[ok] - b[ok]
        pv = wilcoxon(a[ok], b[ok]).pvalue if ok.sum() >= 5 and (diff != 0).any() else np.nan
        cells.append(f"{int((diff > 0).sum())}/{int(ok.sum())}" + ("" if pd.isna(pv) else f" ({pv:.2g})"))
    L.append(f"| {m} | " + " | ".join(cells) + " |")

L += ["", "## Strand consistency (strand-symmetric tasks; mean ± SD over seeds)", "",
      "| model | NT-12 fwd | NT-12 RC | NT-12 both | NT-12 flips | GB-7 fwd | GB-7 RC | GB-7 both | GB-7 flips |",
      "|---|---|---|---|---|---|---|---|---|"]
for m in models:
    cells = []
    for ts in (NT_SYM, GB_SYM):
        for col in ("fwd", "rc", "both", "flips"):
            v, s = suite(m, ts, col)
            cells.append("–" if pd.isna(v) else (f"{100 * v:.1f}% ± {100 * s:.1f}" if col == "flips" else f3(v, s)))
    L.append(f"| {m} | " + " | ".join(cells) + " |")

text = "\n".join(L) + "\n"
(CTR / "final_report.md").write_text(text, encoding="utf-8")
print(text.replace("±", "+-"))