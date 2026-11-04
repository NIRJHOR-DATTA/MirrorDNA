import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NT18_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
            "H4K20me1", "enhancers", "enhancers_types"]
NT18_DIR = ["promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors", "splice_sites_all",
            "splice_sites_donors"]
GB9 = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "drosophila_enhancers_stark",
       "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
       "human_nontata_promoters", "human_ocr_ensembl"]
GENE = ["bio_gene_strand_tss", "bio_gene_strand_donor"]

p = argparse.ArgumentParser()
p.add_argument("--standard", default="results_scale_tiny_RC-equivariant_caduceus10B")
p.add_argument("--center", default="results_scale_tiny_RC-equivariant_center")
args = p.parse_args()


def load(folder):
    f = Path(folder) / "results_suite.jsonl"
    if not f.exists():
        raise SystemExit(f"{f} not found")
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        extra = r["key"].split("|")[4:]
        if r.get("variant") != "RC-equivariant" or "bf" not in extra or "inv" in extra or "len" in extra \
                or any(e.startswith("n") and e[1:].isdigit() for e in extra) or "ep10" not in extra:
            continue                                  # main protocol only: +base, all labels, 10 epochs
        rows.append(dict(task=r["task"], seed=r["seed"], val=r.get("val_mcc"), test=r["mcc"],
                         both=r.get("consistent_mcc"), flips=r.get("flip_rate")))
    return pd.DataFrame(rows).drop_duplicates(["task", "seed"], keep="last")


std, ctr = load(args.standard), load(args.center)
m = std.merge(ctr, on=["task", "seed"], suffixes=("_std", "_ctr"))
if not len(m):
    raise SystemExit("no task and seed finished with both heads yet")
if m.val_std.isna().any() or m.val_ctr.isna().any():
    raise SystemExit("validation MCC missing in some results; cannot select on validation")
seeds = sorted(int(s) for s in m.seed.unique())
t = m.groupby("task", as_index=False).agg(n_seeds=("seed", "nunique"), val_std=("val_std", "mean"),
                                          val_ctr=("val_ctr", "mean"), test_std=("test_std", "mean"),
                                          test_ctr=("test_ctr", "mean"), flips_ctr=("flips_ctr", "mean"))
t["chosen"] = np.where(t.val_ctr > t.val_std, "centre", "standard")
t["test_sel"] = np.where(t.chosen == "centre", t.test_ctr, t.test_std)

groups = {"NT 18": t.task.isin(NT18_SYM + NT18_DIR), "NT 12 symmetric": t.task.isin(NT18_SYM),
          "NT 6 direction-dependent": t.task.isin(NT18_DIR), "NT splice sites": t.task.str.startswith("splice"),
          "NT promoters": t.task.str.startswith("promoter"), "Genomic Benchmarks": t.task.isin(GB9),
          "Gene strand": t.task.isin(GENE), "All finished": t.task.notna()}
lines = ["# Head chosen per task on the validation split (test MCC reported)", "",
         f"Seeds both runs have: {seeds}. For each task the head with the higher mean validation MCC is chosen; the "
         "test set is not used for the choice. Wilcoxon signed-rank test over tasks: selected vs standard head.", "",
         "| tasks | n | standard head | centre readout | selected (by validation) | selected - standard | "
         "centre chosen | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
for g, sel in groups.items():
    d = t[sel]
    if not len(d):
        continue
    diff = d.test_sel - d.test_std
    pv = wilcoxon(d.test_sel, d.test_std).pvalue if len(d) >= 5 and (diff != 0).any() else float("nan")
    lines.append(f"| {g} | {len(d)} | {d.test_std.mean():.3f} | {d.test_ctr.mean():.3f} | {d.test_sel.mean():.3f} | "
                 f"{diff.mean():+.3f} | {(d.chosen == 'centre').sum()}/{len(d)} | {pv:.3g} |")
lines += ["", "| task | val: standard | val: centre | chosen | test: standard | test: centre | test: selected |",
          "|---|---|---|---|---|---|---|"]
order = {k: i for i, k in enumerate(NT18_SYM + NT18_DIR + GB9 + GENE)}
for _, r in t.sort_values("task", key=lambda s: s.map(lambda x: order.get(x, 999))).iterrows():
    lines.append(f"| {r.task} | {r.val_std:.3f} | {r.val_ctr:.3f} | {r.chosen} | {r.test_std:.3f} | {r.test_ctr:.3f} | "
                 f"**{r.test_sel:.3f}** |")
if (t.flips_ctr.fillna(0) > 0.001).any():
    lines += ["", "Note: the centre-readout runs show strand flips on: "
              + ", ".join(t[t.flips_ctr.fillna(0) > 0.001].task)]
if len(seeds) < 3:
    lines += ["", f"Only {len(seeds)} seed(s): treat as preliminary."]
text = "\n".join(lines) + "\n"
out = Path(args.center)
(out / "head_selection.md").write_text(text, encoding="utf-8")
t.round(4).to_csv(out / "head_selection.csv", index=False)
print(text)