import argparse
import json
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--all", action="store_true", help="include runs from the older protocol (5 epochs / capped data)")
p.add_argument("--out", default="results_csv")
args = p.parse_args()

NT_DIR = {"promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors", "splice_sites_all",
          "splice_sites_donors"}
GB = {"dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn",
      "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl",
      "drosophila_enhancers_stark"}
GB_DIR = {"demo_coding_vs_intergenomic_seqs", "human_nontata_promoters"}
METRICS = ["mcc", "mcc_rc_input", "mcc_rc_avg", "consistent_mcc", "f1", "f1_rc", "consistent_f1", "acc", "acc_rc", "consistent_acc",
           "flip_rate", "best_epoch", "train_min"]


def suite_of(t):
    return "bio" if t.startswith("bio_") else ("gb" if t in GB else "nt")


def read(f):
    rows = []
    for line in Path(f).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass                                    # a line cut short by a crash or a write in progress
    return rows


rows = []
for f in sorted(Path(".").glob("results_scale_*/results_suite.jsonl")):
    if "smoke" in f.parent.name:
        continue
    for r in read(f):
        if not args.all and "|ep10" not in r.get("key", ""):
            continue
        t_ = r.get("task", "")
        if (not args.all and "RCaug" in r.get("key", "") and "|dirfix" not in r["key"] and "|inv" not in r["key"]
                and (t_ in NT_DIR | GB_DIR or t_.startswith("bio_"))):
            continue                                # old run: RC augmentation was (wrongly) on for a directional task
        name = f.parent.name.replace("results_scale_", "")
        rows.append({**r, "source": "ours", "model": name + (" +base" if "|bf" in r["key"] else "") + (" [inv head]" if "|inv" in r["key"] else "")})
for f in sorted(Path(".").glob("results_public*/results.jsonl")):
    for r in read(f):
        k = r.get("key", "")
        if not args.all and not ("|ep10p3" in k and "|full" in k):
            continue
        rows.append({**r, "source": "public", "model": r["variant"]})

if not rows:
    raise SystemExit("no results found yet (run from the project folder; use --all for older runs)")
df = pd.DataFrame(rows)
df["suite"] = df.task.map(suite_of)
df["directional"] = df.task.isin(NT_DIR | GB_DIR) | df.task.str.startswith("bio_")
for m in METRICS:
    if m not in df:
        df[m] = float("nan")
df["robust_mcc"] = (df.mcc + df.mcc_rc_input) / 2
out = Path(args.out)
out.mkdir(exist_ok=True)

cols = ["source", "model", "suite", "task", "directional", "seed", "robust_mcc"] + METRICS
df = df.sort_values(["suite", "task", "model", "seed"])
df[[c for c in cols if c in df]].to_csv(out / "all_runs.csv", index=False, float_format="%.4f")

agg = df.groupby(["suite", "task", "directional", "model"])
by = agg[["robust_mcc"] + METRICS].mean()
by["mcc_std"] = agg.mcc.std()
by["seeds"] = agg.seed.nunique()
by = by.reset_index()
by.to_csv(out / "by_task.csv", index=False, float_format="%.4f")

# side-by-side: forward MCC per model (+ flip rate), average over tasks finished by every model
wide = by.pivot_table(index=["suite", "task"], columns="model", values="mcc")
flips = by.pivot_table(index=["suite", "task"], columns="model", values="flip_rate").add_suffix(" flips")
seeds = by.pivot_table(index=["suite", "task"], columns="model", values="seeds").add_suffix(" seeds")
comp = pd.concat([wide, flips, seeds], axis=1)
common = wide.dropna()
avg = pd.DataFrame([common.mean()], index=pd.MultiIndex.from_tuples([("ALL", f"average over {len(common)} tasks "
                                                                              "done by every model")]))
comp = pd.concat([comp, avg])
comp.to_csv(out / "compare_mcc.csv", float_format="%.3f")

print(f"{len(df)} runs | models: {', '.join(sorted(df.model.unique()))}")
print(f"tasks finished by every model: {len(common)}")
if len(common):
    print("\nForward MCC, mean over seeds, tasks done by all models:")
    print(common.round(3).to_string())
    print("\naverage:", common.mean().round(3).to_dict())

# every finished result, per suite (also tasks not yet done by all models; '-' = not finished yet)
SUITE_NAME = {"nt": "Nucleotide Transformer", "gb": "Genomic Benchmarks", "bio": "Biology (gene strand)"}
for su in ["nt", "gb", "bio"]:
    t = by[by.suite == su].pivot_table(index="task", columns="model", values="mcc")
    if len(t):
        n_done = t.notna().sum()
        print(f"\n=== {SUITE_NAME[su]}: forward MCC (mean over seeds), every finished task ===")
        print(t.round(3).to_string(na_rep="-"))
        print("tasks finished per model:", n_done.to_dict())
# strand view: strand-symmetric tasks only (direction-dependent tasks have no RC label, so flips mean nothing there)
sym = by[~by.directional]
lines = []
for metric, title in [("mcc", "forward MCC"), ("mcc_rc_input", "MCC on the reverse-complement strand"),
                      ("robust_mcc", "strand-robust MCC (mean of both strands)"),
                      ("consistent_mcc", "CONSISTENT MCC (correct on both strands)"),
                      ("mcc_rc_avg", "MCC with test-time strand averaging (predict both strands, average; 2x cost)"),
                      ("flip_rate", "flip rate")]:
    t = sym.pivot_table(index=["suite", "task"], columns="model", values=metric)
    t = t.loc[:, t.notna().sum() >= 0.6 * t.notna().sum().max()]     # leave out runs that are far from finished
    t = t.dropna()
    if len(t):
        t.loc[("ALL", f"average ({len(t)} tasks)"), :] = t.mean()
        t.to_csv(out / f"strand_{metric}.csv", float_format="%.3f")
        lines.append(f"\n=== {title}, strand-symmetric tasks done by every model ===\n" + t.round(3).to_string())
print("\n".join(lines))
print(f"\nwritten: {out / 'all_runs.csv'}, {out / 'by_task.csv'}, {out / 'compare_mcc.csv'}, {out}/strand_*.csv")
