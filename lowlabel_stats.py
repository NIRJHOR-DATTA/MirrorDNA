import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EQ = "results_scale_tiny_duplex_RC-equivariant_caduceus10B"
AUG = "results_scale_tiny_duplex_baseline_RCaug_caduceus10B"
NT_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
          "H4K20me1", "enhancers", "enhancers_types"]
GB = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "drosophila_enhancers_stark",
      "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
      "human_nontata_promoters", "human_ocr_ensembl"]
GROUPS = {"NT 12 strand-symmetric": NT_SYM, "Genomic Benchmarks (9)": GB, "All 21": NT_SYM + GB}
DIRECTIONAL = {"demo_coding_vs_intergenomic_seqs", "human_nontata_promoters"}


def read(folder, label):
    f = Path(folder) / "results_suite.jsonl"
    out = []
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        if not line.strip():
            continue
        r = json.loads(line)
        m, k = r.get("model", ""), r.get("key", "")
        if r.get("task") not in NT_SYM + GB or "+base" not in m:
            continue
        if "[inv" in m or " +length" in m or "|ep10" not in k:
            continue
        if label == "aug" and r["task"] in DIRECTIONAL and "|dirfix" not in k:
            continue                                           # directional tasks: the run without augmentation
        n = re.search(r" n=(\d+)", m)
        out.append(dict(model=label, budget=int(n.group(1)) if n else 10 ** 9, task=r["task"], seed=r["seed"],
                        fwd=r["mcc"], both=r.get("consistent_mcc"), flips=r.get("flip_rate"),
                        avg=r.get("mcc_rc_avg")))
    return out


df = pd.DataFrame(read(EQ, "equivariant") + read(AUG, "aug")).drop_duplicates(["model", "budget", "task", "seed"],
                                                                                keep="last")
if df.empty:
    raise SystemExit("no results found: run from the project folder")
budgets = sorted(df.budget.unique())
name = lambda b: "full" if b >= 10 ** 9 else str(b)
lines = ["# Label efficiency: exact equivariance vs RC augmentation (same architecture and pretraining)", "",
         "Mean ± SD over seeds of the per-seed group average; wins = tasks where the equivariant model's mean is higher; "
         "Wilcoxon signed-rank over tasks, two-sided. Directional tasks have no flip / both-strands metric. 'averaged over strands' = MCC of the class probabilities averaged over a sequence and its reverse complement (test-time averaging).", ""]
rows = []
for gname, tasks in GROUPS.items():
    lines += [f"## {gname}", "",
              "| labels | metric | equivariant | RC aug. | difference | wins | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for b in budgets:
        for metric in ["fwd", "both", "flips", "avg"]:
            ts = [t for t in tasks if not (metric != "fwd" and t in DIRECTIONAL)]
            d = df[(df.budget == b) & df.task.isin(ts)].dropna(subset=[metric])
            e, a = d[d.model == "equivariant"], d[d.model == "aug"]
            common = sorted(set(e.task) & set(a.task))
            if len(common) < 2:
                continue
            em, am = e.groupby("task")[metric].mean()[common], a.groupby("task")[metric].mean()[common]

            def per_seed(x):
                full = [s for s in sorted(x.seed.unique()) if set(x[x.seed == s].task) >= set(common)]
                v = [x[(x.seed == s) & x.task.isin(common)][metric].mean() for s in full]
                return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else np.nan)
            (me, se), (ma, sa) = per_seed(e), per_seed(a)
            diff = em - am
            try:
                p = stats.wilcoxon(em, am).pvalue if (diff != 0).any() else 1.0
            except ValueError:
                p = np.nan
            better = (diff < 0) if metric == "flips" else (diff > 0)
            fmt = (lambda m, s: f"{100 * m:.1f}% ± {100 * s:.1f}") if metric == "flips" else (lambda m, s: f"{m:.3f} ± {s:.3f}")
            lines.append(f"| {name(b)} | {dict(fwd='forward MCC', both='both strands', flips='flips', avg='averaged over strands')[metric]} | "
                         f"{fmt(me, se)} | {fmt(ma, sa)} | {diff.mean():+.3f} | {int(better.sum())}/{len(common)} | "
                         f"{p:.2g} |")
            rows.append(dict(group=gname, labels=name(b), metric=metric, equivariant=me, equivariant_sd=se, aug=ma,
                             aug_sd=sa, diff=diff.mean(), wins=int(better.sum()), tasks=len(common), wilcoxon_p=p))
    lines.append("")
pd.DataFrame(rows).to_csv("lowlabel_stats.csv", index=False)
# per task (supplement): mean and SD over seeds for each budget, model and task
pt = df.groupby(["budget", "task", "model"]).agg(fwd=("fwd", "mean"), fwd_sd=("fwd", "std"), both=("both", "mean"),
                                                 both_sd=("both", "std"), flips=("flips", "mean"),
                                                 n=("seed", "nunique")).reset_index()
pt["labels"] = pt.budget.map(name)
pt.drop(columns="budget").to_csv("lowlabel_per_task.csv", index=False)
Path("lowlabel_stats.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(len(budgets)); lab = [name(b) for b in budgets]
    for i, metric in enumerate(["fwd", "both"]):
        s = r[(r.group == "All 21") & (r.metric == metric)] if metric == "fwd" else \
            r[(r.group == "NT 12 strand-symmetric") & (r.metric == metric)]
        s = s.set_index("labels").reindex(lab)
        for col, c, mk, nm in [("equivariant", "#c0392b", "o", "exact equivariance"), ("aug", "#e67e22", "s", "RC augmentation")]:
            ax[i].errorbar(x, s[col], yerr=s[col + "_sd"], color=c, marker=mk, capsize=3, label=nm)
        top = np.nanmax(s[["equivariant", "aug"]].values + s[["equivariant_sd", "aug_sd"]].fillna(0).values, axis=1)
        for xi, (p, t) in enumerate(zip(s.wilcoxon_p, top)):          # significance of the paired test over tasks
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax[i].text(xi, t + 0.008, star, ha="center", va="bottom", fontsize=7, color="0.3")
        ax[i].set_xticks(x); ax[i].set_xticklabels([l if l == "full" else f"{int(l):,}" for l in lab])
        ax[i].margins(y=0.15)
        ax[i].set_xlabel("training labels per task")
        ax[i].set_ylabel("MCC")
        ax[i].set_title(["(a) forward MCC, 21 tasks", "(b) correct on both strands, 12 tasks"][i], loc="left", fontsize=9)
        ax[i].spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.text(0.5, -0.02, "Wilcoxon signed-rank test over tasks: * p<0.05, ** p<0.01, *** p<0.001, n.s. not significant",
             ha="center", fontsize=6.5, color="0.3")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"lowlabel.{ext}", dpi=300, bbox_inches="tight")
except Exception as e:
    print("no figure:", e)
