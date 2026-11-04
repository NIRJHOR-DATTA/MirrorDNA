import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import COLORS, MARKERS, OURS, plt, save

SYM12 = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
         "H4K20me1", "enhancers", "enhancers_types"]
p = argparse.ArgumentParser()
p.add_argument("--ours", default="results_scale_tiny_RC-equivariant_caduceus10B")
p.add_argument("--aug", default="results_scale_tiny_duplex_baseline_RCaug_caduceus10B")
p.add_argument("--tasks", nargs="+", default=SYM12)
p.add_argument("--out", default="fig_strand")
args = p.parse_args()


def rows(f):
    f = Path(f)
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def pick(rs, label, need=lambda r: True):
    out = []
    for r in rs:
        if r.get("task") in args.tasks and need(r) and "[inv" not in r.get("model", "") and "oriented" not in r.get("key", ""):
            out.append(dict(model=label, task=r["task"], seed=r["seed"], fwd=r["mcc"], rc=r["mcc_rc_input"],
                            both=r.get("consistent_mcc", np.nan), flips=r.get("flip_rate", np.nan)))
    return out


data = []
data += pick(rows(Path(args.ours) / "results_suite.jsonl"), OURS, lambda r: "+base" in r.get("model", ""))
data += pick(rows(Path(args.aug) / "results_suite.jsonl"), "Chunker + RC aug.", lambda r: "+base" in r.get("model", ""))
PUBLIC = {"NT-v2-100M (6-mer) lr3e-05": "NT-v2", "DNABERT-2 (BPE) lr3e-05": "DNABERT-2", "HyenaDNA-small (char)": "HyenaDNA",
          "Caduceus-PS-1k (char, RC-equivariant)": "Caduceus-PS-1k",
          # published protocol: Caduceus-PH is fine-tuned with reverse-complement augmentation
          "Caduceus-PH-1k (char, RC-aug pretrained) +RCaug": "Caduceus-PH-1k"}
for f in sorted(Path(".").glob("results_public*/results.jsonl")):
    for r in rows(f):
        if r.get("variant") in PUBLIC and ("|full" in r.get("key", "") or "Caduceus" in r.get("variant", "")):
            data += pick([r], PUBLIC[r["variant"]])
if not data:
    raise SystemExit("no results found: run from the project folder (where results_scale_* and results_public* are)")

df = pd.DataFrame(data).drop_duplicates(["model", "task", "seed"])
per_task = df.groupby(["model", "task"]).mean(numeric_only=True).reset_index()
# models missing some tasks would bias the averages: report it and keep only complete models in panel (b)
n_tasks = per_task.groupby("model").task.nunique()
complete = [m for m in n_tasks.index if n_tasks[m] == len(args.tasks)]
for m in n_tasks.index:
    if m not in complete:
        print(f"note: {m} has {n_tasks[m]}/{len(args.tasks)} tasks -> shown in (a) only")
summ = per_task[per_task.model.isin(complete)].groupby("model").mean(numeric_only=True)
order = [m for m in ["NT-v2", "DNABERT-2", "HyenaDNA", "Caduceus-PH-1k", "Caduceus-PS-1k", "Chunker + RC aug.", OURS]
         if m in summ.index]
summ = summ.loc[order].drop(columns="seed", errors="ignore")
summ.round(4).to_csv(args.out + ".csv")
print(summ.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1, 1.35]))
a = ax[0]
lo = min(per_task.fwd.min(), per_task.rc.min()) - 0.03
hi = max(per_task.fwd.max(), per_task.rc.max()) + 0.03
a.plot([lo, hi], [lo, hi], color="0.75", lw=0.8, ls="--", zorder=0)
a.text(lo + 0.03, lo + 0.05, "same accuracy on both strands", ha="left", va="bottom", fontsize=6, color="0.5",
       rotation=45, rotation_mode="anchor")
for m in order[::-1] + [m for m in per_task.model.unique() if m not in order]:
    d = per_task[per_task.model == m]
    a.scatter(d.fwd, d.rc, s=55 if m == OURS else 22, marker=MARKERS.get(m, "o"), color=COLORS.get(m, "0.4"),
              edgecolor="white", lw=0.4, label=m, zorder=3 if m == OURS else 2)
a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
a.set_xlabel("MCC, forward strand"); a.set_ylabel("MCC, reverse-complement strand")
a.set_title("(a) each task, both strands", loc="left")
a.legend(frameon=False, loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0.1)

b = ax[1]
x = np.arange(len(order)); w = 0.26
for k, (col, lab, shade) in enumerate([("fwd", "forward", 0.45), ("rc", "reverse complement", 0.7),
                                         ("both", "correct on both strands", 1.0)]):
    b.bar(x + (k - 1) * w, summ[col], w, color=[COLORS.get(m, "0.4") for m in order], alpha=shade,
          edgecolor="white", lw=0.4, label=lab)
for i, m in enumerate(order):
    b.text(i + w, summ["both"].iloc[i] + 0.012, f"{summ['both'].iloc[i]:.2f}", ha="center", fontsize=6,
           fontweight="bold" if m == OURS else None)
    fl = summ["flips"].iloc[i]
    b.text(i, max(summ.loc[m, ["fwd", "rc", "both"]]) + 0.07, f"{fl:.0%} flips", ha="center", fontsize=6,
           color=COLORS.get(m, "0.4"), fontweight="bold")
b.set_xticks(x); b.set_xticklabels([m.replace(" (ours)", "\n(ours)").replace(" + ", "\n+ ").replace("Caduceus-", "Caduceus\n") for m in order], fontsize=6.5)
b.set_ylabel("MCC (mean over tasks)"); b.set_ylim(min(0, summ[["fwd", "rc", "both"]].values.min() - 0.02),
                                                   summ[["fwd", "rc", "both"]].values.max() + 0.14)
b.axhline(0, color="0.6", lw=0.6)
from matplotlib.patches import Patch
b.legend(handles=[Patch(color="0.3", alpha=s, label=l) for s, l in
                  [(0.45, "forward"), (0.7, "reverse complement"), (1.0, "correct on both strands")]],
         frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=6.5)
b.set_title(f"(b) mean over {len(args.tasks)} tasks", loc="left", pad=22)
fig.tight_layout()
save(fig, args.out)
