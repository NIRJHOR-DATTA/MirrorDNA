import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--out", default="stats")
p.add_argument("--ours", default="MirrorDNA (ours)")
args = p.parse_args()

NT_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
          "H4K20me1", "enhancers", "enhancers_types"]
NT_DIR = ["promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors", "splice_sites_all",
          "splice_sites_donors"]
GB = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "drosophila_enhancers_stark",
      "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory",
      "human_nontata_promoters", "human_ocr_ensembl"]
BIO = ["bio_gene_strand_tss", "bio_gene_strand_donor"]
SUITES = {"NT-18": NT_SYM + NT_DIR, "NT 12 sym.": NT_SYM, "NT 6 dir.": NT_DIR, "GB-9": GB, "Gene strand": BIO}
DIRECTIONAL = set(NT_DIR) | set(BIO)
ORDER = ["NT-v2", "DNABERT-2", "HyenaDNA-small", "HyenaDNA-tiny-1k", "Caduceus-PS-1k", "Caduceus-PS-1k (oriented head)", "Caduceus-PH-1k",
         "Chunker + RC aug.", "MirrorDNA duplex", "MirrorDNA, fixed cuts (ablation)", "MirrorDNA pico (0.46M)", args.ours]


def read(f):
    f = Path(f)
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


# ------------------------------------------------------------------ which runs belong to which reported model
def ours_label(folder, r):
    """Our models: the reported head (+base), 10 epochs, full data, not the invariant-head control."""
    m, k = r.get("model", ""), r.get("key", "")
    if "+base" not in m or "[inv" in m or " n=" in m or " +length" in m or "|ep10" not in k:
        return None
    if folder.endswith("tiny_RC-equivariant_caduceus10B"):
        return args.ours
    if folder.startswith("results_scale_tiny_fixed") and folder.endswith("RC-equivariant_caduceus10B"):
        return "MirrorDNA, fixed cuts (ablation)"
    if folder.startswith("results_scale_pico") and "RC-equivariant" in folder:
        return "MirrorDNA pico (0.46M)"
    if folder.endswith("tiny_duplex_RC-equivariant_caduceus10B"):
        return "MirrorDNA duplex"
    if folder.endswith("tiny_duplex_baseline_RCaug_caduceus10B"):
        # augmentation only on strand-symmetric tasks: directional tasks use the rerun without it (|dirfix)
        return "Chunker + RC aug." if (r["task"] not in DIRECTIONAL or "|dirfix" in k) else None
    return None


def public_label(r):
    v, k, t = r.get("variant", ""), r.get("key", ""), r.get("task", "")
    if "|full" not in k:
        return None
    if "[oriented" in v:                 # control: Caduceus-PS with an orientation-aware head, strand-dependent tasks only
        return "Caduceus-PS-1k (oriented head)" if v.startswith("Caduceus-PS-1k") and t in BIO else None
    if v.startswith("NT-v2-100M") and "lr3e-05" in v:
        return "NT-v2"
    if v.startswith("DNABERT-2") and "lr3e-05" in v:
        return "DNABERT-2"
    if v == "HyenaDNA-small (char)":
        return "HyenaDNA-small"
    if v == "HyenaDNA-tiny-1k (char)":
        return "HyenaDNA-tiny-1k"
    if v == "Caduceus-PS-1k (char, RC-equivariant)":
        return "Caduceus-PS-1k"
    if v.startswith("Caduceus-PH-1k"):
        # published protocol (RC augmentation) on NT and GB; gene strand (our task) without augmentation
        aug = "+RCaug" in v
        return "Caduceus-PH-1k" if (aug != (t in BIO)) else None
    return None


rows = []
for f in sorted(Path(".").glob("results_scale_*/results_suite.jsonl")):
    for r in read(f):
        lab = ours_label(f.parent.name, r)
        if lab:
            rows.append(dict(model=lab, task=r["task"], seed=r["seed"], fwd=r["mcc"], rc=r.get("mcc_rc_input"),
                             both=r.get("consistent_mcc"), flips=r.get("flip_rate"),
                             avg=r.get("mcc_rc_avg")))
for f in sorted(Path(".").glob("results_public*/results.jsonl")):
    for r in read(f):
        lab = public_label(r)
        if lab:
            rows.append(dict(model=lab, task=r["task"], seed=r["seed"], fwd=r["mcc"], rc=r.get("mcc_rc_input"),
                             both=r.get("consistent_mcc"), flips=r.get("flip_rate"),
                             avg=r.get("mcc_rc_avg")))
if not rows:
    raise SystemExit("no results found: run from the project folder")
df = pd.DataFrame(rows).drop_duplicates(["model", "task", "seed"], keep="last")
models = [m for m in ORDER if m in set(df.model)] + sorted(set(df.model) - set(ORDER))
out = Path(args.out)
out.mkdir(exist_ok=True)
report = ["# Statistical summary (MCC; mean +- SD over seeds)", ""]


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else (f"{v.mean():.{d}f}" if len(v) else "–")


# ------------------------------------------------------------------ per-task tables
per_task = {}
for suite in ["NT-18", "GB-9", "Gene strand"]:
    tasks = SUITES[suite]
    tab = pd.DataFrame(index=tasks, columns=models, dtype=object)
    for m in models:
        for t in tasks:
            v = df[(df.model == m) & (df.task == t)].fwd.values
            tab.loc[t, m] = ms(v) + (f" (n={len(v)})" if 0 < len(v) < 3 else "") if len(v) else "–"
    per_task[suite] = tab
    tab.to_csv(out / f"per_task_{suite.replace(' ', '_').replace('-', '')}.csv")
    report += [f"## {suite}: per task", "", tab.to_markdown(), ""]

# ------------------------------------------------------------------ suite averages with SD over seeds
summ = []
for suite, tasks in SUITES.items():
    for m in models:
        d = df[(df.model == m) & df.task.isin(tasks)]
        if d.task.nunique() < len(tasks):
            continue                                       # incomplete suite: no average (would be biased)
        seeds = [s for s in sorted(d.seed.unique()) if d[d.seed == s].task.nunique() == len(tasks)]
        per_seed = [d[d.seed == s].fwd.mean() for s in seeds]
        task_means = d.groupby("task").fwd.mean()
        summ.append(dict(suite=suite, model=m, mean=task_means.mean(),
                         sd=np.std(per_seed, ddof=1) if len(per_seed) > 1 else np.nan, n_seeds=len(per_seed)))
summ = pd.DataFrame(summ)
summ.to_csv(out / "summary.csv", index=False)
piv = summ.assign(cell=[f"{a:.3f} ± {b:.3f}" if not np.isnan(b) else f"{a:.3f}" for a, b in zip(summ["mean"], summ.sd)])
piv = piv.pivot(index="model", columns="suite", values="cell").reindex([m for m in models if m in set(summ.model)])
report += ["## Suite averages (SD = spread of the per-seed suite average)", "",
           piv[[c for c in SUITES if c in piv.columns]].fillna("–").to_markdown(), ""]

# ------------------------------------------------------------------ our model vs each baseline
def compare(model_a, model_b, tasks, metric="fwd"):
    """model_a vs model_b over tasks: mean difference, wins/losses, Wilcoxon over tasks, Holm-corrected per-task t."""
    a = df[(df.model == model_a) & df.task.isin(tasks)].dropna(subset=[metric])
    b = df[(df.model == model_b) & df.task.isin(tasks)].dropna(subset=[metric])
    common = sorted(set(a.task) & set(b.task))
    if len(common) < 2:
        return None
    ma = a.groupby("task")[metric].mean()[common]
    mb = b.groupby("task")[metric].mean()[common]
    diff = ma - mb
    try:
        wp = stats.wilcoxon(ma, mb, alternative="two-sided").pvalue if (diff != 0).any() else 1.0
    except ValueError:
        wp = np.nan
    pt = []                                                # per-task Welch t-test over seeds
    for t in common:
        x, y = a[a.task == t][metric].values, b[b.task == t][metric].values
        pt.append(stats.ttest_ind(x, y, equal_var=False).pvalue if len(x) > 1 and len(y) > 1 else np.nan)
    pt = np.array(pt, float)
    ok = ~np.isnan(pt)
    holm = np.full(len(pt), np.nan)
    if ok.any():                                           # Holm step-down correction across the tasks
        idx = np.argsort(pt[ok]); k = ok.sum()
        adj = np.maximum.accumulate(np.minimum(1, (k - np.arange(k)) * pt[ok][idx]))
        tmp = np.empty(k); tmp[idx] = adj; holm[ok] = tmp
    return dict(model=model_a, baseline=model_b, metric=metric, tasks=len(common), mean_diff=diff.mean(),
                wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), wilcoxon_p=wp,
                sig_wins=int(((holm < 0.05) & (diff.values > 0)).sum()),
                sig_losses=int(((holm < 0.05) & (diff.values < 0)).sum()))


pair = []
for suite, tasks in SUITES.items():
    for m in models:
        if m != args.ours:
            r = compare(args.ours, m, tasks)
            if r:
                pair.append(dict(suite=suite, **r))
for m in models:                                           # accuracy that holds on both strands of the same DNA
    if m != args.ours:
        r = compare(args.ours, m, NT_SYM, "both")
        if r:
            pair.append(dict(suite="NT 12 sym.", **r))
# controlled comparison: same architecture (duplex) and budget, only exact equivariance vs augmentation differs
for suite, tasks in SUITES.items():
    for metric in (["fwd", "both"] if suite == "NT 12 sym." else ["fwd"]):
        r = compare("MirrorDNA duplex", "Chunker + RC aug.", tasks, metric)
        if r:
            pair.append(dict(suite=suite, **r))
pair = pd.DataFrame(pair)
pair.to_csv(out / "pairwise.csv", index=False)
report += ["## Pairwise comparisons", "",
           "metric fwd = forward MCC, both = MCC counting only predictions correct on both strands; mean_diff = model − baseline (task means); wins/losses over tasks; Wilcoxon signed-rank over tasks "
           "(two-sided); sig_wins / sig_losses = tasks where a per-task Welch t-test over seeds is significant after "
           "Holm correction (p < 0.05). With 3 seeds per task, per-task tests have little power.", "",
           pair.assign(wilcoxon_p=pair.wilcoxon_p.map(lambda v: f"{v:.2g}")).round(4).to_markdown(index=False), ""]

# ------------------------------------------------------------------ strand metrics, 12 strand-symmetric tasks
strand = []
for m in models:
    d = df[(df.model == m) & df.task.isin(NT_SYM)].dropna(subset=["rc"])
    if d.task.nunique() < len(NT_SYM):
        continue
    seeds = [s for s in sorted(d.seed.unique()) if d[d.seed == s].task.nunique() == len(NT_SYM)]
    row = dict(model=m)
    for c in ["fwd", "rc", "both", "flips", "avg"]:
        ps = [d[d.seed == s][c].mean() for s in seeds]
        row[c] = np.mean(ps); row[c + "_sd"] = np.std(ps, ddof=1) if len(ps) > 1 else np.nan
    strand.append(row)
strand = pd.DataFrame(strand)
strand.to_csv(out / "strand.csv", index=False)

# ------------------------------------------------------------------ supplement: every run, and strand metrics per task
df.sort_values(["model", "task", "seed"]).to_csv(out / "all_runs.csv", index=False)     # one row per model/task/seed
GB_SYM = [t for t in GB if t not in ("demo_coding_vs_intergenomic_seqs", "human_nontata_promoters")]
st_rows = []
for t in NT_SYM + GB_SYM:
    for m in models:
        d = df[(df.model == m) & (df.task == t)].dropna(subset=["rc"])
        if len(d):
            st_rows.append(dict(task=t, model=m, n=len(d), **{c: d[c].mean() for c in ["fwd", "rc", "both", "flips", "avg"]},
                                **{c + "_sd": d[c].std(ddof=1) if len(d) > 1 else np.nan for c in ["fwd", "rc", "both", "flips", "avg"]}))
pd.DataFrame(st_rows).to_csv(out / "per_task_strand.csv", index=False)

# ------------------------------------------------------------------ test-time averaging over both strands (reviewer control)
# averaging the class probabilities of a sequence and its reverse complement makes ANY model strand-invariant, at twice
# the inference cost; mcc_rc_avg is that averaged prediction. Only meaningful for strand-symmetric tasks.
SYM = {"NT 12 sym.": NT_SYM, "GB 7 sym.": GB_SYM, "All 19 sym.": NT_SYM + GB_SYM}
avg_rows, avg_tests = [], []
for g, tasks in SYM.items():
    for m in models:
        d = df[(df.model == m) & df.task.isin(tasks)].dropna(subset=["avg"])
        if d.task.nunique() < len(tasks):
            continue
        seeds = [x for x in sorted(d.seed.unique()) if d[d.seed == x].task.nunique() == len(tasks)]
        pf = [d[d.seed == x].fwd.mean() for x in seeds]; pa = [d[d.seed == x].avg.mean() for x in seeds]
        avg_rows.append(dict(group=g, model=m, fwd=np.mean(pf), fwd_sd=np.std(pf, ddof=1) if len(pf) > 1 else np.nan,
                             avg=np.mean(pa), avg_sd=np.std(pa, ddof=1) if len(pa) > 1 else np.nan))
        if m != args.ours:
            r = compare(args.ours, m, tasks, "avg")
            if r:
                avg_tests.append(dict(group=g, **r))
avg_df, avg_tests = pd.DataFrame(avg_rows), pd.DataFrame(avg_tests)
avg_df.to_csv(out / "rc_averaged.csv", index=False)
avg_tests.to_csv(out / "rc_averaged_tests.csv", index=False)
report += ["## Test-time averaging over both strands (strand-symmetric tasks)", "",
           "fwd = forward MCC; avg = MCC of the prediction averaged over the sequence and its reverse complement "
           "(strand-invariant for every model, at twice the inference cost). Tests: MirrorDNA (ours) vs each model, "
           "both averaged, Wilcoxon over tasks.", "",
           avg_df.round(3).to_markdown(index=False) if len(avg_df) else "(no averaged results found)", "",
           avg_tests.assign(wilcoxon_p=avg_tests.wilcoxon_p.map(lambda v: f"{v:.2g}")).round(4).to_markdown(index=False)
           if len(avg_tests) else "", ""]
report += ["## Both strands, 12 strand-symmetric NT tasks (mean ± SD over seeds)", "",
           strand.round(3).to_markdown(index=False), ""]


# ------------------------------------------------------------------ LaTeX
def tex_cell(v, best):
    s = ms(v)
    return f"\\textbf{{{s}}}" if best else s


tex = ["% Generated by stats_tables.py -- do not edit by hand. Needs \\usepackage{booktabs}.", ""]
for suite in ["NT-18", "GB-9", "Gene strand"]:
    tasks = SUITES[suite]
    cols = [m for m in models if df[(df.model == m) & df.task.isin(tasks)].shape[0]]
    tex += [f"% ---- {suite}", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
            "Task & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for t in tasks:
        means = {m: df[(df.model == m) & (df.task == t)].fwd.mean() for m in cols}
        best = max((v for v in means.values() if not np.isnan(v)), default=np.nan)
        cells = [tex_cell(df[(df.model == m) & (df.task == t)].fwd.values, means[m] == best)
                 if not np.isnan(means[m]) else "--" for m in cols]
        tex.append(t.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    s = summ[summ.suite == suite].set_index("model")
    tex.append("\\midrule")
    tex.append("Mean & " + " & ".join(f"{s.loc[m, 'mean']:.3f} ± {s.loc[m, 'sd']:.3f}" if m in s.index else "--"
                                      for m in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
(out / "stats_tables.tex").write_text("\n".join(tex).replace("±", "$\\pm$"), encoding="utf-8")
(out / "stats_report.md").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
print(f"\nwritten: {out}/ (per_task_*.csv, per_task_strand.csv, all_runs.csv, summary.csv, pairwise.csv, strand.csv, stats_tables.tex, stats_report.md)")
