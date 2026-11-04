import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--into", default=None, help="supplementary.tex whose generated-tables block is replaced")
args = ap.parse_args()
A = Path("paper_assets")
OUT = A / "supplement_tables.tex"
SHORT = {"NT-v2": "NT-v2", "DNABERT-2": "DNABERT-2", "HyenaDNA-small": "HyenaDNA", "HyenaDNA-tiny-1k": "HyenaDNA-1k",
         "Caduceus-PS-1k": "Cad.-PS", "Caduceus-PS-1k (oriented head)": "Cad.-PS+or.", "Caduceus-PH-1k": "Cad.-PH", "Chunker + RC aug.": "Chunk.+aug.",
         "MirrorDNA duplex": "MDNA-dup.", "MirrorDNA, fixed cuts (ablation)": "MDNA-fixed", "MirrorDNA (ours)": "MirrorDNA"}
TASK = {"demo_coding_vs_intergenomic_seqs": "coding vs intergenomic", "demo_human_or_worm": "human or worm",
        "drosophila_enhancers_stark": "Drosophila enhancers", "dummy_mouse_enhancers_ensembl": "mouse enhancers",
        "human_enhancers_cohn": "enhancers (Cohn)", "human_enhancers_ensembl": "enhancers (Ensembl)",
        "human_ensembl_regulatory": "Ensembl regulatory", "human_nontata_promoters": "non-TATA promoters",
        "human_ocr_ensembl": "open chromatin", "bio_gene_strand_tss": "gene strand, TSS",
        "bio_gene_strand_donor": "gene strand, splice donor"}
out = []


def tex(s):
    return (str(s).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
            .replace("±", "$\\pm$").replace("×", "$\\times$").replace("−", "$-$").replace("**", ""))


def task(t):
    return tex(TASK.get(t, t))


def cell(s):
    """'0.495 ± 0.026 (n=2)' -> $.495_{\\pm.026}$ (n=2)"""
    s = str(s)
    m = re.match(r"\s*(-?\d*\.\d+)\s*±\s*(\d*\.\d+)(.*)", s)
    if not m:
        return tex(s)
    f = lambda v: v[1:] if v.startswith("0.") else (v.replace("-0.", "-.") if v.startswith("-0.") else v)
    return f"${f(m.group(1))}_{{\\pm{f(m.group(2))}}}${tex(m.group(3))}"


def ms(m, s, d=3, pct=False):
    if pd.isna(m):
        return "--"
    if pct:
        return f"{100 * m:.1f}" + ("" if pd.isna(s) else f"$_{{\\pm{100 * s:.1f}}}$")
    v = f"{m:.{d}f}".replace("0.", ".", 1) if abs(m) < 1 else f"{m:.{d}f}"
    sd = "" if pd.isna(s) else f"_{{\\pm{s:.{d}f}}}".replace("0.", ".", 1)
    return f"${v}{sd}$"


def table(caption, label, header, rows, align, resize=True, note=None):
    body = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    tab = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    tab += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    body += ([r"\resizebox{\textwidth}{!}{%"] + tab + ["}"]) if resize else tab
    if note:
        body.append(rf"\par\smallskip{{\footnotesize {note}}}")
    body.append(r"\end{table}")
    out.extend(body + [""])


def missing(what):
    out.append(rf"\noindent\textcolor{{red}}{{[{tex(what)} not found: rerun make\_paper\_assets.py]}}\par" + "\n")


def md_tables(path):
    """All markdown tables of a report: list of (heading, header, rows)."""
    res, head, cur = [], None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            head = line.lstrip("# ").strip()
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = (head, cells, [])
            elif not set("".join(cells)) <= set("-: "):
                cur[2].append(cells)
        elif cur is not None:
            res.append(cur); cur = None
    if cur is not None:
        res.append(cur)
    return res


def per_task(fname, title, label):
    f = A / fname
    if not f.exists():
        return missing(fname)
    d = pd.read_csv(f, index_col=0)
    cols = list(d.columns)
    table(title, label, ["Task"] + [tex(SHORT.get(c, c)) for c in cols],
          [[task(t)] + [cell(d.loc[t, c]) for c in cols] for t in d.index], "l" + "c" * len(cols))


# ================================================================================================ accuracy
out.append(r"\subsection{Accuracy per task}" + "\n")
out.append("Forward MCC on the official test sets, mean $\\pm$ SD over three seeds (one fine-tuning protocol for all "
           "models; Section~S2). Model names: Cad.-PS/PH = Caduceus-PS/PH-1k; Chunk.+aug. = learned chunker with "
           "reverse-complement augmentation; MDNA-dup. = MirrorDNA-duplex; HyenaDNA = HyenaDNA-small.\n")
per_task("per_task_NT18.csv", "Nucleotide Transformer benchmark (revised), 18 tasks: forward MCC.", "tab:s-nt")
per_task("per_task_GB9.csv", "Genomic Benchmarks, 9 tasks: forward MCC.", "tab:s-gb")
per_task("per_task_Gene_strand.csv", "Gene-strand tasks (1-kb windows): forward MCC. MirrorDNA with base readout, "
         "as in the main results.", "tab:s-gene")

f = A / "summary.csv"
if f.exists():
    s = pd.read_csv(f)
    suites = list(dict.fromkeys(s.suite))
    models = list(dict.fromkeys(s.model))
    rows = []
    for m in models:
        r = [tex(m)]
        for su in suites:
            x = s[(s.model == m) & (s.suite == su)]
            r.append(ms(x["mean"].iloc[0], x["sd"].iloc[0]) if len(x) else "--")
        rows.append(r)
    table("Suite averages (forward MCC; SD of the per-seed suite average).", "tab:s-summary",
          ["Model"] + [tex(x) for x in suites], rows, "l" + "c" * len(suites), resize=False)
else:
    missing("summary.csv")

# ================================================================================================ strands
out.append(r"\subsection{Accuracy on both strands, per task}" + "\n")
f = A / "per_task_strand.csv"
if f.exists():
    d = pd.read_csv(f)
    models = [m for m in SHORT if m in set(d.model)]
    tasks = list(dict.fromkeys(d.task))
    for c, what, pct in [("rc", "MCC on the reverse-complemented test set", False),
                         ("both", "MCC counting only predictions correct on both strands", False),
                         ("flips", "share of test sequences (\\%) whose prediction changes with the strand", True)]:
        rows = []
        for t in tasks:
            r = [task(t)]
            for m in models:
                x = d[(d.task == t) & (d.model == m)]
                r.append(ms(x[c].iloc[0], x[c + "_sd"].iloc[0], pct=pct) if len(x) else "--")
            rows.append(r)
        table(f"Strand-symmetric tasks (12 NT, 7 Genomic Benchmarks): {what}; mean $\\pm$ SD over seeds.",
              f"tab:s-{c}", ["Task"] + [tex(SHORT[m]) for m in models], rows, "l" + "c" * len(models))
else:
    missing("per_task_strand.csv")

# ================================================================================================ tests
f = A / "pairwise.csv"
if f.exists():
    d = pd.read_csv(f)
    d = d[d.model.str.contains("ours")]
    rows = [[tex(r.suite), tex(r.baseline), "both strands" if r.metric == "both" else "forward", f"{r.mean_diff:+.3f}",
             f"{int(r.wins)}/{int(r.losses)}", f"{r.wilcoxon_p:.2g}", f"{int(r.sig_wins)}/{int(r.sig_losses)}"]
            for r in d.itertuples()]
    table("MirrorDNA against every model: mean MCC difference over tasks (MirrorDNA minus baseline), tasks won/lost, "
          "two-sided Wilcoxon signed-rank test over tasks, and tasks with a significant per-task Welch test over seeds "
          "after Holm correction (won/lost).", "tab:s-tests",
          ["Suite", "Baseline", "Metric", "$\\Delta$", "W/L", "$p$", "sig. W/L"], rows, "lllrrrr", resize=False)
else:
    missing("pairwise.csv")

# ================================================================================================ label efficiency
out.append(r"\subsection{Exact equivariance versus augmentation at limited labels}" + "\n")
f = A / "lowlabel_stats.md"
if f.exists():
    for head, hdr, rows in md_tables(f):
        table(f"Label efficiency, {tex(head)}: MirrorDNA-duplex (exact equivariance) against the augmented chunker "
              "(same architecture and pretraining). Mean $\\pm$ SD over seeds; Wilcoxon signed-rank test over tasks.",
              "tab:s-low-" + re.sub(r"\W+", "", head.lower())[:12], [tex(h) for h in hdr],
              [[tex(c) for c in r] for r in rows], "ll" + "c" * (len(hdr) - 2), resize=False)
else:
    missing("lowlabel_stats.md")
f = A / "lowlabel_per_task.csv"
if f.exists():
    d = pd.read_csv(f)
    labs = list(dict.fromkeys(d["labels"]))
    rows = []
    for t in dict.fromkeys(d.task):
        r = [task(t)]
        for b in labs:
            for m in ["equivariant", "aug"]:
                x = d[(d.task == t) & (d["labels"] == b) & (d.model == m)]
                r.append(ms(x.fwd.iloc[0], x.fwd_sd.iloc[0]) if len(x) else "--")
        rows.append(r)
    hdr = ["Task"] + [f"{b}, {m}" for b in labs for m in ["equiv.", "aug."]]
    table("Label efficiency per task: forward MCC of MirrorDNA-duplex (equiv.) and the augmented chunker (aug.) at "
          "each number of training labels.", "tab:s-low-task", hdr, rows, "l" + "c" * (len(hdr) - 1))
else:
    missing("lowlabel_per_task.csv")

# ================================================================================================ length
out.append(r"\subsection{Longer inputs}" + "\n")
f = A / "compare_long.md"
if f.exists():
    for head, hdr, rows in md_tables(f):
        table("Gene strand at increasing window length (MCC, mean $\\pm$ SD over seeds; $n$ = seeds). Rows marked "
              "+base use the base readout (main results); the length experiment uses chunk features only.",
              "tab:s-long", [tex(h) for h in hdr], [[tex(c) for c in r] for r in rows], "ll" + "c" * (len(hdr) - 2))
else:
    missing("compare_long.md")

# ================================================================================================ DeepSTARR
out.append(r"\subsection{Regression: DeepSTARR}" + "\n" + r"\label{sec:s-deepstarr}" + "\n")
f = A / "deepstarr.csv"
if f.exists():
    d = pd.read_csv(f, index_col=0)
    rows = [[tex(m), f"{r.params_M:.2f}M", ms(r.pearson_dev, r.pearson_dev_sd), ms(r.pearson_hk, r.pearson_hk_sd),
             ms(r.pearson_mean, r.pearson_mean_sd), ms(r.pearson_mean_rc, r.pearson_mean_rc_sd),
             ms(r.get("pearson_mean_avg", np.nan), r.get("pearson_mean_avg_sd", np.nan)),
             ms(r.strand_r, r.strand_r_sd), ms(r.strand_gap, r.strand_gap_sd), ms(r.strand_gap_big, r.strand_gap_big_sd, pct=True)]
            for m, r in d.iterrows()]
    table("DeepSTARR enhancer-activity regression (official test set; mean $\\pm$ SD over 3 seeds). Pearson's $r$ for "
          "developmental (Dev) and housekeeping (Hk) activity, their mean on the test set as given and "
          "reverse-complemented (RC), and of the prediction averaged over both strands; strand $r$: correlation of the predictions for a sequence and its reverse "
          "complement; gap: mean absolute difference of the two predictions in target SD units; $>$0.5 SD: \\% of "
          "sequences whose two predictions differ by more than half an SD. Published DeepSTARR network: Dev 0.68, "
          "Hk 0.74.", "tab:s-deepstarr",
          ["Model", "Params", "Dev $r$", "Hk $r$", "mean $r$", "RC mean $r$", "averaged $r$", "strand $r$", "gap (SD)",
           "$>$0.5 SD"], rows, "lrcccccccc")
else:
    missing("deepstarr.csv")

# ================================================================================================ tokenization
out.append(r"\subsection{Tokenization of the two strands}" + "\n")
f, g = A / "fig_motif_strand_summary.csv", A / "fig_tokens.csv"
if f.exists():
    d = pd.read_csv(f, index_col=0)
    jac = pd.read_csv(g, index_col=0) if g.exists() else None
    rows = []
    for m, r in d.iterrows():
        j = jac.loc[m, "jaccard_mean"] if jac is not None and m in jac.index else np.nan
        rows.append([tex(m), f"{r.bp_per_token:.2f}", f"{r.pieces_per_motif:.2f}", f"{r.vs_random_cuts:.2f}",
                     f"{100 * r.kept_whole:.1f}", f"{100 * r.strand_inconsistent:.1f}",
                     "--" if pd.isna(j) else f"{j:.3f}"])
    table("Tokenization of JASPAR~2024 motif occurrences (chromosomes 8 and 9): mean token length, tokens per motif "
          "occurrence, the same relative to random cuts of equal density, \\% of occurrences kept in one token, \\% "
          "tokenized differently on the two strands, and the mean Jaccard index of the two strands' cut sets in "
          "2{,}000 random 1-kb windows.", "tab:s-motif",
          ["Tokenizer", "bp/token", "tokens/motif", "vs.\\ random", "whole (\\%)", "differ (\\%)", "Jaccard"],
          rows, "lcccccc", resize=False)
else:
    missing("fig_motif_strand_summary.csv")

f = A / "bio_boundaries_report.md"
if f.exists():
    for head, hdr, rows in md_tables(f)[:1]:
        rows = [[re.sub(r"\s*\[.*?\]", "", c) for c in r] for r in rows]
        table("Router cut rate at gene-structure sites (RefSeq, 1-kb windows centred on the site) relative to the "
              "background rate, and the strand agreement of the cuts.", "tab:s-bound",
              [tex(h) for h in hdr], [[tex(c) for c in r] for r in rows], "ll" + "c" * (len(hdr) - 2))
else:
    missing("bio_boundaries_report.md")

f = A / "splice_tokens_report.md"
if f.exists():
    for head, hdr, rows in md_tables(f):
        table(f"Tokens formed at {tex(head)}: the chunk containing the first base after the site, read on the "
              "gene's strand, with the model's cuts and with the same cuts circularly shifted by 100--900 bp; offset "
              "= chunk start relative to the site.", "tab:s-tok-" + re.sub(r"\W+", "", head.lower())[:10],
              [tex(h) for h in hdr], [[tex(c) for c in r] for r in rows], "lcccc", resize=False)
else:
    missing("splice_tokens_report.md")

text = "\n".join(out)
text = re.sub(r"(?<!\A)\\subsection\{", r"\\clearpage\n\\subsection{", text).replace("\\clearpage\n", "", 1)
OUT.write_text("% generated by make_supplement.py; do not edit by hand\n" + text, encoding="utf-8")
print(f"wrote {OUT}")
if args.into:
    f = Path(args.into)
    s = f.read_text(encoding="utf-8")
    b, e = "% BEGIN GENERATED TABLES", "% END GENERATED TABLES"
    i, j = s.index(b), s.index(e)
    i = s.index("\n", i) + 1
    f.write_text(s[:i] + text.strip() + "\n" + s[j:], encoding="utf-8")
    print(f"updated the tables in {f}")
