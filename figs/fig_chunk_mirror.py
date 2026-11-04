import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import rcchunk as rk
from figstyle import COLORS, OURS, plt, save

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="+", help="model folders; the first is ours, the others are baselines")
p.add_argument("--labels", nargs="+", default=None, help="display names, same order as folders")
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--chrom", default="chr8")
p.add_argument("--pos", type=int, default=None, help="start of the 1-kb window for panel (a); default: random")
p.add_argument("--show", type=int, default=80, help="bp of the window drawn in panel (a) (<=90 shows the bases)")
p.add_argument("--n", type=int, default=2000, help="windows for panel (b)")
p.add_argument("--window", type=int, default=1024)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--motif-summary", default=None,
               help="fig_motifs.py summary CSV: adds panel (c), motif occurrences tokenized differently on the two strands")
p.add_argument("--out", default="fig_chunk_mirror")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
labels = args.labels or [OURS] + ["Chunker + RC aug."] * (len(args.folders) - 1)
L = args.window


@torch.no_grad()
def cuts(model, X):
    """Boolean cuts per gap (gap j = between base j and j+1), forward strand and reverse strand mirrored back."""
    F, R = [], []
    for i in range(0, len(X), args.bs):
        x = torch.from_numpy(X[i:i + args.bs]).long().to(dev)
        F.append(model(x, decode=False)["cut"].bool().cpu().numpy())
        R.append(model(rk.rc_tokens(x), decode=False)["cut"].bool().flip(1).cpu().numpy())
    return np.concatenate(F), np.concatenate(R)


tok = rk.load_chrom(args.genome_dir, args.chrom)
if tok is None:
    raise SystemExit(f"{args.genome_dir}/{args.chrom}.tok.npy not found")
rng = np.random.default_rng(0)
X = []
while len(X) < args.n:
    s = int(rng.integers(0, len(tok) - L))
    w = np.array(tok[s:s + L], np.uint8)
    if (w == rk.N).mean() == 0:
        X.append(w)
X = np.stack(X)
if args.pos is None:                         # a typical-looking window: no long homopolymer, ordinary GC
    def typical(w):
        mid = w[(L - args.show) // 2:(L + args.show) // 2]
        best, cur = 1, 1
        for i in range(1, len(mid)):
            cur = cur + 1 if mid[i] == mid[i - 1] else 1
            best = max(best, cur)
        gc = np.isin(mid, [1, 2]).mean()
        return best <= 4 and 0.4 <= gc <= 0.6
    show = next((w for w in X if typical(w)), X[0])
else:
    show = np.array(tok[args.pos:args.pos + L], np.uint8)

models = []
for folder, lab in zip(args.folders, labels):
    for variant, mcfg, m, name in rk.load_pretrained(folder, dev):
        models.append((lab, m))
        print(f"{lab}: {name} from {folder}")

rows, jac = [], {}
for lab, m in models:
    f, r = cuts(m, X)
    j = (f & r).sum(1) / np.maximum((f | r).sum(1), 1)
    jac[lab] = j
    sf, sr = cuts(m, show[None])
    rows.append((lab, sf[0], sr[0]))
    print(f"{lab:22s} strand Jaccard mean {j.mean():.3f} | windows fully identical {np.mean(j == 1):.1%} | "
          f"bp per chunk {L / (1 + f.sum(1).mean()):.2f}")
with open(args.out + ".csv", "w") as fh:
    fh.write("model,jaccard_mean,jaccard_median,frac_windows_identical,bp_per_chunk\n")
    for lab, m in models:
        fh.write(f"{lab},{jac[lab].mean():.4f},{np.median(jac[lab]):.4f},{np.mean(jac[lab] == 1):.4f},\n")

# ---------------------------------------------------------------- panel (a): one window, both strands
S = args.show
o = (L - S) // 2                                     # draw the middle of the window
base = "ACGTN"
comp = "TGCAN"
fig = plt.figure(figsize=(7.2, 1.3 + 0.62 * len(rows) + 1.6))
gs = fig.add_gridspec(2, 2 if args.motif_summary else 1, height_ratios=[0.62 * len(rows) + 0.4, 1.4], hspace=0.75,
                      width_ratios=[1, 1] if args.motif_summary else [1], wspace=0.75)
a = fig.add_subplot(gs[0, :])
y = 0
yt, yl = [], []
for lab, cf, cr in rows:
    col = COLORS.get(lab, "0.4")
    for strand, c, seqchars in [("+", cf, [base[v] for v in show[o:o + S]]),
                                ("−", cr, [comp[v] for v in show[o:o + S]])]:
        gaps = np.where(c[o:o + S - 1])[0]                 # cut after base o+g
        edges = [0] + list(gaps + 1) + [S]
        for k in range(len(edges) - 1):
            a.add_patch(plt.Rectangle((edges[k], y - 0.36), edges[k + 1] - edges[k], 0.72,
                                      facecolor=col if k % 2 == 0 else "white", alpha=0.28 if k % 2 == 0 else 1,
                                      edgecolor=col, lw=0.7))
        if S <= 90:
            for i, ch in enumerate(seqchars):
                a.text(i + 0.5, y, ch, ha="center", va="center", fontsize=4.5, family="monospace")
        yt.append(y); yl.append(f"{lab}  {strand} strand")
        y -= 1
    agree = (cf[o:o + S - 1] == cr[o:o + S - 1]).all()
    a.text(S + 1, y + 1.5, "identical" if agree else "differ", va="center", fontsize=7,
           color=col, fontweight="bold")
    y -= 0.5
a.set_xlim(0, S + 14); a.set_ylim(y + 0.3, 0.6)
a.set_yticks(yt); a.set_yticklabels(yl, fontsize=6.5)
a.set_xlabel("position (bp); the reverse strand is drawn in forward coordinates, read right to left")
a.spines[["left"]].set_visible(False); a.tick_params(axis="y", length=0)
a.set_title("(a) chunks of the same DNA read on its two strands", loc="left")

b = fig.add_subplot(gs[1, 0])
labs = [lab for lab, _ in models]
yy = np.arange(len(labs))
mean = [jac[l].mean() for l in labs]
ident = [np.mean(jac[l] == 1) for l in labs]
b.barh(yy, mean, 0.55, color=[COLORS.get(l, "0.4") for l in labs])
for i in range(len(labs)):
    b.text(mean[i] + 0.02, yy[i], f"{mean[i]:.2f} ({ident[i]:.0%} identical)", va="center", fontsize=6.5,
           fontweight="bold" if labs[i] == OURS else None)
b.set_yticks(yy); b.set_yticklabels(labs, fontsize=6.5); b.invert_yaxis()
b.set_xlim(0, 1.0 if not args.motif_summary else 1.0)
b.set_xticks([0, 0.5, 1.0])
b.set_xlabel(f"mean Jaccard of the two strands' cut sets\n({len(X):,} random 1-kb windows)")
b.set_title("(b) same cuts on both strands?", loc="left")
b.set_xlim(0, 1.9)
b.spines["bottom"].set_bounds(0, 1)
if args.motif_summary:                       # panel (c): the same question for motif occurrences, all tokenizers
    import pandas as pd
    m = pd.read_csv(args.motif_summary, index_col=0)
    c = fig.add_subplot(gs[1, 1])
    names = list(m.index); yy = np.arange(len(names))
    v = m["strand_inconsistent"].values
    c.barh(yy, v, 0.55, color=[COLORS.get(n, "0.5") for n in names])
    for i, x in enumerate(v):
        c.text(x + 0.02, i, f"{x:.0%}", va="center", fontsize=6.5, fontweight="bold" if names[i] == OURS else None)
    c.set_yticks(yy); c.set_yticklabels(names, fontsize=6.5); c.invert_yaxis()
    c.set_xlim(0, 1.2); c.set_xticks([0, 0.5, 1.0]); c.spines["bottom"].set_bounds(0, 1)
    c.set_xlabel("JASPAR motif occurrences tokenized\ndifferently on the two strands")
    c.set_title("(c) same tokens for a motif on both strands?", loc="left")
save(fig, args.out)
