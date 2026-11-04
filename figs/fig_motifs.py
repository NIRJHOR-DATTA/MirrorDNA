import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import rcchunk as rk
from figstyle import COLORS, OURS, plt, save

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="*", help="model folders; the first is ours")
p.add_argument("--labels", nargs="+", default=None)
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--jaspar", default=None)
p.add_argument("--chroms", nargs="+", default=["chr8", "chr9"])
p.add_argument("--windows", type=int, default=3000)
p.add_argument("--window", type=int, default=1024)
p.add_argument("--rel", type=float, default=0.85, help="relative PWM score for a motif hit")
p.add_argument("--max-hits", type=int, default=300, help="occurrences used per motif")
p.add_argument("--min-hits", type=int, default=30, help="motifs with fewer occurrences are skipped")
p.add_argument("--top", type=int, default=25)
p.add_argument("--kmer", type=int, default=6)
p.add_argument("--no-bpe", action="store_true")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--only-strand", action="store_true", help="draw only the strand-consistency panel")
p.add_argument("--out", default="fig_motifs")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
L = args.window
rng = np.random.default_rng(0)
labels = args.labels or ([OURS] + ["Chunker + RC aug."] * (len(args.folders) - 1))


# ---------------------------------------------------------------- data
def sample_windows():
    toks = [(c, rk.load_chrom(args.genome_dir, c)) for c in args.chroms]
    toks = [(c, t) for c, t in toks if t is not None]
    if not toks:
        raise SystemExit("no chromosomes found in " + args.genome_dir)
    X = []
    while len(X) < args.windows:
        c, t = toks[int(rng.integers(len(toks)))]
        s = int(rng.integers(0, len(t) - L))
        w = np.array(t[s:s + L], np.uint8)
        if (w == rk.N).sum() == 0:
            X.append(w)
    return np.stack(X)


def load_jaspar():
    f = Path(args.jaspar) if args.jaspar else Path(args.genome_dir) / "JASPAR2024_CORE_vertebrates_nr.txt"
    if not f.exists():
        raise SystemExit(f"{f} not found: run bio_sae.py once (it downloads JASPAR) or pass --jaspar FILE")
    motifs, cur = [], None
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            cur = [line[1:].split()[0], " ".join(line[1:].split()[1:]), []]
            motifs.append(cur)
        elif line and cur is not None and line[0] in "ACGT":
            cur[2].append([float(v) for v in line[line.index("[") + 1:line.index("]")].split()])
    out = []
    for mid, name, rows in motifs:
        pfm = np.array(rows).T + 0.8
        pwm = np.log2(pfm / pfm.sum(1, keepdims=True) / 0.25)
        if 5 <= len(pwm) <= 30:
            out.append((mid, name, pwm))
    return out


OH = None


@torch.no_grad()
def hits(X, pwm):
    """(window, start) of occurrences on either strand (a motif on the - strand occupies the same bases)."""
    global OH
    if OH is None:
        OH = F.one_hot(torch.from_numpy(X).long().to(dev), 5)[..., :4].permute(0, 2, 1).float()
    oh, found = OH, []
    for w in (pwm, pwm[::-1, ::-1].copy()):
        k = torch.tensor(w.T, dtype=torch.float32, device=dev)[None]
        lo, hi = w.min(1).sum(), w.max(1).sum()
        s = F.conv1d(oh, k)[:, 0]
        i, j = torch.where((s - lo) / (hi - lo) >= args.rel)
        found += list(zip(i.tolist(), j.tolist()))
    return sorted(set(found))


# ---------------------------------------------------------------- segmentations: cut[g] = boundary between base g and g+1
@torch.no_grad()
def model_cuts(model, X):
    Fw, Rv = [], []
    for i in range(0, len(X), args.bs):
        x = torch.from_numpy(X[i:i + args.bs]).long().to(dev)
        Fw.append(model(x, decode=False)["cut"].bool().cpu().numpy())
        Rv.append(model(rk.rc_tokens(x), decode=False)["cut"].bool().flip(1).cpu().numpy())
    return np.concatenate(Fw), np.concatenate(Rv)


def string_cuts(token_lists):
    out = np.zeros((len(token_lists), L - 1), bool)
    for n, toks in enumerate(token_lists):
        ends = np.cumsum([len(t) for t in toks])[:-1] - 1
        if len(toks) and ends.size and ends[-1] < L - 1:
            out[n, ends] = True
    return out


def kmer_cuts(X):
    c = np.zeros((len(X), L - 1), bool)
    c[:, args.kmer - 1::args.kmer] = True
    return c, np.ascontiguousarray(c[:, ::-1])           # reverse strand tokenized from its own start


def bpe_cuts(X):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    fw = ["".join("ACGTN"[v] for v in x) for x in X]
    rv = ["".join("TGCAN"[v] for v in x[::-1]) for x in X]
    clean = lambda t: t.replace("##", "").replace("▁", "")
    tf = [[clean(t) for t in tok.tokenize(s)] for s in fw]
    tr = [[clean(t) for t in tok.tokenize(s)] for s in rv]
    bad = sum(sum(map(len, t)) != L for t in tf + tr)
    if bad:
        print(f"warning: {bad} BPE tokenizations do not cover the window exactly")
    return string_cuts(tf), string_cuts(tr)[:, ::-1]


def pieces(cut, w, s, k):
    return 1 + int(cut[w, s:s + k - 1].sum())            # gaps strictly inside the occurrence


# ---------------------------------------------------------------- run
X = sample_windows()
motifs = load_jaspar()
print(f"{len(X)} windows from {args.chroms}, {len(motifs)} JASPAR motifs")
segs = {}
for folder, lab in zip(args.folders, labels):
    for variant, mcfg, m, name in rk.load_pretrained(folder, dev):
        segs[lab] = model_cuts(m, X)
        print(f"{lab}: {name}, {L / (1 + segs[lab][0].sum(1).mean()):.2f} bp per chunk")
        del m
if not args.no_bpe:
    segs["BPE (DNABERT-2)"] = bpe_cuts(X)
segs[f"{args.kmer}-mer (NT)"] = kmer_cuts(X)
RATE = {k: float(f.mean()) for k, (f, _) in segs.items()}         # boundaries per gap
for k, (f, _) in segs.items():
    print(f"{k:20s} mean token length {L / (1 + f.sum(1).mean()):.2f} bp")

rows = []
for mid, name, pwm in motifs:
    hs = hits(X, pwm)
    if len(hs) < args.min_hits:
        continue
    hs = [hs[i] for i in rng.permutation(len(hs))[:args.max_hits]]
    k = len(pwm)
    r = dict(id=mid, name=name, length=k, n=len(hs))
    for lab, (cf, cr) in segs.items():
        pf = np.array([pieces(cf, w, s, k) for w, s in hs])
        r[lab + "|vs_random"] = pf.mean() / (1 + (k - 1) * RATE[lab])     # 1 = like random cuts at the same rate
        same = np.array([(cf[w, s:s + k - 1] == cr[w, s:s + k - 1]).all() for w, s in hs])
        r[lab + "|pieces"] = pf.mean()
        r[lab + "|whole"] = (pf == 1).mean()
        r[lab + "|inconsistent"] = 1 - same.mean()
    rows.append(r)
import pandas as pd
df = pd.DataFrame(rows).sort_values("n", ascending=False)
df.to_csv(args.out + ".csv", index=False)
names = list(segs)
summ = pd.DataFrame({lab: dict(pieces_per_motif=df[lab + "|pieces"].mean(), vs_random_cuts=df[lab + "|vs_random"].mean(),
                               bp_per_token=1 / (RATE[lab] + 1 / L), kept_whole=df[lab + "|whole"].mean(),
                               strand_inconsistent=df[lab + "|inconsistent"].mean()) for lab in names}).T
summ.round(4).to_csv(args.out + "_summary.csv")
print(f"\n{len(df)} motifs with >= {args.min_hits} occurrences (mean over motifs):")
print(summ.round(3).to_string())

# ---------------------------------------------------------------- figure
if args.only_strand:
    fig, b = plt.subplots(figsize=(3.5, 2.0))
else:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    top = df.head(args.top)
    x = np.arange(len(top)); w = 0.8 / len(names)
    for i, lab in enumerate(names):
        ax[0].bar(x + (i - (len(names) - 1) / 2) * w, top[lab + "|vs_random"], w, color=COLORS.get(lab, "0.5"),
                  label=f"{lab}: {summ.loc[lab, 'vs_random_cuts']:.2f}")
    ax[0].axhline(1, color="0.6", lw=0.6, ls="--")
    ax[0].set_xticks(x); ax[0].set_xticklabels([n[:10] for n in top.name], rotation=70, fontsize=5.5)
    ax[0].set_ylabel("pieces per motif occurrence /\nexpected for random cuts at the same rate")
    ax[0].set_title(f"(a) motif fragmentation, {args.top} most frequent JASPAR motifs (<1: cuts avoid motifs)",
                    loc="left", fontsize=8)
    ax[0].legend(frameon=False, ncol=2, fontsize=6, loc="upper left", title="mean over all motifs", title_fontsize=6)
    b = ax[1]

yy = np.arange(len(names))
inc = [summ.loc[l, "strand_inconsistent"] for l in names]
b.barh(yy, inc, 0.6, color=[COLORS.get(l, "0.5") for l in names])
for i, v in enumerate(inc):
    b.text(v + 0.01, i, f"{v:.0%}" if v > 0 else "0% (exact)", va="center", fontsize=6.5,
           fontweight="bold" if names[i] == OURS else None)
b.set_yticks(yy); b.set_yticklabels(names, fontsize=6.5); b.invert_yaxis()
b.set_xlim(0, max(inc) * 1.35 + 0.05)
b.set_xlabel("motif occurrences split differently\non the two strands")
b.set_title("strand consistency of motif tokenization" if args.only_strand else "(b) strand consistency", loc="left")
fig.tight_layout()
save(fig, args.out)
