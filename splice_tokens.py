import argparse
import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--chroms", nargs="+", default=["chr8", "chr9"])
p.add_argument("--n-sites", type=int, default=3000)
p.add_argument("--window", type=int, default=1024)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--top", type=int, default=12)
p.add_argument("--out", default="splice_tokens_report.md")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
L, C = args.window, args.window // 2
COMP = str.maketrans("ACGTN", "TGCAN")


def sites():
    """(chrom, g, plus): g = 0-based index of the first base AFTER the site on the + strand coordinate."""
    out = {"splice donor": set(), "splice acceptor": set(), "TSS": set()}
    with gzip.open(Path(args.genome_dir) / "ncbiRefSeq.txt.gz", "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            name, chrom, strand = c[1], c[2], c[3]
            if chrom not in args.chroms or not name.startswith("NM_"):
                continue
            starts = [int(v) for v in c[9].strip(",").split(",")]
            ends = [int(v) for v in c[10].strip(",").split(",")]
            plus = strand == "+"
            if plus:
                out["splice donor"].update((chrom, g, True) for g in ends[:-1])
                out["splice acceptor"].update((chrom, g, True) for g in starts[1:])
                out["TSS"].add((chrom, int(c[4]), True))
            else:
                out["splice donor"].update((chrom, g, False) for g in starts[1:])
                out["splice acceptor"].update((chrom, g, False) for g in ends[:-1])
                out["TSS"].add((chrom, int(c[5]), False))
    rng = np.random.default_rng(0)
    return {k: [sorted(v)[i] for i in rng.permutation(len(v))[:args.n_sites]] for k, v in out.items()}


def windows(sl):
    cache, X, keep = {}, [], []
    for chrom, g, plus in sl:
        if chrom not in cache:
            cache[chrom] = rk.load_chrom(args.genome_dir, chrom)
        t = cache[chrom]
        if t is None or g - C < 0 or g - C + L > len(t):
            continue
        w = np.array(t[g - C:g - C + L], np.uint8)
        if (w == rk.N).mean() < 0.01:
            X.append(w); keep.append(plus)
    return np.stack(X), keep


@torch.no_grad()
def cuts(model, X):
    out = []
    for i in range(0, len(X), args.bs):
        out.append(model(torch.from_numpy(X[i:i + args.bs]).long().to(dev), decode=False)["cut"].bool().cpu().numpy())
    return np.concatenate(out)


def site_chunks(X, plus, cut):
    """chunk containing the first base after the site in the gene's direction; read on the gene's strand."""
    rows = []
    for w, pl, c in zip(X, plus, cut):
        b = np.where(c)[0]                      # cut after base j
        q = C if pl else C - 1                  # first base after the site in the gene's reading direction
        start = (b[b < q].max() + 1) if (b < q).any() else 0
        end = (b[b >= q].min() + 1) if (b >= q).any() else L
        s = "".join("ACGTN"[v] for v in w[start:end])
        if pl:
            rows.append((s, start - C))
        else:                                   # gene on the - strand: read the chunk on that strand
            rows.append((s[::-1].translate(COMP), C - end))
    return rows


S = sites()
v, cfg, model, name = [t for t in rk.load_pretrained(args.folder, dev) if t[0] == "RC-equivariant"][0]
rng = np.random.default_rng(1)
csv_rows = []
lines = ["# Tokens formed at gene-structure sites", "",
         f"RefSeq NM_ transcripts on {', '.join(args.chroms)}; model {name}. For each site, the chunk containing the "
         "first base after the site, read on the gene's strand. Offset = chunk start relative to the site "
         "(negative = the chunk spans the site).", "",
         "**Control (shifted cuts):** the same model's cuts in the same window, circularly shifted by a random 100-900 bp, "
         "so chunk lengths and cut density are unchanged but the cuts are unrelated to the site. Enrichment = share with "
         "the model's own cuts / share with shifted cuts.", ""]
for site, sl in S.items():
    X, plus = windows(sl)
    cut = cuts(model, X)
    shift = np.stack([np.roll(c, int(rng.integers(100, 900))) for c in cut])
    obs, ctl = site_chunks(X, plus, cut), site_chunks(X, plus, shift)
    co, cc = Counter(s for s, _ in obs), Counter(s for s, _ in ctl)
    lens = np.array([len(s) for s, _ in obs]); offs = np.array([o for _, o in obs])
    lines += [f"## {site} ({len(X)} sites)", "",
              f"Model's cuts: mean chunk length {lens.mean():.2f} bp; the chunk spans the site in {(offs < 0).mean():.0%} "
              f"of cases (shifted cuts: {np.mean([o < 0 for _, o in ctl]):.0%}).", "",
              "| chunk (gene strand) | share, model's cuts | share, shifted cuts | enrichment | typical offset |",
              "|---|---|---|---|---|"]
    for s_, n in co.most_common(args.top):
        o = Counter(o for t, o in obs if t == s_).most_common(1)[0][0]
        c = cc.get(s_, 0) / len(ctl)
        csv_rows.append(dict(site=site, token=s_, share_model=n / len(obs), share_shifted=c, offset=o, sites=len(obs)))
        lines.append(f"| `{s_}` | {n / len(obs):.1%} | {c:.1%} | "
                     f"{(n / len(obs)) / c:.0f}x |" if c > 0 else f"| `{s_}` | {n / len(obs):.1%} | 0.0% | n/a |")
        lines[-1] += f" {o} |"
    lines.append("")
Path(args.out).write_text("\n".join(lines), encoding="utf-8")
import pandas as pd
pd.DataFrame(csv_rows).to_csv(Path(args.out).with_suffix(".csv"), index=False)
print("\n".join(lines))
