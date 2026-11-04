import argparse
import gzip
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="+")
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--annotation", default=None, help="UCSC genePred table (default: download ncbiRefSeq for hg38)")
p.add_argument("--n-sites", type=int, default=3000, help="sites per class")
p.add_argument("--window", type=int, default=1024)
p.add_argument("--flank", type=int, default=100)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--out", default="bio_boundaries")
args = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
REFSEQ = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/ncbiRefSeq.txt.gz"
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def annotation_sites():
    """Gap coordinates (0-based: the boundary between base g-1 and base g) of donors, acceptors and TSSs."""
    f = Path(args.annotation) if args.annotation else Path(args.genome_dir) / "ncbiRefSeq.txt.gz"
    if not f.exists():
        log("downloading RefSeq annotation from UCSC ...")
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        urllib.request.urlretrieve(REFSEQ, tmp)
        os.replace(tmp, f)
    opener = gzip.open if f.suffix == ".gz" else open
    sites = {"splice donor": set(), "splice acceptor": set(), "TSS": set()}
    with opener(f, "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            name, chrom, strand = c[1], c[2], c[3]
            if chrom not in CHROMS or not name.startswith("NM_"):          # curated protein-coding transcripts
                continue
            starts = [int(v) for v in c[9].strip(",").split(",")]
            ends = [int(v) for v in c[10].strip(",").split(",")]
            inner_ends, inner_starts = ends[:-1], starts[1:]                 # exon/intron boundaries
            if strand == "+":
                sites["splice donor"].update((chrom, g) for g in inner_ends)
                sites["splice acceptor"].update((chrom, g) for g in inner_starts)
                sites["TSS"].add((chrom, int(c[4])))
            else:
                sites["splice donor"].update((chrom, g) for g in inner_starts)
                sites["splice acceptor"].update((chrom, g) for g in inner_ends)
                sites["TSS"].add((chrom, int(c[5])))
    rng = np.random.default_rng(0)
    out = {}
    for k, v in sites.items():
        v = sorted(v)
        idx = rng.permutation(len(v))[:args.n_sites]
        out[k] = [v[i] for i in idx]
        log(f"{k}: {len(v)} unique sites, using {len(out[k])}")
    return out


def random_sites(n, rng):
    lens = {c: len(rk.load_chrom(args.genome_dir, c)) for c in CHROMS if rk.load_chrom(args.genome_dir, c) is not None}
    names = list(lens)
    prob = np.array([lens[c] for c in names], float)
    prob /= prob.sum()
    return [(names[i], int(rng.integers(args.window, lens[names[i]] - args.window)))
            for i in rng.choice(len(names), n, p=prob)]


def windows(sites):
    """Windows with the site at gap index window//2 - 1 (between bases c-1 and c); skips windows rich in N."""
    L, c = args.window, args.window // 2
    cache, X = {}, []
    for chrom, g in sites:
        if chrom not in cache:
            cache[chrom] = rk.load_chrom(args.genome_dir, chrom)
        tok = cache[chrom]
        if tok is None or g - c < 0 or g - c + L > len(tok):
            continue
        w = np.array(tok[g - c:g - c + L], dtype=np.uint8)
        if (w == rk.N).mean() < 0.05:
            X.append(w)
    return np.stack(X) if X else np.zeros((0, L), np.uint8)


@torch.no_grad()
def router(model, X):
    """Cut probability per gap on the forward window, and cut decisions on both strands (RC mirrored back)."""
    P, cut_f, cut_r = [], [], []
    for i in range(0, len(X), args.bs):
        x = torch.from_numpy(X[i:i + args.bs]).long().to(device)
        o, orc = model(x, decode=False), model(rk.rc_tokens(x), decode=False)
        P.append(o["p"].float().cpu().numpy())
        cut_f.append(o["cut"].cpu().numpy())
        cut_r.append(orc["cut"].flip(1).cpu().numpy())             # gap j on RC = gap L-2-j on forward
    return np.concatenate(P), np.concatenate(cut_f), np.concatenate(cut_r)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    sites = annotation_sites()
    sites["random"] = random_sites(args.n_sites, rng)
    W = {k: windows(v) for k, v in sites.items()}
    c, F = args.window // 2 - 1, args.flank
    offs = np.arange(-F, F + 1)
    results, lines = [], ["# Where the learned tokenizer cuts the genome", "",
                          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. RefSeq NM_ transcripts, hg38; "
                          f"{args.window}-bp windows centred on each site; cut probability of the router per gap.", "",
                          "| Model | Site | sites | cut rate at site | background | enrichment | "
                          "cut rate +-1 bp | strand agreement of cuts (Jaccard) |", "|---|---|---|---|---|---|---|---|"]
    for folder in args.folders:
        for variant, mcfg, model, name in rk.load_pretrained(folder, device):
            label = f"{name} [{Path(folder).name}]"
            prof, bg = {}, None
            for k in ["random", "splice donor", "splice acceptor", "TSS"]:
                P, cf, cr = router(model, W[k])
                inter, union = (cf & cr).sum(), (cf | cr).sum()
                jac = float(inter / max(union, 1))
                rate = cf.mean(0)
                if k == "random":
                    bg = float(cf.mean())
                prof[k] = rate[c - F:c + F + 1]
                at, near = float(rate[c]), float(rate[c - 1:c + 2].mean())
                results.append(dict(model=label, site=k, n=len(W[k]), at=at, bg=bg, enr=at / max(bg, 1e-9), near=near,
                                    jac=jac, mean_p=float(P[:, c].mean())))
                r = results[-1]
                lines.append(f"| {label} | {k} | {r['n']} | {at:.3f} | {bg:.3f} | **{r['enr']:.2f}x** | {near:.3f} | "
                             f"{jac:.3f} |")
                log(f"{label} {k:16s}: cut rate at site {at:.3f} vs background {bg:.3f} ({r['enr']:.2f}x) | "
                    f"strand Jaccard {jac:.3f}")
            np.savetxt(f"{args.out}_{variant}_{Path(folder).name}_profiles.csv",
                       np.column_stack([offs] + [prof[k] for k in prof]), delimiter=",",
                       header="offset," + ",".join(prof), comments="")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 3.2))
                for k, col in zip(prof, ["0.6", "#c0392b", "#2c7fb8", "#27ae60"]):
                    ax.plot(offs, prof[k], label=k, color=col, lw=1.2)
                ax.axvline(0, color="0.8", lw=0.8, ls="--")
                ax.set_xlabel("position relative to site (bp)"); ax.set_ylabel("chunk boundary rate")
                ax.set_title(label, fontsize=9); ax.legend(fontsize=8, frameon=False)
                fig.tight_layout()
                fig.savefig(f"{args.out}_{variant}_{Path(folder).name}_profiles.png", dpi=200)
                plt.close(fig)
            except Exception as e:
                log(f"no figure ({e})")
    lines += ["", "- **Enrichment > 1**: the tokenizer places chunk boundaries at that kind of site more often than "
              "elsewhere, without ever seeing an annotation.",
              "- **Strand agreement**: 1.0 means the two strands of the same DNA are cut at exactly mirrored positions."]
    Path(f"{args.out}_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE -> {args.out}_report.md")
