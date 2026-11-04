import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="*")
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--chroms", nargs="+", default=["chr8", "chr9"])
p.add_argument("--n-sites", type=int, default=2000)
p.add_argument("--window", type=int, default=1024)
p.add_argument("--shifts", type=int, nargs="+", default=[-32, -16, -8, -4, -2, -1, 1, 2, 3, 4, 8, 16, 32])
p.add_argument("--margin", type=int, default=128)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--no-bpe", action="store_true")
p.add_argument("--out", default="shift_stability")
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
L, C, M = args.window, args.window // 2, args.margin
OUT = Path(args.out)
OUT.mkdir(exist_ok=True)
SHIFTS = [0] + [s for s in args.shifts if s != 0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- sites and windows
def donors():
    """(chrom, g, plus) with g = first intron base on the + coordinate for + genes; for - genes the intron is
    [.., g-1] and its first base in the gene's direction is g-1. The exon-intron junction is the gap after base g-1."""
    out = set()
    with gzip.open(Path(args.genome_dir) / "ncbiRefSeq.txt.gz", "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            name, chrom, strand = c[1], c[2], c[3]
            if chrom not in args.chroms or not name.startswith("NM_"):
                continue
            starts = [int(v) for v in c[9].strip(",").split(",")]
            ends = [int(v) for v in c[10].strip(",").split(",")]
            if strand == "+":
                out.update((chrom, g, True) for g in ends[:-1])
            else:
                out.update((chrom, g, False) for g in starts[1:])
    out = sorted(out)
    rng = np.random.default_rng(0)
    return [out[i] for i in rng.permutation(len(out))]


def load_windows():
    """Windows for every shift; a site is kept only if all its shifted windows are inside the chromosome and
    almost free of N. Returns X[shift] (n x L uint8), window starts[shift] (n), site base, plus flag."""
    chroms = {c: rk.load_chrom(args.genome_dir, c) for c in args.chroms}
    smax = max(abs(s) for s in SHIFTS)
    keep = []
    for chrom, g, plus in donors():
        t = chroms[chrom]
        if t is None or g - C - smax < 0 or g - C + smax + L > len(t):
            continue
        if (np.asarray(t[g - C - smax:g - C + smax + L]) == rk.N).mean() >= 0.01:
            continue
        keep.append((chrom, g, plus))
        if len(keep) == args.n_sites:
            break
    X = {s: np.stack([np.asarray(chroms[c][g - C + s:g - C + s + L], np.uint8) for c, g, _ in keep]) for s in SHIFTS}
    starts = {s: np.array([g - C + s for _, g, _ in keep]) for s in SHIFTS}
    site = np.array([g if pl else g - 1 for _, g, pl in keep])          # first intronic base (genome index)
    junction = np.array([g - 1 for _, g, _ in keep])                    # cut "after base g-1" = the junction
    log(f"{len(keep)} splice donors on {', '.join(args.chroms)}; shifts {SHIFTS[1:]}")
    return X, starts, site, junction


# ----------------------------------------------------------------------------------------------- tokenizers
# Each returns, per window, the sorted gap indices j (0..L-2) with a boundary after base j of the window.
def model_cuts(model):
    @torch.no_grad()
    def f(X):
        out = []
        for i in range(0, len(X), args.bs):
            c = model(torch.from_numpy(X[i:i + args.bs]).long().to(dev), decode=False)["cut"].bool().cpu().numpy()
            out += [np.where(r)[0] for r in c]
        return out
    return f


def kmer_cuts(X, k=6):
    j = np.arange(k - 1, L - 1, k)                                       # tokens [0,6), [6,12), ... from the start
    return [j] * len(X)


def bpe_tokenizer():
    if args.no_bpe:
        return None
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    except Exception as e:                                                # noqa: BLE001
        log(f"DNABERT-2 tokenizer not available ({type(e).__name__}); BPE skipped")
        return None

    def f(X):
        out = []
        for w in X:
            toks = tk.tokenize("".join("ACGTN"[v] for v in w))
            lens = np.array([len(t.replace("#", "").replace("▁", "")) for t in toks])
            if lens.sum() != L:
                raise RuntimeError("BPE tokens do not add up to the window length")
            out.append(np.cumsum(lens)[:-1] - 1)
        return out
    return f


# ----------------------------------------------------------------------------------------------- measures
def compare(cuts, starts, site, junction):
    """cuts[s] = list of gap-index arrays. Returns one row per shift."""
    rows = []
    g0 = [set((st + c).tolist()) for st, c in zip(starts[0], cuts[0])]   # genome coordinate of each boundary
    for s in SHIFTS:
        jc, ja, same, jun = [], [], [], []
        for i, (st, c) in enumerate(zip(starts[s], cuts[s])):
            gs = set((st + c).tolist())
            lo, hi = max(starts[0][i], st), min(starts[0][i], st) + L - 2  # shared gaps (genome coordinates)
            for lst, m in ((ja, 0), (jc, M)):
                a = {v for v in g0[i] if lo + m <= v <= hi - m}
                b = {v for v in gs if lo + m <= v <= hi - m}
                lst.append(len(a & b) / max(1, len(a | b)))
            q = site[i]
            tok = lambda S: (max([v for v in S if v < q], default=None), min([v for v in S if v >= q], default=None))
            same.append(tok(g0[i]) == tok(gs))
            jun.append(junction[i] in gs)
        rows.append(dict(shift=s, jaccard_core=np.mean(jc), jaccard_all=np.mean(ja), site_token=np.mean(same),
                         junction_cut=np.mean(jun)))
    return rows


# ----------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    import pandas as pd
    X, starts, site, junction = load_windows()
    toks = [("6-mers (NT-v2)", kmer_cuts)]
    bpe = bpe_tokenizer()
    if bpe:
        toks.append(("BPE (DNABERT-2)", bpe))
    for folder in args.folders:
        for variant, mcfg, model, _ in rk.load_pretrained(folder, dev):
            label = {"RC-equivariant": "MirrorDNA" + ("-duplex" if mcfg.get("enc") == "duplex" else "")
                     + (", fixed cuts" if mcfg.get("router") == "fixed" else ""),
                     "baseline+RCaug": "Chunker + RC aug.", "baseline": "Chunker (no aug.)"}[variant]
            toks.append((label, model_cuts(model)))
    rows = []
    for label, f in toks:
        cuts = {s: f(X[s]) for s in SHIFTS}
        r = compare(cuts, starts, site, junction)
        rows += [dict(tokenizer=label, **x) for x in r]
        nz = [x for x in r if x["shift"] != 0]
        log(f"{label:24s} Jaccard core {np.mean([x['jaccard_core'] for x in nz]):.3f} | "
            f"site token unchanged {np.mean([x['site_token'] for x in nz]):.3f}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "per_shift.csv", index=False)
    nz = df[df["shift"] != 0]
    summ = nz.groupby("tokenizer", sort=False).agg(jaccard_core=("jaccard_core", "mean"),
                                                  jaccard_core_min=("jaccard_core", "min"),
                                                  jaccard_all=("jaccard_all", "mean"),
                                                  site_token=("site_token", "mean"),
                                                  site_token_min=("site_token", "min"))
    jun = df.pivot(index="tokenizer", columns="shift", values="junction_cut").loc[summ.index]
    lines = [f"# Token boundaries under window shifts ({len(site)} RefSeq splice donors, {', '.join(args.chroms)}; "
             f"{L}-bp windows; shifts {SHIFTS[1:]})", "",
             f"Compared with the unshifted window. Jaccard of boundary positions in genome coordinates over the shared "
             f"region, excluding {M} bp at each window edge (core) or not (all); site token = token containing the first "
             f"intronic base has the same genomic start and end. Means over shifts (min = worst shift).", "",
             "| tokenizer | Jaccard core (min) | Jaccard all | site token unchanged (min) |", "|---|---|---|---|"]
    for t, r in summ.iterrows():
        lines.append(f"| {t} | **{r.jaccard_core:.3f}** ({r.jaccard_core_min:.3f}) | {r.jaccard_all:.3f} | "
                     f"{r.site_token:.3f} ({r.site_token_min:.3f}) |")
    lines += ["", "Share of sites with a boundary exactly at the exon-intron junction, per shift:", "",
              "| tokenizer | " + " | ".join(str(s) for s in jun.columns) + " |",
              "|---|" + "---|" * len(jun.columns)]
    for t, r in jun.iterrows():
        lines.append(f"| {t} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))