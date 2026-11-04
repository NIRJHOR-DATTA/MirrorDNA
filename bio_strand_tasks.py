import argparse
import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--window", type=int, default=1024)
p.add_argument("--max-donors", type=int, default=60000)
p.add_argument("--test-chroms", nargs="+", default=["chr8", "chr9"])
args = p.parse_args()
REFSEQ = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/ncbiRefSeq.txt.gz"
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
TEST = set(args.test_chroms)


def annotation():
    f = Path(args.genome_dir) / "ncbiRefSeq.txt.gz"
    if not f.exists():
        print("downloading RefSeq annotation from UCSC ...")
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        urllib.request.urlretrieve(REFSEQ, tmp)
        os.replace(tmp, f)
    tss, donors = {}, set()
    with gzip.open(f, "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            name, chrom, strand = c[1], c[2], c[3]
            if chrom not in CHROMS or not name.startswith("NM_"):
                continue
            plus = strand == "+"
            starts = [int(v) for v in c[9].strip(",").split(",")]
            ends = [int(v) for v in c[10].strip(",").split(",")]
            gene = c[12]
            t = int(c[4]) if plus else int(c[5])
            tss.setdefault((gene, chrom, plus), t)                       # one TSS per gene and strand
            for g in (ends[:-1] if plus else starts[1:]):
                donors.add((chrom, g, plus))
    return [(c, t, s) for (_, c, s), t in tss.items()], sorted(donors)


def build(sites, name, rng, cap=None):
    L, half = args.window, args.window // 2
    cache, split = {}, {"train": ([], []), "test": ([], [])}
    sites = [sites[i] for i in rng.permutation(len(sites))]
    used = {}
    for chrom, g, plus in sites:
        if cap and sum(len(v[0]) for v in split.values()) >= cap:
            break
        key = (chrom, g // 2000)                                            # no near-duplicate windows
        if key in used:
            continue
        if chrom not in cache:
            cache[chrom] = rk.load_chrom(args.genome_dir, chrom)
        tok = cache[chrom]
        if tok is None or g - half < 0 or g + half > len(tok):
            continue
        w = np.array(tok[g - half:g - half + L], np.uint8)
        if (w == rk.N).mean() > 0.05:
            continue
        used[key] = True
        part = "test" if chrom in TEST else "train"
        split[part][0].append(w)
        split[part][1].append(int(plus))
    if args.window != 1024:                     # longer windows: separate task files, the 1-kb tasks stay untouched
        name = f"{name}_w{args.window}"
    out = Path(rk.BIO_DIR)
    out.mkdir(exist_ok=True)
    arr = {}
    for part, (X, y) in split.items():
        arr[part + "_seq"] = np.array(X, dtype=object)
        arr[part + "_y"] = np.array(y, np.int64)
    tmp = out / f"{name}.tmp.npz"
    with open(tmp, "wb") as fh:                                          # atomic: parallel jobs may read it
        np.savez_compressed(fh, **arr)
    os.replace(tmp, out / f"{name}.npz")
    for part in ("train", "test"):
        y = arr[part + "_y"]
        print(f"{name} {part}: {len(y)} windows, {y.mean():.1%} on the + strand")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tss, donors = annotation()
    build(tss, "bio_gene_strand_tss", rng)
    build(donors, "bio_gene_strand_donor", rng, cap=args.max_donors)
    print("done -> bio_tasks/   (evaluate with: python run_suite.py FOLDER --suite bio)")