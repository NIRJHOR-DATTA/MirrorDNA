"""
Builds the minimal test dataset shipped in example/data/ from the full gene-strand task
(bio_tasks/bio_gene_strand_tss.npz, made by bio_strand_tasks.py from hg38 and RefSeq).

    python example/make_example_data.py
Output: example/data/gene_strand_tss_small.npz (400 training and 200 test windows of 1,024 bp; held-out chr8/chr9 test)
"""
import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--source", default="bio_tasks/bio_gene_strand_tss.npz")
p.add_argument("--n-train", type=int, default=400)
p.add_argument("--n-test", type=int, default=200)
args = p.parse_args()

d = np.load(args.source, allow_pickle=True)
rng = np.random.default_rng(0)
out = {}
for part, n in (("train", args.n_train), ("test", args.n_test)):
    y = d[part + "_y"]
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:n // 2] for c in (0, 1)])   # balanced classes
    idx = rng.permutation(idx)
    out[part + "_seq"] = np.stack([np.asarray(d[part + "_seq"][i], np.uint8) for i in idx])
    out[part + "_y"] = y[idx].astype(np.int64)
dst = Path(__file__).parent / "data" / "gene_strand_tss_small.npz"
dst.parent.mkdir(exist_ok=True)
np.savez_compressed(dst, **out)
print(f"wrote {dst} ({dst.stat().st_size / 1e3:.0f} kB): "
      + ", ".join(f"{k} {v.shape}" for k, v in out.items()))
