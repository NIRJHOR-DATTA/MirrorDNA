import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ours", default="results_scale_tiny_RC-equivariant_caduceus10B")
p.add_argument("--public", nargs="+", default=["results_public_long", "results_public_1k"])
p.add_argument("--out", default="compare_long")
args = p.parse_args()


def rows(f):
    f = Path(f)
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def split(task):
    """bio_gene_strand_tss_w4096 -> ('TSS', 4096); bio_gene_strand_donor -> ('donor', 1024)"""
    m = re.match(r"bio_gene_strand_(tss|donor)(?:_w(\d+))?(_core\d+)?$", task)
    if not m:
        return None, None
    site = ("TSS" if m.group(1) == "tss" else "donor") + (" (random flanks)" if m.group(3) else "")
    return site, int(m.group(2) or 1024)


res = defaultdict(list)                                     # (model, site, window) -> [mcc per seed]
for r in rows(Path(args.ours) / "results_suite.jsonl"):
    site, w = split(r.get("task", ""))
    name = r.get("model", "")
    if site and "[inv" not in name and " n=" not in name and " +length" not in name:
        # keep the two readouts apart: a fair length curve uses the same head at every window
        res["MirrorDNA (ours)" + (" +base" if "+base" in name else ""), site, w].append(r["mcc"])
for folder in args.public:
    for r in rows(Path(folder) / "results.jsonl"):
        site, w = split(r.get("task", ""))
        v = r.get("variant", "")
        if site and "RCaug" not in v and "[oriented" not in v and "Caduceus" in v:
            res[v.split(" (")[0], site, w].append(r["mcc"])
if not res:
    raise SystemExit("no gene-strand results found: run from the project folder")

models = sorted({k[0] for k in res}, key=lambda m: (not m.startswith("MirrorDNA"), m))
windows = sorted({k[2] for k in res})
lines = ["# Gene strand at increasing window length (MCC, mean +- sd over seeds; n = seeds)", "",
         "| model | site | " + " | ".join(f"{w} bp" for w in windows) + " |",
         "|---|---|" + "---|" * len(windows)]
csv = ["model,site,window,mean,sd,n"]
for m in models:
    for site in ("TSS", "donor", "TSS (random flanks)", "donor (random flanks)"):
        cells = []
        for w in windows:
            v = res.get((m, site, w), [])
            if v:
                cells.append(f"{np.mean(v):.3f} +- {np.std(v):.3f} (n={len(v)})")
                csv.append(f"{m},{site},{w},{np.mean(v):.4f},{np.std(v):.4f},{len(v)}")
            else:
                cells.append("-")
        if any(c != "-" for c in cells):
            lines.append(f"| {m} | {site} | " + " | ".join(cells) + " |")
Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path(args.out + ".csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
print("\n".join(lines))
