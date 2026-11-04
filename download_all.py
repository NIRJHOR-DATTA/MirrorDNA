import argparse
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--public", action="store_true", help="also download the public baseline models")
p.add_argument("--tries", type=int, default=3)
args = p.parse_args()
G = Path(args.genome_dir)
G.mkdir(parents=True, exist_ok=True)
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
NT18 = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
        "H4K20me1", "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata",
        "splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]
status = []


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step(name, fn):
    for k in range(1, args.tries + 1):
        try:
            fn()
            status.append((name, "OK"))
            return
        except Exception as e:
            log(f"{name}: attempt {k}/{args.tries} failed: {e}")
            if k == args.tries:
                traceback.print_exc()
                status.append((name, f"FAILED ({type(e).__name__}: {str(e)[:80]})"))
            else:
                time.sleep(10 * k)


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  {dest.name}: already there")
        return
    log(f"  downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def genome():
    rk.prepare_genome(CHROMS, G, log, mmap=True)
    for c in CHROMS:                                  # the .fa.gz files are no longer needed (saves ~900 MB)
        gz = G / f"{c}.fa.gz"
        if gz.exists() and (G / f"{c}.tok.npy").exists():
            gz.unlink()


def nt():
    from datasets import load_dataset
    for t in NT18:
        try:
            d = load_dataset(rk.NT_REPO, t)
        except Exception:
            d = load_dataset(rk.NT_REPO)
            d = d.filter(lambda e: e["task"] == t)
        log(f"  NT {t}: train {len(d['train'])}, test {len(d['test'])}")


def gb():
    for name in rk.GB_TASKS:
        rk.load_gb_task(name, 1, 1, log=log)
        log(f"  GB {name}: ready")


def clinvar():
    from datasets import load_dataset
    d = load_dataset("songlab/clinvar", split="test")
    log(f"  ClinVar: {len(d)} variants")


def bio():
    if all((rk.BIO_DIR / f"{t}.npz").exists() for t in rk.BIO_TASKS):
        log("  biology tasks: already built")
        return
    subprocess.run([sys.executable, "bio_strand_tasks.py", "--genome-dir", str(G)], check=True)


def public():
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import snapshot_download
    for repo in ["kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
                 "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
                 "LongSafari/hyenadna-small-32k-seqlen-hf",
                 "zhihan1996/DNABERT-2-117M",
                 "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"]:
        snapshot_download(repo)                       # files only; no model code is run here
        log(f"  {repo}: downloaded")


if __name__ == "__main__":
    t0 = time.time()
    log("1/7 hg38 genome (first time ~30-60 min)")
    step("hg38 genome", genome)
    log("2/7 Enformer / Caduceus split")
    step("Enformer split (bed)", lambda: fetch(rk.ENFORMER_BED, G / "human-sequences.bed"))
    log("3/7 RefSeq annotation")
    step("RefSeq annotation", lambda: fetch(
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/ncbiRefSeq.txt.gz", G / "ncbiRefSeq.txt.gz"))
    log("4/7 Nucleotide Transformer benchmark (18 tasks)")
    step("NT benchmark (18)", nt)
    log("5/7 Genomic Benchmarks (9 tasks)")
    step("Genomic Benchmarks (9)", gb)
    log("6/7 ClinVar")
    step("ClinVar", clinvar)
    log("7/7 gene-strand biology tasks")
    step("Biology tasks", bio)
    if args.public:
        log("+ public model weights")
        step("Public models", public)
    print("\n" + "=" * 60)
    for name, s in status:
        print(f"  {name:28s} {s}")
    print("=" * 60)
    bad = [n for n, s in status if s != "OK"]
    print(f"took {(time.time() - t0) / 60:.0f} min. " +
          ("ALL DONE - you can start START_WEEK.bat" if not bad else
           "Some items failed: rerun  python download_all.py  (finished items are skipped)."))
    sys.exit(1 if bad else 0)
