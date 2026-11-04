import argparse
import os
import urllib.request
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk
from test_equivariance import check as equiv_check

SIZES = {  # name: (model config, pretraining peak lr)
    "pico": (dict(d=44, enc_layers=3, main_layers=4, dec_layers=1, heads=4, target_ratio=4.0), 2.5e-3),   # ~0.46M (= Caduceus-PS-1k)
    "nano": (dict(d=64, enc_layers=3, main_layers=4, dec_layers=1, heads=4, target_ratio=4.0), 2e-3),     # ~1M
    "tiny": (dict(d=96, enc_layers=4, main_layers=4, dec_layers=2, heads=4, target_ratio=4.0), 1.5e-3),   # ~2.5M
    "mini": (dict(d=128, enc_layers=4, main_layers=6, dec_layers=2, heads=8, target_ratio=4.0), 1.2e-3),  # ~6M
    "small": (dict(d=160, enc_layers=4, main_layers=8, dec_layers=2, heads=8, target_ratio=4.0), 1e-3),   # 11M
    "base": (dict(d=224, enc_layers=4, main_layers=10, dec_layers=2, heads=8, target_ratio=4.0), 6e-4),   # 27M
    "large": (dict(d=256, enc_layers=4, main_layers=12, dec_layers=2, heads=8, target_ratio=4.0), 5e-4),  # 42M
}

p = argparse.ArgumentParser()
p.add_argument("--variant", default="RC-equivariant", choices=list(rk.VARIANTS))
p.add_argument("--size", default="base", choices=list(SIZES))
p.add_argument("--chunk", type=float, default=0, help="target chunk size in bp (default 4; larger = lighter model)")
p.add_argument("--ratio-weight", type=float, default=0.03, help="chunk-size loss weight (0.03 so far; ~0.3 holds the target)")
p.add_argument("--encoder", default="conv", choices=["conv", "duplex"],
               help="duplex = cross-strand gated encoder (new); conv = the encoder used so far")
p.add_argument("--router", default="learned", choices=["learned", "fixed"],
               help="fixed = ABLATION: fixed mirror-symmetric cuts every --fixed-len bp instead of the learned router")
p.add_argument("--fixed-len", type=int, default=3, help="chunk length of the fixed router (MirrorDNA averages 3.1 bp)")
p.add_argument("--hours", type=float, default=15.0, help="pretraining budget (fine-tuning comes on top)")
p.add_argument("--steps", type=int, default=0,
               help="fixed number of pretraining steps (overrides --hours); use the RC run's steps for a fair baseline")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--recipe", default="ours", choices=["ours", "caduceus"],
               help="caduceus = Caduceus/HyenaDNA 1k setup: Enformer train split only, 2^20 tokens per step "
                    "(1,024 windows of 1 kb), 10,000 steps = 10.5B tokens")
p.add_argument("--micro-bs", type=int, default=128, help="windows per forward pass (gradient accumulation up to --bs)")
p.add_argument("--lr", type=float, default=0, help="peak pretraining learning rate (default: per model size)")
p.add_argument("--L", type=int, default=1024)
p.add_argument("--tasks", nargs="+", default=["H3K4me3", "H3K27ac", "H3K36me3", "enhancers"])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--no-gb", action="store_true", help="skip the Genomic Benchmarks suite after pretraining")
p.add_argument("--base-features", action="store_true",
               help="fine-tuning head also pools the per-base decoder output (passed on to run_suite.py)")
p.add_argument("--suites", nargs="+", default=None, choices=["nt4", "nt18", "gb", "bio"],
               help="fine-tuning suites after pretraining (default: nt4 gb). bio also runs the variant and boundary analyses")
p.add_argument("--pretrain-seed", type=int, default=0, help="seed of the pretraining run (repeat pretraining for error bars)")
p.add_argument("--no-mmap", action="store_true", help="load the genome fully into RAM instead of sharing it via mmap")
p.add_argument("--tag", default="", help="extra name for the output folder, e.g. 35B (keeps runs with different budgets apart)")
p.add_argument("--genome-dir", default="hg38_data", help="reuses the chr1-7 files from Day 1")
p.add_argument("--day1", default="results_day1/results.jsonl", help="Day-1 results, for the comparison")
p.add_argument("--smoke", action="store_true")
p.add_argument("--offline", action="store_true")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MCFG, LR_PRE = SIZES[args.size]
if args.lr:
    LR_PRE = args.lr
if args.recipe == "caduceus":
    if args.bs == 64:
        args.bs = 1024                                 # 2^20 tokens per step at 1 kb
    if not args.steps:
        args.steps = 10000
    if not args.tag:
        args.tag = "caduceus10B"
CHROMS, L, BS = [f"chr{i}" for i in range(1, 23)] + ["chrX"], args.L, args.bs
FT = dict(max_train=20000, max_test=5000, epochs=5, bs=64, lr=3e-4)       # = Day-1 Phase C
SECONDS = args.hours * 3600
if args.smoke:
    MCFG, LR_PRE = dict(d=32, enc_layers=2, main_layers=2, dec_layers=1, heads=4, target_ratio=4.0), 1e-3
    CHROMS, L, BS = ["chr21"], 512, 16
    FT = dict(max_train=800, max_test=300, epochs=2, bs=32, lr=3e-4)
    SECONDS = 180
    args.seeds = args.seeds[:1]
    if args.recipe == "caduceus":
        BS, args.micro_bs = 64, 16
ACCUM = max(1, BS // max(1, min(args.micro_bs, BS)))
BS = (BS // ACCUM) * ACCUM
rk.RATIO_W = args.ratio_weight
MCFG = dict(MCFG, enc=args.encoder, **({"target_ratio": args.chunk} if args.chunk else {}),
            **({"router": "fixed", "fixed_len": args.fixed_len} if args.router == "fixed" else {}))

OUT = Path(f"results_scale_{args.size}{'_duplex' if args.encoder == 'duplex' else ''}{f'_fixed{args.fixed_len}' if args.router == 'fixed' else ''}{f'_chunk{args.chunk:g}' if args.chunk else ''}{f'_rw{args.ratio_weight:g}' if args.ratio_weight != 0.03 else ''}_{args.variant.replace('+', '_')}" + (f"_{args.tag}" if args.tag else "") + (f"_ps{args.pretrain_seed}" if args.pretrain_seed else "") + ("_smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
RES, LOG = OUT / "results.jsonl", OUT / "log.txt"
_LOCK = open(OUT / ".lock", "a+")                   # one run per folder: a second copy would corrupt checkpoints
try:
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(_LOCK.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(_LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    raise SystemExit(f"{OUT} is already being used by another running job; not starting a second copy.")
NAME = f"{args.variant} ({args.size}{', duplex encoder' if args.encoder == 'duplex' else ''}{f', fixed {args.fixed_len}-bp cuts' if args.router == 'fixed' else ''}, full genome)"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_rows(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def make_report():
    rows = read_rows(RES)
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    old = pd.DataFrame([r for r in read_rows(args.day1) if r.get("phase") == "C"])
    lines = [f"# Scale-up: {NAME}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Pretrained on hg38 {CHROMS[0]}-{CHROMS[-1]}, "
             f"{rk.n_params(rk.build(args.variant, MCFG)) / 1e6:.1f}M parameters; see pretrain_info.json for tokens seen. "
             "Fine-tuning = Day-1 Phase C recipe. Mean over seeds.", "",
             "| Task | Model | fwd MCC | RC MCC | Strand-robust | Flips | seeds |", "|---|---|---|---|---|---|---|"]
    avg = {}
    for t in args.tasks:
        cands = [(NAME, df[df.task == t])]
        if len(old):
            cands += [(f"Day 1: {v} (11M, chr1-7)", old[(old.task == t) & (old.variant == v)]) for v in rk.VARIANTS]
        for name, d in cands:
            if not len(d):
                continue
            f, r = d.mcc.mean(), d.mcc_rc_input.mean()
            avg.setdefault(name, []).append((f + r) / 2)
            lines.append(f"| {t} | {name} | {f:.3f} | {r:.3f} | **{(f + r) / 2:.3f}** | {d.flip_rate.mean():.1%} | {len(d)} |")
    lines += ["", "## Average strand-robust MCC over finished tasks", "", "| Model | Strand-robust | tasks |", "|---|---|---|"]
    lines += [f"| {n} | **{np.mean(v):.3f}** | {len(v)} |" for n, v in avg.items()]
    lines += ["", "Reference points (public checkpoints, same recipe): DNABERT-2 0.502, Caduceus-PS 0.451, "
              "HyenaDNA 0.419 (4-task averages)."]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    t_start = time.time()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    log(f"{gpu} | {NAME} | {rk.n_params(rk.build(args.variant, MCFG)) / 1e6:.1f}M params | out {OUT}")

    if rk.VARIANTS[args.variant][0]:
        for seed in range(2):
            r = equiv_check(True, seed, device=device, enc=args.encoder, router=args.router)
            assert r["cuts_mirrored"] and r["pooled_feature_rel_err"] < 1e-3, "equivariance broken; stop"
        log("equivariance self-test passed")

    final = OUT / f"pretrain_{args.variant}.final.pt"
    pretrained = rk.safe_load(final) is not None
    if pretrained:
        log("pretraining already finished; going straight to fine-tuning")
        genome = None
    else:
        log("genome: loading (first time downloads ~900 MB from UCSC, ~30-60 min)")
        genome = rk.SyntheticGenome() if args.offline else rk.prepare_genome(CHROMS, Path(args.genome_dir), log, mmap=not args.no_mmap)
        if args.recipe == "caduceus":
            bed = Path(args.genome_dir) / "human-sequences.bed"
            if args.offline:
                bed.parent.mkdir(parents=True, exist_ok=True)
                n = len(genome.tok[0])
                bed.write_text("".join(f"{CHROMS[0]}\t{s}\t{s + 20000}\t{'train' if k % 5 else 'test'}\n"
                                       for k, s in enumerate(range(0, n - 20000, 20000))))
            elif not bed.exists():
                log("downloading the Enformer / HyenaDNA hg38 split (sequences_human.bed) ...")
                tmp = bed.with_suffix(".part")
                urllib.request.urlretrieve(rk.ENFORMER_BED, tmp)
                os.replace(tmp, bed)
            genome = rk.IntervalGenome(genome, CHROMS, bed, "train", log)

    info_f = OUT / "pretrain_info.json"
    try:
        steps = json.loads(info_f.read_text())["steps"]
    except Exception:                                          # missing, or cut short by a power loss
        steps = None
    if steps is None and not pretrained:
        sps = rk.time_steps(args.variant, MCFG, genome, BS, L, device, n=4 if ACCUM > 1 else 15, accum=ACCUM)
        steps = args.steps if args.steps and not args.smoke else max(50, int(0.95 * SECONDS / sps))
        info = dict(steps=steps, sec_per_step=sps, tokens=steps * BS * L, chroms=CHROMS, mcfg=MCFG, lr=LR_PRE,
                    recipe=args.recipe, batch=BS, accum=ACCUM)
        tmp = info_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(info_f)
        log(f"recipe {args.recipe}: batch {BS} x {L} bp ({ACCUM} micro-batches of {BS // ACCUM}) | "
            f"{sps:.3f} s/step -> {steps} steps = {steps * BS * L / 1e9:.2f}B nt "
            f"(Day 1 was 2.58B), pretraining ends ~{time.strftime('%a %H:%M', time.localtime(time.time() + steps * sps))}")

    if not pretrained:
        log("pretraining (regular checkpoints; mlm should fall steadily; the chunk size shown includes the "
            "forced cuts around masked bases, so ~2 bp here is expected)")
        final = rk.pretrain(args.variant, MCFG, genome, steps, BS, L, LR_PRE, device, OUT, log, accum=ACCUM,
                            seed=args.pretrain_seed)
        del genome

    # fine-tuning + evaluation on the benchmark suites (crash-safe, resumes per task / seed / epoch)
    suites = args.suites or (["nt4"] + ([] if args.no_gb else ["gb"]))
    log(f"fine-tuning on {', '.join(suites)} via run_suite.py -> {OUT / 'suite_report.md'}")
    cmd = [sys.executable, str(Path(__file__).with_name("run_suite.py")), str(OUT), "--suite", *suites,
           "--seeds", *map(str, args.seeds), "--out", str(OUT / "suite_report.md")]
    bf = ["--base-features"] if args.base_features else []
    cmd += bf
    if args.smoke:
        cmd += ["--smoke", "--tasks", *args.tasks, "human_nontata_promoters", "human_enhancers_cohn"]
    if args.offline:
        cmd += ["--offline"]
    if "bio" in suites and not args.offline and not args.smoke:
        missing = [t for t in rk.BIO_TASKS if not (rk.BIO_DIR / f"{t}.npz").exists()]
        if missing:
            log("building the gene-strand biology tasks (bio_strand_tasks.py) ...")
            subprocess.run([sys.executable, str(Path(__file__).with_name("bio_strand_tasks.py")),
                            "--genome-dir", args.genome_dir], check=True)
    steps_ = [cmd]
    if "bio" in suites:                       # same tasks with a strand-invariant head (what Caduceus-PS-style heads can do)
        steps_.append([*cmd[:3], "--suite", "bio", "--seeds", *map(str, args.seeds), "--out",
                       str(OUT / "suite_report_invariant_head.md"), "--invariant-head", *bf]
                      + (["--smoke", "--tasks", *args.tasks] if args.smoke else []) + (["--offline"] if args.offline else []))
    for c in steps_:
        rc = subprocess.run(c).returncode
        if rc != 0:
            raise SystemExit(f"run_suite.py stopped (exit {rc}); rerun this same command to continue where it stopped")
    if "bio" in suites and not args.offline and not args.smoke:
        for script, extra in (("bio_variants.py", []), ("bio_boundaries.py", [])):
            stem = OUT / script.replace(".py", "")
            if Path(f"{stem}_report.md").exists():
                continue
            log(f"running {script} -> {stem}_report.md")
            rc = subprocess.run([sys.executable, str(Path(__file__).with_name(script)), str(OUT), "--genome-dir",
                                 args.genome_dir, "--out", str(stem), *extra]).returncode
            if rc != 0:
                log(f"{script} failed (exit {rc}); the benchmark results above are unaffected")
    log(f"DONE in {(time.time() - t_start) / 3600:.1f} h -> {OUT / 'suite_report.md'}")
