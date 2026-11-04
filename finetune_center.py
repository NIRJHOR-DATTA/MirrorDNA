import argparse
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk

NT18_DIR = ["promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors", "splice_sites_all",
            "splice_sites_donors"]
NT18_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
            "H4K20me1", "enhancers", "enhancers_types"]

p = argparse.ArgumentParser()
p.add_argument("--main", default="results_scale_tiny_RC-equivariant_caduceus10B", help="pretrained main model")
p.add_argument("--widths", nargs="+", type=int, default=[4, 16, 64], help="half-widths w of the central windows")
p.add_argument("--tasks", nargs="+", default=["dir"], help="'dir' (6 direction-dependent), 'all' (18) or task names")
p.add_argument("--seeds", nargs="+", type=int, default=[0])
p.add_argument("--full", action="store_true",
               help="the full main evaluation: NT-18, all 9 Genomic Benchmarks and the 2 gene-strand tasks")
p.add_argument("--check-centre", action="store_true")
p.add_argument("--compare", action="store_true")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("results_scale_tiny_RC-equivariant_center" + ("_smoke" if args.smoke else ""))
WIDTHS = sorted(set(args.widths))
TASKS = NT18_DIR if args.tasks == ["dir"] else NT18_SYM + NT18_DIR if args.tasks == ["all"] else args.tasks


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if OUT.exists():
        with open(OUT / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------------------------- centre readout
def central(L, w):
    """Index range of the central window: closed under i -> L-1-i (so it maps onto itself on the other strand)."""
    n = min(L, 2 * w + (L % 2))
    lo = (L - n) // 2
    return lo, lo + n


class CenterClassifier(rk.Classifier):
    """rk.Classifier + mean/max of the decoder output Y over central windows. Same interface as rk.Classifier."""

    def __init__(self, backbone, n_classes, oriented=False, use_length=False, use_base=False):
        super().__init__(backbone, n_classes, oriented=oriented, use_length=use_length, use_base=True)
        D = 2 * backbone.D * 2 + (1 if use_length else 0)
        Dc = 2 * backbone.D * len(WIDTHS)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(D + Dc), torch.nn.Linear(D + Dc, n_classes))

    def forward(self, x):
        out = self.backbone(x, decode=True)
        z = torch.cat([self.backbone.pooled_features(out, self.oriented),
                       self.backbone.pooled_base(out, self.oriented)], -1)
        if self.use_length:
            n = (x < 4).sum(1, keepdim=True).float().clamp(min=1)
            z = torch.cat([z, torch.log(n) / 7.0], -1)
        Y = out["Y"].float()
        cz = []
        for w in WIDTHS:
            a, b = central(Y.shape[1], w)
            Yc = Y[:, a:b]
            cz.append(self.backbone._pool_sym(Yc.mean(1), Yc.amax(1), self.oriented))
        return self.head(torch.cat([z] + cz, -1)), out


def check_equivariance():
    torch.manual_seed(0)
    m = rk.Chunker(d=16, enc_layers=2, main_layers=2, dec_layers=1, heads=2, sym=True).eval()
    ok = True
    for L in (300, 301):
        x = torch.randint(0, 4, (4, L))
        for oriented in (False, True):
            clf = CenterClassifier(m, 2, oriented=oriented, use_base=True).eval()
            with torch.no_grad():
                out1, out2 = m(x, decode=True), m(rk.rc_tokens(x), decode=True)
                cz = []
                for o in (out1, out2):
                    Y = o["Y"].float()
                    a, b = central(L, 16)
                    cz.append(m._pool_sym(Y[:, a:b].mean(1), Y[:, a:b].amax(1), oriented))
                l1, _ = clf(x)
                l2, _ = clf(rk.rc_tokens(x))
            if oriented:                                   # orientation-aware: the strand halves must be swapped
                h = cz[0].shape[-1] // 4
                sw = lambda v: torch.cat([v[:, h:2 * h], v[:, :h], v[:, 3 * h:], v[:, 2 * h:3 * h]], -1)
                err = float((cz[0] - sw(cz[1])).abs().max())
                log(f"  L={L} orientation-aware: centre features are swapped strand halves (error {err:.1e})")
            else:
                err = float((l1 - l2).abs().max())
                log(f"  L={L} invariant: max logit difference between strands {err:.1e}")
            ok &= err < 1e-4
    return ok


# ----------------------------------------------------------------------------------------------- centring check
def check_centre():
    """Positive examples of the splice tasks: positions of the most enriched GT (donor) / AG (acceptor) dinucleotide
    relative to the window centre, to confirm the sites sit at the centre."""
    for t, dinuc in (("splice_sites_donors", "GT"), ("splice_sites_acceptors", "AG")):
        d = rk.load_nt_task(t, 10 ** 9, 10 ** 9, log=log)
        X, y = d["train"]
        pos = [x for x, l in zip(X, y) if l == 1][:5000]
        L = len(pos[0])
        a, b = (0, 1, 2, 3)["ACGT".index(dinuc[0])], (0, 1, 2, 3)["ACGT".index(dinuc[1])]
        S = np.stack([x for x in pos if len(x) == L])
        f = ((S[:, :-1] == a) & (S[:, 1:] == b)).mean(0)
        top = np.argsort(f)[::-1][:3]
        log(f"  {t}: window {L} bp, centre {L / 2:.1f}; {dinuc} most frequent at positions " +
            ", ".join(f"{i} ({100 * f[i]:.0f}% of positives, offset {i - L / 2:+.1f})" for i in top) +
            f"; background {100 * np.median(f):.1f}%")


# ----------------------------------------------------------------------------------------------- comparison
def compare():
    import pandas as pd
    from scipy.stats import wilcoxon

    def load(folder):
        f = Path(folder) / "results_suite.jsonl"
        rows = []
        for line in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []):
            if not line.strip():
                continue
            r = json.loads(line)
            extra = r["key"].split("|")[4:]
            if r.get("variant") != "RC-equivariant" or "bf" not in extra or "inv" in extra or "len" in extra \
                    or any(e.startswith("n") and e[1:].isdigit() for e in extra) or "ep10" not in extra:
                continue
            rows.append(dict(task=r["task"], seed=r["seed"], fwd=r["mcc"], flips=r.get("flip_rate")))
        return pd.DataFrame(rows).drop_duplicates(["task", "seed"], keep="last")

    b = load(OUT)
    if not len(b):
        raise SystemExit(f"no finished runs in {OUT} yet")
    m = load(args.main).merge(b, on=["task", "seed"], suffixes=("_base", "_ctr"))
    seeds = sorted(int(s) for s in m.seed.unique())
    t = m.groupby("task", as_index=False)[["fwd_base", "fwd_ctr"]].mean()
    t["dir"] = t.task.isin(NT18_DIR)
    nt = t.task.isin(NT18_SYM + NT18_DIR)
    groups = {"all finished": t.task.notna(), "NT 18": nt, "NT direction-dependent": t.dir,
              "NT splice sites": t.task.str.startswith("splice"), "NT promoters": t.task.str.startswith("promoter"),
              "NT strand-symmetric": nt & ~t.dir, "Genomic Benchmarks": t.task.isin(list(rk.GB_TASKS)),
              "Gene strand": t.task.str.startswith("bio_gene_strand")}
    lines = ["# MirrorDNA with and without the centre readout (forward MCC)", "",
             f"Seeds present in both runs: {seeds}; central windows of {', '.join(str(2 * w) for w in WIDTHS)} bp. "
             "Wilcoxon signed-rank test over tasks where there are at least 5.", "",
             "| tasks | n | MirrorDNA | + centre readout | difference | better / worse | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for g, sel in groups.items():
        d = t[sel]
        if not len(d):
            continue
        diff = d.fwd_ctr - d.fwd_base
        pv = wilcoxon(d.fwd_ctr, d.fwd_base).pvalue if len(d) >= 5 and (diff != 0).any() else float("nan")
        lines.append(f"| {g} | {len(d)} | {d.fwd_base.mean():.3f} | {d.fwd_ctr.mean():.3f} | {diff.mean():+.3f} | "
                     f"{(diff > 0).sum()} / {(diff < 0).sum()} | {pv:.3g} |")
    lines += ["", "| task | MirrorDNA | + centre readout | difference |", "|---|---|---|---|"]
    for _, r in t.sort_values("task").iterrows():
        lines.append(f"| {r.task} | {r.fwd_base:.3f} | {r.fwd_ctr:.3f} | {r.fwd_ctr - r.fwd_base:+.3f} |")
    if len(seeds) < 3:
        lines += ["", f"Only {len(seeds)} seed(s): treat as preliminary."]
    (OUT / "compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    if args.compare:
        compare()
        raise SystemExit
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check_centre:
        check_centre()
        raise SystemExit
    log(f"centre readout: central windows of {', '.join(str(2 * w) for w in WIDTHS)} bp; tasks {TASKS}")
    if not check_equivariance():
        raise SystemExit("equivariance broken; stop")
    log("equivariance check passed (centre readout)")
    src = Path(args.main)
    if args.smoke:
        mcfg = dict(d=16, enc_layers=2, main_layers=2, dec_layers=1, heads=2, target_ratio=4.0, enc="conv")
        if not (OUT / "pretrain_RC-equivariant.final.pt").exists():
            torch.save(rk.build("RC-equivariant", mcfg).state_dict(), OUT / "pretrain_RC-equivariant.final.pt")
            (OUT / "pretrain_info.json").write_text(json.dumps(dict(mcfg=mcfg)))
    else:
        for name in ("pretrain_RC-equivariant.final.pt", "pretrain_info.json"):
            if not (src / name).exists():
                raise SystemExit(f"{src / name} not found")
            if not (OUT / name).exists():
                shutil.copy2(src / name, OUT / name)                     # copy: the main run folder is untouched
    rk.Classifier = CenterClassifier           # this process only: rk.finetune now builds the centre-readout head
    argv = ["run_suite.py", str(OUT), "--suite", "nt18", "--tasks", *TASKS, "--seeds", *map(str, args.seeds),
            "--out", str(OUT / "suite_report.md"), "--base-features"]
    if args.full:                                                       # same suites as the main run
        argv = ["run_suite.py", str(OUT), "--suite", "nt18", "gb", "bio", "--seeds", *map(str, args.seeds),
                "--out", str(OUT / "suite_report.md"), "--base-features"]
    if args.smoke:
        argv = ["run_suite.py", str(OUT), "--suite", "nt18", "--tasks", "H3K4me3", "promoter_all", "--seeds", "0",
                "--out", str(OUT / "suite_report.md"), "--base-features", "--smoke", "--offline"]
    log("fine-tuning: " + " ".join(argv))
    sys.argv = argv
    try:
        runpy.run_path(str(Path(__file__).with_name("run_suite.py")), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise SystemExit(f"run_suite.py stopped ({e.code}); rerun this same command to continue where it stopped")
    if not args.smoke:
        log("fine-tuning done")
        compare()