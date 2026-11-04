import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk

SMALL = dict(d=160, enc_layers=4, main_layers=8, dec_layers=2, heads=8, target_ratio=4.0)   # Day-1 size
NT4 = {"H3K4me3": False, "H3K27ac": False, "H3K36me3": False, "enhancers": False}           # name -> directional?
NT18 = {t: False for t in ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac",
                           "H3K9me3", "H4K20me1", "enhancers", "enhancers_types"]}
NT18.update({t: True for t in ["promoter_all", "promoter_no_tata", "promoter_tata", "splice_sites_acceptors",
                               "splice_sites_all", "splice_sites_donors"]})                  # direction-dependent

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="+", help="folders holding pretrain_<variant>.final.pt")
p.add_argument("--suite", nargs="+", default=["gb"], choices=["nt4", "nt18", "gb", "bio", "biolong"])
p.add_argument("--tasks", nargs="+", default=None, help="subset of task names")
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--gb-max-train", type=int, default=10 ** 9, help="cap on training sequences per Genomic Benchmarks task (default: all)")
p.add_argument("--epochs", type=int, default=10, help="max epochs (Caduceus: 10), with early stopping")
p.add_argument("--patience", type=int, default=3, help="stop after this many epochs without validation gain (0 = off)")
p.add_argument("--nt-max-train", type=int, default=10 ** 9, help="cap on NT training sequences (default: all; earlier runs used 20000)")
p.add_argument("--nt-max-test", type=int, default=10 ** 9)
p.add_argument("--invariant-head", action="store_true",
               help="force the strand-invariant head even on direction-dependent tasks (shows what an RC-invariant model like Caduceus-PS can do)")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--out", default="suite_report.md")
p.add_argument("--length-feature", action="store_true", help="give the classifier the sequence length (strand-invariant)")
p.add_argument("--base-features", action="store_true",
               help="classifier also pools the per-base decoder output (single-base resolution; still strand-exact)")
p.add_argument("--report-only", action="store_true", help="rebuild the report from saved results, no training")
p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, 1 seed: code check")
p.add_argument("--offline", action="store_true", help="synthetic data instead of downloads")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
if args.smoke:
    args.seeds, args.epochs = args.seeds[:1], 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_rows(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def models_in(folder):
    folder = Path(folder)
    info_f = folder / "pretrain_info.json"
    info = json.loads(info_f.read_text()) if info_f.exists() else {}
    mcfg = dict(info.get("mcfg", SMALL))
    for f in sorted(folder.glob("pretrain_*.final.pt")):
        variant = f.name[len("pretrain_"):-len(".final.pt")]
        if variant in rk.VARIANTS:
            keys = torch.load(f, map_location="cpu").keys()        # the weights say which encoder was trained
            mcfg["enc"] = "duplex" if any(".gate." in k for k in keys) else "conv"
            size = f"{rk.n_params(rk.build(variant, mcfg)) / 1e6:.0f}M"
            enc = ", duplex" if mcfg.get("enc") == "duplex" else ""
            yield variant, mcfg, f, f"{variant} ({size}{enc})"


def task_list():
    out = []
    for suite in args.suite:
        reg = {"nt4": NT4, "nt18": NT18, "bio": rk.BIO_TASKS, "biolong": getattr(rk, "BIO_LONG_TASKS", {})}.get(suite, rk.GB_TASKS)
        out += [(suite, t, directional) for t, directional in reg.items() if args.tasks is None or t in args.tasks]
    return out


def load(suite, t, cache={}):
    if (suite, t) not in cache:
        cache.clear()                                    # keep only the current task in memory
        if args.offline:
            cache[suite, t] = rk.synthetic_task(t, n_train=600, n_test=200)
        elif suite in ("nt4", "nt18"):
            n_tr, n_te = (800, 300) if args.smoke else (args.nt_max_train, args.nt_max_test)
            cache[suite, t] = rk.load_nt_task(t, n_tr, n_te, log=log)
        else:
            n_tr, n_te = (800, 300) if args.smoke else (args.gb_max_train, 10 ** 9)
            cache[suite, t] = rk.load_task(t, n_tr, n_te, log=log)
    return cache[suite, t]


def cap_tag(suite):
    """Label-budget runs (--nt-max-train / --gb-max-train below the full data) get their own key and model name."""
    cap = args.nt_max_train if suite in ("nt4", "nt18") else args.gb_max_train if suite == "gb" else 10 ** 9
    return f"|n{cap}" if cap < 10 ** 9 else ""


def fmt(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def make_report(rows):
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df.model))
    lines = ["# Benchmark suite: every metric on both strands", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Fine-tuning: AdamW lr {args.lr:g}, {args.epochs} epochs, "
             f"best epoch on a 10% validation split (early stopping, patience {args.patience}); test in fp32. Mean over seeds.", "",
             "- **fwd**: test set as given. **RC**: the same sequences reverse-complemented. "
             "**robust** = mean of fwd and RC (for each metric).",
             "- **Consistent**: correct only if correct on BOTH strands. **Flips**: predictions that change with the strand.",
             "- Direction-dependent tasks (marked *) use the orientation-aware head; only forward metrics apply.", ""]
    METRICS = [("MCC", "mcc", "mcc_rc_input", "consistent_mcc"), ("F1", "f1", "f1_rc", "consistent_f1"),
               ("Acc", "acc", "acc_rc", "consistent_acc")]
    head = "| Task | Model | " + " | ".join(f"{n} fwd | {n} RC | {n} robust | {n} cons." for n, *_ in METRICS)
    head += " | Flips | seeds |"
    sep = "|" + "---|" * (2 + 4 * len(METRICS) + 2)

    def cells(s, dirn):
        out = []
        for _, fw, rc, cons in METRICS:
            f = s[fw].mean()
            if dirn:                                  # direction-dependent label: only the forward metric applies
                out += [fmt(f), "—", "—", "—"]
            else:
                r = s[rc].mean()
                out += [fmt(f), fmt(r), fmt((f + r) / 2), f"**{fmt(s[cons].mean())}**"]
        return out

    for suite in args.suite:
        sd = df[df.suite == suite]
        if not len(sd):
            continue
        title = {"nt4": "NT-revised (4 tasks)", "nt18": "NT-revised (all 18 tasks)",
                 "bio": "Biology: which strand is the gene on? (strand-dependent)"}.get(suite, "Genomic Benchmarks")
        lines += [f"## {title}", "", head, sep]
        for t in dict.fromkeys(sd.task):
            for m in order:
                s = sd[(sd.task == t) & (sd.model == m)]
                if not len(s):
                    continue
                dirn = bool(s.directional.iloc[0])
                flips = "—" if dirn else f"{s.flip_rate.mean():.1%}"
                lines.append(f"| {t}{'*' if dirn else ''} | {m} | " + " | ".join(cells(s, dirn)) + f" | {flips} | {len(s)} |")
        g = sd.groupby(["model", "task"]).mean(numeric_only=True).reset_index()
        g["directional"] = g.directional.astype(bool)
        lines += ["", "**Averages over tasks.** Forward metrics: all tasks. RC / robust / consistent / flips: "
                  "strand-symmetric tasks only (direction-dependent tasks have no RC label).", "",
                  "| Model | " + " | ".join(f"{n} fwd (all) | {n} fwd | {n} RC | {n} robust | {n} cons." for n, *_ in METRICS)
                  + " | Flips | tasks |", "|" + "---|" * (2 + 5 * len(METRICS) + 1)]
        for m in order:
            a = g[g.model == m]
            b = a[~a.directional]
            if not len(a):
                continue
            row = []
            for _, fw, rc, cons in METRICS:
                if len(b):
                    f, r = b[fw].mean(), b[rc].mean()
                    row += [fmt(a[fw].mean()), fmt(f), fmt(r), fmt((f + r) / 2), f"**{fmt(b[cons].mean())}**"]
                else:
                    row += [fmt(a[fw].mean()), "—", "—", "—", "—"]
            flips = f"{b.flip_rate.mean():.1%}" if len(b) else "—"
            lines.append(f"| {m} | " + " | ".join(row) + f" | {flips} | {len(a)} |")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    tasks = task_list()
    if args.report_only:
        rows = [r for f in args.folders for r in read_rows(Path(f) / ("results_suite_smoke.jsonl" if args.smoke
                                                                         else "results_suite.jsonl"))]
        make_report(rows)
        raise SystemExit(f"report rebuilt from {len(rows)} saved results -> {args.out}")
    log(f"device {device} | {len(tasks)} tasks | seeds {args.seeds} | models from {args.folders}")
    all_rows = []
    for folder in args.folders:
        folder = Path(folder)
        res_f = folder / ("results_suite_smoke.jsonl" if args.smoke else "results_suite.jsonl")
        (folder / "preds").mkdir(exist_ok=True)
        (folder / "ft_ckpt").mkdir(exist_ok=True)
        found = list(models_in(folder))
        if not found:
            log(f"no pretrain_*.final.pt in {folder}, skipping")
        for variant, mcfg, wpath, name in found:
            rows = [r for r in read_rows(res_f) if r["variant"] == variant]
            done = {r["key"] for r in rows}
            state = torch.load(wpath, map_location="cpu")
            for suite, t, directional in tasks:
                for seed in args.seeds:
                    key = f"{suite}|{variant}|{t}|{seed}" + cap_tag(suite) + ("|len" if args.length_feature else "") + ("|bf" if args.base_features else "") + ("|inv" if args.invariant_head else "") + (f"|ep{args.epochs}" if args.epochs != 5 else "") + ("|dirfix" if directional and not args.invariant_head and rk.VARIANTS[variant][1] else "")
                    if key in done:
                        continue
                    data = load(suite, t)
                    ck = folder / "ft_ckpt" / (f"{suite}_{variant}_{t}_seed{seed}" + (cap_tag(suite) + ("|len" if args.length_feature else "") + ("|bf" if args.base_features else "") + ("|inv" if args.invariant_head else "") + (f"|ep{args.epochs}" if args.epochs != 5 else "") + ("|dirfix" if directional and not args.invariant_head and rk.VARIANTS[variant][1] else "")).replace("|", "_") + ".pt")
                    r = rk.finetune(variant, mcfg, data, device, seed, args.lr, args.epochs, args.bs, init_state=state,
                                    log=log, eval_amp=False, oriented=directional and not args.invariant_head, ckpt=ck, use_length=args.length_feature, use_base=args.base_features,
                                    patience=args.patience,
                                    save_preds=folder / "preds" / (f"{suite}_{variant}_{t}_seed{seed}" + (cap_tag(suite) + ("|len" if args.length_feature else "") + ("|bf" if args.base_features else "") + ("|inv" if args.invariant_head else "")).replace("|", "_") + ".npz"))
                    row = dict(key=key, suite=suite, model=name + cap_tag(suite).replace("|n", " n=") + (" +length" if args.length_feature else "") + (" +base" if args.base_features else "") + (" [invariant head]" if args.invariant_head else ""),
                               variant=variant, task=t, seed=seed,
                               directional=directional, **r)
                    with open(res_f, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    ck.unlink(missing_ok=True)                 # the result is saved; the epoch checkpoint is not needed
                    rows.append(row)
                    log(f"{name:28s} {t:34s} seed {seed}: MCC {r['mcc']:.3f} | F1 {r['f1']:.3f} | acc {r['acc']:.3f}"
                        + ("" if directional else f" | RC: MCC {r['mcc_rc_input']:.3f} F1 {r['f1_rc']:.3f} "
                           f"acc {r['acc_rc']:.3f} | cons. MCC {r['consistent_mcc']:.3f} | flips {r['flip_rate']:.1%}")
                        + f" | {r['train_min']:.1f} min")
                    make_report(all_rows + rows)
            all_rows += rows
    make_report(all_rows)
    log(f"DONE -> {args.out}")