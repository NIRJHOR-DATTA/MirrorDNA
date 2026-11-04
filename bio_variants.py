import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="+", help="run folders holding pretrain_<variant>.final.pt")
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--dataset", default="songlab/clinvar")
p.add_argument("--variants-csv", default=None)
p.add_argument("--max-variants", type=int, default=0, help="0 = all")
p.add_argument("--window", type=int, default=1024)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--out", default="bio_variants")
args = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
LUT = {"A": 0, "C": 1, "G": 2, "T": 3}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_variants():
    import pandas as pd
    if args.variants_csv:
        df = pd.read_csv(args.variants_csv)
    else:
        from datasets import load_dataset
        df = load_dataset(args.dataset, split="test").to_pandas()
    df = df[df.ref.str.len().eq(1) & df.alt.str.len().eq(1)].copy()
    df["chrom"] = df.chrom.astype(str).map(lambda c: c if c.startswith("chr") else "chr" + c)
    df["label"] = df.label.astype(int)
    if args.max_variants and len(df) > args.max_variants:
        df = df.sample(args.max_variants, random_state=0)
    return df.reset_index(drop=True)


def windows(df):
    """Reference windows with the variant at index c; drops variants near chromosome ends or with ref mismatch."""
    L, c = args.window, args.window // 2
    X, keep, mism = [], [], 0
    cache = {}
    for i, r in df.iterrows():
        if r.chrom not in cache:
            cache[r.chrom] = rk.load_chrom(args.genome_dir, r.chrom)
        tok = cache[r.chrom]
        if tok is None:
            continue
        s = int(r.pos) - 1 - c
        if s < 0 or s + L > len(tok):
            continue
        w = np.array(tok[s:s + L], dtype=np.uint8)
        if w[c] != LUT.get(r.ref.upper(), -1):
            mism += 1
            continue
        X.append(w)
        keep.append(i)
    log(f"{len(keep)} variants with windows ({mism} dropped: reference allele does not match hg38)")
    return np.stack(X), np.array(keep)


@torch.no_grad()
def score(model, X, ref, alt, rc):
    """MLM log-likelihood ratio at the (mirrored) variant position, fp32."""
    L, c = X.shape[1], X.shape[1] // 2
    out = np.empty(len(X), np.float64)
    for i in range(0, len(X), args.bs):
        x = torch.from_numpy(X[i:i + args.bs]).long().to(device)
        r = torch.from_numpy(ref[i:i + args.bs]).long().to(device)
        a = torch.from_numpy(alt[i:i + args.bs]).long().to(device)
        pos = c
        if rc:
            x, pos, r, a = rk.rc_tokens(x), L - 1 - c, 3 - r, 3 - a
        inp = x.clone()
        inp[:, pos] = rk.MASK
        m = torch.zeros_like(inp, dtype=torch.bool)
        m[:, pos] = True
        lp = F.log_softmax(model(inp, mask_pos=m)["logits"][:, pos].float(), -1)
        out[i:i + len(x)] = (lp.gather(1, a[:, None]) - lp.gather(1, r[:, None])).squeeze(1).cpu().numpy()
    return out


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, -s))            # lower log-likelihood ratio = more damaging


if __name__ == "__main__":
    from scipy.stats import spearmanr
    df = load_variants()
    log(f"{len(df)} SNVs ({df.label.sum()} pathogenic) from {args.variants_csv or args.dataset}")
    X, keep = windows(df)
    df = df.iloc[keep].reset_index(drop=True)
    y = df.label.values
    ref = df.ref.str.upper().map(LUT).values.astype(np.int64)
    alt = df.alt.str.upper().map(LUT).values.astype(np.int64)
    rows, scores = [], {"chrom": df.chrom, "pos": df.pos, "ref": df.ref, "alt": df.alt, "label": y}
    for folder in args.folders:
        for variant, mcfg, model, name in rk.load_pretrained(folder, device):
            t0 = time.time()
            sf, sr = score(model, X, ref, alt, False), score(model, X, ref, alt, True)
            thr = np.median((sf + sr) / 2)
            disagree = float(((sf < thr) != (sr < thr)).mean())
            rows.append(dict(model=f"{name} [{Path(folder).name}]", auc_fwd=auroc(y, sf), auc_rc=auroc(y, sr),
                             auc_avg=auroc(y, (sf + sr) / 2), spearman=float(spearmanr(sf, sr).correlation),
                             mad=float(np.abs(sf - sr).mean()), disagree=disagree))
            scores[f"{name} fwd"], scores[f"{name} rc"] = sf, sr
            r = rows[-1]
            log(f"{r['model']}: AUROC fwd {r['auc_fwd']:.3f} / RC {r['auc_rc']:.3f} | Spearman fwd-RC {r['spearman']:.4f}"
                f" | mean |diff| {r['mad']:.4f} | called differently {disagree:.1%} | {(time.time() - t0) / 60:.1f} min")

    import pandas as pd
    pd.DataFrame(scores).to_csv(f"{args.out}_scores.csv", index=False)
    lines = ["# Variant effect prediction on both strands", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. {len(y)} SNVs ({int(y.sum())} pathogenic, "
             f"{int(len(y) - y.sum())} benign), {args.window}-bp hg38 windows, zero-shot masked-LM log-likelihood ratio.",
             "", "| Model | AUROC fwd | AUROC RC | AUROC avg | Spearman fwd vs RC | mean abs. difference | "
             "called differently on the two strands |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['auc_fwd']:.3f} | {r['auc_rc']:.3f} | {r['auc_avg']:.3f} | {r['spearman']:.4f} | "
              f"{r['mad']:.4f} | {r['disagree']:.1%} |" for r in rows]
    # published scores shipped with the dataset (forward strand only), on exactly the same variants
    ref_cols = [c for c in ("NT", "HyenaDNA", "GPN-MSA", "ESM-1b") if c in df]
    if ref_cols:
        lines += ["", "Reference: published zero-shot scores in the dataset (forward strand only, lower = more damaging), "
                  "same variants:", "", "| Score | AUROC | variants scored |", "|---|---|---|"]
        for c_ in ref_cols:
            ok = df[c_].notna().values
            if ok.sum() > 100 and len(set(y[ok])) == 2:
                lines.append(f"| {c_} | {auroc(y[ok], df[c_].values[ok].astype(float)):.3f} | {int(ok.sum())} |")
                log(f"reference {c_}: AUROC {auroc(y[ok], df[c_].values[ok].astype(float)):.3f} ({int(ok.sum())} variants)")
    lines += ["", "A strand-consistent model has Spearman 1, difference 0 and 0% called differently: the same mutation "
              "gets the same score whichever strand is read."]
    Path(f"{args.out}_report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [k[:-4] for k in scores if k.endswith(" fwd")]
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
        for ax, n in zip(axes[0], names):
            ax.scatter(scores[n + " fwd"], scores[n + " rc"], s=2, alpha=0.3, c=np.where(y, "#c0392b", "#2c7fb8"))
            lo, hi = np.percentile(np.r_[scores[n + " fwd"], scores[n + " rc"]], [0.5, 99.5])
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(n, fontsize=9); ax.set_xlabel("score, forward strand"); ax.set_ylabel("score, reverse strand")
        fig.tight_layout()
        fig.savefig(f"{args.out}_scatter.png", dpi=200)
    except Exception as e:
        log(f"no figure ({e})")
    log(f"DONE -> {args.out}_report.md")