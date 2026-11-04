import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("models", nargs="*", help="run folders and/or caduceus_ps_1k, caduceus_ph_1k")
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--no-base-features", action="store_true", help="MirrorDNA: chunk features only (default: + base)")
p.add_argument("--max-train", type=int, default=0, help="cap on training sequences (0 = all)")
p.add_argument("--data-dir", default="deepstarr_data")
p.add_argument("--out", default="results_deepstarr")
p.add_argument("--smoke", action="store_true", help="2,000 training sequences, 1 epoch, 1 seed")
p.add_argument("--report-only", action="store_true")
args = p.parse_args()
if args.smoke:
    args.seeds, args.epochs, args.max_train = args.seeds[:1], 1, 2000
    args.out += "_smoke"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RES, LOG = OUT / "results.jsonl", OUT / "log.txt"
TARGETS = ["dev", "hk"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_rows():
    return [json.loads(l) for l in RES.read_text(encoding="utf-8").splitlines() if l.strip()] if RES.exists() else []


# ----------------------------------------------------------------------------------------------- data
ZENODO = "https://zenodo.org/records/5502060/files/{}?download=1"
SPLITS = ["Train", "Val", "Test"]


def read_fasta(path):
    seqs, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        elif line:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def fetch(name, d):
    f = d / name
    if not f.exists():
        log(f"  downloading {name} ...")
        try:
            urllib.request.urlretrieve(ZENODO.format(name), f)
        except Exception as e:
            if f.exists():
                f.unlink()
            raise SystemExit(f"Could not download {name} ({e}).\nDownload the DeepSTARR files from "
                             f"https://zenodo.org/records/5502060 into {d}/ and rerun.")
    return f


def load_deepstarr():
    """dict(train/val/test = (list of uint8 token arrays, float32 [n, 2] targets dev/hk)), cached as one npz."""
    import pandas as pd
    d = Path(args.data_dir)
    d.mkdir(exist_ok=True)
    cache = d / "deepstarr.npz"
    if not cache.exists():
        out = {}
        if (d / "Sequences_activity_all.txt").exists():                 # alternative single-table format
            t = pd.read_table(d / "Sequences_activity_all.txt")
            for s in SPLITS:
                u = t[t["set"] == s]
                out[s] = (list(u["Sequence"]), u[["Dev_log2_enrichment", "Hk_log2_enrichment"]].to_numpy(np.float32))
        else:
            for s in SPLITS:
                seqs = read_fasta(fetch(f"Sequences_{s}.fa", d))
                act = pd.read_table(fetch(f"Sequences_activity_{s}.txt", d))
                if len(seqs) != len(act):
                    raise SystemExit(f"{s}: {len(seqs)} sequences but {len(act)} activity rows")
                out[s] = (seqs, act[["Dev_log2_enrichment", "Hk_log2_enrichment"]].to_numpy(np.float32))
        arrays = {}
        for s, (seqs, y) in out.items():
            arrays[s + "_x"] = np.stack([rk.encode(q.upper()) for q in seqs]).astype(np.uint8)
            arrays[s + "_y"] = y
        np.savez_compressed(cache, **arrays)
    z = np.load(cache)
    data = {}
    for s, k in zip(SPLITS, ["train", "val", "test"]):
        X, y = z[s + "_x"], z[s + "_y"]
        if k == "train" and args.max_train:
            idx = np.random.default_rng(0).permutation(len(X))[:args.max_train]
            X, y = X[idx], y[idx]
        if args.smoke and k != "train":
            X, y = X[:1000], y[:1000]
        data[k] = ([x for x in X], y)
    return data


# ----------------------------------------------------------------------------------------------- metrics
def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def metrics(Y, P, Prc, sd):
    """Y, P, Prc: [n, 2] in original units; sd: target SD on the training set."""
    r = {}
    for i, t in enumerate(TARGETS):
        r[f"pearson_{t}"] = pearson(P[:, i], Y[:, i])
        r[f"pearson_{t}_rc"] = pearson(Prc[:, i], Y[:, i])
        r[f"pearson_{t}_avg"] = pearson((P[:, i] + Prc[:, i]) / 2, Y[:, i])
        r[f"strand_r_{t}"] = pearson(P[:, i], Prc[:, i]) if np.ptp(P[:, i] - Prc[:, i]) > 0 else 1.0
        gap = np.abs(P[:, i] - Prc[:, i]) / sd[i]
        r[f"strand_gap_{t}"] = float(gap.mean())
        r[f"strand_gap_big_{t}"] = float((gap > 0.5).mean())
    r["pearson_mean"] = (r["pearson_dev"] + r["pearson_hk"]) / 2
    r["pearson_mean_rc"] = (r["pearson_dev_rc"] + r["pearson_hk_rc"]) / 2
    r["strand_gap"] = (r["strand_gap_dev"] + r["strand_gap_hk"]) / 2
    r["strand_gap_big"] = (r["strand_gap_big_dev"] + r["strand_gap_big_hk"]) / 2
    r["strand_r"] = (r["strand_r_dev"] + r["strand_r_hk"]) / 2
    return r


# ----------------------------------------------------------------------------------------------- backends
class Ours:
    """A pretrained MirrorDNA / chunker variant with the same pooled head as for classification (2 outputs)."""
    amp_eval = False                                              # fp32 at test time: exact equivariance

    def __init__(self, variant, mcfg, state):
        self.variant, self.rcaug = variant, rk.VARIANTS[variant][1]
        bb = rk.build(variant, mcfg)
        bb.load_state_dict(state)
        self.net = rk.Classifier(bb, 2, oriented=False, use_base=not args.no_base_features).to(device)

    def __call__(self, xb):                                       # xb: [B, L] our token ids on the device
        y, out = self.net(xb)
        return y, rk.RATIO_W * out["ratio"]


CADUCEUS = {"caduceus_ps_1k": ("yairschiff/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3", True,
                               "Caduceus-PS-1k"),
            "caduceus_ph_1k": ("yairschiff/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3", False,
                               "Caduceus-PH-1k +RCaug")}
CADUCEUS_CODE = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"


class CaduceusHead(nn.Module):
    def __init__(self, bb, ps, dim):
        super().__init__()
        self.bb, self.ps = bb, ps
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 2))

    def forward(self, ids):
        out = self.bb(input_ids=ids)
        h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).float()
        mean, mx = h.mean(1), h.amax(1)
        if self.ps:                                               # published RC-invariant pooling (Caduceus-PS)
            mean, mx = 0.5 * (mean + mean.flip(-1)), 0.5 * (mx + mx.flip(-1))
        return self.head(torch.cat([mean, mx], -1))


class Caduceus:
    """1-kb Caduceus checkpoints (same pretraining data and budget as ours), loaded as in run_public.py."""
    amp_eval = True

    def __init__(self, name):
        import importlib.util
        if importlib.util.find_spec("mamba_ssm") is None:
            raise SystemExit("Caduceus needs mamba-ssm: run this in the WSL environment with Caduceus installed")
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        repo, ps, _ = CADUCEUS[name]
        self.rcaug = not ps                                       # PH: published protocol (RC augmentation)
        Cfg = get_class_from_dynamic_module("configuration_caduceus.CaduceusConfig", CADUCEUS_CODE)
        Mdl = get_class_from_dynamic_module("modeling_caduceus.Caduceus", CADUCEUS_CODE)
        bb, info = Mdl.from_pretrained(repo, config=Cfg.from_pretrained(repo), output_loading_info=True)
        if info.get("missing_keys"):
            raise RuntimeError(f"{repo}: {len(info['missing_keys'])} weights missing, e.g. {info['missing_keys'][:3]}")
        tok = AutoTokenizer.from_pretrained(CADUCEUS_CODE, trust_remote_code=True)
        lut = []
        for ch in "ACGTN":                                       # our ids 0..4 -> Caduceus ids, checked per letter
            ids = tok(ch * 3, add_special_tokens=False)["input_ids"]
            if len(ids) != 3 or len(set(ids)) != 1:
                raise RuntimeError("unexpected Caduceus tokenizer")
            lut.append(ids[0])
        lut += [lut[4]] * (int(max(rk.VOCAB, 8)) - 5)             # any other id (pad / mask) -> N
        self.lut = torch.tensor(lut, device=device)
        with torch.no_grad():
            bb = bb.to(device)                                    # mamba kernels run on the GPU only
            o = bb(input_ids=self.lut[:4][None])
            dim = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).shape[-1]
        self.net = CaduceusHead(bb, ps, dim).to(device)

    def __call__(self, xb):
        return self.net(self.lut[xb.clamp(max=len(self.lut) - 1)]), 0.0


def backends(spec):
    """(display name, run tag, factory) for every model named by one command-line entry."""
    if spec in CADUCEUS:
        return [(CADUCEUS[spec][2], spec, lambda: Caduceus(spec))]
    folder = Path(spec)
    out = []
    for f in sorted(folder.glob("pretrain_*.final.pt")):
        variant = f.name[len("pretrain_"):-len(".final.pt")]
        if variant not in rk.VARIANTS:
            continue
        info = json.loads((folder / "pretrain_info.json").read_text()) if (folder / "pretrain_info.json").exists() else {}
        mcfg = dict(info.get("mcfg", dict(d=160, enc_layers=4, main_layers=8, dec_layers=2, heads=8, target_ratio=4.0)))
        state = torch.load(f, map_location="cpu")
        mcfg["enc"] = "duplex" if any(".gate." in k for k in state) else "conv"
        label = {"RC-equivariant": "MirrorDNA" + ("-duplex" if mcfg["enc"] == "duplex" else " (ours)"),
                 "baseline+RCaug": "Chunker + RC aug.", "baseline": "Chunker (no aug.)"}[variant]
        out.append((label, f"{folder.name}/{variant}",
                    (lambda v=variant, m=mcfg, s=state: Ours(v, m, s))))
    if not out:
        raise SystemExit(f"{spec}: no pretrain_*.final.pt in this folder and not a Caduceus model name")
    return out


# ----------------------------------------------------------------------------------------------- training
def batches(X, Y, bs, rng=None):
    idx = np.arange(len(X))
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield torch.from_numpy(np.stack([X[j] for j in b])).long(), torch.from_numpy(Y[b])


@torch.no_grad()
def predict(model, X, Y, amp):
    model.net.eval()
    P, Prc = [], []
    for xb, _ in batches(X, Y, 4 * args.bs):
        xb = xb.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device == "cuda")):
            P.append(model(xb)[0].float().cpu())
            Prc.append(model(rk.rc_tokens(xb))[0].float().cpu())
    return torch.cat(P).numpy(), torch.cat(Prc).numpy()


def run(factory, data, seed, ckpt):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = factory()
    (Xtr, ytr), (Xv, yv), (Xte, yte) = data["train"], data["val"], data["test"]
    mu, sd = ytr.mean(0), ytr.std(0)                              # standardized targets for the loss
    ztr = ((ytr - mu) / sd).astype(np.float32)
    opt = torch.optim.AdamW(model.net.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * math.ceil(len(Xtr) / args.bs)
    step, best, best_ep, best_state, ep0, spent = 0, -2.0, 0, None, 0, 0.0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        step, best, best_ep, best_state, ep0, spent = (st[k] for k in ("step", "best", "best_ep", "best_state",
                                                                         "epoch", "spent"))
        rng.bit_generator.state = st["rng"]
        log(f"    resumed after epoch {ep0}")
    t0 = time.time()
    for ep in range(ep0, args.epochs):
        if args.patience and ep - best_ep >= args.patience:
            log(f"    early stop after epoch {ep}")
            break
        model.net.train()
        losses = []
        for xb, yb in batches(Xtr, ztr, args.bs, rng):
            for g in opt.param_groups:
                g["lr"] = rk.lr_at(step, total, args.lr, warmup=max(1, total // 20))
            xb, yb = xb.to(device), yb.to(device)
            if model.rcaug:                                       # label is strand-independent: augmentation is valid
                flip = torch.rand(len(xb), device=device) < 0.5
                xb = torch.where(flip[:, None], rk.rc_tokens(xb), xb)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                pred, extra = model(xb)
            loss = F.mse_loss(pred.float(), yb) + extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(loss.item())
        P, _ = predict(model, Xv, yv, amp=True)
        v = (pearson(P[:, 0], yv[:, 0]) + pearson(P[:, 1], yv[:, 1])) / 2
        log(f"    epoch {ep + 1}/{args.epochs}: train loss {np.mean(losses):.4f} | val Pearson {v:.3f} "
            f"({(spent + time.time() - t0) / 60:.1f} min)")
        if v > best:
            best, best_ep = v, ep + 1
            best_state = {k: t.detach().cpu().clone() for k, t in model.net.state_dict().items()}
        tmp = ckpt.with_suffix(".tmp")
        torch.save(dict(net=model.net.state_dict(), opt=opt.state_dict(), step=step, best=best, best_ep=best_ep,
                        best_state=best_state, epoch=ep + 1, spent=spent + time.time() - t0,
                        rng=rng.bit_generator.state), tmp)
        tmp.replace(ckpt)
    model.net.load_state_dict(best_state)
    P, Prc = predict(model, Xte, yte, amp=model.amp_eval)
    P, Prc = P * sd + mu, Prc * sd + mu                           # back to log2 enrichment
    res = dict(**metrics(yte, P, Prc, sd), val_pearson=best, best_epoch=best_ep,
               train_min=(spent + time.time() - t0) / 60, params=sum(q.numel() for q in model.net.parameters()))
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------------------------- report
def report():
    import pandas as pd
    rows = read_rows()
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "pearson_dev_avg" in df:                      # prediction averaged over both strands (test-time averaging)
        df["pearson_mean_avg"] = (df.pearson_dev_avg + df.pearson_hk_avg) / 2
    cols = ["pearson_dev", "pearson_hk", "pearson_mean", "pearson_mean_rc", "pearson_mean_avg", "strand_r", "strand_gap",
            "strand_gap_big"]
    cols = [c for c in cols if c in df]
    g = df.groupby("model")
    tab = pd.DataFrame({c: g[c].mean() for c in cols})
    sds = pd.DataFrame({c + "_sd": g[c].std(ddof=1) for c in cols})
    tab = tab.join(sds)
    tab["seeds"] = g["seed"].nunique()
    tab["params_M"] = g["params"].first() / 1e6
    tab = tab.sort_values("pearson_mean", ascending=False)
    tab.round(4).to_csv(OUT / "deepstarr.csv")

    def f(r, c, pct=False):
        m, s = r[c], r[c + "_sd"]
        if pct:
            return f"{100 * m:.1f}%" + ("" if pd.isna(s) else f" ± {100 * s:.1f}")
        return f"{m:.3f}" + ("" if pd.isna(s) else f" ± {s:.3f}")

    lines = ["# DeepSTARR enhancer-activity regression, on both strands", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Official test set; mean ± SD over seeds. "
             "Pearson r on the test sequences as given (Dev, Hk, mean) and reverse-complemented (RC mean). "
             "strand r: correlation between the predictions for a sequence and for its reverse complement; "
             "gap: mean |fwd − RC| in target SD units; >0.5 SD: share of sequences whose two predictions differ by "
             "more than half a standard deviation.", "",
             "| model | params | Dev r | Hk r | mean r | RC mean r | averaged mean r | strand r | gap (SD) | >0.5 SD | seeds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in tab.iterrows():
        lines.append(f"| {m} | {r.params_M:.2f}M | {f(r, 'pearson_dev')} | {f(r, 'pearson_hk')} | "
                     f"**{f(r, 'pearson_mean')}** | {f(r, 'pearson_mean_rc')} | {f(r, 'pearson_mean_avg')} | {f(r, 'strand_r')} | "
                     f"{f(r, 'strand_gap')} | {f(r, 'strand_gap_big', pct=True)} | {int(r.seeds)} |")
    ours = [m for m in tab.index if m.startswith("MirrorDNA (ours)")]
    if ours:
        try:
            from scipy.stats import ttest_ind
            lines += ["", "Welch t-test over seeds, MirrorDNA (ours) vs each model (mean Pearson r; with 3 seeds per "
                      "model the test has little power):", ""]
            a = df[df.model == ours[0]].pearson_mean
            for m in tab.index:
                if m != ours[0]:
                    b = df[df.model == m].pearson_mean
                    if len(a) > 1 and len(b) > 1:
                        lines.append(f"- vs {m}: Δ = {a.mean() - b.mean():+.3f}, p = {ttest_ind(a, b, equal_var=False).pvalue:.3g}")
        except ImportError:
            pass
    lines += ["", "Reference: DeepSTARR (CNN trained from scratch) reports test Pearson r = 0.68 (Dev) and 0.74 (Hk)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if args.report_only or not args.models:
        report()
        raise SystemExit
    data = load_deepstarr()
    log(f"DeepSTARR: train {len(data['train'][0]):,}, val {len(data['val'][0]):,}, test {len(data['test'][0]):,} "
        f"sequences of {len(data['train'][0][0])} bp | device {device} | seeds {args.seeds}")
    done = {r["key"] for r in read_rows()}
    (OUT / "ckpt").mkdir(exist_ok=True)
    for spec in args.models:
        for name, tag, factory in backends(spec):
            for seed in args.seeds:
                key = f"{tag}|seed{seed}|lr{args.lr:g}|ep{args.epochs}p{args.patience}" + \
                      ("" if args.no_base_features or tag.startswith("caduceus") else "|bf") + \
                      (f"|n{args.max_train}" if args.max_train else "")
                if key in done:
                    continue
                log(f"{name} | seed {seed}")
                ck = OUT / "ckpt" / (key.replace("|", "_").replace("/", "_") + ".pt")
                r = run(factory, data, seed, ck)
                with open(RES, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict(key=key, model=name, seed=seed, **r)) + "\n")
                log(f"{name} seed {seed}: Pearson Dev {r['pearson_dev']:.3f} | Hk {r['pearson_hk']:.3f} | "
                    f"RC mean {r['pearson_mean_rc']:.3f} | strand gap {r['strand_gap']:.3f} SD "
                    f"({100 * r['strand_gap_big']:.1f}% > 0.5 SD) | {r['train_min']:.1f} min | "
                    f"{r['params'] / 1e6:.2f}M params")
                ck.unlink(missing_ok=True)
                report()
    report()
    log(f"DONE -> {OUT / 'report.md'}")
