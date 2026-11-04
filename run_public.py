import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rcchunk as rk

MODELS = {  # name -> HF repo, tokenizer kind, mixed precision allowed?
    "hyenadna": dict(hf="LongSafari/hyenadna-small-32k-seqlen-hf", label="HyenaDNA-small (char)", amp=False,
                     special=False),   # Hyena FFT convolutions: keep fp32
    "nt100m": dict(hf="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", label="NT-v2-100M (6-mer)",
                   amp=True, special=True),
    "dnabert2": dict(hf="zhihan1996/DNABERT-2-117M", label="DNABERT-2 (BPE)", amp=True, special=True),
    # Caduceus needs the mamba-ssm CUDA kernels: Linux or WSL2 only.
    "caduceus_ps": dict(hf="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
                        label="Caduceus-PS (char, RC-equivariant)", amp=True, special=False, ps=True),
    "caduceus_ph": dict(hf="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
                        label="Caduceus-PH (char, RC-aug pretrained)", amp=True, special=False),
    # 1-kb checkpoints trained with the same recipe as ours (10k steps x 2^20 tokens, Caduceus paper): budget-matched
    "caduceus_ps_1k": dict(hf="yairschiff/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3", tok="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16", 
                           label="Caduceus-PS-1k (char, RC-equivariant)", amp=True, special=False, ps=True),
    "caduceus_ph_1k": dict(hf="yairschiff/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3", tok="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16", 
                           label="Caduceus-PH-1k (char, RC-aug pretrained)", amp=True, special=False),
    "hyenadna_1k": dict(hf="LongSafari/hyenadna-tiny-1k-seqlen-hf", label="HyenaDNA-tiny-1k (char)", amp=False,
                        special=False, max_bp=1024),   # fixed-length filters: longer inputs are cropped to the centre
}

p = argparse.ArgumentParser()
p.add_argument("--model", default="hyenadna", choices=list(MODELS))
p.add_argument("--tasks", nargs="+", default=["H3K4me3", "H3K27ac", "H3K36me3", "enhancers"])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--patience", type=int, default=0, help="early stopping: epochs without validation gain (0 = off)")
p.add_argument("--suite", nargs="+", default=None, choices=["nt4", "nt18", "gb", "bio", "biolong"],
               help="run whole suites instead of --tasks (uses full training sets)")
p.add_argument("--full-data", action="store_true", help="no 20k/5k cap on NT tasks")
p.add_argument("--bs", type=int, default=32)
p.add_argument("--max-attn", type=float, default=64e6,
               help="memory cap: a batch of b sequences of n tokens is split into micro-batches with b*n^2 <= this "
                    "(gradients accumulated, so the effective batch and the results are unchanged)")
p.add_argument("--out", default="results_public")
p.add_argument("--day1", default="results_day1/results.jsonl", help="our Phase C results, for the comparison table")
p.add_argument("--rcaug", action="store_true", help="reverse-complement augmentation during fine-tuning")
p.add_argument("--oriented", action="store_true",
               help="Caduceus-PS: keep the two strand halves apart (orientation-aware head, like ours) instead of the "
                    "published strand-invariant pooling; needed for direction-dependent labels")
p.add_argument("--fp32", action="store_true", help="disable bf16 mixed precision (use if losses become NaN)")
p.add_argument("--smoke", action="store_true")
p.add_argument("--offline", action="store_true")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
SPEC = MODELS[args.model]
OUT = Path(args.out + ("_smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
RES, LOG = OUT / "results.jsonl", OUT / "log.txt"
MAX_TRAIN, MAX_TEST = (600, 200) if args.smoke else ((10 ** 9, 10 ** 9) if (args.full_data or args.suite) else (20000, 5000))
if args.suite:
    _nt18 = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
             "H4K20me1", "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata",
             "splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"]
    _s = {"nt4": ["H3K4me3", "H3K27ac", "H3K36me3", "enhancers"], "nt18": _nt18, "gb": list(rk.GB_TASKS),
          "bio": list(rk.BIO_TASKS), "biolong": list(rk.BIO_LONG_TASKS)}
    args.tasks = [t for su in args.suite for t in _s[su]]
if args.smoke:
    args.seeds, args.epochs, args.tasks = args.seeds[:1], 1, args.tasks[:2]
NAME = ("stand-in (offline)" if args.offline else SPEC["label"]) + ("" if args.lr == 3e-4 else f" lr{args.lr:g}") + (" +RCaug" if args.rcaug else "") + (" [oriented head]" if args.oriented else "")
ALPH = np.array(list("ACGTN"))
COMP = str.maketrans("ACGTN", "TGCAN")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_rows(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def to_str(tok):
    return "".join(ALPH[np.minimum(tok, 4)])


def rc_str(s):
    return s.translate(COMP)[::-1]


# ----------------------------------------------------------------------------- backbones
class OfflineTok:
    """Character tokenizer stand-in (for --offline code checks)."""
    def __call__(self, seqs, **kw):
        ids = torch.tensor([[ "ACGTN".index(c) for c in s] for s in seqs])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class OfflineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.conv = nn.Embedding(5, 64), nn.Conv1d(64, 64, 9, padding=4)
        self.config = type("C", (), {"hidden_size": 64})()

    def forward(self, input_ids, attention_mask=None):
        h = torch.relu(self.conv(self.emb(input_ids).transpose(1, 2))).transpose(1, 2)
        return type("O", (), {"last_hidden_state": h, "hidden_states": None})()


CADUCEUS_CODE = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"


def load_caduceus_1k(repo):
    """The 1-kb Caduceus checkpoints ship model code written for mamba-ssm 1.x (it imports `Block` from
    mamba_ssm.modules.mamba_simple, which mamba-ssm 2.x moved). Load their weights with the updated code of the
    131k checkpoint (same architecture, same parameter names), keeping the 1-kb config (width 118, 4 layers, rcps)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    Cfg = get_class_from_dynamic_module("configuration_caduceus.CaduceusConfig", CADUCEUS_CODE)
    Mdl = get_class_from_dynamic_module("modeling_caduceus.Caduceus", CADUCEUS_CODE)
    cfg = Cfg.from_pretrained(repo)
    model, info = Mdl.from_pretrained(repo, config=cfg, output_loading_info=True)
    if info.get("missing_keys"):                    # never fine-tune a silently half-random model
        raise RuntimeError(f"{repo}: {len(info['missing_keys'])} weights not found in the checkpoint, e.g. "
                           f"{info['missing_keys'][:3]}")
    return model


def load_backbone():
    if args.offline:
        return OfflineTok(), OfflineModel()
    if args.model.startswith("caduceus"):
        import importlib.util
        if importlib.util.find_spec("mamba_ssm") is None:
            raise SystemExit("Caduceus needs `mamba-ssm` (Linux/WSL2 only): pip install causal-conv1d mamba-ssm --no-build-isolation")
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
    # the 1-kb Caduceus checkpoints ship no tokenizer; they use the same character tokenizer as the 131k ones
    tok = AutoTokenizer.from_pretrained(SPEC.get("tok", SPEC["hf"]), trust_remote_code=True)
    kw = dict(trust_remote_code=True)
    if args.model == "dnabert2":
        # Known DNABERT-2 issue with newer transformers: its remote code expects the standard BertConfig class.
        from transformers.models.bert.configuration_bert import BertConfig
        kw["config"] = BertConfig.from_pretrained(SPEC["hf"])
    if "seqlen-1k" in SPEC["hf"] and args.model.startswith("caduceus"):
        model = load_caduceus_1k(SPEC["hf"])
        log(f"  {SPEC['hf']}: loaded with the 131k checkpoint's model code ({sum(q.numel() for q in model.parameters()) / 1e6:.2f}M params)")
        return tok, model
    errors = []
    for cls in (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(SPEC["hf"], **kw)
            if args.model == "dnabert2":
                # DNABERT-2 ships a Triton flash-attention kernel written for an old Triton API
                # (tl.dot(..., trans_a=True)); with any recent Triton it crashes in the backward pass.
                # Disabling it makes the model use its own plain PyTorch attention (same math, slower).
                import sys
                n = 0
                for name, mod in list(sys.modules.items()):
                    if "DNABERT" in name and getattr(mod, "flash_attn_qkvpacked_func", None) is not None:
                        mod.flash_attn_qkvpacked_func = None
                        n += 1
                log(f"  DNABERT-2: Triton flash-attention disabled in {n} module(s); using PyTorch attention")
            return tok, model
        except Exception as e:
            errors.append(f"{cls.__name__}: {type(e).__name__}: {str(e)[:300]}")
    raise RuntimeError("Could not load the model. Every attempt failed:\n  " + "\n  ".join(errors))


def hidden(model, ids, mask):
    """Last-layer hidden states [B, T, D] across remote-code model classes."""
    attempts = [dict(input_ids=ids, attention_mask=mask, output_hidden_states=True),
                dict(input_ids=ids, attention_mask=mask), dict(input_ids=ids, output_hidden_states=True),
                dict(input_ids=ids)]
    err = None
    for kw in attempts:
        try:
            out = model(**kw)
            break
        except TypeError as e:
            err = e
    else:
        raise err
    hs = getattr(out, "hidden_states", None)
    if hs is not None:
        return hs[-1]
    if getattr(out, "last_hidden_state", None) is not None:
        return out.last_hidden_state
    return out[0]


class Clf(nn.Module):
    """Backbone + the same mean+max pooled head as our models."""
    def __init__(self, backbone, n_cls):
        super().__init__()
        self.bb = backbone
        self.head = None
        self.n_cls = n_cls

    def forward(self, ids, mask, keep):
        h = hidden(self.bb, ids, mask).float()
        k = keep[..., None].float()
        mean = (h * k).sum(1) / k.sum(1).clamp(min=1)
        mx = h.masked_fill(k == 0, float("-inf")).amax(1)
        if SPEC.get("ps") and not args.oriented:
            # Caduceus-PS: pooled(RC x) = channel-flip(pooled(x)); symmetrize -> exactly strand-invariant,
            # as in the Caduceus paper's invariant classification head
            mean, mx = 0.5 * (mean + mean.flip(-1)), 0.5 * (mx + mx.flip(-1))
        z = torch.cat([mean, mx], -1)
        if self.head is None:                                   # created lazily: hidden size varies by model
            self.head = nn.Sequential(nn.LayerNorm(z.shape[-1]), nn.Linear(z.shape[-1], self.n_cls)).to(z.device)
        return self.head(z)


def encode_batch(tok, seqs):
    m = SPEC.get("max_bp")
    if m:                                        # models that cannot read past their pretraining length: centre crop
        seqs = [s if len(s) <= m else s[(len(s) - m) // 2:(len(s) - m) // 2 + m] for s in seqs]
    enc = tok(seqs, return_tensors="pt", padding=True, return_special_tokens_mask=True,
              add_special_tokens=SPEC["special"]) if not args.offline else tok(seqs)
    ids = enc["input_ids"]
    mask = enc.get("attention_mask", torch.ones_like(ids))
    smask = enc.get("special_tokens_mask")
    keep = mask.clone() if smask is None or smask.shape != ids.shape else mask * (1 - smask)
    return ids.to(device), mask.to(device), keep.to(device)


def micro(n_tok, b):
    """micro-batch size so that (micro-batch) x n_tok^2 stays under --max-attn (attention memory grows with n^2)"""
    return max(1, min(b, int(args.max_attn // max(1, n_tok * n_tok))))


def batches(X, y, bs, rng=None):
    by_len = {}
    for i, s in enumerate(X):
        by_len.setdefault(len(s), []).append(i)
    out = []
    for idx in by_len.values():
        idx = np.array(idx)
        if rng is not None:
            rng.shuffle(idx)
        out += [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if rng is not None:
        rng.shuffle(out)
    for b in out:
        yield [X[i] for i in b], torch.from_numpy(y[b])


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda" and SPEC["amp"] and not args.offline and not args.fp32))


@torch.no_grad()
def predict(clf, tok, X, y, rc=False):
    clf.eval()
    P, Y = [], []
    for xb, yb in batches(X, y, 2 * args.bs):
        if rc:
            xb = [rc_str(s) for s in xb]
        ids, mask, keep = encode_batch(tok, xb)
        for i in range(0, len(xb), micro(ids.shape[1], len(xb))):
            j = slice(i, i + micro(ids.shape[1], len(xb)))
            with amp():
                lg = clf(ids[j], mask[j], keep[j])
            P.append(lg.float().softmax(-1).cpu())
        Y.append(yb)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def run(data, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, bb = load_backbone()
    n_cls = int(max(data["train"][1].max(), data["test"][1].max())) + 1
    clf = Clf(bb, n_cls).to(device)
    with torch.no_grad():                                       # build the lazy head before creating the optimizer
        clf(*encode_batch(tok, data["train"][0][:2]))
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=0.01)
    Xtr, ytr = data["train"]
    total, step, best, best_state = args.epochs * math.ceil(len(Xtr) / args.bs), 0, -2.0, None
    t0, best_ep = time.time(), 0
    for ep in range(args.epochs):
        if args.patience and ep - best_ep >= args.patience:
            log(f"    early stop after epoch {ep}")
            break
        clf.train()
        losses = []
        for xb, yb in batches(Xtr, ytr, args.bs, rng):
            for g in opt.param_groups:
                g["lr"] = rk.lr_at(step, total, args.lr, warmup=max(1, total // 20))
            if args.rcaug:
                xb = [rc_str(s_) if rng.random() < 0.5 else s_ for s_ in xb]
            ids, mask, keep = encode_batch(tok, xb)
            yb = yb.to(device)
            mb, tot = micro(ids.shape[1], len(xb)), 0.0
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(xb), mb):                    # long sequences: split, accumulate gradients
                j = slice(i, i + mb)
                with amp():
                    logits = clf(ids[j], mask[j], keep[j])
                loss = F.cross_entropy(logits.float(), yb[j], reduction="sum") / len(xb)
                loss.backward()
                tot += loss.item()
            losses.append(tot)
            if not math.isfinite(losses[-1]):
                log("    WARNING: loss is NaN/inf -> rerun with --fp32 (and a smaller --lr)")
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
            step += 1
        P, Y = predict(clf, tok, *data["val"])
        v = rk.mcc(Y, P.argmax(1))
        share = np.bincount(P.argmax(1), minlength=P.shape[1]).max() / len(P)
        log(f"    epoch {ep + 1}/{args.epochs}: train loss {np.nanmean(losses):.4f} | val MCC {v:.3f} | "
            f"largest predicted class {share:.0%}  ({(time.time() - t0) / 60:.1f} min)")
        if v > best:
            best, best_ep = v, ep + 1
            best_state = {k: t.detach().cpu().clone() for k, t in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    P, Y = predict(clf, tok, *data["test"])
    Prc, _ = predict(clf, tok, *data["test"], rc=True)
    yf, yr = P.argmax(1), Prc.argmax(1)
    res = dict(**rk.strand_metrics(Y, P, Prc), val_mcc=best, best_epoch=best_ep, train_min=(time.time() - t0) / 60,
               params=sum(p_.numel() for p_ in clf.parameters()))
    del clf, bb, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------- report
def make_report():
    import pandas as pd
    rows = read_rows(RES) + [dict(r, variant="ours: " + r["variant"]) for r in read_rows(args.day1)
                             if r.get("phase") == "C"]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[df.task.isin(args.tasks)]
    df["robust"] = (df.mcc + df.mcc_rc_input) / 2
    g = df.groupby(["task", "variant"]).agg(fwd=("mcc", "mean"), rc=("mcc_rc_input", "mean"),
                                            robust=("robust", "mean"), flips=("flip_rate", "mean"),
                                            n=("seed", "count")).reset_index()
    lines = ["# Published models vs ours: accuracy on each strand of the same test DNA", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Same tasks, splits and fine-tuning recipe "
             f"(lr {args.lr}, {args.epochs} epochs, best epoch on validation). Mean over seeds.", "",
             "- **fwd**: official test set. **RC**: the same sequences reverse-complemented.",
             "- **Strand-robust** = mean of the two. **Flips**: predictions that change between the strands.", ""]
    for t in [t for t in args.tasks if t in set(g.task)]:
        lines += [f"## {t}", "", "| Model | fwd MCC | RC MCC | drop | Strand-robust | Flips | seeds |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g[g.task == t].sort_values("robust", ascending=False).iterrows():
            lines.append(f"| {r.variant} | {r.fwd:.3f} | {r.rc:.3f} | {r.fwd - r.rc:+.3f} | **{r.robust:.3f}** | "
                         f"{r.flips:.1%} | {int(r.n)} |")
        lines.append("")
    lines += ["Note: published papers tune the recipe per task; here every model uses one fixed recipe, so compare "
              "models with each other, not with published leaderboard numbers.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    log(f"model {NAME} | device {device} | tasks {args.tasks} | seeds {args.seeds} | lr {args.lr} | bs {args.bs}")
    done = {r["key"] for r in read_rows(RES)}
    for t in args.tasks:
        data = None
        for seed in args.seeds:
            key = (f"{args.model}|{t}|{seed}" if args.lr == 3e-4 else f"{args.model}|lr{args.lr:g}|{t}|{seed}") + ("|rcaug" if args.rcaug else "") + ("|oriented" if args.oriented else "") + (f"|ep{args.epochs}p{args.patience}" if args.patience else "") + ("|full" if MAX_TRAIN > 20000 else "")
            if key in done:
                continue
            if data is None:
                raw = (rk.synthetic_task(t, n_train=MAX_TRAIN, n_test=MAX_TEST) if args.offline
                       else rk.load_task(t, MAX_TRAIN, MAX_TEST, log=log))
                data = {k: ([to_str(s) for s in X], y) for k, (X, y) in raw.items()}
                log(f"{t}: train {len(data['train'][0])}, val {len(data['val'][0])}, test {len(data['test'][0])}, "
                    f"{len(data['train'][0][0])} bp")
            r = run(data, seed)
            with open(RES, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(key=key, task=t, variant=NAME, seed=seed, **r)) + "\n")
            log(f"{t:10s} {NAME} seed {seed}: fwd {r['mcc']:.3f} | RC {r['mcc_rc_input']:.3f} | "
                f"flips {r['flip_rate']:.1%} | {r['train_min']:.1f} min | {r['params'] / 1e6:.1f}M params")
            make_report()
    make_report()
    log(f"DONE -> {OUT / 'report.md'}")