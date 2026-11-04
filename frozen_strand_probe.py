import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import rcchunk as rk

NT_SYM = ["H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3",
          "H4K20me1", "enhancers", "enhancers_types"]
PUBLIC = {   # name -> (HF repo, label, mixed precision ok, special tokens, Caduceus-PS symmetric pooling)
    "nt100m": ("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "NT-v2", True, True, False),
    "dnabert2": ("zhihan1996/DNABERT-2-117M", "DNABERT-2", True, True, False),
    "hyenadna": ("LongSafari/hyenadna-small-32k-seqlen-hf", "HyenaDNA-small", False, False, False),
    "caduceus_ps_1k": ("yairschiff/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3", "Caduceus-PS-1k", True, False, True),
    "caduceus_ph_1k": ("yairschiff/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3", "Caduceus-PH-1k", True, False, False),
}
CADUCEUS_CODE = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

p = argparse.ArgumentParser()
p.add_argument("models", nargs="*", help="run folders of our models and/or: " + ", ".join(PUBLIC))
p.add_argument("--tasks", nargs="+", default=NT_SYM)
p.add_argument("--max-train", type=int, default=20000, help="training sequences per task for the probe")
p.add_argument("--max-test", type=int, default=10 ** 9)
p.add_argument("--nn-sample", type=int, default=2000, help="test sequences used for the retrieval measure")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--out", default="results_frozen")
p.add_argument("--report-only", action="store_true")
p.add_argument("--smoke", action="store_true", help="2 tasks, 300/200 sequences: code check")
args = p.parse_args()
if args.smoke:
    args.tasks, args.max_train, args.max_test, args.out = args.tasks[:2], 300, 200, args.out + "_smoke"
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(args.out)
(OUT / "emb").mkdir(parents=True, exist_ok=True)
RES = OUT / "results.jsonl"
ALPH = np.array(list("ACGTN"))
COMP = str.maketrans("ACGTN", "TGCAN")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- embedders
class Ours:
    """Our pretrained models, frozen. Features = the pooled features the fine-tuning head reads (chunk + base readout).
    For the symmetric models these are the invariant combinations [a+b, |a-b|]; for the augmented chunker [mean; max]."""

    def __init__(self, folder):
        models = rk.load_pretrained(folder, device)
        if not models:
            raise SystemExit(f"{folder}: no pretrained model")
        self.variant, mcfg, self.m, name = models[0]
        self.label = {"RC-equivariant": "MirrorDNA" + ("-duplex" if mcfg.get("enc") == "duplex" else " (ours)")
                      + (", fixed cuts" if mcfg.get("router") == "fixed" else ""),
                      "baseline+RCaug": "Chunker + RC aug.", "baseline": "Chunker (no aug.)"}[self.variant]
        self.tag = Path(folder).name

    @torch.no_grad()
    def embed(self, X):
        out = []
        dummy = np.zeros(len(X), np.int64)
        for xb, _ in rk.batches(X, dummy, args.bs):
            o = self.m(xb.to(device), decode=True)
            f = torch.cat([self.m.pooled_features(o), self.m.pooled_base(o)], -1)
            out.append(f.float().cpu())
        return torch.cat(out).numpy()


class Public:
    """Published models, frozen, mean+max pooled over sequence tokens (the same pooling as our fine-tuning head)."""

    def __init__(self, name):
        repo, self.label, self.amp, self.special, self.ps = PUBLIC[name]
        self.tag = name
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
        if name.startswith("caduceus"):
            import importlib.util
            if importlib.util.find_spec("mamba_ssm") is None:
                raise SystemExit("Caduceus needs mamba-ssm: run this model in the WSL environment")
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            Cfg = get_class_from_dynamic_module("configuration_caduceus.CaduceusConfig", CADUCEUS_CODE)
            Mdl = get_class_from_dynamic_module("modeling_caduceus.Caduceus", CADUCEUS_CODE)
            self.model, info = Mdl.from_pretrained(repo, config=Cfg.from_pretrained(repo), output_loading_info=True)
            if info.get("missing_keys"):
                raise RuntimeError(f"{repo}: weights missing, e.g. {info['missing_keys'][:3]}")
            self.tok = AutoTokenizer.from_pretrained(CADUCEUS_CODE, trust_remote_code=True)
        else:
            kw = dict(trust_remote_code=True)
            if name == "dnabert2":
                from transformers.models.bert.configuration_bert import BertConfig
                kw["config"] = BertConfig.from_pretrained(repo)
            self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            self.model = None
            for cls in (AutoModel, AutoModelForMaskedLM):
                try:
                    self.model = cls.from_pretrained(repo, **kw)
                    break
                except Exception as e:                                   # noqa: BLE001
                    err = e
            if self.model is None:
                raise err
            if name == "dnabert2":                                       # old Triton kernel: use PyTorch attention
                for mname, mod in list(sys.modules.items()):
                    if "DNABERT" in mname and getattr(mod, "flash_attn_qkvpacked_func", None) is not None:
                        mod.flash_attn_qkvpacked_func = None
        self.model = self.model.to(device).eval()

    def _hidden(self, ids, mask):
        for kw in (dict(input_ids=ids, attention_mask=mask, output_hidden_states=True),
                   dict(input_ids=ids, attention_mask=mask), dict(input_ids=ids)):
            try:
                out = self.model(**kw)
                break
            except TypeError:
                continue
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            return hs[-1]
        return out.last_hidden_state if getattr(out, "last_hidden_state", None) is not None else out[0]

    @torch.no_grad()
    def embed(self, X):
        seqs = ["".join(ALPH[np.minimum(x, 4)]) for x in X]
        out = []
        for i in range(0, len(seqs), args.bs):
            batch = seqs[i:i + args.bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=self.special,
                           return_special_tokens_mask=True)
            ids, mask = enc["input_ids"].to(device), enc.get("attention_mask", torch.ones_like(enc["input_ids"])).to(device)
            sm = enc.get("special_tokens_mask")
            keep = mask * (1 - sm.to(device)) if sm is not None and sm.shape == ids.shape else mask
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda" and self.amp)):
                h = self._hidden(ids, mask).float()
            k = keep[..., None].float()
            mean = (h * k).sum(1) / k.sum(1).clamp(min=1)
            mx = h.masked_fill(k == 0, float("-inf")).amax(1)
            if self.ps:                                  # Caduceus-PS: published strand-invariant pooling
                mean, mx = 0.5 * (mean + mean.flip(-1)), 0.5 * (mx + mx.flip(-1))
            out.append(torch.cat([mean, mx], -1).cpu())
        return torch.cat(out).numpy()


# ----------------------------------------------------------------------------------------------- measures
def rc_arrays(X):
    return [rk.rc_tokens(torch.from_numpy(np.asarray(x, np.int64))[None])[0].numpy().astype(np.uint8) for x in X]


def evaluate(Etr, ytr, Ete, Erc, yte):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
    Ztr, Zte, Zrc = (Etr - mu) / sd, (Ete - mu) / sd, (Erc - mu) / sd
    cos = (Zte * Zrc).sum(1) / (np.linalg.norm(Zte, axis=1) * np.linalg.norm(Zrc, axis=1) + 1e-12)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Zte))[:args.nn_sample]
    A = Zte[idx] / (np.linalg.norm(Zte[idx], axis=1, keepdims=True) + 1e-12)
    B = Zrc[idx] / (np.linalg.norm(Zrc[idx], axis=1, keepdims=True) + 1e-12)
    nn = float((np.argmax(B @ A.T, axis=1) == np.arange(len(idx))).mean())
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Ztr, ytr)
    P, Prc = clf.predict_proba(Zte), clf.predict_proba(Zrc)
    r = rk.strand_metrics(yte, P, Prc)
    return dict(cos=float(cos.mean()), nn=nn, fwd=r["mcc"], rc=r["mcc_rc_input"], both=r["consistent_mcc"],
                flips=r["flip_rate"], avg=r["mcc_rc_avg"])


def read_rows():
    return [json.loads(l) for l in RES.read_text(encoding="utf-8").splitlines() if l.strip()] if RES.exists() else []


def report():
    import pandas as pd
    rows = read_rows()
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(["model", "task"], keep="last")
    df.to_csv(OUT / "frozen_probe.csv", index=False)
    cols = ["cos", "nn", "fwd", "rc", "both", "flips", "avg"]
    g = df.groupby("model")
    summ = g[cols].mean().join(g.task.nunique().rename("tasks")).sort_values("both", ascending=False)
    summ.round(4).to_csv(OUT / "frozen_summary.csv")
    lines = ["# Frozen representations on the two strands (12 strand-symmetric NT tasks, mean over tasks)", "",
             "cos: cosine similarity of the embeddings of a sequence and its reverse complement; nn: share of reverse "
             "complements whose nearest forward neighbour is the same DNA; probe trained on forward embeddings: "
             "forward / RC / both-strands MCC, flip rate, and MCC of the probe averaged over both strands.", "",
             "| model | tasks | cos | nn | fwd MCC | RC MCC | both-strands MCC | flips | averaged MCC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, r in summ.iterrows():
        lines.append(f"| {m} | {int(r.tasks)} | {r.cos:.3f} | {r.nn:.3f} | {r.fwd:.3f} | {r.rc:.3f} | **{r.both:.3f}** | "
                     f"{100 * r.flips:.1f}% | {r.avg:.3f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    if args.report_only or not args.models:
        report()
        raise SystemExit
    done = {(r["tag"], r["task"]) for r in read_rows()}
    data = {}
    for spec in args.models:
        emb = None
        for t in args.tasks:
            tag = Path(spec).name if spec not in PUBLIC else spec
            if (tag, t) in done:
                continue
            if t not in data:
                d = rk.load_task(t, args.max_train, args.max_test, log=log)
                Xtr = list(d["train"][0]) + list(d["val"][0])
                ytr = np.concatenate([d["train"][1], d["val"][1]])
                data[t] = (Xtr, ytr, list(d["test"][0]), d["test"][1])
            Xtr, ytr, Xte, yte = data[t]
            if emb is None:
                emb = Public(spec) if spec in PUBLIC else Ours(spec)
                log(f"model {emb.label} ({spec}) on {device}")
            cache = OUT / "emb" / f"{tag}_{t}.npz"
            if cache.exists():
                z = np.load(cache)
                Etr, Ete, Erc = z["tr"], z["te"], z["rc"]
            else:
                t0 = time.time()
                Etr, Ete, Erc = emb.embed(Xtr), emb.embed(Xte), emb.embed(rc_arrays(Xte))
                np.savez_compressed(cache, tr=Etr, te=Ete, rc=Erc)
                log(f"  {t}: embeddings {Etr.shape[1]}-d for {len(Xtr)} + 2 x {len(Xte)} sequences "
                    f"({(time.time() - t0) / 60:.1f} min)")
            r = evaluate(Etr, ytr, Ete, Erc, yte)
            with open(RES, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(model=emb.label, tag=tag, task=t, **r)) + "\n")
            log(f"  {emb.label:22s} {t:16s} cos {r['cos']:.3f} | nn {r['nn']:.3f} | fwd {r['fwd']:.3f} | "
                f"RC {r['rc']:.3f} | both {r['both']:.3f} | flips {100 * r['flips']:.1f}%")
        del emb
        if device == "cuda":
            torch.cuda.empty_cache()
    report()