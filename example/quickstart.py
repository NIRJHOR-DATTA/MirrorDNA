"""
MirrorDNA quickstart: load a pretrained model, look at its strand-mirrored tokens, check the equivariance guarantee,
and fine-tune on the small test dataset in example/data/.

    python example/quickstart.py --weights weights/mirrordna-tiny
    python example/quickstart.py --weights weights/mirrordna-tiny --finetune     # + 2-minute fine-tuning demo

Runs on CPU (slow but fine for the demo) or GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("--weights", default="weights/mirrordna-tiny",
               help="folder with pretrain_RC-equivariant.final.pt and pretrain_info.json")
p.add_argument("--data", default=str(Path(__file__).parent / "data" / "gene_strand_tss_small.npz"))
p.add_argument("--finetune", action="store_true")
p.add_argument("--epochs", type=int, default=3)
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load ------------------------------------------------------------------------------------------------------
models = [m for m in rk.load_pretrained(args.weights, dev) if m[0] == "RC-equivariant"]
if not models:
    raise SystemExit(f"no pretrain_RC-equivariant.final.pt in {args.weights} (see weights/README.md)")
variant, mcfg, model, name = models[0]
print(f"loaded {name} from {args.weights} on {dev}")

# 2. tokens of a sequence and of its reverse complement ---------------------------------------------------------
seq = "ACGTTAGCCATGGCTTAACGTACGATTGCATGCCGTAAGCTTGACCATGGTACCGATTACAGGCTTAG"
x = torch.from_numpy(rk.encode(seq)).long()[None].to(dev)
with torch.no_grad():
    cut_f = model(x, decode=False)["cut"][0].cpu().numpy()
    cut_r = model(rk.rc_tokens(x), decode=False)["cut"][0].cpu().numpy()


def chunks(s, cut):
    out, start = [], 0
    for j, c in enumerate(cut):                       # gap j lies between base j and j+1
        if c:
            out.append(s[start:j + 1]); start = j + 1
    return out + [s[start:]]


rc_seq = seq[::-1].translate(str.maketrans("ACGT", "TGCA"))
print("\nforward strand :", " | ".join(chunks(seq, cut_f)))
print("reverse strand :", " | ".join(chunks(rc_seq, cut_r)))
print("mirror images  :", np.array_equal(cut_f, cut_r[::-1]))

# 3. the guarantee on random genomic-like input ------------------------------------------------------------------
xs = torch.randint(0, 4, (8, 1000), device=dev)
with torch.no_grad():
    a, b = model(xs, decode=False), model(rk.rc_tokens(xs), decode=False)
    fa, fb = model.pooled_features(a), model.pooled_features(b)
print(f"\n8 random 1-kb sequences: cuts mirrored = {torch.equal(a['cut'], b['cut'].flip(1))}, "
      f"invariant features equal up to {((fa - fb).abs().max() / fa.abs().max()).item():.1e} (float rounding)")

# 4. fine-tuning demo on the minimal test dataset ----------------------------------------------------------------
if args.finetune:
    d = np.load(args.data, allow_pickle=True)
    tr = ([np.asarray(s, np.uint8) for s in d["train_seq"]], d["train_y"].astype(np.int64))
    te = ([np.asarray(s, np.uint8) for s in d["test_seq"]], d["test_y"].astype(np.int64))
    data = rk.split_val(tr, te, seed=0)
    print(f"\nfine-tuning on {Path(args.data).name}: {len(data['train'][1])} train, {len(data['val'][1])} val, "
          f"{len(te[1])} test sequences (gene strand from the TSS window: a direction-dependent task, "
          f"so the orientation-aware head is used)")
    state = torch.load(Path(args.weights) / "pretrain_RC-equivariant.final.pt", map_location="cpu")
    r = rk.finetune(variant, mcfg, data, dev, seed=0, lr=3e-4, epochs=args.epochs, bs=32, init_state=state,
                    oriented=True, use_base=True, log=print)
    print(f"test MCC {r['mcc']:.3f} | accuracy {r['acc']:.3f}")
