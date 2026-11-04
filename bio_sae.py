import argparse
import gzip
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rcchunk as rk

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--genome-dir", default="hg38_data")
p.add_argument("--chroms", nargs="+", default=["chr8", "chr9"], help="held-out chromosomes (not used for bio tasks training)")
p.add_argument("--windows", type=int, default=4000, help="1,024-bp windows to read")
p.add_argument("--chunks-per-window", type=int, default=48)
p.add_argument("--pairs", type=int, default=768, help="feature pairs (dictionary size = 2 x pairs)")
p.add_argument("--k", type=int, default=32, help="active features per chunk (TopK)")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--top", type=int, default=200, help="top-activating chunks used to interpret a feature")
p.add_argument("--ctx", type=int, default=15, help="context = chunk centre +- ctx bp")
p.add_argument("--jaspar", default=None, help="local JASPAR file (jaspar format); default: download")
p.add_argument("--annotation", default=None, help="UCSC genePred table (default hg38_data/ncbiRefSeq.txt.gz)")
p.add_argument("--site-frac", type=float, default=0.25,
               help="share of windows centred (+-300 bp jitter) on annotated TSS / splice sites, so these rare sites "
                    "are represented; the rest are random")
p.add_argument("--dedup-bp", type=int, default=100, help="top chunks of a feature: at most one per this many bp")
p.add_argument("--fdr", type=float, default=0.01)
p.add_argument("--min-fold", type=float, default=3.0)
p.add_argument("--min-hits", type=int, default=10)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
p.add_argument("--random-init", action="store_true",
               help="CONTROL: same architecture with random (untrained) weights -> what the pipeline finds by itself")
p.add_argument("--shuffle-motifs", action="store_true",
               help="CONTROL: scan column-shuffled JASPAR motifs (same base composition, no real motif) -> false matches")
p.add_argument("--null-shuffles", type=int, default=20,
               help="per-motif null: each significant real motif must beat this many column-shuffled copies of "
                    "itself on the same feature (0 = off)")
p.add_argument("--debug-plant", action="store_true", help=argparse.SUPPRESS)   # self-test of the enrichment code
args = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(args.out) if args.out else Path(args.folder) / ("bio_sae" + ("_random_init" if args.random_init else "")
                                                             + ("_shuffled_motifs" if args.shuffle_motifs else ""))
OUT.mkdir(parents=True, exist_ok=True)
JASPAR_URLS = [
    "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "https://jaspar.genereg.net/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
]
BASES = "ACGT"
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def swap(z):
    """sigma on a chunk vector: exchange the two strand halves."""
    a, b = z.chunk(2, -1)
    return torch.cat([b, a], -1)


def annotation_file():
    f = Path(args.annotation) if args.annotation else Path(args.genome_dir) / "ncbiRefSeq.txt.gz"
    if not f.exists():
        raise SystemExit(f"{f} not found: run download_all.py first")
    return f


def site_positions():
    """TSS, splice donor and acceptor coordinates of RefSeq NM_ transcripts on the chosen chromosomes."""
    out = {c: set() for c in args.chroms}
    f = annotation_file()
    with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if c[2] not in out or not c[1].startswith("NM_"):
                continue
            starts = [int(v) for v in c[9].strip(",").split(",")]
            ends = [int(v) for v in c[10].strip(",").split(",")]
            out[c[2]].update([int(c[4]) if c[3] == "+" else int(c[5])] + ends[:-1] + starts[1:])
    return {c: np.array(sorted(v)) for c, v in out.items()}


# ----------------------------------------------------------------------------- 1. chunk representations
@torch.no_grad()
def collect(model):
    L, C = 1024, args.ctx
    Z, meta = [], []
    for chrom in args.chroms:
        tok = rk.load_chrom(args.genome_dir, chrom)
        if tok is None:
            raise SystemExit(f"{chrom} not found in {args.genome_dir}: run download_all.py first")
        n_w = args.windows // len(args.chroms)
        sites = SITES.get(chrom, [])
        n_site = int(n_w * args.site_frac) if len(sites) else 0
        starts = []
        while len(starts) < n_w:
            if len(starts) < n_site:
                s = int(sites[rng.integers(len(sites))]) - L // 2 + int(rng.integers(-300, 301))
            else:
                s = int(rng.integers(0, len(tok) - L))
            if 0 <= s <= len(tok) - L and (np.asarray(tok[s:s + L]) == rk.N).mean() < 0.01:
                starts.append(s)
        for i in range(0, len(starts), args.bs):
            sb = starts[i:i + args.bs]
            x = torch.from_numpy(np.stack([np.asarray(tok[s:s + L]) for s in sb])).long().to(device)
            out = model(x, decode=False)
            cid, zm = out["cid"].cpu().numpy(), out["Zm"].float().cpu()
            for j, s in enumerate(sb):
                c = cid[j]
                edges = np.flatnonzero(np.diff(c)) + 1                          # first base of chunks 1..n-1
                st = np.concatenate([[0], edges]); en = np.concatenate([edges, [L]])
                centre = (st + en) // 2
                ok = np.flatnonzero((centre >= C + 1) & (centre <= L - C - 1))
                pick = rng.choice(ok, min(len(ok), args.chunks_per_window), replace=False) if len(ok) else []
                for q in pick:
                    Z.append(zm[j, q]); meta.append((chrom, s + int(centre[q]), int(en[q] - st[q])))
        log(f"  {chrom}: {len(starts)} windows read")
    return torch.stack(Z), meta


# ----------------------------------------------------------------------------- 2. strand-paired TopK SAE
class PairedSAE(nn.Module):
    """Dictionary = [W ; sigma(W)]: features come in reverse-complement pairs (i, i + m). Exactly equivariant:
    codes(sigma z) = codes(z) with the two halves of the dictionary exchanged."""

    def __init__(self, D, m, k):
        super().__init__()
        self.m, self.k = m, k
        W = torch.randn(m, D) / math.sqrt(D)
        self.enc = nn.Parameter(W.clone())
        self.dec = nn.Parameter(W.clone())
        self.b = nn.Parameter(torch.zeros(m))

    def full(self, W):
        return torch.cat([W, swap(W)], 0)

    def codes(self, z):
        pre = z @ self.full(self.enc).T + torch.cat([self.b, self.b])
        v, i = F.relu(pre).topk(self.k, -1)
        return torch.zeros_like(pre).scatter_(-1, i, v)

    def forward(self, z):
        a = self.codes(z)
        return a @ self.full(F.normalize(self.dec, dim=-1)), a


def train_sae(Z):
    X = torch.cat([Z, swap(Z)], 0)                       # both strands: mean and scale stay strand-symmetric
    mu, sc = X.mean(0), X.norm(dim=-1).mean()
    X = ((X - mu) / sc).to(device)
    sae = PairedSAE(X.shape[1], args.pairs, args.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    n, bs = len(X), 4096
    for ep in range(args.epochs):
        perm, tot = torch.randperm(n, device=device), 0.0
        for i in range(0, n, bs):
            xb = X[perm[i:i + bs]]
            rec, _ = sae(xb)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if ep % 5 == 0 or ep == args.epochs - 1:
            fvu = tot / n / X.pow(2).mean().item()
            log(f"  SAE epoch {ep + 1}/{args.epochs}: fraction of variance unexplained {fvu:.3f}")
    with torch.no_grad():
        N = len(Z)                                                                            # forward strand
        A = torch.cat([sae.codes(X[i:min(i + bs, N)]) for i in range(0, N, bs)]).cpu()
        A_rc = torch.cat([sae.codes(X[N + i:N + min(i + bs, N)]) for i in range(0, N, bs)]).cpu()
    m = args.pairs
    err = (A_rc - torch.cat([A[:, m:], A[:, :m]], 1)).abs().max().item()
    log(f"  strand pairing check: max |code(sigma z) - swapped code(z)| = {err:.2e}")
    torch.save(dict(state=sae.state_dict(), mu=mu, scale=sc, pairs=m, k=args.k), OUT / "sae.pt")
    return sae, A.numpy(), fvu, err


# ----------------------------------------------------------------------------- 3. annotation and motifs
SITE = {"exon": 1, "gene body": 2, "TSS +-50": 4, "splice donor +-3": 8, "splice acceptor +-3": 16}


def annotation_masks(lengths):
    f = annotation_file()
    masks = {c: np.zeros(n, np.uint8) for c, n in lengths.items()}
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if c[2] not in masks or not c[1].startswith("NM_"):
                continue
            mk, plus = masks[c[2]], c[3] == "+"
            t0, t1 = int(c[4]), int(c[5])
            starts = [int(v) for v in c[9].strip(",").split(",")]
            ends = [int(v) for v in c[10].strip(",").split(",")]
            mk[t0:t1] |= SITE["gene body"]
            for a, b in zip(starts, ends):
                mk[a:b] |= SITE["exon"]
            tss = t0 if plus else t1
            mk[max(0, tss - 50):tss + 50] |= SITE["TSS +-50"]
            don, acc = (ends[:-1], starts[1:]) if plus else (starts[1:], ends[:-1])
            for g in don:
                mk[max(0, g - 3):g + 3] |= SITE["splice donor +-3"]
            for g in acc:
                mk[max(0, g - 3):g + 3] |= SITE["splice acceptor +-3"]
    return masks


def annotate(meta, seqs):
    lengths = {c: len(rk.load_chrom(args.genome_dir, c)) for c in args.chroms}
    masks = annotation_masks(lengths)
    reps = {c: np.load(Path(args.genome_dir) / f"{c}.rep.npy", mmap_mode="r") for c in args.chroms}
    cat = {k: np.zeros(len(meta), bool) for k in ["repeat (soft-masked)", "exon", "intron", "intergenic",
                                                   "TSS +-50", "splice donor +-3", "splice acceptor +-3",
                                                   "CpG-rich", "poly-A/T (>=8)"]}
    for i, (chrom, pos, _) in enumerate(meta):
        m = masks[chrom][pos]
        cat["repeat (soft-masked)"][i] = bool(reps[chrom][pos])
        cat["exon"][i] = bool(m & SITE["exon"])
        cat["intron"][i] = bool(m & SITE["gene body"]) and not (m & SITE["exon"])
        cat["intergenic"][i] = not (m & SITE["gene body"])
        for k in ["TSS +-50", "splice donor +-3", "splice acceptor +-3"]:
            cat[k][i] = bool(m & SITE[k])
    s = seqs
    gc = np.isin(s, [1, 2]).mean(1)
    cpg = ((s[:, :-1] == 1) & (s[:, 1:] == 2)).sum(1)
    exp = (s == 1).sum(1) * (s == 2).sum(1) / s.shape[1]
    cat["CpG-rich"] = (gc >= 0.55) & (cpg / np.maximum(exp, 1e-9) >= 0.6) & (cpg >= 2)
    run = np.zeros(len(s), bool)
    for b in (0, 3):
        r = np.zeros(len(s), int); best = np.zeros(len(s), int)
        for j in range(s.shape[1]):
            r = np.where(s[:, j] == b, r + 1, 0); best = np.maximum(best, r)
        run |= best >= 8
    cat["poly-A/T (>=8)"] = run
    return cat


def load_jaspar():
    f = Path(args.jaspar) if args.jaspar else Path(args.genome_dir) / "JASPAR2024_CORE_vertebrates_nr.txt"
    if not f.exists():
        for url in JASPAR_URLS:
            try:
                log(f"downloading JASPAR from {url} ...")
                tmp = f.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, f)
                break
            except Exception as e:
                log(f"  failed ({e})")
        if not f.exists():
            raise SystemExit("could not download JASPAR: download the 'CORE vertebrates non-redundant, JASPAR format' "
                             "file from jaspar.elixir.no/downloads in a browser and pass it with --jaspar FILE")
    motifs, cur = [], None
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            cur = [line[1:].split()[0], " ".join(line[1:].split()[1:]), []]
            motifs.append(cur)
        elif line and cur is not None and line[0] in "ACGT":
            cur[2].append([float(v) for v in line[line.index("[") + 1:line.index("]")].split()])
    out = []
    for mid, name, rows in motifs:
        pfm = np.array(rows).T + 0.8                                    # [L, 4] counts + pseudocount
        pwm = np.log2(pfm / pfm.sum(1, keepdims=True) / 0.25)
        if 5 <= len(pwm) <= 2 * args.ctx:
            cons = pwm.argmax(1)
            if args.shuffle_motifs:                        # control: same columns in random order
                pwm = pwm[rng.permutation(len(pwm))]
                name = "shuffled " + name
            low = any(len(cons) - q >= 4 and (cons[:-q] == cons[q:]).mean() >= 0.9
                      for q in (1, 2, 3, 4))                    # (GGAA)n- or poly-A-like consensus; TATAAA is not
            out.append((mid, name, pwm, low))
    log(f"JASPAR: {len(out)} motifs (length 5..{2 * args.ctx}), {sum(m[3] for m in out)} of them low-complexity "
        f"(short tandem-repeat consensus)")
    return out


@torch.no_grad()
def motif_hits(seqs, motifs, rel=0.85):
    """hits[i, m] for the motif on the + strand and on the - strand of each context (relative score >= rel)."""
    X = F.one_hot(torch.from_numpy(seqs).long(), 4).float().permute(0, 2, 1).to(device)      # [N, 4, W]
    fw = np.zeros((len(seqs), len(motifs)), bool); rv = np.zeros_like(fw)
    for m, (_, _, pwm, _) in enumerate(motifs):
        w = torch.tensor(pwm.T, dtype=torch.float32, device=device)[None]                     # [1, 4, L]
        lo, hi = pwm.min(1).sum(), pwm.max(1).sum()
        thr = lo + rel * (hi - lo)
        for strand, dst in ((w, fw), (w.flip(1).flip(2), rv)):           # reverse complement of the motif
            for i in range(0, len(X), 65536):
                sc = F.conv1d(X[i:i + 65536], strand).amax(-1).squeeze(1)
                dst[i:i + 65536, m] = (sc >= thr).cpu().numpy()
    return fw, rv


# ----------------------------------------------------------------------------- 4. statistics
def binom_sf(k, n, p0):
    """P(X >= k), X ~ Binomial(n, p0); k and p0 may be arrays."""
    k, p0 = np.asarray(k, float), np.asarray(p0, float)
    try:
        from scipy.stats import binom
        return binom.sf(k - 1, n, p0)
    except Exception:                                                     # normal approximation
        sd = np.sqrt(np.maximum(n * p0 * (1 - p0), 1e-12))
        z = (k - 0.5 - n * p0) / sd / math.sqrt(2)
        return 0.5 * np.vectorize(math.erfc)(z)


def bh(pv, n_total=None):
    """Benjamini-Hochberg; n_total = number of hypotheses tested (>= len(pv) when weak ones were not stored)."""
    pv = np.asarray(pv, float)
    o = np.argsort(pv); q = pv[o] * max(n_total or 0, len(pv)) / np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1)
    return out


def logos(rows, seqs, A, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.textpath import TextPath
        from matplotlib.patches import PathPatch
        from matplotlib.font_manager import FontProperties
        from matplotlib import transforms
    except Exception as e:
        log(f"no logos ({e})"); return
    col = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
    fp = FontProperties(family="DejaVu Sans", weight="bold")

    def draw(ax, freq, title):
        ic = 2 + (freq * np.log2(np.clip(freq, 1e-9, 1))).sum(1)
        for x, (f, h) in enumerate(zip(freq, ic)):
            y = 0.0
            for b in np.argsort(f):
                hgt = f[b] * h
                if hgt < 0.01:
                    continue
                tp = TextPath((0, 0), BASES[b], size=1, prop=fp)
                bb = tp.get_extents()
                t = transforms.Affine2D().translate(-bb.x0, -bb.y0).scale(0.9 / bb.width, hgt / bb.height) \
                    .translate(x + 0.05, y) + ax.transData
                ax.add_patch(PathPatch(tp, transform=t, color=col[BASES[b]], lw=0)); y += hgt
        ax.set_xlim(0, len(freq)); ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2]); ax.set_xticks([])
        ax.set_title(title, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    def aligned(top, pwm, minus):
        """Reference-strand sequence around the best motif match in each context (motif on + or - strand)."""
        w = pwm[::-1, ::-1] if minus else pwm                     # reverse complement of the motif
        L, out = len(w), []
        for s in seqs[top]:
            sc = [w[np.arange(L), s[i:i + L]].sum() for i in range(len(s) - L + 1)]
            i = int(np.argmax(sc))
            lo, hi = i - 2, i + L + 2
            if lo >= 0 and hi <= len(s):
                out.append(s[lo:hi])
        out = np.array(out) if out else seqs[top][:, :L + 4]
        return np.stack([(out == b).mean(0) for b in range(4)], 1)

    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 1.6 * len(rows)), squeeze=False)
    for r, (fa, fb, label, pwm, plus_a) in enumerate(rows):
        for c, f in enumerate((fa, fb)):
            top = np.argsort(-A[:, f])[:args.top]
            minus = (plus_a < 0.5) if c == 0 else (plus_a >= 0.5)  # the partner sees the motif on the other strand
            draw(axes[r, c], aligned(top, pwm, minus),
                 f"feature {f}" + (" (strand partner, reference strand)" if c else f": {label}"))
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    models = [m for m in rk.load_pretrained(args.folder, device) if m[0] == "RC-equivariant"]
    if not models:
        raise SystemExit("no RC-equivariant pretrained model in this folder")
    variant, mcfg, model, name = models[0]
    if args.random_init:                                   # control: identical architecture, untrained weights
        model = rk.build(variant, mcfg).to(device).eval()
        name = "CONTROL random init: " + name
    log(f"model {name} | chromosomes {args.chroms} | {args.windows} windows ({args.site_frac:.0%} centred on sites)")
    SITES = site_positions() if args.site_frac > 0 else {}
    Z, meta = collect(model)
    log(f"{len(Z):,} chunks, dimension {Z.shape[1]}, mean chunk {np.mean([m[2] for m in meta]):.1f} bp")

    log("training the strand-paired sparse autoencoder ...")
    sae, A, fvu, perr = train_sae(Z)
    M, m = A.shape[1], args.pairs
    density = (A > 0).mean(0)
    alive = density > 0
    with torch.no_grad():
        W = F.normalize(sae.enc, dim=-1)
        selfsim = (W * F.normalize(swap(sae.enc), dim=-1)).sum(-1).cpu().numpy()   # 1 = palindromic feature

    log("reading contexts and annotation ...")
    C = args.ctx
    toks = {c: rk.load_chrom(args.genome_dir, c) for c in args.chroms}
    seqs = np.stack([np.asarray(toks[c][p - C:p + C]) for c, p, _ in meta]).astype(np.int64)
    good = (seqs < 4).all(1)
    seqs[~good] = 0
    cats = annotate(meta, seqs)
    motifs = load_jaspar()
    log(f"scanning {len(motifs)} motifs on both strands of {len(seqs):,} contexts ...")
    hf, hr = motif_hits(seqs, motifs)
    hf &= good[:, None]; hr &= good[:, None]
    hb = hf | hr
    if args.debug_plant:                    # self-test: feature 0 = "motif 0 on + strand", partner = "on - strand"
        A[:, 0] = hf[:, 1] * (1 + rng.random(len(A))); A[:, m] = hr[:, 1] * (1 + rng.random(len(A)))
        alive[0] = alive[m] = True

    # composition-matched background: the expected motif rate of a feature is the average rate in chunks with the
    # same GC, CpG count, sequence complexity and repeat status as its own top chunks (controls for "this feature
    # just likes AT-rich / GC-rich / repetitive DNA")
    gcb = np.minimum((np.isin(seqs, [1, 2]).mean(1) * 5).astype(int), 4)
    cpgb = np.minimum(((seqs[:, :-1] == 1) & (seqs[:, 1:] == 2)).sum(1), 2)
    k3 = seqs[:, :-2] * 16 + seqs[:, 1:-1] * 4 + seqs[:, 2:]
    cplx = np.array([len(np.unique(r)) for r in k3])
    cxb = np.digitize(cplx, np.quantile(cplx, [1 / 3, 2 / 3]))
    binid = ((gcb * 3 + cpgb) * 3 + cxb) * 2 + cats["repeat (soft-masked)"].astype(int)
    nb = int(binid.max()) + 1
    R = np.zeros((nb, hb.shape[1]), np.float32)
    for b in range(nb):
        sel = (binid == b) & good
        if sel.any():
            R[b] = hb[sel].mean(0)
    low = np.array([mo[3] for mo in motifs])
    chrom_id = {c: i for i, c in enumerate(args.chroms)}
    loc = np.array([chrom_id[c] * 10 ** 10 + p_ for c, p_, _ in meta], np.int64)

    def top_chunks(f):
        """Top-activating chunks of feature f, at most one per dedup window (no double counting of one locus)."""
        order = np.argsort(-A[:, f])
        taken, out = set(), []
        for i in order:
            if A[i, f] <= 0 or len(out) >= args.top:
                break
            if not good[i]:
                continue
            b = loc[i] // args.dedup_bp
            if b in taken or b - 1 in taken or b + 1 in taken:
                continue
            taken.add(b); out.append(i)
        return np.array(out, int)

    log("enrichment tests ...")
    T = args.top
    tests, n_tested = [], 0                             # (feature, kind, name, hits, fold, p, plus_share)
    base_m = hb[good].mean(0)
    base_c = {k: v[good].mean() for k, v in cats.items()}
    TOP = {}
    for f in np.flatnonzero(alive):
        top = top_chunks(f)
        TOP[f] = top
        if len(top) < 50:
            continue
        n = len(top)
        k = hb[top].sum(0)
        js = np.flatnonzero(k >= args.min_hits)
        n_tested += len(js)
        if len(js):
            p0 = np.maximum(R[binid[top]][:, js].mean(0), 1e-6)           # composition-matched expectation
            fold = k[js] / n / p0
            keep = fold >= args.min_fold                    # only these can pass; the rest would be filtered anyway
            js, p0, fold = js[keep], p0[keep], fold[keep]
            pv = binom_sf(k[js], n, p0) if len(js) else []
            nf, nr = hf[top][:, js].sum(0), hr[top][:, js].sum(0)
            for j, fo, pp, a_, b_ in zip(js, fold, pv, nf, nr):
                tests.append((f, "motif", j, int(k[j]), float(fo), float(pp), a_ / max(1, a_ + b_)))
        for cname, v in cats.items():
            kk = int(v[top].sum())
            if kk >= args.min_hits:
                n_tested += 1
                p0 = max(base_c[cname], 1e-6)
                tests.append((f, "annotation", cname, kk, kk / n / p0, float(binom_sf(kk, n, p0)), np.nan))
    q = bh([t[5] for t in tests], n_tested) if tests else np.array([])
    # common categories (repeats ~50% of the genome) cannot reach 3x: require a fold that is attainable
    need = {c: min(args.min_fold, max(1.3, 0.5 / max(b, 1e-6))) for c, b in base_c.items()}
    sig = [t + (qq,) for t, qq in zip(tests, q)
           if qq < args.fdr and t[4] >= (args.min_fold if t[1] == "motif" else need[t[2]])]

    # ---- per-motif matched null: does the REAL motif beat column-shuffled copies of itself on the SAME feature?
    # A column shuffle keeps the motif's length, information content and base composition, so a feature that only
    # likes a k-mer or a composition matches the shuffles as well as the real motif. "Specific" = real fold above
    # every shuffle (one-sided empirical p <= 1/(S+1)); expected false share under the null = 1/(S+1).
    S = 0 if args.shuffle_motifs else args.null_shuffles
    specific = {}
    if S and sig:
        nrng = np.random.default_rng(args.seed + 1)
        by_motif = {}
        for t in sig:
            if t[1] == "motif" and not low[t[2]]:
                by_motif.setdefault(t[2], []).append(t)
        log(f"matched null: {S} shuffles x {len(by_motif)} motifs ...")
        for j, ts in by_motif.items():
            mid, mname, pwm, lw = motifs[j]
            shuf = []
            for _ in range(S):
                perm = nrng.permutation(len(pwm))
                while len(pwm) > 1 and (perm == np.arange(len(pwm))).all():
                    perm = nrng.permutation(len(pwm))
                shuf.append((mid, mname, pwm[perm], lw))
            sf, sr = motif_hits(seqs, shuf)
            sh = (sf | sr) & good[:, None]                                  # [N, S]
            Rs = np.zeros((nb, S), np.float32)
            for b in range(nb):
                sel = (binid == b) & good
                if sel.any():
                    Rs[b] = sh[sel].mean(0)
            for t in ts:
                top = TOP[t[0]]
                fold_s = sh[top].sum(0) / len(top) / np.maximum(Rs[binid[top]].mean(0), 1e-6)
                specific[(t[0], t[2])] = (t[4] > fold_s.max(), float(fold_s.max()), float(np.median(fold_s)))
    n_cx_sig = len(specific)
    n_spec = sum(v[0] for v in specific.values())

    best_m, best_a, best_cx = {}, {}, {}           # best_cx: best match among complex (non-repeat) motifs
    for t in sig:
        d = best_m if t[1] == "motif" else best_a
        if t[0] not in d or t[4] > d[t[0]][4]:
            d[t[0]] = t
        if t[1] == "motif" and not low[t[2]] and (t[0] not in best_cx or t[4] > best_cx[t[0]][4]):
            best_cx[t[0]] = t
    best_sp = {}                                   # best complex motif that also beats its own shuffles
    for t in sig:
        if t[1] == "motif" and specific.get((t[0], t[2]), (False,))[0] and (t[0] not in best_sp or t[4] > best_sp[t[0]][4]):
            best_sp[t[0]] = t

    import csv
    with open(OUT / "sae_features.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "strand_partner", "self_similarity", "density", "best_motif_id", "best_motif_tf",
                    "motif_fold", "motif_q", "motif_plus_strand_share", "motif_low_complexity",
                    "best_complex_motif_tf", "complex_motif_fold", "complex_motif_q",
                    "best_annotation", "annotation_fold", "annotation_q"])
        for f in range(M):
            partner = f + m if f < m else f - m
            bm, ba, bc = best_m.get(f), best_a.get(f), best_cx.get(f)
            w.writerow([f, partner, f"{selfsim[f % m]:.3f}", f"{density[f]:.5f}",
                        motifs[bm[2]][0] if bm else "", motifs[bm[2]][1] if bm else "",
                        f"{bm[4]:.2f}" if bm else "", f"{bm[7]:.2e}" if bm else "", f"{bm[6]:.2f}" if bm else "",
                        bool(low[bm[2]]) if bm else "",
                        f"{motifs[bc[2]][1]} ({motifs[bc[2]][0]})" if bc else "", f"{bc[4]:.2f}" if bc else "",
                        f"{bc[7]:.2e}" if bc else "",
                        ba[2] if ba else "", f"{ba[4]:.2f}" if ba else "", f"{ba[7]:.2e}" if ba else ""])

    # strand check: pairs whose two members match the same motif -> that motif should sit on opposite strands
    opp, same = 0, 0
    for f in range(m):
        a, b = best_m.get(f), best_m.get(f + m)
        if a and b and a[2] == b[2] and abs(a[6] - 0.5) > 0.2 and abs(b[6] - 0.5) > 0.2:
            if (a[6] > 0.5) != (b[6] > 0.5):
                opp += 1
            else:
                same += 1

    n_alive = int(alive.sum())
    tf_all = {motifs[t[2]][1] for t in best_m.values()}
    tf_cx = {motifs[t[2]][1] for t in best_cx.values()}
    n_low = sum(1 for t in best_m.values() if low[t[2]])

    def pair_table(best, n_rows):
        """One row per feature PAIR (the two strands of one feature), strongest first, one pair per TF."""
        rows, seen_tf, seen_pair = [], set(), set()
        for t in sorted(best.values(), key=lambda t: -t[4]):
            f = t[0]; pr = min(f, f + m if f < m else f - m)
            tf = motifs[t[2]][1]
            if pr in seen_pair or tf in seen_tf:
                continue
            seen_pair.add(pr); seen_tf.add(tf)
            partner = f + m if f < m else f - m
            pt = best.get(partner)
            ba = best_a.get(f)
            rows.append(f"| {f} / {partner} | {tf} ({motifs[t[2]][0]}){' *' if low[t[2]] else ''} | {t[4]:.1f}x | "
                        f"{t[7]:.1e} | {t[6]:.2f} / {pt[6]:.2f} |" if pt and pt[2] == t[2] else
                        f"| {f} / {partner} | {tf} ({motifs[t[2]][0]}){' *' if low[t[2]] else ''} | {t[4]:.1f}x | "
                        f"{t[7]:.1e} | {t[6]:.2f} / - |")
            rows[-1] += f" {(ba[2] + f' {ba[4]:.1f}x') if ba else ''} |"
            if len(rows) >= n_rows:
                break
        return rows

    lines = ["# What the learned chunks encode: strand-paired sparse autoencoder", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Model: {name}. {len(Z):,} chunks from held-out "
             f"{', '.join(args.chroms)} ({args.site_frac:.0%} of windows centred near TSS / splice sites); "
             f"contexts = chunk centre +- {C} bp; top {T} chunks per feature, at most one per {args.dedup_bp} bp.", "",
             f"- Dictionary: {M} features in {m} reverse-complement pairs, TopK k = {args.k}. "
             f"Fraction of variance unexplained: **{fvu:.3f}**. Active features: {n_alive}/{M}.",
             f"- Strand pairing is exact by construction (max error {perr:.1e}); self-similar pairs (> 0.9): "
             f"{int((selfsim > 0.9).sum())} of {m}.",
             f"- Motif enrichment is measured against a **composition-matched background** (same GC, CpG, sequence "
             f"complexity and repeat status as the feature's own top chunks), FDR < {args.fdr}, >= {args.min_fold:g}x, "
             f">= {args.min_hits} hits.",
             f"- Features with any significant JASPAR motif: **{len(best_m)}** of {n_alive}; of these {n_low} match best "
             f"a low-complexity motif (short tandem repeat, e.g. GGAA repeats = EWSR1-FLI1), marked *.",
             f"- Features with a significant **complex** (non-repeat) TF motif: **{len(best_cx)}** of {n_alive}, "
             f"covering **{len(tf_cx)}** distinct TFs (all motifs: {len(tf_all)} TFs).",
             *([f"- **Matched null ({S} column-shuffled copies per motif)**: of {n_cx_sig:,} significant "
                f"(feature, complex motif) pairs, **{n_spec:,}** beat every shuffled copy of their own motif "
                f"(expected by chance: {n_cx_sig / (S + 1):,.0f}). Features with at least one such motif: "
                f"**{len(best_sp)}**, covering **{len({motifs[t[2]][1] for t in best_sp.values()})}** TFs. "
                f"Only these support a claim of TF-specific motifs."] if S else []),
             f"- Features with a significant annotation category: **{len(best_a)}** of {n_alive}.",
             f"- Sanity check (expected from the pairing): pairs matched to the same motif with a clear strand "
             f"preference: {opp} on opposite strands, {same} on the same strand.", "",
             "## Strongest complex-motif features (one pair per TF)", "",
             "| feature / partner | TF (JASPAR) | fold vs matched bg | q | + strand share (feature / partner) | annotation |",
             "|---|---|---|---|---|---|"] + pair_table(best_cx, 25) + ([
             "", f"## Motif-specific features: real motif beats all {S} shuffled copies of itself", "",
             "| feature / partner | TF (JASPAR) | fold vs matched bg | q | + strand share (feature / partner) | annotation |",
             "|---|---|---|---|---|---|"] + pair_table(best_sp, 25) if S else []) + [
             "", "## Strongest features overall, including repeat-like motifs (one pair per TF)", "",
             "| feature / partner | TF (JASPAR) | fold vs matched bg | q | + strand share (feature / partner) | annotation |",
             "|---|---|---|---|---|---|"] + pair_table(best_m, 10)
    lines += ["", "## Annotation categories", "",
              "| category | background share | fold needed | features enriched (best fold) |", "|---|---|---|---|"]
    for cname in cats:
        hits = [t for t in best_a.values() if t[2] == cname]
        lines.append(f"| {cname} | {base_c[cname]:.3f} | {need[cname]:.2f}x | {len(hits)}" +
                     (f" ({max(t[4] for t in hits):.1f}x)" if hits else "") + " |")
    lines += ["", "- **Strand partner**: the same feature read on the other DNA strand (dictionary is built in pairs).",
              "- **+ strand share**: share of the motif hits in the top chunks that lie on the + strand of the "
              "reference; the two members of a pair show opposite shares (same motif, two strands).",
              "- **\\***: low-complexity motif (consensus is a short tandem repeat); such matches mostly mean "
              "'this feature marks a microsatellite', not a TF binding site.",
              f"- Logos: sae_logos.png (complex motifs, aligned on the best match). Runtime "
              f"{(time.time() - t0) / 60:.1f} min."]
    (OUT / "sae_report.md").write_text("\n".join(lines), encoding="utf-8")

    rows, seen = [], set()
    for t in sorted((best_sp or best_cx).values(), key=lambda t: -t[4]):   # logos: motif-specific features first
        f, tf = t[0], motifs[t[2]][1]
        pr = min(f, f + m if f < m else f - m)
        if pr in seen or tf in seen or len(rows) >= 6:
            continue
        seen.update([pr, tf])
        partner = f + m if f < m else f - m
        rows.append((f, partner, f"{tf} {t[4]:.1f}x", motifs[t[2]][2], t[6]))
    if rows:
        logos(rows, seqs, A, OUT / "sae_logos.png")
    log(f"DONE -> {OUT / 'sae_report.md'}")