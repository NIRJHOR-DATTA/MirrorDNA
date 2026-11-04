from __future__ import annotations

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

# ----------------------------------------------------------------------------- tokens
A, C, G, T, N, MASK = 0, 1, 2, 3, 4, 5
VOCAB = 6
_LUT = np.full(256, N, dtype=np.uint8)
for _ch, _v in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _LUT[_ch] = _v
_COMP = torch.tensor([T, G, C, A, N, MASK])


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def rc_tokens(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement of token ids [B, L] (N and MASK map to themselves)."""
    return _COMP.to(x.device)[x].flip(-1)


# ----------------------------------------------------------------------------- symmetry helpers
def swap_halves(h: torch.Tensor) -> torch.Tensor:
    d = h.shape[-1] // 2
    return torch.cat([h[..., d:], h[..., :d]], -1)


def sigma(h: torch.Tensor) -> torch.Tensor:
    """RC action on nucleotide-level hidden states [B, L, D]."""
    return swap_halves(h).flip(1)


def rev_valid(z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Reverse the first n[b] rows of z[b] (chunk sequences are right-padded)."""
    k = torch.arange(z.shape[1], device=z.device)[None]
    idx = torch.where(k < n[:, None], n[:, None] - 1 - k, k)
    return z.gather(1, idx[..., None].expand(-1, -1, z.shape[-1]))


def sigma_chunks(z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return swap_halves(rev_valid(z, n))


# ----------------------------------------------------------------------------- building blocks
class ConvBlock(nn.Module):
    """Pre-norm depthwise dilated conv + MLP, 'same' padding (reversal-friendly)."""

    def __init__(self, d: int, k: int = 9, dil: int = 1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil, groups=d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.dw(self.n1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.mlp(self.n2(x))


class LocalNet(nn.Module):
    def __init__(self, d: int, n: int, dils=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock(d, dil=dils[i % len(dils)]) for i in range(n)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class DuplexBlock(nn.Module):
    """Cross-strand gated dilated convolution (the "duplex" encoder block).

    Two streams, one per strand, each in its own reading orientation: u for x, v for RC(x).
    Each stream's update is gated by its own local context AND by the other strand's context at the SAME base
    pair (the partner stream reversed into this stream's coordinates):
        c_u = dwconv(norm(u)),   c_v = dwconv(norm(v))
        u  += W_o( GELU(W_a c_u) * sigmoid(W_g [c_u ; reverse(c_v)]) ),   v likewise with u and v swapped
    Shared weights make the pair update commute with swapping the strands, so the encoder stays exactly
    RC-equivariant (Lemma 1 holds for any swap-commuting encoder). With v=None (baseline) the gate sees only u.
    """

    def __init__(self, d: int, k: int = 9, dil: int = 1, cross: bool = True):
        super().__init__()
        self.cross = cross
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil, groups=d)
        self.val, self.out = nn.Linear(d, d), nn.Linear(d, d)
        self.gate = nn.Linear(2 * d if cross else d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def _conv(self, u):
        return self.dw(self.n1(u).transpose(1, 2)).transpose(1, 2)

    def _mix(self, c, partner):
        g = self.gate(torch.cat([c, partner], -1) if self.cross else c)
        return self.out(F.gelu(self.val(c)) * torch.sigmoid(g))

    def forward(self, u, v=None):
        cu = self._conv(u)
        if v is None:
            u = u + self._mix(cu, torch.zeros_like(cu) if self.cross else None)
            return u + self.mlp(self.n2(u)), None
        cv = self._conv(v)
        u, v = u + self._mix(cu, cv.flip(1)), v + self._mix(cv, cu.flip(1))
        return u + self.mlp(self.n2(u)), v + self.mlp(self.n2(v))


class DuplexNet(nn.Module):
    def __init__(self, d: int, n: int, dils=(1, 2, 4, 8), cross: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([DuplexBlock(d, dil=dils[i % len(dils)], cross=cross) for i in range(n)])
        self.norm = nn.LayerNorm(d)

    def forward(self, u, v=None):
        for b in self.blocks:
            u, v = b(u, v)
        return self.norm(u), (self.norm(v) if v is not None else None)


class MainNet(nn.Module):
    """Pre-norm transformer over chunks with sinusoidal chunk-index positions."""

    def __init__(self, D: int, n: int, heads: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, heads, 4 * D, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)
        self.D = D

    def pos(self, n, device):
        i = torch.arange(n, device=device, dtype=torch.float32)[:, None]
        f = torch.exp(torch.arange(0, self.D, 2, device=device, dtype=torch.float32) * (-math.log(1e4) / self.D))
        pe = torch.zeros(n, self.D, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(i * f), torch.cos(i * f)
        return pe

    def forward(self, z, valid):
        z = z + self.pos(z.shape[1], z.device)
        out = self.tf(z, src_key_padding_mask=~valid)
        return self.norm(out) * valid[..., None]


def ratio_loss(raw_cut, p, keep, target):
    """H-Net compression loss; minimal (=1) when the cut fraction and mean p both equal 1/target."""
    k = keep.float()
    n = k.sum().clamp(min=1.0)
    Fr = (raw_cut.float() * k).sum() / n
    Gr = (p.float() * k).sum() / n
    return target / (target - 1) * ((target - 1) * Fr * Gr + (1 - Fr) * (1 - Gr))


# ----------------------------------------------------------------------------- the model
class Chunker(nn.Module):
    def __init__(self, d=160, enc_layers=4, main_layers=8, dec_layers=2, heads=8, target_ratio=4.0, sym=True,
                 enc="conv", router="learned", fixed_len=3):
        super().__init__()
        self.sym, self.target, D = sym, float(target_ratio), 2 * d
        self.D, self.enc = D, enc
        # router="fixed": ABLATION. Cuts at fixed, mirror-symmetric positions (gap j is cut when min(j, L-j) is a
        # multiple of fixed_len), so chunks are ~fixed_len bp, identical for every sequence, and still exactly
        # RC-equivariant. Everything else (encoder, group-averaged transformer, heads) is unchanged.
        if router not in ("learned", "fixed"):
            raise ValueError(f"router must be 'learned' or 'fixed', not {router!r}")
        self.router, self.fixed_len = router, int(fixed_len)
        self.embed = nn.Embedding(VOCAB, d)
        # "conv": per-strand dilated conv encoder; "duplex": cross-strand gated encoder (DuplexNet)
        self.g = DuplexNet(d, enc_layers) if enc == "duplex" else LocalNet(d, enc_layers)
        self.proj = None if sym else nn.Linear(d, D)          # baseline: widen d -> D (tiny extra params)
        self.q, self.k = nn.Linear(D, D), nn.Linear(D, D)
        for lin in (self.q, self.k):                          # H-Net init: identity projections
            nn.init.eye_(lin.weight); nn.init.zeros_(lin.bias)
        self.main = MainNet(D, main_layers, heads)
        self.res = nn.Linear(D, D)
        self.dec = LocalNet(D, dec_layers)
        self.head = nn.Linear(D, 4)

    # -- symmetrization wrapper: 0.5 * (f(args) + sigma(f(sigma(args)))), batched in one call
    def _sym(self, f, *args):
        out = f(*[torch.cat([a, sigma(a)], 0) for a in args])
        a, b = out.chunk(2, 0)
        return 0.5 * (a + sigma(b))

    def encode(self, x):
        if self.enc == "duplex":
            if self.sym:
                u, v = self.g(self.embed(x), self.embed(rc_tokens(x)))
                return torch.cat([u, v.flip(1)], -1)
            return self.proj(self.g(self.embed(x))[0])
        if self.sym:
            gx = self.g(self.embed(torch.cat([x, rc_tokens(x)], 0)))
            a, b = gx.chunk(2, 0)
            return torch.cat([a, b.flip(1)], -1)
        return self.proj(self.g(self.embed(x)))

    def _route_raw(self, H):
        q = F.normalize(self.q(H[:, 1:]).float(), dim=-1)
        k = F.normalize(self.k(H[:, :-1]).float(), dim=-1)
        return (0.5 * (1 - (q * k).sum(-1))).clamp(0, 1)      # [B, L-1]; entry j = gap between j and j+1

    def _fixed_cuts(self, H):
        """Mirror-symmetric fixed segmentation: score 1 at cut gaps, 0 elsewhere (gap j = between bases j and j+1)."""
        L = H.shape[1]
        j = torch.arange(1, L, device=H.device)
        cut = (torch.minimum(j, L - j) % self.fixed_len) == 0
        return cut.float()[None].expand(H.shape[0], -1)

    def route(self, H):
        if self.router == "fixed":
            return self._fixed_cuts(H)
        if not self.sym:
            return self._route_raw(H)
        r = self._route_raw(torch.cat([H, sigma(H)], 0))
        a, b = r.chunk(2, 0)
        return 0.5 * (a + b.flip(1))

    def _main(self, Z, valid, nch):
        if not self.sym:
            return self.main(Z, valid)
        out = self.main(torch.cat([Z, sigma_chunks(Z, nch)], 0), torch.cat([valid, valid], 0))
        a, b = out.chunk(2, 0)
        return 0.5 * (a + sigma_chunks(b, nch))

    def _dec(self, H, U):
        return self.dec(U + self.res(H))

    def _head(self, Y):
        if not self.sym:
            return self.head(Y)
        out = self.head(torch.cat([Y, sigma(Y)], 0))
        a, b = out.chunk(2, 0)
        return 0.5 * (a + b.flip(1).flip(-1))                 # flip(-1) = complement for order A,C,G,T

    def forward(self, x, mask_pos=None, decode=True):
        B, L = x.shape
        H = self.encode(x)
        p = self.route(H)
        raw_cut = p >= 0.5
        forced = (mask_pos[:, 1:] | mask_pos[:, :-1]) if mask_pos is not None else torch.zeros_like(raw_cut)
        cut = raw_cut | forced                                # masked bases always form their own chunk
        ratio = ratio_loss(raw_cut, p, ~forced, self.target) if self.router == "learned" else p.new_zeros(())

        cid = torch.cat([x.new_zeros(B, 1), cut.long().cumsum(1)], 1)       # chunk id per nucleotide
        nch = cid[:, -1] + 1
        # round the chunk dimension up to a multiple of 64: far fewer distinct tensor shapes, so the CUDA
        # caching allocator stops fragmenting (extra rows are padding, masked out everywhere below)
        Cmax = int(math.ceil(int(nch.max()) / 64) * 64)
        Hf = H.float()
        Z = Hf.new_zeros(B, Cmax, self.D).scatter_add_(1, cid[..., None].expand(-1, -1, self.D), Hf)
        cnt = Hf.new_zeros(B, Cmax).scatter_add_(1, cid, torch.ones_like(cid, dtype=Hf.dtype))
        Z = Z / cnt.clamp(min=1)[..., None]

        # chunk confidence (H-Net-style straight-through signal for the router), mirror-symmetric
        v = torch.where(cut, p, 1 - p)
        v = torch.where(forced, torch.ones_like(v), v)
        S = v.new_zeros(B, Cmax).scatter_add_(1, cid[:, :-1], v).scatter_add_(1, cid[:, 1:], v)
        Cn = v.new_zeros(B, Cmax).scatter_add_(1, cid[:, :-1], torch.ones_like(v)).scatter_add_(1, cid[:, 1:], torch.ones_like(v))
        conf = torch.where(Cn > 0, S / Cn.clamp(min=1), torch.ones_like(S))

        valid = torch.arange(Cmax, device=x.device)[None] < nch[:, None]
        Zm = self._main(Z, valid, nch).float()
        out = dict(p=p, cut=cut, cid=cid, nch=nch, Zm=Zm, valid=valid, ratio=ratio,
                   chunk_bp=(L / nch.float()).mean())
        if decode:
            ste = conf + (1 - conf).detach()                   # forward value 1, gradient reaches the router
            U = Zm.gather(1, cid[..., None].expand(-1, -1, self.D)) * ste.gather(1, cid)[..., None]
            Y = self._sym(self._dec, H, U) if self.sym else self._dec(H, U)
            out["Y"] = Y                                       # per-base decoder features (full resolution)
            out["logits"] = self._head(Y)
        return out

    def pooled_features(self, out, oriented=False):
        """Sequence feature for classification. Symmetric model: RC-invariant by construction.
        oriented=True keeps the two strand halves apart, so the (still equivariant) backbone can tell which strand
        it reads: use it for direction-dependent labels (promoters, splice sites, coding sequence)."""
        v = out["valid"].float()[..., None]
        mean = (out["Zm"] * v).sum(1) / v.sum(1)
        mx = out["Zm"].masked_fill(v == 0, float("-inf")).amax(1)     # max pooling catches single motifs
        return self._pool_sym(mean, mx, oriented)

    def pooled_base(self, out, oriented=False):
        """Same pooling over the per-base decoder output Y (no chunking: keeps single-base resolution).
        Y(rc x) = sigma(Y(x)), and mean/max over all positions ignore the order, so the symmetric combinations
        below are exactly strand-invariant as well. Batches hold equal-length sequences, so no padding mask."""
        Y = out["Y"].float()
        return self._pool_sym(Y.mean(1), Y.amax(1), oriented)

    def _pool_sym(self, mean, mx, oriented):
        if not self.sym or oriented:
            return torch.cat([mean, mx], -1)
        feats = []
        for z in (mean, mx):                                            # sigma swaps halves -> symmetric combos
            a, b = z.chunk(2, -1)
            feats += [a + b, (a - b).abs()]
        return torch.cat(feats, -1)


class Classifier(nn.Module):
    """use_length: also give the head the sequence length (log scale). Length is the same on both strands, so the
    model stays exactly strand-invariant. Padding-based models (CNNs, HyenaDNA, Caduceus) see length implicitly."""

    def __init__(self, backbone: Chunker, n_classes: int, oriented: bool = False, use_length: bool = False,
                 use_base: bool = False):
        super().__init__()
        self.backbone, self.oriented, self.use_length, self.use_base = backbone, oriented, use_length, use_base
        D = 2 * backbone.D * (2 if use_base else 1) + (1 if use_length else 0)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, n_classes))

    def forward(self, x):
        out = self.backbone(x, decode=self.use_base)
        z = self.backbone.pooled_features(out, self.oriented)
        if self.use_base:                                               # + base-resolution features
            z = torch.cat([z, self.backbone.pooled_base(out, self.oriented)], -1)
        if self.use_length:
            n = (x < 4).sum(1, keepdim=True).float().clamp(min=1)          # real bases (pads / N excluded)
            z = torch.cat([z, torch.log(n) / 7.0], -1)
        return self.head(z), out


RATIO_W = 0.03              # weight of the chunk-size (ratio) loss; raise it to hold chunks nearer the target size

VARIANTS = {                 # name -> (symmetric architecture?, reverse-complement augmentation?)
    "baseline": (False, False),
    "baseline+RCaug": (False, True),
    "RC-equivariant": (True, False),
}


def atomic_save(obj, path: Path):
    """Write to a temp file, flush it to disk, then swap it in: a power cut leaves either the old or the new file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def safe_load(path: Path, device="cpu"):
    """torch.load that returns None for a missing or unreadable (half-written) file."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None


def build(variant: str, mcfg: dict) -> Chunker:
    return Chunker(sym=VARIANTS[variant][0], **mcfg)


def load_pretrained(folder, device="cpu"):
    """All pretrained models in a run folder: list of (variant, model config, model in eval mode, display name).
    The encoder type is read from the weights themselves, so folders from older script versions load correctly."""
    import json as _json
    folder = Path(folder)
    info_f = folder / "pretrain_info.json"
    info = _json.loads(info_f.read_text()) if info_f.exists() else {}
    out = []
    for f in sorted(folder.glob("pretrain_*.final.pt")):
        variant = f.name[len("pretrain_"):-len(".final.pt")]
        if variant not in VARIANTS:
            continue
        state = torch.load(f, map_location="cpu")
        mcfg = dict(info.get("mcfg", dict(d=160, enc_layers=4, main_layers=8, dec_layers=2, heads=8, target_ratio=4.0)))
        mcfg["enc"] = "duplex" if any(".gate." in k for k in state) else "conv"
        m = build(variant, mcfg)
        m.load_state_dict(state)
        m.to(device).eval()
        name = f"{variant} ({n_params(m) / 1e6:.0f}M{', duplex' if mcfg['enc'] == 'duplex' else ''})"
        out.append((variant, mcfg, m, name))
    return out


def load_chrom(data_dir, chrom):
    """Encoded chromosome (A,C,G,T,N = 0..4), memory-mapped from the hg38_data files written by prepare_genome."""
    f = Path(data_dir) / f"{chrom}.tok.npy"
    return np.load(f, mmap_mode="r") if f.exists() else None


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- genome data
UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{}.fa.gz"


def prepare_genome(chroms, data_dir: Path, log=print, mmap=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = []
    for c in chroms:
        tok_f, rep_f = data_dir / f"{c}.tok.npy", data_dir / f"{c}.rep.npy"
        if not tok_f.exists():
            gz = data_dir / f"{c}.fa.gz"
            if not gz.exists():
                log(f"  downloading {c} ...")
                tmp = gz.with_suffix(".part")                       # atomic: a cut download is never mistaken for done
                urllib.request.urlretrieve(UCSC.format(c), tmp)
                os.replace(tmp, gz)
            try:
                with gzip.open(gz, "rb") as fh:
                    raw = b"".join(line.strip() for line in fh if not line.startswith(b">"))
            except (EOFError, OSError, gzip.BadGzipFile):
                gz.unlink(missing_ok=True)                             # damaged download: fetch it again
                raise RuntimeError(f"{gz.name} was damaged and has been deleted; run the command again")
            b = np.frombuffer(raw, dtype=np.uint8)
            np.save(rep_f, (b >= 97).astype(np.uint8))   # hg38 soft-masks repeats in lowercase
            tmp_tok = data_dir / f"{c}.tok.part.npy"
            np.save(tmp_tok, _LUT[b])                     # tok written last + atomically = the "done" marker
            os.replace(tmp_tok, tok_f)
            del raw, b
        mode = "r" if mmap else None          # mmap: one shared copy in the OS file cache for all parallel runs
        arrays.append((np.load(tok_f, mmap_mode=mode), np.load(rep_f, mmap_mode=mode)))
        log(f"  {c}: {len(arrays[-1][0]):,} bp")
    return Genome(arrays)


class Genome:
    def __init__(self, arrays):
        self.tok = [a for a, _ in arrays]
        self.rep = [r for _, r in arrays]
        lens = np.array([len(a) for a in self.tok], dtype=np.float64)
        self.prob = lens / lens.sum()

    def sample(self, B, L, rng):
        toks, reps = np.empty((B, L), np.uint8), np.empty((B, L), np.uint8)
        i = 0
        while i < B:
            c = rng.choice(len(self.tok), p=self.prob)
            s = int(rng.integers(0, len(self.tok[c]) - L))
            w = np.asarray(self.tok[c][s:s + L])
            if (w == N).mean() > 0.1:
                continue
            toks[i], reps[i] = w, self.rep[c][s:s + L]
            i += 1
        return toks, reps


ENFORMER_BED = "https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed"   # HyenaDNA / Caduceus split


class IntervalGenome(Genome):
    """Samples windows only from the training intervals of the Enformer hg38 split (as HyenaDNA and Caduceus do).
    Each sample: an interval chosen with probability proportional to its length, then a random window inside it."""

    def __init__(self, genome: Genome, chroms, bed_path: Path, split="train", log=print):
        self.tok, self.rep = genome.tok, genome.rep
        idx = {c: i for i, c in enumerate(chroms)}
        iv = []
        for line in Path(bed_path).read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 4 and f[3].strip() == split and f[0] in idx:
                iv.append((idx[f[0]], int(f[1]), int(f[2])))
        if not iv:
            raise ValueError(f"no '{split}' intervals for {chroms[:3]}... in {bed_path}")
        self.iv = np.array(iv, dtype=np.int64)
        lens = (self.iv[:, 2] - self.iv[:, 1]).astype(np.float64)
        self.p_iv = lens / lens.sum()
        log(f"  Enformer split: {len(iv)} '{split}' intervals, {lens.sum() / 1e9:.2f} Gb")

    def sample(self, B, L, rng):
        toks, reps = np.empty((B, L), np.uint8), np.empty((B, L), np.uint8)
        i = 0
        while i < B:
            c, a, b = self.iv[rng.choice(len(self.iv), p=self.p_iv)]
            a, b = max(0, int(a)), min(len(self.tok[c]), int(b))
            if b - a < L:
                continue
            s = int(rng.integers(a, b - L + 1))
            w = np.asarray(self.tok[c][s:s + L])
            if (w == N).mean() > 0.1:
                continue
            toks[i], reps[i] = w, self.rep[c][s:s + L]
            i += 1
        return toks, reps


class SyntheticGenome(Genome):
    """Offline stand-in for tests: random sequence with planted motifs."""

    def __init__(self, length=2_000_000, seed=0):
        rng = np.random.default_rng(seed)
        tok = rng.integers(0, 4, length).astype(np.uint8)
        motif = np.array([0, 3, 2, 1, 2, 1, 0, 3], np.uint8)
        for s in rng.integers(0, length - 8, length // 200):
            tok[s:s + 8] = motif
        super().__init__([(tok, (rng.random(length) < 0.3).astype(np.uint8))])


def mlm_mask(x, rate=0.15):
    m = (torch.rand(x.shape, device=x.device) < rate) & (x < 4)
    r = torch.rand(x.shape, device=x.device)
    inp = x.clone()
    inp[m & (r < 0.8)] = MASK
    rnd = m & (r >= 0.8) & (r < 0.9)
    inp[rnd] = torch.randint(0, 4, (int(rnd.sum()),), device=x.device)
    return inp, m


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))


def pretrain_step(model, opt, genome, rng, bs, L, rc_aug, device, accum=1):
    """One optimizer step on bs windows, processed as `accum` micro-batches of bs // accum (gradient accumulation)."""
    opt.zero_grad(set_to_none=True)
    mb, mlms, cbps = bs // accum, [], []
    for _ in range(accum):
        toks, reps = genome.sample(mb, L, rng)
        x = torch.from_numpy(toks).long().to(device)
        w = torch.from_numpy(reps).to(device).float().mul(-0.9).add(1.0)       # repeats weighted 0.1
        if rc_aug:
            flip = torch.rand(mb, device=device) < 0.5
            x = torch.where(flip[:, None], rc_tokens(x), x)
            w = torch.where(flip[:, None], w.flip(1), w)
        inp, m = mlm_mask(x)
        with autocast(device):
            out = model(inp, mask_pos=m)
        logits = out["logits"].float()[m]
        tgt, ww = x[m], w[m]
        mlm = (F.cross_entropy(logits, tgt, reduction="none") * ww).sum() / ww.sum().clamp(min=1)
        ((mlm + RATIO_W * out["ratio"]) / accum).backward()
        mlms.append(mlm.item()); cbps.append(float(out["chunk_bp"]))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(np.mean(mlms)), float(np.mean(cbps))


def pretrain(variant, mcfg, genome, steps, bs, L, peak_lr, device, ckpt_dir: Path, log=print, seed=0, accum=1,
             betas=(0.9, 0.95)):
    """Resumable MLM pretraining. Returns the path of the final weights."""
    final = ckpt_dir / f"pretrain_{variant}.final.pt"
    if final.exists():
        log(f"  {variant}: final weights exist, skipping")
        return final
    torch.manual_seed(seed)
    model = build(variant, mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=betas, weight_decay=0.1)
    rng = np.random.default_rng(seed)
    log_every, ck_every = max(1, min(200, steps // 500)), max(10, min(500, steps // 200))
    step, ck = 0, ckpt_dir / f"pretrain_{variant}.ckpt.pt"
    prev = ck.with_suffix(".prev.pt")
    for cand in (ck, prev):                    # newest first; a checkpoint cut short by a power loss is skipped
        if not cand.exists():
            continue
        try:
            st = torch.load(cand, map_location=device, weights_only=False)
        except Exception as e:
            log(f"  {variant}: {cand.name} is unreadable ({type(e).__name__}), trying an older checkpoint")
            continue
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step, rng.bit_generator.state = st["step"], st["rng"]
        log(f"  {variant}: resumed at step {step} from {cand.name}")
        break
    else:
        if ck.exists() or prev.exists():
            log(f"  {variant}: no readable checkpoint, starting from step 0")
    rc_aug = VARIANTS[variant][1]
    t0, hist = time.time(), []
    model.train()
    while step < steps:
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step, steps, peak_lr, warmup=min(1000, steps // 10 + 1))
        mlm, cbp = pretrain_step(model, opt, genome, rng, bs, L, rc_aug, device, accum)
        hist.append(mlm)
        step += 1
        if device == "cuda" and step % 100 == 0:
            torch.cuda.empty_cache()          # hand fragmented cache back; stops Windows spilling into system RAM
        if step % log_every == 0 or step == steps:
            el = time.time() - t0
            mem = f"  gpu mem {torch.cuda.memory_reserved() / 1e9:.1f} GB" if device == "cuda" else ""
            log(f"  {variant} step {step}/{steps}  mlm {np.mean(hist[-log_every:]):.4f}  chunk {cbp:.1f} bp  "
                f"{el / 60:.1f} min{mem}")
        if step % ck_every == 0 or step == steps:
            if ck.exists():
                os.replace(ck, prev)               # keep the previous checkpoint as a fallback
            atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                             rng=rng.bit_generator.state), ck)
    atomic_save(model.state_dict(), final)
    return final


def time_steps(variant, mcfg, genome, bs, L, device, n=15, accum=1):
    """Seconds per pretraining step (used to fit the pretraining budget)."""
    model = build(variant, mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    for i in range(n + 3):
        if i == 3:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        pretrain_step(model, opt, genome, rng, bs, L, VARIANTS[variant][1], device, accum)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return dt


# ----------------------------------------------------------------------------- downstream tasks
NT_REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def load_nt_task(task, max_train, max_test, seed=0, log=print):
    from datasets import load_dataset
    try:
        ds = load_dataset(NT_REPO, task)
    except Exception:
        ds = load_dataset(NT_REPO)
        ds = ds.filter(lambda e: e["task"] == task)
    cols = ds["train"].column_names
    sc = "sequence" if "sequence" in cols else cols[0]
    lc = "label" if "label" in cols else [c for c in cols if c not in (sc, "task", "name")][-1]

    def take(split, n):
        d = ds[split].shuffle(seed=seed)
        d = d.select(range(min(n, len(d))))
        return [encode(s.upper()) for s in d[sc]], np.asarray(d[lc]).astype(np.int64)

    tr, te = take("train", max_train), take("test", max_test)
    return split_val(tr, te, seed)


GB_URL = "https://zenodo.org/records/16605299/files/{}_v0.zip?download=1"     # official Genomic Benchmarks files
GB_TASKS = {  # name -> label depends on the strand read? (then use the orientation-aware head)
    "dummy_mouse_enhancers_ensembl": False,
    "demo_coding_vs_intergenomic_seqs": True,     # coding sequence has a direction
    "demo_human_or_worm": False,
    "human_enhancers_cohn": False,
    "human_enhancers_ensembl": False,
    "human_ensembl_regulatory": False,
    "human_nontata_promoters": True,              # promoters point toward their gene
    "human_ocr_ensembl": False,
    "drosophila_enhancers_stark": False,
}


def load_gb_task(name, max_train, max_test, data_dir=Path("gb_data"), seed=0, log=print):
    """Genomic Benchmarks (Gresova et al. 2023): downloads the official zip once, caches the parsed sequences."""
    import shutil
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"{name}.npz"
    if not cache.exists():
        root = data_dir / name
        if not (root / "train").exists():
            z = data_dir / f"{name}.zip"
            if not z.exists():
                log(f"  downloading Genomic Benchmarks {name} ...")
                tmp = z.with_suffix(".part")
                urllib.request.urlretrieve(GB_URL.format(name), tmp)
                os.replace(tmp, z)
            ex = data_dir / f"{name}_unzipped"
            shutil.rmtree(ex, ignore_errors=True)                     # a half-finished unzip is redone
            shutil.unpack_archive(str(z), str(ex))
            hits = sorted((p.parent for p in ex.rglob("train") if p.is_dir() and (p.parent / "test").is_dir()),
                          key=lambda q: len(q.parts))
            if not hits:
                raise FileNotFoundError(f"no train/ and test/ folders inside {z}")
            root = hits[0]
        classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
        arrays = {}
        for split in ("train", "test"):
            seqs, labels = [], []
            for ci, c in enumerate(classes):
                for f in sorted((root / split / c).glob("*.txt")):
                    seqs.append(f.read_text().strip().upper()); labels.append(ci)
            arrays[split + "_seq"] = np.array(seqs, dtype=object)
            arrays[split + "_y"] = np.array(labels, dtype=np.int64)
        np.savez_compressed(cache, classes=np.array(classes), **arrays)
        log(f"  {name}: classes {classes}, train {len(arrays['train_y'])}, test {len(arrays['test_y'])}")
    d = np.load(cache, allow_pickle=True)
    rng = np.random.default_rng(seed)

    def take(split, n):
        idx = rng.permutation(len(d[split + "_y"]))[:n]
        return [encode(s) for s in d[split + "_seq"][idx]], d[split + "_y"][idx]
    return split_val(take("train", max_train), take("test", max_test), seed)


def synthetic_task(name, n_train=600, n_test=200, L=200, seed=0, **_):
    """Offline stand-in: label = whether a motif occurs on either strand (an RC-symmetric label)."""
    rng = np.random.default_rng(sum(name.encode()) + seed)
    motif = np.array([0, 3, 2, 1, 2, 2], np.uint8)

    def make(n):
        X, y = [], []
        for _ in range(n):
            s = rng.integers(0, 4, L).astype(np.uint8)
            lab = int(rng.random() < 0.5)
            if lab:
                pos = rng.integers(0, L - 6)
                s[pos:pos + 6] = motif if rng.random() < 0.5 else (3 - motif)[::-1]
            X.append(s); y.append(lab)
        return X, np.array(y)
    return split_val(make(n_train), make(n_test), seed)


BIO_DIR = Path("bio_tasks")
BIO_TASKS = {"bio_gene_strand_tss": True, "bio_gene_strand_donor": True}     # built by bio_strand_tasks.py
# longer-window versions (bio_strand_tasks.py --window 4096 ...): length-generalization test, separate "biolong" suite
BIO_LONG_TASKS = {f.stem: True for f in sorted(Path("bio_tasks").glob("bio_gene_strand_*_w*.npz"))}


def load_bio_task(name, max_train, max_test, seed=0, log=print):
    f = BIO_DIR / f"{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run  python bio_strand_tasks.py  first")
    d = np.load(f, allow_pickle=True)
    rng = np.random.default_rng(seed)

    def take(split, n):
        idx = rng.permutation(len(d[split + "_y"]))[:n]
        return [np.asarray(x, np.uint8) for x in d[split + "_seq"][idx]], d[split + "_y"][idx].astype(np.int64)
    return split_val(take("train", max_train), take("test", max_test), seed)


def load_task(name, max_train=10 ** 9, max_test=10 ** 9, seed=0, log=print):
    """Any downstream task by name: Genomic Benchmarks, our biology tasks (bio_*), else NT-revised."""
    if name in GB_TASKS:
        return load_gb_task(name, max_train, max_test, seed=seed, log=log)
    if name in BIO_TASKS or name in BIO_LONG_TASKS:
        return load_bio_task(name, max_train, max_test, seed=seed, log=log)
    return load_nt_task(name, max_train, max_test, seed=seed, log=log)


def split_val(tr, te, seed):
    X, y = tr
    idx = np.random.default_rng(seed).permutation(len(X))
    nv = max(1, len(X) // 10)
    vi, ti = idx[:nv], idx[nv:]
    return dict(train=([X[i] for i in ti], y[ti]), val=([X[i] for i in vi], y[vi]), test=te)


def batches(X, y, bs, rng=None, max_tokens=65536):
    """Batches of equal-length sequences (reversal needs no padding inside a sequence).
    Long sequences get smaller batches (at most max_tokens bases per batch); 1 kb tasks keep bs=64."""
    by_len = {}
    for i, s in enumerate(X):
        by_len.setdefault(len(s), []).append(i)
    out = []
    for L in sorted(by_len):
        idx = np.array(by_len[L])
        if rng is not None:
            rng.shuffle(idx)
        b = max(1, min(bs, max_tokens // max(1, L)))
        out += [idx[i:i + b] for i in range(0, len(idx), b)]
    if rng is not None:
        rng.shuffle(out)
    for b in out:
        yield torch.from_numpy(np.stack([X[i] for i in b])).long(), torch.from_numpy(y[b])


@torch.no_grad()
def predict(clf, X, y, bs, device, amp=True):
    """Class probabilities on the test sequences as given (P) and reverse-complemented (Prc).
    amp=False evaluates in full 32-bit precision (removes near-threshold flips caused by bf16 rounding)."""
    clf.eval()
    P, Prc, Y, cbp = [], [], [], []
    for xb, yb in batches(X, y, bs):
        xb = xb.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device == "cuda")):
            lf, out = clf(xb)
            lr_, _ = clf(rc_tokens(xb))
        P.append(lf.float().softmax(-1).cpu()); Prc.append(lr_.float().softmax(-1).cpu())
        Y.append(yb); cbp.append(float(out["chunk_bp"]))
    return torch.cat(P).numpy(), torch.cat(Prc).numpy(), torch.cat(Y).numpy(), float(np.mean(cbp))


def mcc(y, yhat):
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(y, yhat))


def f1(y, yhat, n_cls=2):
    """F1 of the positive class for binary tasks (as in the NT benchmark), macro-F1 for multi-class tasks."""
    from sklearn.metrics import f1_score
    return float(f1_score(y, yhat, average="binary" if n_cls == 2 else "macro", zero_division=0))


def strand_metrics(Y, P, Prc):
    """All strand metrics from per-sequence probabilities on both strands.
    strand-consistent: a prediction counts as correct only if it is correct on BOTH strands
      consistent_acc = P(pred_fwd == y and pred_rc == y)
      consistent_mcc = MCC of a prediction that equals the (shared) answer when both strands agree and are right,
                       and is scored wrong otherwise (the model gave the other strand a different or wrong answer)
    For an exactly RC-invariant model these equal plain accuracy / MCC."""
    yf, yr = P.argmax(1), Prc.argmax(1)
    n_cls = P.shape[1]
    both = (yf == Y) & (yr == Y)
    # strand-consistent prediction: the shared answer when both strands agree, otherwise counted as wrong
    y_cons = np.where(yf == yr, yf, np.where(yf != Y, yf, yr))
    y_cons = np.where((yf != yr) & (y_cons == Y), (Y + 1) % n_cls, y_cons)
    return dict(mcc=mcc(Y, yf), mcc_rc_input=mcc(Y, yr), mcc_rc_avg=mcc(Y, (P + Prc).argmax(1)),
                f1=f1(Y, yf, n_cls), f1_rc=f1(Y, yr, n_cls), consistent_f1=f1(Y, y_cons, n_cls),
                acc=float((yf == Y).mean()), acc_rc=float((yr == Y).mean()),
                consistent_acc=float(both.mean()), consistent_mcc=mcc(Y, y_cons),
                flip_rate=float((yf != yr).mean()))


def finetune(variant, mcfg, data, device, seed, lr, epochs, bs, init_state=None, log=print, eval_amp=True,
             save_preds=None, oriented=False, ckpt=None, use_length=False, patience=0, use_base=False):
    """Fine-tune and test on both strands. ckpt: a file path; the run then saves its state after every epoch and,
    if interrupted (crash, power cut), resumes from the last finished epoch when called again."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bb = build(variant, mcfg)
    if init_state is not None:
        bb.load_state_dict(init_state)
    n_cls = int(max(data["train"][1].max(), data["test"][1].max())) + 1
    clf = Classifier(bb, n_cls, oriented=oriented, use_length=use_length, use_base=use_base).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    Xtr, ytr = data["train"]
    total = epochs * sum(1 for _ in batches(Xtr, ytr, bs))
    # RC augmentation with an unchanged label is only valid when the label does not depend on the strand; for
    # direction-dependent tasks (oriented head) it would teach the model to ignore exactly what it must predict
    rc_aug, step, best, best_state, ep0, spent = VARIANTS[variant][1] and not oriented, 0, -2.0, None, 0, 0.0
    best_ep = 0
    st = safe_load(ckpt, device) if ckpt is not None else None
    if st is not None:
        clf.load_state_dict(st["clf"]); opt.load_state_dict(st["opt"])
        step, best, best_state, ep0, spent = st["step"], st["best"], st["best_state"], st["epoch"], st["spent"]
        best_ep = st.get("best_ep", ep0)
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"].cpu().to(torch.uint8))
        log(f"    resumed fine-tuning after epoch {ep0}/{epochs}")
    t0 = time.time()
    for ep in range(ep0, epochs):
        if patience and ep - best_ep >= patience:          # early stopping (also after a resume)
            break
        clf.train()
        for xb, yb in batches(Xtr, ytr, bs, rng):
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step, total, lr, warmup=max(1, total // 20))
            xb, yb = xb.to(device), yb.to(device)
            if rc_aug:
                flip = torch.rand(len(xb), device=device) < 0.5
                xb = torch.where(flip[:, None], rc_tokens(xb), xb)
            with autocast(device):
                logits, out = clf(xb)
            loss = F.cross_entropy(logits.float(), yb) + RATIO_W * out["ratio"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
            step += 1
        P, _, Y, _ = predict(clf, *data["val"], bs, device)
        v = mcc(Y, P.argmax(1))
        if v > best:
            best, best_ep = v, ep + 1
            best_state = {k: t.detach().cpu().clone() for k, t in clf.state_dict().items()}
        if ckpt is not None:
            atomic_save(dict(clf=clf.state_dict(), opt=opt.state_dict(), step=step, best=best, best_state=best_state,
                             best_ep=best_ep, epoch=ep + 1, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                             spent=spent + time.time() - t0), ckpt)
    clf.load_state_dict(best_state)
    P, Prc, Y, cbp = predict(clf, *data["test"], bs, device, amp=eval_amp)
    if save_preds is not None:
        np.savez_compressed(save_preds, y=Y, p=P, p_rc=Prc)
    return dict(**strand_metrics(Y, P, Prc), val_mcc=best, best_epoch=best_ep, chunk_bp=cbp, oriented=oriented,
                train_min=(spent + time.time() - t0) / 60, params=n_params(clf))
