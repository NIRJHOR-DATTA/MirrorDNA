import torch
from rcchunk import Chunker, Classifier, rc_tokens, mlm_mask


def check(sym, seed=0, L=301, B=4, device="cpu", enc="conv", router="learned"):
    torch.manual_seed(seed)
    m = Chunker(d=16, enc_layers=2, main_layers=2, dec_layers=1, heads=2, sym=sym, enc=enc,
                router=router).to(device).eval()
    for p in m.parameters():                        # random weights so the router makes nontrivial cuts
        p.data.normal_(0, 0.3)
    x = torch.randint(0, 4, (B, L), device=device)
    inp, mask = mlm_mask(x)
    with torch.no_grad():
        o1 = m(inp, mask_pos=mask)
        o2 = m(rc_tokens(inp), mask_pos=mask.flip(1))
        f1, f2 = m.pooled_features(o1), m.pooled_features(o2)
    cuts_mirrored = torch.equal(o1["cut"], o2["cut"].flip(1))
    logit_err = (o1["logits"] - o2["logits"].flip(1).flip(-1)).abs().max().item()
    feat_err = (f1 - f2).abs().max().item() / f1.abs().max().item()
    return dict(cuts_mirrored=cuts_mirrored, n_cuts=int(o1["cut"].sum()), mlm_logit_err=logit_err,
                pooled_feature_rel_err=feat_err)


if __name__ == "__main__":
    ok = True
    for seed, enc in [(0, "conv"), (1, "conv"), (2, "conv"), (0, "duplex"), (1, "duplex")]:
        s, b = check(True, seed, enc=enc), check(False, seed, enc=enc)
        print(f"seed {seed} [{enc}]  RC-equivariant: {s}")
        print(f"        baseline:       {b}")
        ok &= s["cuts_mirrored"] and s["mlm_logit_err"] < 1e-4 and s["pooled_feature_rel_err"] < 1e-4
        ok &= not b["cuts_mirrored"] or b["pooled_feature_rel_err"] > 1e-3
    for seed in (0, 1):                             # ablation: fixed, mirror-symmetric cuts
        for L in (301, 1024):
            s = check(True, seed, L=L, router="fixed")
            print(f"seed {seed} [fixed router, L={L}]  RC-equivariant: {s}")
            ok &= s["cuts_mirrored"] and s["mlm_logit_err"] < 1e-4 and s["pooled_feature_rel_err"] < 1e-4
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
