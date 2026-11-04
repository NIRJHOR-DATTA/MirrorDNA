import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import COLORS, OURS, plt, save

p = argparse.ArgumentParser()
p.add_argument("--ours", default="results_scale_tiny_RC-equivariant_caduceus10B")
p.add_argument("--aug", default="results_scale_tiny_duplex_baseline_RCaug_caduceus10B")
p.add_argument("--tokens", default="splice_tokens_report.csv")
p.add_argument("--flank", type=int, default=20)
p.add_argument("--top", type=int, default=6)
p.add_argument("--out", default="fig_splice")
args = p.parse_args()


def profile(folder):
    f = sorted(Path(".").glob(f"bio_boundaries_*_{Path(folder).name}_profiles.csv"))
    return pd.read_csv(f[0]) if f else None


fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]
for folder, ls, lab in [(args.ours, "-", OURS), (args.aug, "--", "Chunker + RC aug.")]:
    d = profile(folder)
    if d is None:
        print(f"no bio_boundaries profile for {folder}")
        continue
    bg = d["random"].mean()
    d = d[d.offset.abs() <= args.flank]
    for site, col in [("splice donor", "#c0392b"), ("splice acceptor", "#2c7fb8")]:
        a.plot(d.offset, d[site] / bg, ls=ls, color=col, lw=1.4 if ls == "-" else 1.0,
               label=f"{site} ({'ours' if folder == args.ours else 'RC aug.'})")
a.axhline(1, color="0.6", lw=0.6)
a.axvline(0, color="0.8", lw=0.6, ls=":")
a.set_xlabel("position relative to the splice junction (bp, reference strand)")
a.set_ylabel("chunk boundary rate / background")
a.set_title("(a) where the tokenizer cuts", loc="left")
a.legend(frameon=False, fontsize=6, loc="upper left")

b = ax[1]
t = pd.read_csv(args.tokens)
t = t[t.site == "splice donor"].sort_values("share_model", ascending=False).head(args.top)
y = np.arange(len(t))
b.barh(y - 0.2, 100 * t.share_model, 0.38, color=COLORS[OURS], label="model's cuts")
b.barh(y + 0.2, 100 * t.share_shifted, 0.38, color="0.7", label="cuts shifted away from the site")
for i, (m, c) in enumerate(zip(t.share_model, t.share_shifted)):
    if c > 0:
        b.text(100 * m + 0.3, i - 0.2, (f"{m / c:.0f}×" if m / c >= 1 else "<1×"), va="center", fontsize=6.5, fontweight="bold")
b.set_yticks(y); b.set_yticklabels(t.token, family="monospace", fontsize=7); b.invert_yaxis()
b.set_xlabel("share of splice donors (%)")
b.set_title("(b) token formed at splice donors", loc="left")
b.legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout()
save(fig, args.out)
