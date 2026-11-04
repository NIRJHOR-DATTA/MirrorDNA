import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})

OURS = "MirrorDNA (ours)"
COLORS = {OURS: "#c0392b", "Chunker + RC aug.": "#e67e22", "Caduceus-PS-1k": "#2c7fb8", "Caduceus-PH-1k": "#7fb3d5",
          "HyenaDNA": "#27ae60", "NT-v2": "#7d3c98", "DNABERT-2": "#555555", "BPE (DNABERT-2)": "#555555",
          "6-mer (NT)": "#7d3c98"}
MARKERS = {OURS: "*", "Chunker + RC aug.": "D", "Caduceus-PS-1k": "s", "Caduceus-PH-1k": "s", "HyenaDNA": "^",
           "NT-v2": "o", "DNABERT-2": "v"}


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    print("saved", stem + ".pdf /.png")
