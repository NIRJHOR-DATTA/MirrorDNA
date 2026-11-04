import argparse
import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""), flush=True)


py_files = sorted(f for f in ROOT.rglob("*.py") if "__pycache__" not in f.parts and f.name != "check_release.py")

# 1. compile
bad = []
for f in py_files:
    try:
        compile(f.read_text(encoding="utf-8"), str(f), "exec")
    except SyntaxError as e:
        bad.append(f"{f.relative_to(ROOT)}: {e}")
report("all .py files compile", not bad, "; ".join(bad))

# 2. local imports resolve inside the folder (installed or not, third-party packages are checked in step 3)
local = {f.stem for f in py_files}
PLATFORM = {"msvcrt", "fcntl", "winreg", "_winapi"}                       # imported only on Windows / Linux
THIRD_PARTY = {"torch", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "tabulate", "datasets", "huggingface_hub",
               "transformers", "einops", "safetensors", "mamba_ssm", "causal_conv1d", "seaborn", "pyfaidx", "tqdm"}
missing = []
for f in py_files:
    tree = ast.parse(f.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        for n in names:
            if n in local or n in THIRD_PARTY or n in PLATFORM or importlib.util.find_spec(n) is not None:
                continue
            missing.append(f"{f.relative_to(ROOT)} imports '{n}' (not in the folder, not a known package)")
# also scripts started via subprocess / runpy by file name
for f in py_files:
    for m in re.findall(r"""with_name\(["']([\w./]+\.py)["']\)|["']([\w/]+\.py)["']""", f.read_text(encoding="utf-8")):
        name = m[0] or m[1]
        if name.startswith("figs/") or "/" not in name:
            if not (ROOT / name).exists() and not (f.parent / name).exists() and name != "check_release.py":
                missing.append(f"{f.relative_to(ROOT)} refers to '{name}'")
missing = sorted(set(missing))
report("local imports and script references resolve", not missing,
       "\n       ".join(missing[:30]) + ("" if len(missing) <= 30 else f"\n       ... {len(missing) - 30} more"))

# 3. requirements
req = ROOT / "requirements.txt"
need = []
if req.exists():
    alias = {"scikit-learn": "sklearn", "huggingface_hub": "huggingface_hub", "pyyaml": "yaml"}
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        pkg = re.split(r"[<>=!~\[ ]", line)[0]
        mod = alias.get(pkg.lower(), pkg.replace("-", "_"))
        if importlib.util.find_spec(mod) is None:
            need.append(pkg)
report("packages in requirements.txt are installed", not need,
       ("missing: " + ", ".join(need) + "   ->  pip install -r requirements.txt") if need else "")

# 4. equivariance test on random weights
r = subprocess.run([sys.executable, "test_equivariance.py"], cwd=ROOT, capture_output=True, text=True)
report("test_equivariance.py", r.returncode == 0 and "PASS" in r.stdout, (r.stdout + r.stderr).strip().splitlines()[-1]
       if (r.stdout + r.stderr).strip() else "no output")

# 5. released weights
sys.path.insert(0, str(ROOT))
try:
    import torch
    import rcchunk as rk
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    folders = sorted(d for d in (ROOT / "weights").iterdir() if d.is_dir()) if (ROOT / "weights").exists() else []
    if not folders:
        report("weights/ contains models", False, "no model folders in weights/")
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 4, (4, 1024), generator=g).to(dev)
    for d in folders:
        try:
            models = rk.load_pretrained(d, dev)
            if not models:
                report(f"weights/{d.name} loads", False, "no pretrain_*.final.pt")
                continue
            variant, mcfg, m, name = models[0]
            with torch.no_grad():
                o1, o2 = m(x, decode=False), m(rk.rc_tokens(x), decode=False)
                err = float((m.pooled_features(o1) - m.pooled_features(o2)).abs().max() /
                            m.pooled_features(o1).abs().max())
                mirrored = torch.equal(o1["cut"], o2["cut"].flip(1))
            sym = rk.VARIANTS[variant][0]
            ok = (mirrored and err < 1e-4) if sym else True
            report(f"weights/{d.name} loads ({name}, {rk.n_params(m) / 1e6:.2f}M)", ok,
                   f"{'equivariant' if sym else 'control (not equivariant, as intended)'}: cuts mirrored {mirrored}, "
                   f"feature difference between strands {err:.1e}, chunk length {float(o1['chunk_bp']):.2f} bp")
        except Exception as e:                                           # noqa: BLE001
            report(f"weights/{d.name} loads", False, f"{type(e).__name__}: {e}")
except Exception as e:                                                   # noqa: BLE001
    report("model code imports", False, f"{type(e).__name__}: {e}")

# 6. quickstart
qs = ROOT / "example" / "quickstart.py"
data = ROOT / "example" / "data" / "gene_strand_tss_small.npz"
if qs.exists() and (ROOT / "weights" / "mirrordna-tiny").exists():
    cmd = [sys.executable, str(qs), "--weights", "weights/mirrordna-tiny"]
    if not args.quick and data.exists():
        cmd.append("--finetune")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = "\n       ".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    report("example/quickstart.py" + (" --finetune" if "--finetune" in cmd else ""), r.returncode == 0, tail)
    if not data.exists():
        report("small test dataset example/data/gene_strand_tss_small.npz present", False,
               "build it once from the project folder (see the output of make_github_folder.py)")
else:
    report("example/quickstart.py and weights/mirrordna-tiny present", False)

# 7. size and privacy
big = [f"{f.relative_to(ROOT)} ({f.stat().st_size / 1e6:.0f} MB)" for f in ROOT.rglob("*")
       if f.is_file() and f.stat().st_size > 50e6 and ".git" not in f.parts]
report("no file above 50 MB", not big, ", ".join(big))
pat = re.compile(r"([A-Za-z]:\\\\?Users\\\\?[^\s\"']+|/home/[a-z0-9_]+/|/mnt/[a-z]/|[\w.+-]+@[\w-]+\.[\w.]+|hf_[A-Za-z0-9]{20,})")
hits = []
for f in ROOT.rglob("*"):
    if f.is_file() and f.suffix in (".py", ".md", ".txt", ".json", ".bat", ".sh", ".cfg") and ".git" not in f.parts \
            and f.name != "check_release.py":
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for m in pat.findall(line):
                if "noreply" not in m and "example.com" not in m:
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {m}")
report("no personal paths, e-mail addresses or tokens", not hits, "\n       ".join(hits[:20]))

n_fail = sum(not ok for _, ok in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed" + ("" if not n_fail else " -- fix the FAIL lines above"))
raise SystemExit(1 if n_fail else 0)
