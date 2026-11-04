# MirrorDNA: learning strand-equivariant tokenizations of DNA


This repository contains the model, pretraining and fine-tuning code, the baselines and analyses of the paper, a
pretrained model and a minimal test dataset.

> Paper: *Learning Strand-Equivariant Tokenizations of DNA with MirrorDNA* (RECOMB 2027 submission).

## Installation

```bash
git clone https://github.com/NIRJHOR-DATTA/MirrorDNA.git
cd MirrorDNA
pip install -r requirements.txt
```

Python 3.10 or 3.11 and PyTorch 2.1 or newer. For NVIDIA Blackwell GPUs (RTX 50xx, RTX PRO 6000) install the CUDA 12.8
build: `pip install torch --index-url https://download.pytorch.org/whl/cu128`. The model itself is plain PyTorch
(no custom kernels) and runs on Linux, Windows and macOS, on GPU or CPU.

## Quick start (about 2 minutes)

1. Check the equivariance guarantee on random weights (CPU, seconds):

   ```bash
   python test_equivariance.py          # ends with PASS
   ```

2. Run the main model, which is included in `weights/mirrordna-tiny/` (all released models and why each is there:
   `weights/README.md`):

   ```bash
   python example/quickstart.py --weights weights/mirrordna-tiny --finetune
   ```

   This prints the learned chunks of a sequence and of its reverse complement (mirror images), checks the guarantee on
   random 1-kb inputs, and fine-tunes the model for a few epochs on the minimal test dataset
   `example/data/gene_strand_tss_small.npz` (400 training and 200 test windows of 1,024 bp: predict on which strand
   a gene lies from the sequence around its transcription start site; test windows from held-out chromosomes 8 and 9).

## Using the model

```python
import torch, rcchunk as rk

variant, config, model, name = rk.load_pretrained("weights/mirrordna-tiny", "cuda")[0]
x = torch.from_numpy(rk.encode("ACGT" * 256)).long()[None].cuda()   # [batch, length], A,C,G,T,N = 0..4
out = model(x, decode=False)
out["cut"]                                 # chunk boundaries (gap j = between base j and j+1)
model.pooled_features(out)                 # strand-invariant sequence features
model.pooled_features(out, oriented=True)  # orientation-aware features, for strand-dependent labels
```

Fine-tuning with both heads, early stopping and evaluation on both strands: `rk.finetune(...)` (see `run_suite.py`).

## Fine-tuning the released weights

**On the paper's benchmarks** (no pretraining needed). `run_suite.py` takes any folder with a `pretrain_*.final.pt`,
including the folders in `weights/`:

```bash
# pooled head, NT-18 + Genomic Benchmarks + gene-strand tasks, 3 seeds
python run_suite.py weights/mirrordna-tiny --suite nt18 gb bio --seeds 0 1 2 --base-features
# centre readout and per-task head selection on validation (main results of the paper)
python finetune_center.py --main weights/mirrordna-tiny --full --seeds 0 1 2
python select_head.py
python merge_selected.py
```

Use `--tasks splice_sites_donors` (or any task name) and `--seeds 0` for a quick single run.

**On your own data.** Sequences as strings, integer labels 0..K-1:

```python
import numpy as np, rcchunk as rk

variant, config, model, name = rk.load_pretrained("weights/mirrordna-tiny", "cpu")[0]
enc = lambda seqs: [rk.encode(s.upper()) for s in seqs]
data = rk.split_val((enc(train_seqs), np.array(train_labels)),    # 10% of train becomes validation
                    (enc(test_seqs), np.array(test_labels)), seed=0)
result = rk.finetune(variant, config, data, "cuda", seed=0, lr=3e-4, epochs=10, bs=64, patience=3,
                     init_state=model.state_dict(), use_base=True,
                     oriented=False,          # True if the label depends on the strand (promoters, splice sites)
                     save_preds="my_preds.npz")
print(result["mcc"], result["mcc_rc_input"], result["flip_rate"])   # forward MCC, reverse-complement MCC, flips
```

`rk.finetune` returns the test metrics on both strands; `save_preds` stores the class probabilities for the test
sequences as given (`p`) and reverse-complemented (`p_rc`). Sequences of equal length are batched together.

## Reproducing the paper

All scripts resume where they stopped if interrupted. Results are written to `results_*` folders.

**1. Data** (hg38, RefSeq annotation, Enformer training intervals, Genomic Benchmarks; about 5 GB):

```bash
python download_all.py              # add --public to also download the baseline models
python bio_strand_tasks.py          # gene-strand tasks (1-kb windows); add --window 2048 / 4096 / 8192 / 16384 for the length test
```

The Nucleotide Transformer benchmark (revised, 18 tasks) is loaded from the Hugging Face Hub on first use.

**2. Pretraining and evaluation of MirrorDNA** (the 1-kb Caduceus recipe: 2^20 tokens per step, 10,000 steps =
10.5B tokens), then fine-tuning on all 18 NT tasks, 9 Genomic Benchmarks and 2 gene-strand tasks, 3 seeds:

```bash
python run_scale.py --size tiny --encoder conv --recipe caduceus --suites nt18 gb bio --base-features
```

Controls with the same budget: `--encoder duplex` (cross-strand gated encoder) and
`--encoder duplex --variant baseline+RCaug` (same architecture without the guarantee, trained with reverse-complement
augmentation). Label-efficiency runs: `run_suite.py <model folders> --nt-max-train 500 --gb-max-train 500 ...`.

**3. Baselines** (same tasks, splits and protocol):

```bash
python run_public.py --model nt100m   --suite nt18 gb bio --lr 3e-5 --full-data --epochs 10 --patience 3
python run_public.py --model dnabert2 --suite nt18 gb bio --lr 3e-5 --full-data --epochs 10 --patience 3
python run_public.py --model hyenadna --suite nt18 gb bio --full-data --epochs 10 --patience 3
python run_public.py --model caduceus_ps_1k --suite nt18 gb bio --full-data --epochs 10 --patience 3
python run_public.py --model caduceus_ph_1k --suite nt18 gb --rcaug --full-data --epochs 10 --patience 3
python run_public.py --model caduceus_ph_1k --suite bio --full-data --epochs 10 --patience 3
```

Protocol for every baseline (learning rates, heads, Caduceus-PH protocol, version pins) and the Caduceus environment:
see [`docs/BASELINES.md`](docs/BASELINES.md).

**4. Tables, statistics and figures** (all at once, into `paper_assets/`):

```bash
python make_paper_assets.py
```

or one by one:

```bash
python stats_tables.py        # mean ± SD over seeds, Wilcoxon tests over tasks -> stats/
python lowlabel_stats.py      # exact equivariance vs augmentation at 500 / 2,000 / all labels
python compare_long.py        # gene strand on longer inputs
python figs/fig_strand.py     # accuracy on both strands
python figs/fig_motifs.py       results_scale_tiny_RC-equivariant_caduceus10B results_scale_tiny_duplex_baseline_RCaug_caduceus10B --only-strand
python figs/fig_chunk_mirror.py results_scale_tiny_RC-equivariant_caduceus10B results_scale_tiny_duplex_baseline_RCaug_caduceus10B
python bio_boundaries.py results_scale_tiny_RC-equivariant_caduceus10B   # boundary profiles at gene-structure sites
python splice_tokens.py  results_scale_tiny_RC-equivariant_caduceus10B   # tokens formed at splice sites
python figs/fig_splice.py
python bio_sae.py        results_scale_tiny_RC-equivariant_caduceus10B   # sparse autoencoder vs JASPAR motifs
python run_deepstarr.py  results_scale_tiny_RC-equivariant_caduceus10B   # DeepSTARR enhancer-activity regression
python strand_gb.py      stats/all_runs.csv                              # strand metrics on the 7 symmetric GB tasks
python make_supplement.py                                                # supplementary tables
```

Frozen representations on the two strands (no fine-tuning) and token boundaries under window shifts:

```bash
python frozen_strand_probe.py results_scale_tiny_RC-equivariant_caduceus10B results_scale_tiny_duplex_baseline_RCaug_caduceus10B nt100m dnabert2 hyenadna
python frozen_strand_probe.py caduceus_ps_1k caduceus_ph_1k          # Caduceus environment (mamba-ssm)
python frozen_strand_probe.py --report-only
python shift_stability.py results_scale_tiny_RC-equivariant_caduceus10B results_scale_tiny_duplex_baseline_RCaug_caduceus10B
```

**5. Centre readout and head selection** (main results). MirrorDNA is fine-tuned a second time with a head that also
reads the per-base states at the centre of the window; for each task the head with the higher validation MCC is kept,
so the test sets play no part in the choice:

```bash
python finetune_center.py --full --seeds 0 1 2   # copies the pretrained weights; NT-18, GB-9, gene strand
python select_head.py                            # chooses the head per task on validation
python merge_selected.py                         # final tables: averages, tasks won, Wilcoxon tests, strand table
python suite_without_outliers.py results_scale_tiny_RC-equivariant_center/all_runs_selected.csv
```


## Repository layout

| file | content |
|---|---|
| `rcchunk.py` | model (encoder, symmetric router, chunk transformer, decoder, heads), pretraining, fine-tuning, data loading |
| `test_equivariance.py` | numerical check of the guarantee |
| `run_scale.py` | pretraining + full evaluation of one model |
| `run_suite.py` | fine-tuning and evaluation on NT, Genomic Benchmarks and gene-strand tasks |
| `run_public.py` | baselines: NT-v2, DNABERT-2, HyenaDNA, Caduceus-PS/PH |
| `bio_strand_tasks.py` | gene-strand tasks from RefSeq |
| `bio_boundaries.py`, `splice_tokens.py`, `bio_sae.py`, `shift_stability.py` | tokenizer and biological analyses |
| `frozen_strand_probe.py` | frozen embeddings of the two strands (retrieval, linear probe) |
| `finetune_center.py`, `select_head.py`, `merge_selected.py` | centre readout and per-task head selection on validation |
| `run_deepstarr.py` | DeepSTARR enhancer-activity regression |
| `weights/` | pretrained MirrorDNA, MirrorDNA-duplex and the augmented control |
| `stats_tables.py`, `lowlabel_stats.py`, `compare_long.py`, `collect_results.py`, `strand_gb.py`, `make_supplement.py`, `make_paper_assets.py` | tables, statistics and figures |
| `figs/` | figure scripts |
| `example/` | quick start and minimal test dataset |
| `docs/BASELINES.md` | how the baselines were fine-tuned |
| `download_all.py` | data download |

## Citation

```bibtex
will be updated soon
```

## License

MIT (see `LICENSE`).
