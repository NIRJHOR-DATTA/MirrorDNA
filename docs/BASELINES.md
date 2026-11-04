# Fine-tuning the baselines

All baselines are fine-tuned with `run_public.py` on exactly the same tasks, splits and evaluation as MirrorDNA.
Their pretrained weights are downloaded from the Hugging Face Hub (they are not redistributed here).

## Shared protocol

| setting | value |
|---|---|
| tasks | 18 NT tasks (revised benchmark, official train/test splits), 9 Genomic Benchmarks, 2 gene-strand tasks |
| training data | full training set of each task (`--full-data`); 10% of it held out for validation |
| optimizer | AdamW, weight decay 0.01, linear warm-up over the first 5% of steps, then cosine decay to 10% of the peak rate |
| epochs | at most 10, early stopping on validation MCC with patience 3; the best epoch is evaluated on the test set |
| batch size | 32 (long inputs are split into micro-batches with gradient accumulation, `--max-attn`) |
| head | mean and max pooling over the model's output positions, then LayerNorm and a linear layer |
| seeds | 0, 1, 2 |
| evaluation | MCC on the test set as given and with every test sequence reverse-complemented; consistent MCC; flip rate |

## Per model

| model (`--model`) | weights | learning rate | notes |
|---|---|---|---|
| `nt100m` | `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | `--lr 3e-5` | 6-mer tokens; at most 2,048 tokens per input |
| `dnabert2` | `zhihan1996/DNABERT-2-117M` | `--lr 3e-5` | needs `transformers==4.46.3` (newer versions silently load a plain BERT without ALiBi); the Triton flash-attention kernel is switched off for reproducibility |
| `hyenadna` | `LongSafari/hyenadna-small-32k-seqlen-hf` | 3e-4 (default) | full precision (FFT convolutions) |
| `hyenadna_1k` | `LongSafari/hyenadna-tiny-1k-seqlen-hf` | 3e-4 | fixed 1-kb filters: longer inputs are centre-cropped to 1,024 bp |
| `caduceus_ps_1k` | `yairschiff/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3` | 3e-4 | same pretraining data and token budget as MirrorDNA; published strand-invariant head (both strands averaged) |
| `caduceus_ph_1k` | `yairschiff/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3` | 3e-4 | published protocol: reverse-complement augmentation (`--rcaug`) on NT and Genomic Benchmarks; gene-strand tasks without augmentation |

The 1-kb Caduceus checkpoints ship without a tokenizer and with model code that does not import under current
`mamba-ssm`; `run_public.py` loads them with the tokenizer and model code of the 131k checkpoint (identical character
vocabulary and architecture) and stops with an error if any weight fails to load.

## Commands

```bash
python run_public.py --model nt100m   --suite nt18 gb bio --lr 3e-5 --full-data --epochs 10 --patience 3 --seeds 0 1 2
python run_public.py --model dnabert2 --suite nt18 gb bio --lr 3e-5 --full-data --epochs 10 --patience 3 --seeds 0 1 2
python run_public.py --model hyenadna --suite nt18 gb bio --full-data --epochs 10 --patience 3 --seeds 0 1 2
python run_public.py --model hyenadna_1k --suite nt18 gb bio --full-data --epochs 10 --patience 3 --seeds 0 1 2
python run_public.py --model caduceus_ps_1k --suite nt18 gb bio --full-data --epochs 10 --patience 3 --seeds 0 1 2
python run_public.py --model caduceus_ph_1k --suite nt18 gb --rcaug --full-data --epochs 10 --patience 3 --seeds 0 1 2
python run_public.py --model caduceus_ph_1k --suite bio --full-data --epochs 10 --patience 3 --seeds 0 1 2
```

Results go to `results_public/results.jsonl` (one line per task and seed); finished runs are skipped when a command
is repeated, so an interrupted run continues where it stopped.

## Environment for Caduceus

Caduceus needs the `mamba-ssm` and `causal-conv1d` CUDA kernels (Linux or WSL2). On recent GPUs they must be compiled
for the GPU, which needs the CUDA toolkit (`nvcc`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.46.3" einops ninja packaging
TORCH_CUDA_ARCH_LIST="<your GPU, e.g. 8.9 or 12.0>" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    pip install --no-build-isolation causal-conv1d mamba-ssm
```
