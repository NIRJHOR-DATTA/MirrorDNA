# Pretrained weights

Each folder holds the weights (`pretrain_<variant>.final.pt`) and the configuration (`pretrain_info.json`) of one
model. All were pretrained with the same recipe: hg38 (Basenji/Enformer training intervals), 1,024-bp windows,
2^20 bases per step, 10,000 steps (10.5B bases).

| folder | name in the paper | encoder | parameters | what it is |
|---|---|---|---|---|
| `mirrordna-tiny` | MirrorDNA | conv | 2.35M | the main model: shared-weight convolutional encoder, symmetric router |
| `mirrordna-duplex` | MirrorDNA-duplex | duplex | 2.50M | cross-strand gated (duplex) encoder, symmetric router |
| `chunker-rcaug` | Chunker + RC aug. (control) | duplex | 2.52M | the non-equivariant control: same architecture as MirrorDNA-duplex, not symmetrized, trained with reverse-complement augmentation |

Load any of them with:

```python
import rcchunk as rk
variant, config, model, name = rk.load_pretrained("weights/mirrordna-tiny", "cuda")[0]
```
