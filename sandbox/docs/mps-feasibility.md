# NCM on the M1 Max (Metal / MPS): feasibility findings

Date: 2026-10-20. Machine: Apple M1 Max, 64 GB, macOS 26.5. Venv: `sandbox/.venv`
(Python 3.12.13, torch 2.14.1 with MPS, transformers 5.20.2), created with
`uv venv` / `uv pip`. No checkpoints or datasets have been downloaded.

## Verdict

NCM can run on this Mac's GPU. The model code is CUDA-free; only the eval
harness is CUDA-specific. Two small, sandbox-local workarounds are needed.

## What was checked (static)

- `models/` uses `torch.nn.functional.scaled_dot_product_attention` only. The
  flex_attention path exists but `use_flex_attention: false` in every config
  under `config/eval/`, and sudoku configs omit it (default false).
- No custom CUDA kernels, no triton, no torch.compile, no distributed init.
- All model tensors are created relative to input device (`device=x.device`).
- The CUDA assumptions live in `evaluate.py` (exits if
  `torch.cuda.device_count() == 0`), `scripts/eval_shard.py` (hard-coded
  `.cuda()`, `map_location="cuda"`, float64 sampler) and `scripts/eval_merge.py`
  (`AutoModelForCausalLM.from_pretrained("gpt2").cuda()`).
- No released model uses a pretrained T5 context encoder; the context encoder
  is the repo's own 4-layer transformer.

## What was checked (empirical, random init, no weights)

`sandbox/scripts/mps_smoke.py` builds the VQ-VAE and generator from a config,
runs `generate()` then `idx_to_zhats()` then the VQ-VAE decoder, on a chosen
device. Batch 8, after warm-up:

| config      | gen + vqvae params | MPS ms/batch | CPU ms/batch |
|-------------|--------------------|-------------:|-------------:|
| sudoku-easy | 25.5M + 14.9M      |          237 |          741 |
| tinystories | 88.6M + 36.4M      |          334 |         1507 |
| xsum        | 88.0M + 60.6M      |         1879 |         6062 |
| lm1b-small  | 97.1M + 111.4M     |          404 |         2095 |

All four produce finite logits on MPS.

## Blockers found and fixes

1. `F.interpolate(mode="area")` on non-divisible sizes (`models/quant.py`,
   lines 399, 613, 655, 686, 703) raises on MPS: "Adaptive pool MPS: input
   sizes must be divisible by output sizes" (pytorch/pytorch#96056). The
   multiscale residual VQ resamples 1->2->4->6->9->13->... so nearly every
   level hits it. Fix: `sandbox/scripts/mps_compat.py` replaces that case with a
   matmul against a cached averaging matrix. Max abs diff vs CPU over all level
   transitions of all four configs: 0.0.
2. MPS has no float64. `pick_tokens()` in `scripts/eval_shard.py` calls
   `.double()` on the decoder logits (used by xsum and lm1b, which set
   frequency/adjacency penalties). Fix: run that sampler on a CPU copy of the
   logits (small tensor, B x T x V). Also the hard-coded `.cuda()` calls need a
   device argument.

Cross-device agreement on random-init weights is a weak signal (near-tie logits
cascade across levels): xsum greedy agreed 99.8% on codes; sudoku (temperature 0)
75.6%; sampled configs are not comparable across RNG streams. Real weights are
the proper check.

## Download sizes (from the HF API, nothing fetched yet)

Checkpoints repo total 8.0 GB; datasets repo total 22.7 GB (owt train alone
18 GB). Do not use the README's blanket `hf download`; use `--include`.

| task        | tokenizer.pt | generator.pt | eval data |
|-------------|-------------:|-------------:|----------:|
| sudoku/easy |      60 MB   |     103 MB   |    0.8 MB |
| tinystories |     222 MB   |     356 MB   | none needed (unconditional) |
| xsum        |     300 MB   |     354 MB   |     20 MB |
| wmt14       |     646 MB   |     481 MB   |    0.8 MB |
| lm1b small  |     598 MB   |     390 MB   | none needed |
| lm1b large  |     598 MB   |   2,901 MB   | none needed |
| owt         | not released (repo holds only a TODO.md) | | |

## Recommended next steps

1. Pull one small task first:
   `hf download hazyresearch/ncm-checkpoints --include "sudoku/easy/*" --local-dir sandbox/artifacts`
   and `hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/easy/*" --local-dir sandbox/data`
   (about 165 MB total).
2. Write `sandbox/scripts/run_ncm.py`: a device-generic port of
   `scripts/eval_shard.py` plus scoring (import `mps_compat`, `--device
   mps|cpu|cuda`, `map_location`, CPU-side `pick_tokens`, gpt2 perplexity model
   on the chosen device). Leave upstream files untouched.
3. Validate on real weights: sudoku-easy solved rate on MPS vs CPU vs the
   paper, then tinystories samples (578 MB) to see coarse-to-fine text.
4. Move to the Arch / RTX 5060 box only for: wall-clock comparisons against the
   paper's CUDA numbers, lm1b-large, or if a flex-attention / torch.compile
   path becomes interesting. The RTX 5060 is Blackwell (sm_120), so install a
   CUDA 12.8+ torch wheel there.
5. If transformers 5.x breaks `AutoTokenizer` loading (gpt2, bert-base-uncased,
   Qwen/Qwen3-0.6B), pin `transformers<5` in the venv; the repo only requires
   `>=4.51`.

## Update 2026-10-20 (later): real weights, sudoku-easy

Downloaded `sudoku/easy/*` (103 MB generator, 60 MB tokenizer, 0.7 MB data)
into `sandbox/artifacts` and `sandbox/data`. `sandbox/scripts/run_ncm.py` is the
device-generic runner (generation + all five upstream metrics, `--device
auto|cpu|mps|cuda`, `--score-only` to rescore a pickle). Outputs go to
`sandbox/eval/<task>/<device>.pkl`.

Full 2000-puzzle test set, greedy (config defaults), batch 128:

| device | solved | cell % | valid % | wall time | per puzzle |
|--------|-------:|-------:|--------:|----------:|-----------:|
| mps    | 1988/2000 (99.40%) | 99.97 | 99.40 | 56.4 s | 28 ms |
| cpu    | 1988/2000 (99.40%) | 99.97 | 99.40 | 182.6 s | 91 ms |

All 2000 decoded boards are byte-identical between MPS and CPU. The Metal path
with the shim is correct on real weights; no need to move to the RTX box for
this task.
