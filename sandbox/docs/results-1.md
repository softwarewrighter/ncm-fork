# Results 1: NCM on Apple M1 Max (Metal) — sudoku-easy with real weights

Date: 2026-10-20. Branch: `experiment`. All work lives under `sandbox/`;
upstream files are untouched.

## Setup

- Hardware: Apple M1 Max, 32-core GPU, 64 GB unified memory, macOS 26.5.
- Env: `sandbox/.venv` via `uv venv --python 3.12`; packages via `uv pip`
  (torch 2.14.1 with MPS, transformers 5.20.2, omegaconf, hydra-core,
  huggingface_hub 2.2.0, rouge_score, sacrebleu).
- Weights: `hf download hazyresearch/ncm-checkpoints --include "sudoku/easy/*"`
  into `sandbox/artifacts` (generator.pt 103 MB, tokenizer.pt 60 MB) and the
  matching test data into `sandbox/data` (puzzles + solutions, 0.7 MB).
  The shipped `config.yaml` model section is identical to
  `config/eval/sudoku-easy.yaml`.
- Test data format: 2000 puzzles as 91-token lists (BOS + 81 cells + 8 row
  separators, blanks = 0) and 2000 solutions as 89-token lists.

## Feasibility (no weights) — see `mps-feasibility.md` for detail

- Model code is CUDA-free: SDPA attention only, flex attention disabled in
  every eval config, no custom kernels or torch.compile.
- One Metal blocker in the model: `F.interpolate(mode="area")` with
  non-divisible sizes (multiscale residual VQ level resampling). Fixed by
  `sandbox/scripts/mps_compat.py`, which replaces it on MPS with a cached
  averaging-matrix matmul. Max abs diff vs CPU: 0.0 on all level transitions.
- Harness blockers: upstream `evaluate.py` / `scripts/eval_shard.py` /
  `scripts/eval_merge.py` hard-code CUDA and use float64 (absent on MPS).
  Replaced by `sandbox/scripts/run_ncm.py`.

## Runner: `sandbox/scripts/run_ncm.py`

Single-process, device-generic port of the upstream shard + merge scripts.
`--device auto|cpu|mps|cuda`, same generation overrides as upstream
(`--temp --top_k --top_p --cfg --freq --mu --adj --l0`), `--n`, `--bs`,
`--seeds`, `--print k`, `--no-score`, `--score-only <pkl>`. Implements all
five upstream metrics (exact, bleu, rouge, genppl, genppl_flm) with reference
LMs placed on the chosen device. Writes `sandbox/eval/<task>/<device>.pkl` in
the upstream shard format (plus a `meta` dict with device and timing).

```
cd sandbox
.venv/bin/python scripts/run_ncm.py --config ../config/eval/sudoku-easy.yaml \
  --checkpoint-dir artifacts/sudoku/easy --data-dir data/sudoku/easy --device mps
```

## Results: sudoku-easy, full test set (2000 puzzles)

Greedy decoding per config (temperature 0, top_k 0, top_p 1.0), batch 128.

| device | solved | cell % | valid % | wall time | per puzzle |
|--------|-------:|-------:|--------:|----------:|-----------:|
| mps    | 1988/2000 (99.40%) | 99.97 | 99.40 | 56.4 s | 28 ms |
| cpu    | 1988/2000 (99.40%) | 99.97 | 99.40 | 182.6 s | 91 ms |

All 2000 decoded boards are byte-identical between MPS and CPU. On the first
128 puzzles both devices gave 126/128 (98.44%).

Random-init timing from the smoke test (batch 8, after warm-up), for scale:

| config      | MPS ms/batch | CPU ms/batch |
|-------------|-------------:|-------------:|
| sudoku-easy |          237 |          741 |
| tinystories |          334 |         1507 |
| xsum        |         1879 |         6062 |
| lm1b-small  |          404 |         2095 |

## Conclusions

- NCM inference works on the M1 Max GPU with the shim; outputs are exact.
- Metal is ~3.2x faster than CPU on this task. Absolute speed claims in the
  paper are CUDA numbers and should be reproduced on the RTX 5060 box, not here.
- The RTX box is not needed for correctness or for the small/medium checkpoints.

## Not yet done

- tinystories (578 MB, unconditional sampling, genPPL needs GPT-2 large).
- xsum / wmt14 (conditional text; exercises the float64 frequency-penalty
  sampler on the CPU path and HF tokenizers under transformers 5.x).
- lm1b small/medium/large; owt has no released weights.
