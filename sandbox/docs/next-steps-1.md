# Next steps 1: where things stand after sudoku-easy and tinystories

Date: 2026-10-20. Branch: `experiment` (fork default). Last commit: 846eee4.

## Current state

Done and verified on the M1 Max (Metal / MPS):

- Env: `sandbox/.venv` (uv, Python 3.12, torch 2.14.1 MPS, transformers 5.20.2).
- Metal shim `sandbox/scripts/mps_compat.py` makes the multiscale VQ's
  area-interpolate work on MPS; exact vs CPU.
- Runner `sandbox/scripts/run_ncm.py`: device-generic generation + all five
  upstream metrics. Writes `sandbox/eval/<task>/<device>.pkl`.
- sudoku-easy, full 2000-puzzle test set: 1988/2000 solved on MPS and CPU,
  byte-identical outputs, 56 s vs 183 s. See `results-1.md`.
- tinystories: weights downloaded (578 MB), 16 unconditional samples on MPS,
  seed 0, 1.1 s total (67 ms/sample). Coherent stories with typical
  parallel-decoding glitches. Output in `sandbox/eval/tinystories/mps.pkl`
  (gitignored). Not scored.

Downloaded under `sandbox/artifacts` (gitignored): `sudoku/easy/` (163 MB),
`tinystories/` (578 MB). Under `sandbox/data`: `sudoku/easy/` (0.7 MB).

Not committed: nothing pending under sandbox except this file; eval pickles,
weights and the venv are intentionally ignored.

## Deliberately on hold

- Scoring tinystories (genPPL). It downloads GPT-2 large (~3 GB) as the
  reference LM and the config calls for 5 seeds x 256 samples. Hold until
  there is a reason to compare against the paper's number.

## Open items, in suggested order

1. xsum (354 + 300 MB weights, 20 MB data). Exercises two paths not yet
   checked on real weights: the float64 frequency/adjacency-penalty sampler
   (runs on a CPU copy in the runner) and the HF gpt2 pre-tokenizer under
   transformers 5.x. ROUGE scoring is cheap; its genPPL uses base gpt2.
   `hf download hazyresearch/ncm-checkpoints --include "xsum/*" --local-dir sandbox/artifacts`
   `hf download hazyresearch/ncm-datasets --repo-type dataset --include "xsum/*" --local-dir sandbox/data`
2. wmt14 (481 + 646 MB, 0.8 MB data). Uses the Qwen/Qwen3-0.6B tokenizer and
   sacreBLEU; also the only conditional task where `--cfg 1.0` is documented.
3. sudoku medium / hard (163 MB each). Same runner, same data layout.
4. lm1b small (390 MB generator + 598 MB shared tokenizer). Uses the
   bert-base-uncased tokenizer and the FLM genPPL protocol (GPT-2 large).
5. Cross-device check for a sampled task: fix the seed and compare MPS vs CPU
   outputs. They will differ because RNG streams differ per device; compare
   score distributions rather than texts.
6. RTX 5060 box: only for reproducing the paper's wall-clock numbers or
   lm1b-large (2.9 GB). Needs a CUDA 12.8+ (Blackwell / sm_120) torch wheel.
   The runner should work unchanged with `--device cuda`.

## Known caveats

- transformers 5.x is installed; the repo asks for >=4.51. Tokenizer loading
  for gpt2 / bert / Qwen3 is untested so far. Pin `transformers<5` if it breaks.
- owt has no released weights (the HF folder holds only a TODO.md).
- The README's blanket `hf download` commands fetch 8 GB + 22.7 GB; always use
  `--include`.
- Upstream `evaluate.py` cannot be used on the Mac (exits without CUDA).
