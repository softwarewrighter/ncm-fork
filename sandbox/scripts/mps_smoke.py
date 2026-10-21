"""Build NCM tokenizer VQ-VAE + generator at random init from an eval config and
run the full generate -> decode path on a given device. No checkpoints needed.
Usage: python mps_smoke.py <config.yaml> <device> [batch]
"""
import os, sys, time
ROOT = os.environ.get("NCM_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.abspath(ROOT))
import torch
import mps_compat  # noqa: F401  (MPS area-interpolate shim; no-op elsewhere)
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

cfg_path, dev = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CFG = OmegaConf.load(cfg_path)
torch.manual_seed(0)

t0 = time.time()
vq = VQVAE(**CFG.tokenizer.vqvae).to(dev).eval()
net = NextConceptTransformer(vqvae_local=vq, config=CFG.model).to(dev).eval()
print(f"[{CFG.task}] built on {dev} in {time.time()-t0:.1f}s; "
      f"gen params {sum(p.numel() for p in net.parameters())/1e6:.1f}M, "
      f"vqvae params {sum(p.numel() for p in vq.parameters())/1e6:.1f}M")

P = int(CFG.model.prefix_len)
in_vocab = int(CFG.tokenizer.vqvae.in_vocab_size)
x = torch.randint(0, in_vocab, (B, P), device=dev)
levels = vq.vq.levels
if CFG.model.get("query_level0", False) or CFG.model.get("predict_level0", False):
    l0 = None
else:
    l0 = torch.randint(0, int(CFG.model.vocab_size[0]), (B, levels[0]), device=dev)
G = CFG.generation
kw = dict(temperature=float(G.get("temperature", 1.0)), top_k=int(G.get("top_k", 1)),
          top_p=float(G.get("top_p", 0.0)), cfg_scale=0.0, cfg_null_eos=0)

def run():
    with torch.no_grad():
        g = net.generate(l0, x, **kw)
        _, zl = vq.vq.idx_to_zhats(g, return_zlevels=True)
        logits = vq.decoder(zl[-1])
        ids = logits.argmax(-1)
    if dev == "mps":
        torch.mps.synchronize()
    return g, logits, ids

g, logits, ids = run()  # warm-up (MPS compiles kernels on first call)
t0 = time.time(); n = 3
for _ in range(n):
    run()
dt = (time.time() - t0) / n
print(f"[{CFG.task}] {dev}: codes {tuple(g.shape)} -> logits {tuple(logits.shape)} -> ids {tuple(ids.shape)}; "
      f"{dt*1000:.0f} ms/batch of {B} (after warm-up); finite={bool(torch.isfinite(logits).all())}")

# The eval sampler (scripts/eval_shard.py pick_tokens) uses float64, which MPS lacks.
try:
    logits.double()
    print(f"[{CFG.task}] {dev}: float64 OK")
except Exception as e:
    print(f"[{CFG.task}] {dev}: float64 NOT supported -> {type(e).__name__}: {str(e)[:90]}")
    print(f"[{CFG.task}] {dev}: float64 on CPU copy OK: {logits.cpu().double().dtype}")

# Shim check: area pooling on this device vs CPU reference for every level transition.
import torch.nn.functional as F
worst = 0.0
for a, b in zip(levels[:-1], levels[1:]):
    z = torch.randn(2, 16, a)
    ref = F.interpolate(z, size=(b,), mode="area")
    got = F.interpolate(z.to(dev), size=(b,), mode="area").cpu()
    worst = max(worst, (ref - got).abs().max().item())
print(f"[{CFG.task}] {dev}: area-interpolate max abs diff vs CPU over level transitions = {worst:.2e}")
out = os.environ.get("SMOKE_OUT")
if out:
    torch.save({"codes": g.cpu(), "ids": ids.cpu()}, out)
