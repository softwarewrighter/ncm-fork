"""Metal (MPS) compatibility shim for NCM.

PyTorch's MPS backend lacks adaptive_avg_pool1d for non-divisible sizes
(pytorch/pytorch#96056), which F.interpolate(mode="area") lowers to. NCM's
multiscale residual VQ resamples between levels like 1->2->4->6->9->13, so
nearly every level transition hits this. Adaptive average pooling is a linear
map, so on MPS we apply a cached (out x in) averaging matrix instead. Numerics
match CPU to float32 rounding. All other interpolate calls pass through.

Usage: `import mps_compat` before building the model (safe on CPU/CUDA: no-op).
"""
import math
import torch
import torch.nn.functional as F

_orig_interpolate = F.interpolate
_pool_mats = {}


def _area_matrix(n_in, n_out, device, dtype):
    key = (n_in, n_out, str(device), dtype)
    m = _pool_mats.get(key)
    if m is None:
        m = torch.zeros(n_out, n_in)
        for j in range(n_out):  # same bin rule as adaptive_avg_pool1d
            lo = (j * n_in) // n_out
            hi = math.ceil((j + 1) * n_in / n_out)
            m[j, lo:hi] = 1.0 / (hi - lo)
        m = _pool_mats[key] = m.to(device=device, dtype=dtype)
    return m


def _interpolate(input, size=None, scale_factor=None, mode="nearest", **kw):
    if mode == "area" and input.device.type == "mps" and input.dim() == 3 and size is not None:
        n_out = int(size[0] if isinstance(size, (tuple, list)) else size)
        n_in = input.shape[-1]
        if n_in % n_out != 0:  # the divisible case already works natively
            m = _area_matrix(n_in, n_out, input.device, input.dtype)
            return input @ m.T  # (B, C, L_in) @ (L_in, L_out)
    return _orig_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, **kw)


F.interpolate = _interpolate
PATCHED = True
