"""Inference-only PG fusion; gradients and unsupported layouts use native PyTorch."""

from __future__ import annotations

import ctypes
import functools

import torch

from .network import MultiHeadAttention, MultiHeadCrossAttention

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _rms_epilogue(X, M, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        valid = column < WIDTH
        mean = tl.load(M + row).to(tl.float32)
        reciprocal = libdevice.rsqrt(mean + tl.full((), EPS, tl.float32))
        x = tl.load(X + row * WIDTH + column, valid, other=0).to(tl.float32)
        weight = tl.load(W + column, valid, other=0).to(tl.float32)
        output = (x * reciprocal) * weight
        tl.store(Y + row * WIDTH + column, output.to(Y.dtype.element_ty), valid)

    @triton.jit(do_not_specialize=["XS0", "XS1", "XS2", "YS0", "YS1", "YS2", "H", "L", "N"])
    def _rope_pairs(
        X, FREQ, Y,
        XS0, XS1, XS2, XS3: tl.constexpr,
        YS0, YS1, YS2, YS3: tl.constexpr,
        FS2: tl.constexpr, FS3: tl.constexpr, FS4: tl.constexpr, FS5: tl.constexpr,
        H, L, P: tl.constexpr, N, ROUND_PRODUCTS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < N
        pair = offset % P
        sequence = (offset // P) % L
        head = (offset // (P * L)) % H
        batch = offset // (P * L * H)
        index = (batch.to(tl.int64) * XS0 + head.to(tl.int64) * XS1
                 + sequence.to(tl.int64) * XS2 + (2 * pair).to(tl.int64) * XS3)
        freq_index = sequence.to(tl.int64) * FS2 + pair.to(tl.int64) * FS3
        x0 = tl.load(X + index, valid, other=0).to(tl.float32)
        x1 = tl.load(X + index + XS3, valid, other=0).to(tl.float32)
        c00 = tl.load(FREQ + freq_index, valid, other=0).to(tl.float32)
        c01 = tl.load(FREQ + freq_index + FS5, valid, other=0).to(tl.float32)
        c10 = tl.load(FREQ + freq_index + FS4, valid, other=0).to(tl.float32)
        c11 = tl.load(FREQ + freq_index + FS4 + FS5, valid, other=0).to(tl.float32)
        p00, p01, p10, p11 = c00 * x0, c01 * x1, c10 * x0, c11 * x1
        if ROUND_PRODUCTS:
            # Native BF16 multiply rounds before the two products are added.
            p00 = p00.to(tl.bfloat16).to(tl.float32)
            p01 = p01.to(tl.bfloat16).to(tl.float32)
            p10 = p10.to(tl.bfloat16).to(tl.float32)
            p11 = p11.to(tl.bfloat16).to(tl.float32)
        output_index = (batch.to(tl.int64) * YS0 + head.to(tl.int64) * YS1
                        + sequence.to(tl.int64) * YS2 + (2 * pair).to(tl.int64) * YS3)
        tl.store(Y + output_index, (p00 + p01).to(Y.dtype.element_ty), valid)
        tl.store(Y + output_index + YS3, (p10 + p11).to(Y.dtype.element_ty), valid)


def _rms_forward(original, module, x):
    if (triton is None or torch.is_grad_enabled() or not x.is_cuda or not x.is_contiguous()
            or x.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or module.weight.dtype != torch.float32 or x.shape[-1] > 4096):
        return original(x)
    upcast = x.float()
    # Preserve native reduction and rounding, fuse only the pointwise epilogue.
    mean = upcast.pow(2).mean(-1, keepdim=True)
    output = torch.empty_like(x)
    width = x.shape[-1]
    eps = ctypes.c_float(torch.finfo(torch.float32).eps if module.eps is None else module.eps).value
    with torch.cuda.device(x.device):
        _rms_epilogue[(x.numel() // width,)](
            upcast, mean, module.weight, output, width, eps, triton.next_power_of_2(width),
            num_warps=4 if width <= 1024 else 8, enable_fp_fusion=False,
        )
    return output


def _rope_eligible(x, freq):
    if (not x.is_cuda or x.ndim != 4 or x.dtype not in (torch.float32, torch.bfloat16)
            or freq.device != x.device or freq.dtype not in (torch.float32, torch.bfloat16)):
        return False
    batch, heads, length, width = x.shape
    if width % 2 or tuple(freq.shape) != (1, 1, length, width // 2, 2, 2):
        return False
    contiguous = (heads * length * width, length * width, width, 1)
    hl_transpose = (length * heads * width, width, heads * width, 1)
    pairs = width // 2
    return (tuple(x.stride()) in (contiguous, hl_transpose)
            and tuple(freq.stride()[2:]) in ((pairs * 4, 4, 2, 1), (pairs * 4, 4, 1, 2)))


def _rope(x, freq):
    batch, heads, length, width = x.shape
    contiguous = (heads * length * width, length * width, width, 1)
    if tuple(x.stride()) == contiguous or heads == 1 or length == 1:
        strides = contiguous
    else:
        # Match native output layout, including its unused B=1 stride.
        strides = (heads * width if batch == 1 else heads * length * width, width, heads * width, 1)
    output = torch.empty_strided(x.shape, strides, dtype=x.dtype, device=x.device)
    count = x.numel() // 2
    with torch.cuda.device(x.device):
        _rope_pairs[(triton.cdiv(count, 256),)](
            x, freq, output, *x.stride(), *strides, *freq.stride()[2:],
            heads, length, width // 2, count, x.dtype == torch.bfloat16 and freq.dtype == torch.bfloat16, 256,
            num_warps=4, enable_fp_fusion=False,
        )
    return output


def _rotary_forward(original, cross, xq, xk, q_freqs, k_freqs=None):
    k_freqs = q_freqs if k_freqs is None else k_freqs
    if (triton is None or torch.is_grad_enabled()
            or not _rope_eligible(xq, q_freqs) or not _rope_eligible(xk, k_freqs)):
        return original(xq, xk, q_freqs, k_freqs) if cross else original(xq, xk, q_freqs)
    return _rope(xq, q_freqs), _rope(xk, k_freqs)


class GeneratorFusion:
    """Bind the shared PG kernels to one generator and restore them on disable."""

    def __init__(self, generator):
        self.bindings = []
        for module in generator.modules():
            if type(module) is torch.nn.RMSNorm:
                self._bind(module, "forward", functools.partial(_rms_forward, module.forward, module))
            elif isinstance(module, (MultiHeadAttention, MultiHeadCrossAttention)):
                cross = isinstance(module, MultiHeadCrossAttention)
                self._bind(module, "_rotary_fn", functools.partial(_rotary_forward, module._rotary_fn, cross))

    def _bind(self, module, name, replacement):
        self.bindings.append((module, name, name in module.__dict__, module.__dict__.get(name)))
        setattr(module, name, replacement)

    def restore(self):
        for module, name, present, original in self.bindings:
            if present:
                setattr(module, name, original)
            else:
                delattr(module, name)
