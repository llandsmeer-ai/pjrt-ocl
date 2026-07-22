"""Complex64 support + stablehlo.fft — the "complex" op family (§43).

The arena is 4-byte-slotted f32; complex64 is an 8-byte (real, imag) pair. Rather
than teach the arena an 8-byte complex dtype (a format break touching the buffer
table, executor, and every tile op), we represent a complex-typed SSA value as a
PAIR of ordinary f32 buffers ``(re_id, im_id)`` entirely in the lowering. The
pair lives in ``ctx.cbuf`` (mirroring ``ctx.value_to_buf`` for scalars); it is
propagated across func.call/return by ``lowering._alias_value``. NOTHING below
emits a new opcode or tile op — complex ops decompose into the existing f32
ADD/SUB/MUL/SQRT/DOT the VM already runs, so both device engines and both
numpy validators cover them for free.

Coverage landed:
  - stablehlo.convert  real f32 -> complex (real=input, imag=zeros); complex->complex alias
  - stablehlo.complex(re, im)     -> pair (re, im)
  - stablehlo.real / stablehlo.imag(c) -> the corresponding half (f32)
  - stablehlo.add / subtract / multiply / negate on complex (split-complex algebra)
  - stablehlo.abs(complex) -> sqrt(re^2 + im^2)  (f32)
  - stablehlo.fft, type = FFT, 1-D          -> DFT-as-matmul with constant twiddles

FFT scope (honest): only a 1-D **forward** FFT (``type = FFT``, ``len(fft_length)
== 1``) over the last axis, batched across any leading dims. It is a **direct
DFT** X = x @ W with a constant N×N twiddle matrix (W symmetric), i.e. O(N^2),
not a Cooley-Tukey O(N log N) — correct for every N, and fine at the bench size
(N=512: two 1 MB const matrices). IFFT / RFFT / IRFFT and multi-axis FFT raise
LoweringError (clear, not silently wrong). For a large-N production FFT a
radix-2 butterfly network (still linear bytecode) would replace the DFT matmul;
recorded as future work in docs/decisions.md §43.
"""
from __future__ import annotations

import math

import numpy as np

from .. import lowering as L


# --- helpers ----------------------------------------------------------------

def _complex_elem(mlir_type):
    """If `mlir_type` is a ranked tensor of complex<...>, return (shape, n_elems,
    inner_float_dtype_enum); else None."""
    from jaxlib.mlir import ir
    if not isinstance(mlir_type, ir.RankedTensorType):
        return None
    et = mlir_type.element_type
    if not isinstance(et, ir.ComplexType):
        return None
    inner = et.element_type
    if isinstance(inner, ir.F32Type):
        dt = L.DT_F32
    elif isinstance(inner, ir.F64Type):
        dt = L.DT_F64
    else:
        raise L.LoweringError(f"complex over unsupported float {inner}")
    if dt != L.DT_F32:
        raise L.LoweringError(
            "only complex<f32> (complex64) is supported (arena is f32-slotted)")
    shape = tuple(mlir_type.shape)
    return shape, (math.prod(shape) if mlir_type.rank else 1), dt


def _is_complex(mlir_type) -> bool:
    try:
        return _complex_elem(mlir_type) is not None
    except L.LoweringError:
        return True


def _const_f32(ctx, arr) -> int:
    """Add an f32 constant buffer holding `arr` (any shape, row-major); return
    its buffer id."""
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    buf = ctx.new_buffer(a.size, L.DT_F32)
    ctx.consts.append((buf, a.tobytes()))
    return buf


def _add(ctx, a, b, n):
    dst = ctx.new_buffer(n, L.DT_F32)
    ctx.emit(L.Instr(L.OP_ADD_F32, dst=dst, a=a, b=b, n=n))
    return dst


def _sub(ctx, a, b, n):
    dst = ctx.new_buffer(n, L.DT_F32)
    ctx.emit(L.Instr(L.OP_SUB_F32, dst=dst, a=a, b=b, n=n))
    return dst


def _mul(ctx, a, b, n):
    dst = ctx.new_buffer(n, L.DT_F32)
    ctx.emit(L.Instr(L.OP_MUL_F32, dst=dst, a=a, b=b, n=n))
    return dst


def _sqrt(ctx, a, n):
    dst = ctx.new_buffer(n, L.DT_F32)
    ctx.emit(L.Instr(L.OP_SQRT_F32, dst=dst, a=a, b=0, n=n))
    return dst


def _dot(ctx, a, b, M, N, K):
    """Emit OP_DOT: dst[M,N] = a[M,K] @ b[K,N] (row-major, G=1). Mirrors
    ops/dot.py's field packing (M in n, N/K packed in imm)."""
    if N > 0xFFFF or K > 0xFFFF:
        raise L.LoweringError(f"fft DFT matmul: N={N}, K={K} exceed 16-bit pack")
    dst = ctx.new_buffer(M * N, L.DT_F32)
    ctx.emit(L.Instr(L.OP_DOT, dst=dst, a=a, b=b, n=M,
                     imm=(N << 16) | K, imm2=1))
    return dst


# --- stablehlo.convert (real <-> complex) -----------------------------------
# convert already has a real (f32->f32) handler in ops/making.py; wrap it so a
# complex operand/result takes the split-complex path and everything else
# delegates to the original.

_orig_convert = L.OP_HANDLERS["stablehlo.convert"]


@L.handles("stablehlo.convert")
def _convert(ctx, op):
    in_c = _is_complex(op.operands[0].type)
    out_c = _is_complex(op.results[0].type)
    if not in_c and not out_c:
        return _orig_convert(ctx, op)
    res = op.results[0]
    if in_c and out_c:                       # complex -> complex: alias the pair
        ctx.cbuf[res] = ctx.cbuf[op.operands[0]]
        return
    if out_c and not in_c:                   # real f32 -> complex: imag = zeros
        _, n, _ = L.tensor_info(op.operands[0].type)
        re = ctx.buf_for(op.operands[0])
        im = _const_f32(ctx, np.zeros(n, np.float32))
        ctx.cbuf[res] = (re, im)
        return
    # complex -> real convert: keep the real part (stablehlo emits real() for
    # this, but be permissive).
    ctx.value_to_buf[res] = ctx.cbuf[op.operands[0]][0]


# --- stablehlo.complex / real / imag ----------------------------------------

@L.handles("stablehlo.complex")
def _complex(ctx, op):
    ctx.cbuf[op.results[0]] = (ctx.buf_for(op.operands[0]),
                               ctx.buf_for(op.operands[1]))


@L.handles("stablehlo.real")
def _real(ctx, op):
    src = op.operands[0]
    if src in ctx.cbuf:
        ctx.value_to_buf[op.results[0]] = ctx.cbuf[src][0]
    else:                                    # real(real x) == x
        ctx.value_to_buf[op.results[0]] = ctx.buf_for(src)


@L.handles("stablehlo.imag")
def _imag(ctx, op):
    src = op.operands[0]
    if src in ctx.cbuf:
        ctx.value_to_buf[op.results[0]] = ctx.cbuf[src][1]
    else:                                    # imag(real x) == 0
        _, n, _ = L.tensor_info(src.type)
        ctx.value_to_buf[op.results[0]] = _const_f32(ctx, np.zeros(n, np.float32))


# --- stablehlo.abs (complex modulus) ----------------------------------------

_orig_abs = L.OP_HANDLERS["stablehlo.abs"]


@L.handles("stablehlo.abs")
def _abs(ctx, op):
    if not _is_complex(op.operands[0].type):
        return _orig_abs(ctx, op)
    re, im = ctx.cbuf[op.operands[0]]
    _, n, _ = _complex_elem(op.operands[0].type)
    re2 = _mul(ctx, re, re, n)
    im2 = _mul(ctx, im, im, n)
    s = _add(ctx, re2, im2, n)
    ctx.value_to_buf[op.results[0]] = _sqrt(ctx, s, n)


# --- complex elementwise: add / subtract / multiply / negate ----------------

def _wrap_binop(name, real_fn):
    orig = L.OP_HANDLERS[name]

    @L.handles(name)
    def handler(ctx, op):
        if not _is_complex(op.results[0].type):
            return orig(ctx, op)
        real_fn(ctx, op)


def _c_add(ctx, op):
    ar, ai = ctx.cbuf[op.operands[0]]
    br, bi = ctx.cbuf[op.operands[1]]
    _, n, _ = _complex_elem(op.results[0].type)
    ctx.cbuf[op.results[0]] = (_add(ctx, ar, br, n), _add(ctx, ai, bi, n))


def _c_sub(ctx, op):
    ar, ai = ctx.cbuf[op.operands[0]]
    br, bi = ctx.cbuf[op.operands[1]]
    _, n, _ = _complex_elem(op.results[0].type)
    ctx.cbuf[op.results[0]] = (_sub(ctx, ar, br, n), _sub(ctx, ai, bi, n))


def _c_mul(ctx, op):
    # (ar + i ai)(br + i bi) = (ar br - ai bi) + i(ar bi + ai br)
    ar, ai = ctx.cbuf[op.operands[0]]
    br, bi = ctx.cbuf[op.operands[1]]
    _, n, _ = _complex_elem(op.results[0].type)
    rr = _sub(ctx, _mul(ctx, ar, br, n), _mul(ctx, ai, bi, n), n)
    ii = _add(ctx, _mul(ctx, ar, bi, n), _mul(ctx, ai, br, n), n)
    ctx.cbuf[op.results[0]] = (rr, ii)


_wrap_binop("stablehlo.add", _c_add)
_wrap_binop("stablehlo.subtract", _c_sub)
_wrap_binop("stablehlo.multiply", _c_mul)


_orig_negate = L.OP_HANDLERS["stablehlo.negate"]


@L.handles("stablehlo.negate")
def _negate(ctx, op):
    if not _is_complex(op.results[0].type):
        return _orig_negate(ctx, op)
    re, im = ctx.cbuf[op.operands[0]]
    _, n, _ = _complex_elem(op.results[0].type)
    zero = _const_f32(ctx, np.zeros(n, np.float32))
    ctx.cbuf[op.results[0]] = (_sub(ctx, zero, re, n), _sub(ctx, zero, im, n))


# --- stablehlo.fft (1-D forward DFT via constant twiddle matmul) -------------

@L.handles("stablehlo.fft")
def _fft(ctx, op):
    from jaxlib.mlir import ir
    fft_type = str(op.attributes["fft_type"])
    length = list(ir.DenseI64ArrayAttr(op.attributes["fft_length"]))
    if "FFT" not in fft_type or "IFFT" in fft_type or "RFFT" in fft_type:
        raise L.LoweringError(
            f"stablehlo.fft: only forward complex FFT supported (got {fft_type}); "
            f"IFFT/RFFT/IRFFT are future work (§43)")
    if len(length) != 1:
        raise L.LoweringError(
            f"stablehlo.fft: only 1-D FFT supported (got fft_length={length})")
    Nf = int(length[0])

    in_type = op.operands[0].type
    shape, n_total, _ = _complex_elem(in_type)
    if not shape or shape[-1] != Nf:
        raise L.LoweringError(
            f"stablehlo.fft: last operand dim {shape} != fft length {Nf}")
    B = n_total // Nf                         # flattened leading batch

    xr, xi = ctx.cbuf[op.operands[0]]

    # Twiddle matrix W[k,n] = exp(-2pi i k n / N); symmetric, so batched
    # X(B,N) = x(B,N) @ W(N,N). Wr = cos, Wi = -sin (folded into the imag const).
    k = np.arange(Nf)[:, None]
    n = np.arange(Nf)[None, :]
    ang = (-2.0 * np.pi * k * n / Nf)
    Wr = np.cos(ang).astype(np.float32)
    Wi = np.sin(ang).astype(np.float32)       # note: sin(ang) = -sin(2pi kn/N)
    wr_buf = _const_f32(ctx, Wr)
    wi_buf = _const_f32(ctx, Wi)

    # Xr = xr@Wr - xi@Wi ; Xi = xr@Wi + xi@Wr   (M=B, K=Nf, N=Nf)
    xr_wr = _dot(ctx, xr, wr_buf, B, Nf, Nf)
    xi_wi = _dot(ctx, xi, wi_buf, B, Nf, Nf)
    xr_wi = _dot(ctx, xr, wi_buf, B, Nf, Nf)
    xi_wr = _dot(ctx, xi, wr_buf, B, Nf, Nf)
    Xr = _sub(ctx, xr_wr, xi_wi, n_total)
    Xi = _add(ctx, xr_wi, xi_wr, n_total)
    ctx.cbuf[op.results[0]] = (Xr, Xi)
