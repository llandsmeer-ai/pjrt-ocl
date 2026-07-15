"""Coverage tests: the "making" family (iota, convert) — see
pjrt_ocl/ops/making.py for the aux layout and the int-as-f32-policy caveats.

iota: `jax.lax.broadcasted_iota(dtype, shape, dimension)` is jax's direct,
one-op equivalent of `stablehlo.iota` for arbitrary rank + iota_dimension, so
these run through the normal oputil.check() jax pipeline like every other
op-family test.

convert: jax elides a same-dtype `x.astype(jnp.float32)` at the TRACING
level (verified empirically — no `stablehlo.convert` is ever emitted for a
f32 array cast to f32, not even via `lax.convert_element_type` or a direct
`convert_element_type_p.bind`), so `jax.jit(f).lower(...)` can never produce
the op our OP_COPY_F32 handler exists for. To exercise it we hand-build a
tiny stablehlo module containing a literal `stablehlo.convert` using
jaxlib's mlir bindings directly (same bindings `lowering.py` itself uses),
bypassing jax's tracer-level elision.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from oputil import check

RNG = np.random.default_rng(11)


def arr(*shape, hi=20):
    return jnp.asarray(RNG.integers(0, hi, shape).astype(np.float32))


# --- iota --------------------------------------------------------------

def test_iota_1d():
    check(lambda: lax.broadcasted_iota(jnp.float32, (7,), 0))


def test_iota_2d_dim0():
    check(lambda: lax.broadcasted_iota(jnp.float32, (3, 5), 0))


def test_iota_2d_dim1():
    check(lambda: lax.broadcasted_iota(jnp.float32, (3, 5), 1))


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_iota_3d_dims(dim):
    check(lambda: lax.broadcasted_iota(jnp.float32, (2, 3, 4), dim))


def test_iota_plus_arithmetic():
    check(lambda x: lax.broadcasted_iota(jnp.float32, (3, 4), 0) + x, arr(3, 4))


def test_iota_multi_tile():
    # 200*100 = 20000 elements > TILE_SIZE (16384): exercises >1 tile.
    check(lambda: lax.broadcasted_iota(jnp.float32, (200, 100), 1))


def test_iota_row_and_col_combined():
    check(lambda: lax.broadcasted_iota(jnp.float32, (4, 6), 0)
          + lax.broadcasted_iota(jnp.float32, (4, 6), 1))


# --- convert (hand-built IR; see module docstring) ----------------------

def _make_module(shape, build):
    """Build+serialize a tiny stablehlo module. `build(ir, stablehlo, func_dialect,
    entry) -> result Value` constructs the body; the module takes one
    `shape` f32 tensor arg and returns `build`'s result."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import func as func_dialect, stablehlo

    ctx = ir.Context()
    stablehlo.register_dialect(ctx)
    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()
        tensor_t = ir.RankedTensorType.get(list(shape), f32)
        fn_type = ir.FunctionType.get([tensor_t], [tensor_t])
        with ir.InsertionPoint(module.body):
            fop = func_dialect.FuncOp("main", fn_type)
            fop.attributes["sym_visibility"] = ir.StringAttr.get("public")
            fop.attributes["arg_attrs"] = ir.ArrayAttr.get([ir.DictAttr.get({})])
            fop.attributes["res_attrs"] = ir.ArrayAttr.get([ir.DictAttr.get({})])
            entry = fop.add_entry_block()
            with ir.InsertionPoint(entry):
                result = build(ir, stablehlo, func_dialect, entry)
                func_dialect.ReturnOp([result])
        return stablehlo.serialize_portable_artifact(
            module, stablehlo.get_current_version())


def _check_hand_built(shape, build, x, expected=None):
    """Like oputil.check(), but for a hand-built (non-jax-traceable) module:
    run both validators, feeding `x` as the single input, and compare
    against `expected` (default: `x`, i.e. an identity program)."""
    from pjrt_ocl import scheduler, vmreader

    if expected is None:
        expected = x
    artifact = _make_module(shape, build)
    prog = vmreader.parse(scheduler.lower_and_schedule(artifact))
    got_tensor = vmreader.execute(prog, [x])
    got_sched = vmreader.execute_schedule(prog, [x])
    np.testing.assert_allclose(got_tensor[0].reshape(shape), expected)
    np.testing.assert_allclose(got_sched[0].reshape(shape), expected)


def test_convert_identity_1d():
    shape = (6,)

    def build(ir, stablehlo, func_dialect, entry):
        tensor_t = entry.arguments[0].type
        return stablehlo.ConvertOp(tensor_t, entry.arguments[0]).result

    x = RNG.integers(0, 20, shape).astype(np.float32)
    _check_hand_built(shape, build, x)


def test_convert_identity_2d():
    shape = (3, 4)

    def build(ir, stablehlo, func_dialect, entry):
        tensor_t = entry.arguments[0].type
        return stablehlo.ConvertOp(tensor_t, entry.arguments[0]).result

    x = RNG.integers(0, 20, shape).astype(np.float32)
    _check_hand_built(shape, build, x)


def test_convert_in_expression():
    """convert composed with add (arg + convert(arg)) — closer to how it'd
    appear amid real ops than a bare identity."""
    shape = (5,)

    def build(ir, stablehlo, func_dialect, entry):
        tensor_t = entry.arguments[0].type
        conv = stablehlo.ConvertOp(tensor_t, entry.arguments[0])
        return stablehlo.AddOp(conv.result, entry.arguments[0]).result

    x = RNG.integers(0, 20, shape).astype(np.float32)
    _check_hand_built(shape, build, x, expected=x + x)


def test_convert_unsupported_dtype_raises():
    """stablehlo.convert to a non-f32 dtype is not supported: there is no
    real int/bool arena dtype yet, and f32->int truncation has no
    round-toward-zero op in vm2.cl (only floor/ceil/sign) — must raise
    LoweringError with a clear message rather than silently misinterpreting
    bits."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import func as func_dialect, stablehlo

    from pjrt_ocl import lowering as L

    shape = (4,)
    ctx = ir.Context()
    stablehlo.register_dialect(ctx)
    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()
        i32 = ir.IntegerType.get_signless(32)
        in_t = ir.RankedTensorType.get(list(shape), f32)
        out_t = ir.RankedTensorType.get(list(shape), i32)
        fn_type = ir.FunctionType.get([in_t], [out_t])
        with ir.InsertionPoint(module.body):
            fop = func_dialect.FuncOp("main", fn_type)
            fop.attributes["sym_visibility"] = ir.StringAttr.get("public")
            fop.attributes["arg_attrs"] = ir.ArrayAttr.get([ir.DictAttr.get({})])
            fop.attributes["res_attrs"] = ir.ArrayAttr.get([ir.DictAttr.get({})])
            entry = fop.add_entry_block()
            with ir.InsertionPoint(entry):
                conv = stablehlo.ConvertOp(out_t, entry.arguments[0])
                func_dialect.ReturnOp([conv.result])
        artifact = stablehlo.serialize_portable_artifact(
            module, stablehlo.get_current_version())

    with pytest.raises(L.LoweringError):
        L.lower_artifact(artifact)
