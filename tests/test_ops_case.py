"""Coverage tests: stablehlo.case (N-way switch) + stablehlo.if, lowered as
flat sibling OP_IF region ops sharing result carries. Both validators (tensor
interp + schedule simulator) must match jax on every branch selection."""
import jax
import jax.numpy as jnp
import numpy as np

from oputil import check
from pjrt_ocl import lowering as L


def _lowered_ops(f, *args):
    from oputil import to_artifact
    prog = L.lower_artifact(to_artifact(f, *args))
    return [i.op for i in prog.instrs]


def test_switch_2way_each_index():
    branches = [lambda x: x * 2.0, lambda x: x + 100.0]
    for idx in range(2):
        check(lambda x: jax.lax.switch(idx, branches, x),
              jnp.asarray(np.arange(6, dtype=np.float32)))


def test_switch_3way_each_index_and_clamp():
    branches = [lambda x: x * 2.0, lambda x: x + 100.0, lambda x: -x]
    # valid indices 0,1,2 and out-of-range (clamps to last branch)
    for idx in (0, 1, 2, 5, -1):
        check(lambda x: jax.lax.switch(idx, branches, x),
              jnp.asarray(np.arange(8, dtype=np.float32)))


def test_switch_runtime_index():
    # index computed at runtime from the operand (not a python constant)
    branches = [lambda x: x + 1.0, lambda x: x + 2.0, lambda x: x + 3.0]

    def f(x):
        idx = (x[0] > 0).astype(jnp.int32) + (x[1] > 0).astype(jnp.int32)
        return jax.lax.switch(idx, branches, x)

    for a in ([-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [1.0, 1.0, 5.0]):
        check(f, jnp.asarray(a, dtype=np.float32))


def test_switch_multi_result():
    branches = [lambda x: (x * 2.0, x + 1.0),
                lambda x: (x - 1.0, x * 3.0)]
    for idx in (0, 1):
        check(lambda x: jax.lax.switch(idx, branches, x),
              jnp.asarray(np.arange(5, dtype=np.float32)))


def test_switch_nd_result():
    branches = [lambda x: x @ x.T, lambda x: x + x]
    for idx in (0, 1):
        check(lambda x: jax.lax.switch(idx, branches, x),
              jnp.asarray(np.arange(9, dtype=np.float32).reshape(3, 3)))


def test_cond_true_false():
    def f(p, x):
        return jax.lax.cond(p[0] > 0, lambda a: a * 2.0, lambda a: a - 1.0, x)

    for pv in ([1.0], [-1.0]):
        check(f, jnp.asarray(pv, np.float32),
              jnp.asarray(np.arange(6, dtype=np.float32)))


def test_switch_produces_region_op():
    branches = [lambda x: x * 2.0, lambda x: x + 1.0, lambda x: -x]
    ops = _lowered_ops(lambda x: jax.lax.switch(1, branches, x),
                       jnp.asarray(np.arange(4, dtype=np.float32)))
    assert ops.count(L.OP_IF) == 3, ops
