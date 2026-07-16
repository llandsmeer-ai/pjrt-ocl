"""Access-map fusion for matmul (docs/decisions.md §13/§14): a transpose/
reshape/broadcast that feeds a dot_general folds into the matmul's strided
operand read instead of materializing a full tensor + a gather phase.

Each test asserts BOTH validators (tensor interpreter + schedule simulator)
match jax AND that the fold actually happened (the feeding GATHER task is gone
and the MMA task carries a view aux-offset in p4/p5).
"""
import jax.numpy as jnp
import numpy as np

from oputil import check, to_artifact
from pjrt_ocl import scheduler, vmreader

RNG = np.random.default_rng(7)


def arr(*shape, hi=5):
    return jnp.asarray(RNG.integers(0, hi, shape).astype(np.float32))


def _tasks(f, *args):
    prog = vmreader.parse(scheduler.lower_and_schedule(to_artifact(f, *args)))
    return prog, prog.schedule.tasks


def _mma_tasks(tasks):
    return [t for t in tasks if t.tile_op == scheduler.TILE_MMA]


def _gather_tasks(tasks):
    return [t for t in tasks if t.tile_op == scheduler.TILE_GATHER]


# --- the transpose feeding a matmul folds; result still correct --------------

def test_transpose_lhs_folds():
    # (M,K)->transpose->(K,M) then (K,M)^T ... use einsum that puts a transpose
    # directly on a dot operand: C[n,m] = sum_k B[k,n] * A[k,m] with A transposed.
    def f(a, b):
        return a.T @ b          # a:(K,M) -> a.T:(M,K); dot A operand = transpose
    a, b = arr(6, 4), arr(6, 5)   # a.T @ b : (4,6)@... wait shapes
    # a:(6,4) -> a.T:(4,6); b:(6,5) -> (4,6)@(6,5)=(4,5)
    prog, tasks = _tasks(f, a, b)
    check(f, a, b)
    mma = _mma_tasks(tasks)
    assert len(mma) == 1
    assert mma[0].p4 != 0 or mma[0].p5 != 0, "transpose should fold into the dot"
    assert not _gather_tasks(tasks), "the folded transpose must leave no gather"


def test_attention_qk_folds_both_operands():
    # multi-head attention QK^T: q,k are (B,H,T,hd) after reshape+transpose of
    # (B,T,D). scores = q @ k.transpose(0,1,3,2). Both operands are transposes.
    B, T, H, hd = 2, 8, 2, 4
    D = H * hd

    def f(x):
        def split(t):
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        q = split(x @ wq)
        k = split(x @ wk)
        return q @ k.transpose(0, 1, 3, 2)

    wq = arr(D, D)
    wk = arr(D, D)
    x = arr(B, T, D)
    check(f, x)
    prog, tasks = _tasks(f, x)
    # the QK^T matmul (batched, G=B*H) must carry views on BOTH operands.
    qkt = [t for t in _mma_tasks(tasks) if t.p3 == B * H]
    assert qkt, "expected a batched QK^T matmul"
    assert qkt[0].p4 != 0 and qkt[0].p5 != 0


def test_attention_av_folds_v():
    # softmax(scores) @ v where v is (B,H,T,hd) from reshape+transpose: the v
    # transpose folds into the AV matmul's B operand.
    B, T, H, hd = 2, 6, 2, 3
    D = H * hd

    def f(s, x):
        v = (x @ wv).reshape(B, T, H, hd).transpose(0, 2, 1, 3)   # (B,H,T,hd)
        return s @ v          # (B,H,T,T) @ (B,H,T,hd) -> (B,H,T,hd)

    wv = arr(D, D)
    s = arr(B, H, T, T)
    x = arr(B, T, D)
    check(f, s, x)
    prog, tasks = _tasks(f, s, x)
    av = [t for t in _mma_tasks(tasks) if t.p3 == B * H]
    assert av and av[0].p5 != 0, "v transpose should fold into the AV matmul B"


def test_self_attention_qqT_stays_correct():
    # q @ q.transpose : the SAME split(q) buffer feeds operand A directly AND
    # feeds operand B via the inner transpose. The split gather then must NOT be
    # folded away (its materialized output is B's view source) — regression for
    # the "unwritten view source" bug. The transpose still folds into B.
    B, T, H, hd = 2, 6, 2, 3
    D = H * hd

    def f(x):
        qq = (x @ wq).reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # (B,H,T,hd)
        return qq @ qq.transpose(0, 1, 3, 2)                      # (B,H,T,T)

    wq = arr(D, D)
    x = arr(B, T, D)
    check(f, x)
    prog, tasks = _tasks(f, x)
    qkt = [t for t in _mma_tasks(tasks) if t.p3 == B * H]
    assert qkt, "expected a batched self-QK^T matmul"
    # B operand folds (transpose), A operand reads the still-materialized split.
    assert qkt[0].p5 != 0 and qkt[0].p4 == 0
    # exactly one GATHER survives (the split feeding operand A); the two
    # transposes (split's inner + the outer B transpose) are accounted for:
    # split stays (1 gather), outer transpose folds (0).
    assert len(_gather_tasks(tasks)) == 1


def test_output_merge_transpose_folds():
    # out.transpose(0,2,1,3).reshape(B,T,D) @ wo : the merge transpose folds
    # into the wo matmul's A operand.
    B, T, H, hd = 2, 4, 2, 3
    D = H * hd

    def f(o):
        merged = o.transpose(0, 2, 1, 3).reshape(B, T, D)   # (B,H,T,hd)->(B,T,D)
        return merged @ wo

    wo = arr(D, D)
    o = arr(B, H, T, hd)
    check(f, o)
    prog, tasks = _tasks(f, o)
    mma = _mma_tasks(tasks)
    assert len(mma) == 1 and mma[0].p4 != 0
