"""Expanded elementwise ops (div/max/min/pow, unary transcendentals, compare,
select) — all route through the TILE_EW tile op with a subop, same as the
builtin add/mul/sub/fill fast paths in scheduler.py/vmreader.py.

SUB_* constants below are NOT free choices: they must match
``pjrt_plugin/kernels/vm2.cl``'s ``enum { SUB_ADD = 0, SUB_MUL, ... }``
exactly (the C++ engine is the actual executor; the schedule simulator here
is validator b). Binary ew_bin() covers subop <= SUB_POW, unary ew_un()
covers SUB_COPY..SUB_SIGN, then FILL/IOTA_FLAT/CMP/SELECT/LTS are handled
specially — see vm2.cl's TOP_EW case.

Field convention (matches OP_CMP_F32/OP_SELECT_F32 comments in lowering.py):
- compare: tensor Instr.imm = predicate (0 EQ,1 NE,2 LT,3 LE,4 GT,5 GE);
  Task.p2 carries it for vm2.cl's SUB_CMP (which reads t.p2, not t.imm —
  there is no Task.imm, tasks only have p0..p3).
- select: tensor Instr.imm = predicate BUFFER id; a/b = on_true/on_false.
  Task.p3 carries the predicate buffer id (vm2.cl's SUB_SELECT reads
  arena[t.p3+i] as the predicate, arena[t.a+i]/arena[t.b+i] as the two
  values) — NOT Instr.imm's raw slot, which the scheduler doesn't forward.

Unary ops leave tensor Instr.b = 0 (unused) but MUST register a `reads`
function returning {ins.a} only — else the scheduler's default read-set
{ins.a, ins.b} records a bogus RAW dependency on buffer 0. The Task-level
b field is set to `ins.a` (self-alias) instead of 0: vm2.cl's unary branch
never dereferences t.b, but the python schedule simulator eagerly slices
`view(task.b)[lo:hi]` before dispatch, so an out-of-range sentinel there
would raise; aliasing `a` keeps it always in-bounds and side-effect-free.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_EW

# --- EW subops beyond the scheduler builtins (EW_ADD/MUL/SUB/FILL) ---------
# Must match pjrt_plugin/kernels/vm2.cl's SUB_* enum values exactly.
SUB_DIV = 3
SUB_MAX = 4
SUB_MIN = 5
SUB_POW = 6
SUB_NEG = 8
SUB_EXP = 9
SUB_LOG = 10
SUB_SQRT = 11
SUB_RSQRT = 12
SUB_TANH = 13
SUB_ABS = 14
SUB_FLOOR = 15
SUB_CEIL = 16
SUB_SIGN = 17
SUB_CMP = 20
SUB_SELECT = 21

# predicate ints, matching OP_CMP_F32's documented convention and vm2.cl's
# SUB_CMP switch on t.p2 (0 EQ,1 NE,2 LT,3 LE,4 GT, default(5) GE).
_CMP_DIRECTION_TO_PRED = {"EQ": 0, "NE": 1, "LT": 2, "LE": 3, "GT": 4, "GE": 5}
_CMP_PRED_TO_NPFN = {
    0: np.equal, 1: np.not_equal, 2: np.less, 3: np.less_equal,
    4: np.greater, 5: np.greater_equal,
}


# --- generic binary/unary handler + registration factories -----------------

def _binop_handler(opcode: int) -> Callable:
    def handler(ctx, op):
        _, n_elems, dtype = L.tensor_info(op.results[0].type)
        for operand in op.operands:
            if L.tensor_info(operand.type)[1:] != (n_elems, dtype):
                raise L.LoweringError(
                    f"{op.name}: operand/result shape mismatch "
                    f"(implicit broadcast?)")
        dst = ctx.new_buffer(n_elems, dtype)
        ctx.emit(L.Instr(opcode, dst=dst, a=ctx.buf_for(op.operands[0]),
                         b=ctx.buf_for(op.operands[1]), n=n_elems))
        ctx.value_to_buf[op.results[0]] = dst
    return handler


def _unop_handler(opcode: int) -> Callable:
    def handler(ctx, op):
        _, n_elems, dtype = L.tensor_info(op.results[0].type)
        if L.tensor_info(op.operands[0].type)[1:] != (n_elems, dtype):
            raise L.LoweringError(f"{op.name}: operand/result shape mismatch")
        dst = ctx.new_buffer(n_elems, dtype)
        ctx.emit(L.Instr(opcode, dst=dst, a=ctx.buf_for(op.operands[0]), b=0,
                         n=n_elems))
        ctx.value_to_buf[op.results[0]] = dst
    return handler


def _register_binop(stablehlo_name: str, opcode: int, subop: int,
                    npfn: Callable) -> None:
    L.handles(stablehlo_name)(_binop_handler(opcode))

    def to_task(ins) -> Task:
        return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.b,
                    p0=subop, p1=ins.n, p2=0, p3=0)

    def interp(ins, rt) -> None:
        a = rt.view(ins.a, ins.n)
        b = rt.view(ins.b, ins.n)
        rt.view(ins.dst, ins.n)[:] = npfn(a, b)

    def ew_sim(a, b, task, rt, lo, hi):
        return npfn(a, b)

    opsem.register(opcode, to_task=to_task, interp=interp)
    opsem.register_ew_sim(subop, ew_sim)


def _register_unop(stablehlo_name: str, opcode: int, subop: int,
                   npfn: Callable) -> None:
    L.handles(stablehlo_name)(_unop_handler(opcode))

    def to_task(ins) -> Task:
        # b aliases a (self): unused by vm2.cl's unary branch, but keeps the
        # python schedule simulator's eager view(task.b)[lo:hi] in-bounds.
        return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.a,
                    p0=subop, p1=ins.n, p2=0, p3=0)

    def interp(ins, rt) -> None:
        a = rt.view(ins.a, ins.n)
        rt.view(ins.dst, ins.n)[:] = npfn(a)

    def reads(ins) -> set[int]:
        return {ins.a}

    def ew_sim(a, b, task, rt, lo, hi):
        return npfn(a)

    opsem.register(opcode, to_task=to_task, interp=interp, reads=reads)
    opsem.register_ew_sim(subop, ew_sim)


_BINOPS = [
    ("stablehlo.divide", L.OP_DIV_F32, SUB_DIV, np.divide),
    ("stablehlo.maximum", L.OP_MAX_F32, SUB_MAX, np.maximum),
    ("stablehlo.minimum", L.OP_MIN_F32, SUB_MIN, np.minimum),
    ("stablehlo.power", L.OP_POW_F32, SUB_POW, np.power),
]

_UNOPS = [
    ("stablehlo.negate", L.OP_NEG_F32, SUB_NEG, np.negative),
    ("stablehlo.exponential", L.OP_EXP_F32, SUB_EXP, np.exp),
    ("stablehlo.log", L.OP_LOG_F32, SUB_LOG, np.log),
    ("stablehlo.sqrt", L.OP_SQRT_F32, SUB_SQRT, np.sqrt),
    ("stablehlo.rsqrt", L.OP_RSQRT_F32, SUB_RSQRT,
     lambda x: np.float32(1.0) / np.sqrt(x)),
    ("stablehlo.tanh", L.OP_TANH_F32, SUB_TANH, np.tanh),
    ("stablehlo.abs", L.OP_ABS_F32, SUB_ABS, np.abs),
    ("stablehlo.floor", L.OP_FLOOR_F32, SUB_FLOOR, np.floor),
    ("stablehlo.ceil", L.OP_CEIL_F32, SUB_CEIL, np.ceil),
    ("stablehlo.sign", L.OP_SIGN_F32, SUB_SIGN, np.sign),
]

for _name, _opcode, _subop, _npfn in _BINOPS:
    _register_binop(_name, _opcode, _subop, _npfn)

for _name, _opcode, _subop, _npfn in _UNOPS:
    _register_unop(_name, _opcode, _subop, _npfn)


# --- compare -----------------------------------------------------------

@L.handles("stablehlo.compare")
def _compare(ctx, op):
    from jaxlib.mlir.dialects import stablehlo
    # operands share a dtype; the result is bool (1-byte). The VM reads operands
    # as the operand dtype (task.adtype) and writes bool (task.dtype).
    _, n_elems, dtype = L.tensor_info(op.operands[0].type)
    if L.tensor_info(op.operands[1].type)[1:] != (n_elems, dtype):
        raise L.LoweringError("compare: operand shape mismatch")
    direction = stablehlo.ComparisonDirectionAttr(
        op.attributes["comparison_direction"]).value
    pred = _CMP_DIRECTION_TO_PRED.get(direction)
    if pred is None:
        raise L.LoweringError(f"compare: unsupported direction {direction}")
    dst = ctx.new_buffer(n_elems, L.DT_BOOL)
    ctx.emit(L.Instr(L.OP_CMP_F32, dst=dst, a=ctx.buf_for(op.operands[0]),
                     b=ctx.buf_for(op.operands[1]), n=n_elems, imm=pred))
    ctx.value_to_buf[op.results[0]] = dst


def _cmp_to_task(ins) -> Task:
    return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.b,
                p0=SUB_CMP, p1=ins.n, p2=ins.imm, p3=0)


def _cmp_interp(ins, rt) -> None:
    a = rt.view(ins.a, ins.n)
    b = rt.view(ins.b, ins.n)
    npfn = _CMP_PRED_TO_NPFN[ins.imm]
    rt.view(ins.dst, ins.n)[:] = npfn(a, b)   # dst is bool (uint8 0/1)


def _cmp_ew_sim(a, b, task, rt, lo, hi):
    npfn = _CMP_PRED_TO_NPFN[task.p2]
    return npfn(a, b).astype(np.uint8)


opsem.register(L.OP_CMP_F32, to_task=_cmp_to_task, interp=_cmp_interp)
opsem.register_ew_sim(SUB_CMP, _cmp_ew_sim)


# --- select --------------------------------------------------------------

@L.handles("stablehlo.select")
def _select(ctx, op):
    pred_val, on_true_val, on_false_val = op.operands
    # pred operand has i1 type — do not run tensor_info on it (F32-only
    # check would reject it); shape/n_elems come from on_true instead.
    _, n_elems, dtype = L.tensor_info(on_true_val.type)
    if L.tensor_info(on_false_val.type)[1:] != (n_elems, dtype):
        raise L.LoweringError("select: on_true/on_false shape mismatch")
    dst = ctx.new_buffer(n_elems, dtype)
    ctx.emit(L.Instr(L.OP_SELECT_F32, dst=dst, a=ctx.buf_for(on_true_val),
                     b=ctx.buf_for(on_false_val), n=n_elems,
                     imm=ctx.buf_for(pred_val)))
    ctx.value_to_buf[op.results[0]] = dst


def _select_to_task(ins) -> Task:
    return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.b,
                p0=SUB_SELECT, p1=ins.n, p2=0, p3=ins.imm)


def _select_interp(ins, rt) -> None:
    pred = rt.view(ins.imm, ins.n)
    on_true = rt.view(ins.a, ins.n)
    on_false = rt.view(ins.b, ins.n)
    rt.view(ins.dst, ins.n)[:] = np.where(pred != 0.0, on_true, on_false)


def _select_reads(ins) -> set[int]:
    return {ins.a, ins.b, ins.imm}


def _select_ew_sim(a, b, task, rt, lo, hi):
    # a/b already sliced from task.a/task.b (on_true/on_false); pred is a
    # separate buffer referenced by task.p3.
    pred = rt.view(task.p3)[lo:hi]
    return np.where(pred != 0.0, a, b)


opsem.register(L.OP_SELECT_F32, to_task=_select_to_task,
               interp=_select_interp, reads=_select_reads)
opsem.register_ew_sim(SUB_SELECT, _select_ew_sim)
