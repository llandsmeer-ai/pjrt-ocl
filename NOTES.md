# NOTES — ops/reduce.py (Phase 2 op family: reduce)

Decision-log entry for `docs/decisions.md` (merge into main session).

## What was added
- `python/pjrt_ocl/ops/reduce.py` — stablehlo.reduce handler + OP_REDUCE (26)
  tensor semantics + TILE_REDUCE_PART/COMB schedule-simulator tile sims.
- `tests/test_ops_reduce.py` — full-reduce sum/max/min/prod on 1D/2D/3D,
  reduce-then-arithmetic, multi-chunk large arrays, partial-axis rejection.
- `python/pjrt_ocl/ops/__init__.py` — `from . import reduce`.
- `tests/test_lowering.py` — swapped the `test_unsupported_op_exit2_json`
  canary op from `jnp.sum` (now SUPPORTED) to `jnp.tanh` (still unsupported).
  Necessary because that test used reduce as its representative unsupported op.

## Coverage: what we support vs reject
- SUPPORTED: **full reductions only** — `dimensions` == every input axis, result
  is a scalar. kinds: sum(add)=0, max(maximum)=1, min(minimum)=2, prod(multiply)=3.
  The reducer body is classified from the single non-return op in `regions[0]`.
  init value (operand[1]) must be the kind's identity (0 / -inf / +inf / 1);
  otherwise LoweringError. Verified by reading the defining scalar
  stablehlo.constant.
- REJECTED (LoweringError, precise message): any partial-axis reduction, i.e.
  `dimensions` a strict subset of the axes (e.g. `sum(x, axis=0)`). The flat
  vm2.cl REDUCE model reduces a *contiguous* run per partial; a strided reduction
  over a non-contiguous sub-space needs a permuting GATHER first (another op
  family) — deferred. Also rejects variadic reduce (>1 input/output) and
  non-constant / non-identity inits.

## Instr decomposition decision: TWO tensor instrs (PART + COMB), NOT one
- The scheduler maps one instruction -> one Task (`_instr_to_task` returns a
  single Task and is a core file I must not edit). A real reduction needs two
  tasks (partial + combine) separated by a global barrier, so it MUST be two
  instructions.
- Both instructions use opcode OP_REDUCE (26) — the only reduce opcode in
  OP_NAMES (the parser rejects unknown opcodes, so I can't invent one). Phase +
  kind ride in `Instr.imm` = `(phase << 2) | kind`, recoverable from the
  instruction alone by the phase-free scheduler mapper, the tensor interp and
  the tile sims.
- `chunk` is NOT transported. It is a deterministic function of `n`
  (`_chunk_for(n) = max(16384, ceil(n/256))`) recomputed identically by the
  handler (to size the partials buffer + COMB's n_parts), by `_reduce_to_task`
  (Task.p1), and by the interp/sims. So partials count, task p1, and
  `Task.n_tiles()` (= ceil(p0/p1)) agree by construction — no redundant state to
  drift.
- Task field usage matches vm2.cl EXACTLY:
  - TILE_REDUCE_PART: dst=partials, a=input, p0=n, p1=chunk, p2=kind — writes one
    partial per tile (n_tiles = ceil(n/chunk)).
  - TILE_REDUCE_COMB: dst=out scalar, a=partials, p0=n_parts, p1=kind — 1 tile.
- `b` is set to the source buffer (a valid buffer id, required by the parser's
  range check) and ignored; an explicit `reads = {ins.a}` is registered so
  dependency analysis is exact.

## How the part/comb barrier ordering was verified
- COMB reads the partials buffer that PART writes ⇒ `_depends(comb, part)` is
  true (RAW) ⇒ the scheduler places them in different dataflow levels ⇒ a global
  BARRIER entry is emitted between them automatically (v0 "barrier every level").
- Validator (b), `vmreader.execute_schedule`, runs phase-by-phase split on those
  barriers: it asserts every lane has the same barrier count, that every tile of
  every task is covered exactly once, and — crucially — runs each phase forward
  AND reversed on a fresh arena and asserts identical results (order-independence
  within a phase). The multi-chunk tests (100000 / 60000 elems) split PART across
  several lanes, so this exercises: disjoint partial writes in the PART phase
  (order-independent), a barrier, then the COMB phase reading a fully-populated
  partials buffer. Both validators (tensor interp + schedule simulator) are then
  checked against the JAX CPU backend in `oputil.check`.

## Verification
- `.venv` in the worktree is a partial copy missing jaxlib binaries; used the
  main project venv interpreter with `PYTHONPATH=<worktree>/python` so imports
  resolve to the worktree package (no `pip install -e`).
- `pytest tests/test_ops_reduce.py -q` -> 19 passed.
- `pytest tests/ -q` -> 51 passed, 2 skipped (pre-existing skips).
