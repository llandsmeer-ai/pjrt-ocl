# poc/01 notes — for the decision tree

## What was tried / what happened

- Barrier: Xiao & Feng-style inter-workgroup barrier (arrival counter + phase flag), OpenCL 1.2
  atomics only (`atomic_inc`, coherent reads via `atomic_add(p, 0)`, `mem_fence`). Worked on the
  FIRST attempt on both PoCL and NVIDIA — no deadlocks, no corruption in stress tests up to 2000
  chained cross-workgroup-dependent instructions.
- PoCL did NOT deadlock with 24 spinning workgroups (24 groups = 24 CUs = 24 HW threads; each
  spinning group holds one pool thread). **Do not launch more groups than CUs on PoCL** — a
  spinning group that isn't scheduled would deadlock the barrier. Same co-residency rule as GPUs.
- NVIDIA: 188 groups (1 per SM) × 64 all co-resident; barrier ~1.1 µs. Megakernel beats separate
  launches even at 1M-element ops (2.5x), and 3.2x at 4K-element ops. On CPU/PoCL the win
  disappears (0.93x) because ops are compute-bound and launches are cheap there — fine, CPU is a
  debug target.

## Caveats / follow-ups

- The 1.2-style barrier (relaxed atomics + mem_fence + spin) is technically outside the OpenCL 1.2
  memory model's guarantees for cross-workgroup visibility; it works on PoCL and NVIDIA today.
  Follow-up: feature-detect OpenCL 2.0+ C atomics (`atomic_load_explicit(memory_scope_device)`)
  and use the well-defined path where available (Intel/AMD support 2.0 atomics; NVIDIA supports
  OpenCL 3.0 with optional features — query at init).
- Launch geometry = `min(CUs, needed)` groups for now; a proper occupancy query
  (`CL_KERNEL_WORK_GROUP_SIZE` etc.) should size this once kernels get register-heavy.
- `VM_LOCAL=64` untuned; tune later (likely 128–256 on GPUs).
- Instruction fetch: every work-item reads the instr struct from global memory each step — fine at
  µs scale; could broadcast via local memory later.
- Not yet in the PoC: nested instruction lists for `while`/`if` (the instr struct + interpreter
  loop trivially extend: an instruction whose operands are program offsets; cond scalar read from
  arena by all work-items after a barrier).
