# poc/08 — occupancy discovery: measure co-resident groups, don't guess

**Question:** the megakernel needs `ngroups` ≤ true co-resident capacity or its
spin-barrier starves (Xe2 bring-up: `2×CU`=128 lanes → `clFinish` -5; measured
boundary 32 PASS / 33 FAIL, docs/decisions.md §9). Can we *query* the right
number, or must we *measure* it?

**Answer: measure it — with the Sorensen-Donaldson discovery protocol — and
measure it with the REAL kernel.** Queries get within 2× at best and the gap is
kernel-dependent (SIMD width + GRF mode + SLM), so calculation alone is unsafe.

## Protocol (deadlock-free by construction)

Gate/ticket/lock handshake on one global buffer, 1.2 atomics only (dialect-safe
even on the strict-1.2 `VMO_NO_DEVICE_FENCE` build):
leader takes a ticket while the gate is open; **ticket holders spin until the
gate closes** (that holds their residency slot — essential, otherwise the
scheduler backfills exited groups and the count inflates); groups that arrive
after close exit immediately and never wait on anyone, so any oversized launch
terminates. Ticket 0 closes the gate once the count is stable. Final count =
groups that were simultaneously resident.

## Results (`make && PJRT_OCL_DEVICE=Intel ./poc08`, 2026-07-15)

| | Arc 140V (Xe2, 64 "CUs"=XVEs) | PoCL (8-core LNL) |
|---|---|---|
| [A] slim probe | **256** = whole launch (see below) | 8 |
| [B] vm-like probe (8 KB SLM + reg pressure, SIMD32) | **64** | 8 |
| Intel-attr formula @ probe's SIMD32: 512 HW thr ÷ (256/32) | 64 ✓ | — |
| [C] liveness, 2000 spin-barrier rounds @ discovered | PASS, 1.9 µs/barrier | PASS, 225 µs/barrier |
| shipped heuristic 2×CU | 128 (**deadlocks real vm2**) | 16 (would starve) |
| real vm2 measured boundary (JAX e2e) | 32 | — |

Three findings:

1. **Slim kernels over-discover (256 = everything launched): Xe2 mid-thread
   preemption time-slices them**, so "co-residency" isn't even the right
   concept there. Kernels that use barriers + SLM (like vm2 and probe [B]) are
   not preemptible → discovery returns true residency. Corollary: if a future
   kernel IS preemptible, over-discovery is harmless — preemption is exactly
   what keeps a spin-barrier live without co-residency.
2. **Per-kernel footprint changes the answer 2×**: probe [B] compiles at SIMD32
   → 64 resident; the real vm2 compiles at SIMD16 (fatter register file) → 32.
   Both liveness-check clean *at their own discovered count*. So the probe must
   be **the vm2 kernel itself** (a probe mode inside it), not a lookalike.
3. PoCL discovers CUs (8) for any footprint and its spin-barrier is *live* at
   that count under **balanced** load — the known starvation deadlock (poc/07)
   is about imbalanced lane streams, a separate axis. Host-dispatch stays the
   CPU engine.

`./poc08 --over` runs liveness at discovered+1: on Xe2 it spun >60 s with no
progress and had to be host-killed (the driver recovered cleanly) — i.e.
discovery is **tight**: discovered+0 is live, discovered+1 is not. Not run by
default.

**Decision:** integrate discovery as a probe mode in `vm2` (same compiled
kernel = same SIMD/GRF/SLM), run once at plugin init (~20 ms), and use
`ngroups = min(discovered, 2×CU)` — the cap keeps NVIDIA exactly at its
already-validated sizing until discovery is re-validated there.
