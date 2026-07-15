# poc/07 — a correct cross-workgroup barrier

**Question (user):** can we improve the barrier? Is there a CL extension? Another way?

**Answer:** yes — **device-scope acquire/release fences** (OpenCL 2.0 memory
model) fix the cross-lane data race in-kernel, on NVIDIA *and* everywhere else.
No extension exists or is needed. Kernel-boundary dispatch remains the portable
fallback (and the only fix for PoCL's separate *liveness* problem).

## Why the old barrier was wrong

The shipped `vmo_barrier` synchronised lanes with a global atomic counter and a
`mem_fence(CLK_GLOBAL_MEM_FENCE)`. That fence is **work-group-scoped**: it says
nothing about making one lane's *non-atomic* data writes visible to a *different*
workgroup. The atomic phase flag is coherent (it lives in L2), so lanes agree on
*when* the barrier is crossed — but the data a producer lane wrote can still sit
in its SM's L1, and a consumer lane reads a stale copy. It looked fine for
single-shot two-level programs (a fresh launch has a cold L1, so the first
post-barrier read misses L1 and fetches fresh from L2) and blew up under
iteration (a persistent loop keeps the L1 line warm → stale forever). This is
what forced `n_lanes=1` for `while` programs.

## Experiments (`main.c`)

`G` co-resident workgroups; each writes `a[g]=iter`, barrier, reads a neighbour's
cell and checks it equals `iter`; repeat. Stale reads counted.

| test | what | NVIDIA RTX PRO 6000 (Blackwell) | PoCL (Ryzen 3900X) |
|---|---|---|---|
| A | does `memory_scope_device` **compile**? | ✅ yes | ✅ yes |
| B | 1.2 barrier, **plain** neighbour read | ❌ **1599968 / 200000 stale** | ✅ 0 (CPU mem coherent) |
| **E** | **device-scope-fence** barrier, plain read | ✅ **0 / 200000** | ✅ 0 |
| C | 1.2 barrier, **volatile** neighbour read | ✅ 0 (L1 bypass → L2) | ✅ 0 |
| D | **kernel-boundary** (2 kernels/phase) | ✅ 0, **3.1 µs/phase** | ✅ 0, **46 µs/phase** |

## Findings

1. **The race is systematic, not rare.** On Blackwell essentially *every* plain
   cross-lane read under iteration is stale (test B ≈ G × iters). "Works for
   `relu(matmul)`" was a cold-L1 accident.
2. **`clinfo` lies about NVIDIA.** It advertises only *work-group* atomic scope,
   yet the compiler accepts `memory_scope_device` (A) **and the hardware honours
   it** (E → 0 stale). So capability must be **feature-probed at runtime**, not
   read from `CL_DEVICE_ATOMIC_FENCE_CAPABILITIES`.
3. **Device-scope fences are the fix (test E).** Release our data device-wide
   before signalling arrival; acquire peers' data device-wide after the phase
   flips. In-spec, keeps the megakernel, plain reads become coherent. Cost here
   is ~57% *of the barrier* in a pathological tiny-working-set microbench —
   negligible as a fraction of any real program. **Applied to the shipped
   `vmo_barrier`.**
4. **`volatile` reads (C)** also work on NVIDIA (L1-bypass) but disable L1 reuse
   for those loads — worse for reuse-heavy kernels. Fallback only.
5. **Kernel-boundary (D)** is correct on every vendor with no atomics (the
   OpenCL execution model guarantees global visibility between in-order kernels)
   at 3 µs/phase (NVIDIA) / 46 µs/phase (PoCL). This is **Plan B**, and the
   answer for devices that fail the E probe — and for **PoCL liveness**, which
   device-scope fences do *not* address (memory vs. liveness are separate axes;
   B is already clean on PoCL, but the spin-barrier still deadlocks there at
   high lane counts because CPU workgroups aren't guaranteed co-resident).

## Build / run

```
make && PJRT_OCL_DEVICE=NVIDIA ./poc07
      PJRT_OCL_DEVICE=Portable ./poc07
```
