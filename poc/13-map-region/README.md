# poc/13 — map-region fusion (the §23 / ideas-for-v2 Idea A gate)

**Question (go/no-go):** §23 says our "fusion" (§11) removes the *barrier* but not
the *round-trips* — every EW tile-op still does `arena[a] → compute → arena[dst]`,
so a K-op GELU chain = K global writes + K reads. Does keeping the intermediates
**on-chip** across the run (a fused map-region: load once, interpret the op
sub-list over scratch, store once) actually beat that?

**Method.** Hand-emit the GELU tail (tanh approx, **9 pure-map micro-ops that
reuse `x` 4× — a real DAG**) as a single region, interpreted three ways that
differ *only in the scratch address space*, plus two references:

| variant        | scratch      | models                                   |
|----------------|--------------|------------------------------------------|
| `region_global`| global plane | today's per-op round-trips (scalar)      |
| `region_local` | `__local`    | fused region, **step 1** (local staging) |
| `region_reg`   | switch-regs  | fused region, **step 2** ceiling         |
| `ewchain1x`    | global plane | faithful "today": 9 **vectorized** float4 passes, ONE launch (= §11 chain) |
| `region_local4`/`region_reg4` | local/regs | the **fair** fused competitor: float4-vectorized interpreter |
| `gelu_hard`    | registers    | hardcoded vectorized gelu (absolute ceiling) |

All variants share the micro-op program and are numerically identical
(maxerr 2.95e-7, pure f32). Build: `make`; run `./poc13 [n]`
(`PJRT_OCL_DEVICE=NVIDIA|Portable`).

## Result — GO (fused wins; size-dependent)

Fair comparison = **`region_l4`/`region_r4` (fused) vs `ewchain1x` (today)**,
best-of-20 × 50 iters. GB/s = 2·n·4 B (1 read + 1 write) / time.

**NVIDIA RTX PRO 6000 Blackwell (128 MB L2):**

| n (f32)   | size    | ewchain1x | region_l4 | region_r4 | gelu_hard | fused speedup |
|-----------|---------|-----------|-----------|-----------|-----------|---------------|
| 262 144   | 1 MiB   | 0.0088 ms | 0.0090    | 0.0102    | 0.0032    | **~1.0× (wash)** — overhead-bound |
| 1 048 576 | 4 MiB   | 0.0149    | 0.0113    | 0.0110    | 0.0036    | **1.3×** — L2-resident |
| 16 777 216| 64 MiB  | 0.2916    | 0.0944    | 0.0843    | 0.0386    | **3.1–3.5×** — HBM-bound |
| 67 108 864| 256 MiB | 1.1908    | 0.3891    | 0.3715    | 0.3657    | **3.2×** (reg ≈ hardcoded ceiling) |

**PoCL CPU (Ryzen 9 3900X):** n=1M → ewchain1x 2.20 ms, region_l4 **0.77 ms
(2.85×)**, region_r4 1.11 ms. On CPU the round-trip elimination bites even at
base size (cache pressure), and **local-staging beats switch-registers**.

## Findings that shape the design (docs §24)

1. **GO, but the win is size-regime-dependent.** At the base FFN GELU size
   (4×128×2048 = 1 MiB f32 = 4 MiB, **L2-resident** on this 128 MB-L2 GPU) the
   pure-memory win is a modest **1.3×**; the dramatic **3×+** appears only once
   the working set exceeds L2 (HBM-bound). The microbench also **understates**
   the real win: it has zero VM per-instruction dispatch overhead, whereas the
   real megakernel pays bytecode-dispatch + grid-stride setup for *each* of the 9
   EW ops (§22: ~3–5 µs/instr) — collapsing 9→1 saves that too.
2. **Vectorization is load-bearing.** The *scalar* interpreter is a wash even
   vs the multi-launch chain (dispatch-bound). Float4-vectorizing the
   interpreter (process 4 elems/iter, `float4` scratch slots) is what turns it
   into a clear win. The region tile-op MUST vectorize its inner loop.
3. **Local-staging is the right portable default.** `region_reg4` (switch-
   addressed registers) slightly edges local on NVIDIA at large N but *loses* on
   PoCL CPU (switch dispatch is expensive there). `__local` staging wins or ties
   everywhere and is simpler — the "step 1" of Idea A's two-step plan. Registers
   are a per-device step-2 tuning, not needed to bank the win.
4. **Interpreter ceiling is adequate.** `region_reg4` reaches the hardcoded
   gelu speed at 64 MiB (0.372 vs 0.366 ms) — the interpreted register-file is
   not the bottleneck; memory traffic is. No JIT / device-compile needed.
