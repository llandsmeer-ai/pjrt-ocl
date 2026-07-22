# Ideas for v2

Forward-looking architecture explorations — **not committed, not scheduled.** These are the
"if we rebuilt the execution core" ideas that came out of profiling the transformer workload
(where `base` sits ~13× off native CUDA). They are bigger than the incremental fusions we ship
today (§11 chain fusion, §13 access-map views, §19 fused norm — all in `docs/decisions.md`); this
file is where the structural rethinks live so they aren't re-derived from scratch each time.

Guiding constraint for all of these: they must respect the project's load-bearing invariants —
**persistent megakernel + cross-workgroup spin-barrier requires co-resident lanes** (decisions
§10c / poc/01), and **the OpenCL compiler is never on the hot path** (CLAUDE.md). Where an idea
strains one of those, it's called out explicitly.

---

## The problem these ideas attack

Measured on `base` (4×128×512, 6 layers): **192 VM ops → 107 serial barrier-phases**, ~5.8 ms,
13.4× off native CUDA. Per-op kernels are competitive/faster than CUDA *standalone*, yet the whole
is far behind. Two root causes, both structural, neither about per-op kernel speed:

1. **Our "fusion" removes the barrier, not the round-trips.** Every elementwise tile-op still does
   `arena[a] → compute → arena[dst]`. §11 chain fusion lets a chain run barrier-free on one lane,
   but each op *still materializes its full intermediate to global memory*. A 6-op GELU chain = 6
   global writes + 6 reads. We eliminated the sync, not the DRAM traffic — which is exactly why
   per-op wins don't compound.
2. **Everything is strictly serialized by the cross-workgroup barrier.** 107 phases, nothing
   overlaps, each barrier stalls all ~376 lanes. CUDA fuses a layer into a handful of pipelined
   on-chip kernels; we run ~100 discrete global-synced passes.

The fix in one sentence: **let intermediates live on-chip (registers/L1) across a run of ops, and
only touch global memory + barrier at genuine cross-lane boundaries (reduce / matmul / gather).**

---

## Idea A — register-resident map-region fusion (the incremental path)

*(This is `docs/decisions.md` §23; summarized here as the near-term, principle-preserving step.)*

A general **fused map-region tile-op**:

- **Lowering/scheduler finds maximal map-regions**: contiguous runs of ops whose output tile
  depends only on the input tile at a *static* position — elementwise (add/mul/exp/gelu/affine),
  broadcast-reads, transpose/reshape views. Region boundaries are the genuinely cross-lane ops:
  **reduce, matmul, gather/scatter, dynamic index**.
- **Each region → ONE op** run as a per-tile pipeline: load the region's inputs into on-chip
  scratch once → interpret the region's op sub-list on that scratch → store outputs once.
  Intermediates never leave the chip. K ops + K round-trips + K−1 no-op barriers → **1 load +
  1 store, 1 phase**.
- **Generalizes what we already hand-fuse** (§11 barriers, §13 views, §19 norm) into one mechanism.
  Most of the 107 `base` phases are exactly these map-ops.

**Design decision (settled in discussion): a separate region op, not a per-opcode "chain" flag.**
A flag can say "chained" but not "chained *through temp r3*" — once intermediates are on-chip the
operands inside the run must reference tile-registers, not arena ids, and real map-regions are DAGs
(GELU reuses `x` 4×, binary ops need 2 live inputs, residuals fork), not linear pipes. A flag that
grows a register-addressing mode *is* a region op, with worse ergonomics. The flag's only real
advantage is incrementalism (rides on §11's existing chain detection, degrades gracefully) — but
the hard part (the tile-loop + on-chip scratch address space) is shared by both, so the flag saves
format churn, not engineering.

**Two-step implementation (de-risk):**
1. Region op staging intermediates in **`__local`** first (simpler than register allocation, still
   kills the DRAM round-trips — L1 vs global), reduced tile size. Validates region-detection + the
   VM region-loop end-to-end.
2. Promote hot regions to **pure registers** (last slice of latency), local as the spill fallback.

**Register-pressure math (feasibility):** 256-thread WG, 4 elems/thread = 1024-elem tiles; a region
with ~8 SSA temps = 32 tile-registers/thread — comfortable (GPUs have 64–256). Tile-size ×
region-width is the pressure knob; spill and you lose.

---

## Idea B — pure register-dataflow megakernel (the thought experiment)

Take Idea A to its logical end: instead of a *tensor-tile* interpreter (~50 coarse opcodes over
arena buffers), a **SIMT vector-register ISA** — ~15 RISC opcodes (`LOAD`, `STORE`, `GATHER`,
`FMA`, `EXP`, `TANH`, `RSQRT`, `CMP`, `SELECT`, `MAX`, …), operands are register indices +
immediates, memory touched only via explicit load/store. Each register is a **tile-vector** (a
workgroup's worth of elements) so one interpreted instruction still does tile-worth of work — that
amortizes dispatch. The persistent-megakernel skeleton survives (grid-stride lanes, residency
launch, cross-wg barrier); the inner loop between barriers becomes register dataflow.

### What fundamentally changes

1. **Lowering becomes a compiler backend.** We take on **register allocation** (linear-scan over a
   region's SSA temps), **instruction scheduling** (hide load latency), and **spill**. That's the
   complexity the current tensor-op design was built to avoid. Upside: **access maps (§13) become
   the `LOAD` addressing modes** — broadcast/transpose/strided-gather stop being ops and become
   address generators. Clean unification.

2. **Register file vs hardware registers — the crux.** GPU hardware registers are *not* dynamically
   indexable: an interpreter's `R[rd]` with runtime `rd` spills to local/global, defeating the
   point. Three resolutions, which define the design point:
   - **Interpreted, local-backed `R[]`** → a *scratchpad* dataflow machine (L1, not DRAM). Big win
     over today's global round-trips, but not true registers.
   - **Interpreted, small fixed register file via switch-dispatch** (`float r0…r15;
     wr(i,v){switch(i)…}`) → the compiler *can* keep r0…r15 in real registers; the switch is the
     dispatch cost, amortized over the tile-vector. **The sweet spot** — true registers, no device
     compile, but bounded to ~8–16 vector registers (bounds region size before local spill).
   - **JIT to device code** (emit SPIR-V/PTX per program-shape, cache it) → full hardware registers
     + the vendor's scheduler, but **abandons the "compiler never on the hot path" principle.**
     Per-shape caching softens it; still a real philosophical change.

   Principle-preserving landing: **interpreted vector-register machine, small switch-addressed
   register file + local spill.** Captures most of the win without a JIT.

3. **Reduce/matmul stay special — inherently a hybrid.** Scalar register ops can't express
   cross-element (reduce) or cross-lane (matmul) work. Those remain **intrinsics** (`REDUCE` via
   shuffles, `MMA` calling the tuned tensor-core path, §10c) or **region boundaries** (spill +
   barrier). So the machine partitions into register-dataflow regions separated by coarse
   reduce/matmul/gather nodes — i.e. **Idea B is Idea A with a richer region body.** They converge;
   B is not an alternative to A, it's A's interior taken to the limit.

### The tension that bites
On-chip register/local file **fights occupancy, and occupancy is load-bearing**: the cross-wg
spin-barrier only works with co-resident lanes (§10c). A fat per-lane file → fewer resident
workgroups → the barrier's co-residency assumption weakens or breaks. So "how big a register file"
is coupled to whether the whole persistent-barrier model holds — a sharper, more dangerous knob
than today's. (Plan-B host-dispatch sidesteps the barrier but pays per-phase launch, which §14b
showed dominates — so this doesn't get a free pass by dropping the megakernel.)

### Net assessment
- Right **end-state for the memory-bound (base) regime** — it's what a GPU kernel compiler produces;
  kills barriers *and* round-trips at once.
- Principle-preserving version = interpreted vector-register machine (small switch-addressed regs +
  local spill, coarse reduce/matmul intrinsics, access-maps-as-load-modes).
- Compiled version (JIT+cache) strictly faster, trades the no-compile stance — pursue only if the
  interpreted-register ceiling proves too low.
- Does **not** change the matmul intensity-cap lever (large regime, §10c) — orthogonal.

---

## The one measurement that gates all of this

Before committing to *any* compiler backend, run the decisive microbench: **hand-emit the GELU tail
(a ~6-op pure-map region) as a single on-chip region — first local-staged, then switch-addressed
registers — and compare against today's 6 round-tripping EW ops on `base`-sized data.** That single
delta answers: (a) does removing the round-trips actually move the needle at these sizes, and (b) is
the interpreted-register-file ceiling high enough to justify the machine — *before* we build
register allocation. PoC-first (CLAUDE.md): this belongs under `poc/NN-register-region/`.

## Open questions to resolve if/when this is picked up
- Interpreter dispatch overhead vs tile size (smaller tiles → more instructions → more dispatch;
  bigger tiles → local/regfile pressure). Where's the knee?
- Exact register-file size that stays within co-residency for the spin-barrier on each target
  (NVIDIA / AMD CDNA / Intel Xe2 differ).
- Do gather/scatter (data-dependent addresses) stay region boundaries, or become `GATHER`/`SCATTER`
  addressing modes that can sit *inside* a region (they can't be freely reordered/hoisted)?
- Can the reduce intrinsic fuse its *surrounding* elementwise into the same region (as §19's fused
  norm already does by hand) — i.e. is the region boundary "before the reduce" or "the reduce plus
  its local pre/post EW"?
