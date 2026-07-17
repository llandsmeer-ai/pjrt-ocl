# Megakernel literature survey & gap analysis vs our VM (2026-07-17)

**Question.** Published megakernels beat highly-optimized kernel-per-op systems (Hazy Research
Llama-1B: ~2.5× vs vLLM; MPK: 1.2–6.7×; Hazy TP-Llama-70B: +22% vs SGLang). Our megakernel VM is
9–22× *slower* than kernel-per-op CUDA JAX. This document distills how the real systems are built
(scheduling, opcode granularity, synchronization, pipelining, register management), compares each
axis to our implementation, and derives ranked architecture changes.

Method: multi-agent web research over primary sources (claims adversarially verified 3-vote unless
marked), plus a file-level audit of our engine. Our-side facts cite `file:line` as of commit
`dc693bd`.

---

## 1. The systems

### 1.1 Hazy Research Llama-1B megakernel — "Look Ma, No Bubbles" (May 2025)

Source: <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>,
code <https://github.com/HazyResearch/Megakernels> (built on ThunderKittens).

- **Scheduling:** on-GPU interpreter; **each SM gets its own instruction stream**, statically
  scheduled host-side in Python; the schedule is reused across hundreds of forward passes. No
  dynamic work queue in the single-GPU low-latency version.
- **Granularity:** the whole forward pass is **7 fused instruction types** — e.g.
  RMSNorm+QKV-matvec+RoPE as *one* instruction (`rms_matvec_rope_append.cu`), fused
  RMSNorm+up-gate+SiLU, down-proj+residual, RMSNorm+LM-head; attention split into
  partial + reduction instructions. Fusion is inside the instruction body (register/smem
  locality), not adjacency in a schedule.
- **Synchronization:** **no global barriers.** An array of counters in global memory; an
  instruction completing increments counters, an instruction starting spins until its counters
  reach targets. Intermediate tensors are produced/consumed in **4 chunks, each with its own
  counter**, so consumers start on the first chunk. They avoid CUDA async barriers (~60 ns even
  in the "pass" state — too slow at their scale).
- **Pipelining:** H100 shared memory is divided into **13 pages of 16 KiB**; the interpreter hands
  released pages to the *next* instruction so it starts loading weights while the previous
  instruction is still finishing.
- **Numbers:** 78% of H100 HBM bandwidth at batch-1 decode; <680 µs/forward on B200. Their ~600 µs
  breakdown: 250 µs activation I/O, 200 µs compute, 30 µs weight loads, **40 µs synchronization**,
  80 µs setup. Motivation numbers: kernel launch ≈1.3 µs even under CUDA graphs; a 512-block
  launch on 148 SMs leaves an 80-SM idle tail at each kernel's end.

### 1.2 Hazy Research TP-Llama-70B megakernel (Sep 2025)

Source: <https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main>.

- **Scheduling:** upgraded to a **dynamic global work queue** — SMs fetch the next instruction by
  atomically incrementing a global counter. Ablation at batch 8192: static instead of dynamic
  drops 31,516 → 27,033 tok/s (**−14%**).
- **Granularity:** 9 fused instruction types (RMSNorm+all-gather, QKV+RoPE, attention+distributed
  transpose, down-matmul+reduce-scatter+residual, …), each with **load/compute/store phases run by
  specialized warp groups** (warp specialization).
- **Synchronization:** per-tile dependency counters; granularity varies per op — 128 output rows
  of a matmul, or a single attention head.
- **Pipelining:** inter-instruction smem page remapping; ablation: no pipelining = −6.1%
  (31,516 → 29,607 tok/s).
- **Numbers:** 23,468 vs SGLang's 19,170 tok/s on 8×H100 (>22%).

### 1.3 MPK — Mirage Persistent Kernel (Zhihao Jia et al.)

Sources: <https://arxiv.org/abs/2512.22219>,
<https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17>,
<https://github.com/mirage-project/mirage>. (Blog claims verified by direct fetch; paper claims
3-vote verified.)

- The closest system to ours in *intent*: a **compiler + in-kernel runtime** that automatically
  transforms tensor programs (incl. multi-GPU inference) into one persistent megakernel — not
  hand-written per model.
- **Scheduling:** SMs statically partitioned into **workers** (each with a dedicated task queue)
  and **schedulers** (each a single warp, maintaining a queue of activated events) — a
  decentralized in-kernel runtime, not host-driven, not SPMD lockstep.
- **Granularity:** compiler decomposes each operator into **tasks sized to one SM** over an
  SM-level task graph.
- **Synchronization:** tasks increment event counters; an event whose counter reaches its
  threshold is "activated" and schedulers launch its dependent tasks. **Inter-task overhead:
  1–2 µs.** No global barriers. The task graph enables cross-operator pipelining that kernel
  boundaries cannot express (allreduce starts on partial matmul results; matmul of layer N+1
  overlaps attention of layer N).
- **Numbers:** A100-40GB Llama decode 14.5 ms/token (vLLM/SGLang-level baseline) → 12.5 ms, vs a
  10 ms bandwidth-bound floor; claimed 1.2–6.7× across settings.

### 1.4 ThunderKittens (arXiv 2410.20399)

- **Granularity:** 16×16 tiles (register/shared tiles + global layout descriptors) with a
  PyTorch-like tile-op set; kernels follow an asynchronous **Load-Compute-Store-Finish (LCSF)**
  template with warp specialization and multi-stage buffering.
- Deepening the LCSF pipeline 1→4 stages scales GEMM 260 → 760 TFLOPS at 4096³ — i.e. most of a
  3× on the *same* tiles comes purely from pipelining depth.
- (A further claim that persistent-grid launch raises GEMM 271→309 TFLOPS failed our 3-vote
  verification 1-2 — treat the specific numbers as unconfirmed; the persistent-grid scheduler
  itself is real and in the TK repo.)

### 1.5 Triton persistent kernels (tutorial 09, persistent matmul)

- Grid = `min(NUM_SMS, n_tiles)`; **fully static** strided round-robin over output tiles (no
  queue, no stealing). Residency from `multi_processor_count`, not the iteration space.
- **Epilogue subtiling**: split epilogue stores into pieces to *free shared memory*, reinvested as
  deeper pipeline stages — an explicit example of trading smem footprint for pipelining.
- The deeply pipelined variants have hard smem requirements (won't run on RTX 4090) — the cost
  side of pipelining depth.

### 1.6 Classic persistent-threads & portable-barrier literature

- **Gupta, Stuart & Owens (InPar '12)** — defines PT = maximal launch (residency-sized grid) +
  software scheduling; four use cases (CPU-GPU sync, load balancing, producer-consumer locality,
  global sync). Measured (c. 2012): host round-trip ~400 µs, kernel launch 3–7 µs, in-kernel
  software global barrier 1.3–2.0 µs; PT global barrier ⇒ 2–2.5× on sync-bound workloads. Also:
  PT "can result in performance loss in many cases" — it is not free speedup.
- **Aila & Laine (HPG '09)** — persistent warps fetch ray batches from a global pool via atomic
  counter: 1.5–2.2× vs hardware scheduler on GTX285, with **two-level work fetching** to keep the
  global atomic from serializing. Their later addendum: the win is architecture-dependent (≈0 on
  Fermi, +10% on Kepler) — hardware schedulers improved.
- **Xiao & Feng (IPDPS '10)** — lock-based global barrier costs grow linearly in #blocks;
  lock-free (per-block flag arrays, no atomics) is O(1); in-kernel barrier 8.4×/4.0× faster than
  CPU-side sync on sync-dominated microbenchmarks.
- **Sorensen & Donaldson (OOPSLA '16)** — occupancy-discovery protocol + XF barrier on OpenCL 2.0
  atomics, validated on NVIDIA/Intel/AMD/ARM; deadlock cliff measured at 56→57 workgroups on
  Titan X. Their megakernel prototype uses an **on-device scheduler workgroup + dynamic work
  queue** over global atomics. (We already use their discovery protocol —
  `vm_common.cl:192–215`.)

---

## 2. What every winning system shares

Across Hazy 1B, TP-70B, MPK, ThunderKittens — four invariants:

- **P1 — Synchronization is dataflow-shaped, never phase-shaped.** No modern megakernel uses a
  grid-wide barrier between ops. All use per-task/per-tile counters in global memory (chunked
  producers, 1–2 µs handoffs), so independent work overlaps and stragglers only block their true
  consumers.
- **P2 — An instruction is a fused op-group producing an output tile, not one framework op.**
  Fusion happens *inside* the instruction body (register/smem locality: RMSNorm+matvec+RoPE = one
  read of the activation), not by scheduling adjacent ops near each other.
- **P3 — The interpreter overlaps instruction N+1's loads with N's compute** (smem paging, warp
  specialization, multi-stage LCSF). Measured value: 2.5–6.6% end-to-end for inter-instruction
  pipelining (TP-70B), but ~3× *intra*-op for deep GEMM pipelines (TK 260→760 TFLOPS).
- **P4 — Register/smem budgets are per-instruction-type, not a global max.** Instructions are
  separately compiled functions sharing a thin template (Hazy), tile libraries (TK), or
  per-task compiled code (MPK). Nobody pays a max-over-all-ops register budget in one switch.

And two things the literature *validates about our current design*:

- Static host-side scheduling is respectable: Hazy-1B is fully static and reused; Triton's
  persistent matmul is static; the dynamic-queue upgrade bought TP-70B 14%. Our LPT packing with a
  calibrated cost model (`scheduler.py:330–365`) is at state-of-practice for the static family.
- Residency-sized launch with *measured* occupancy discovery (`runtime.cc:505–548`) is exactly
  what OOPSLA'16 prescribes, and stricter than CUDA-side practice (Triton trusts the SM count).
  The host-dispatch fallback engine mirrors the literature's safety argument.

---

## 3. Gap analysis

| Axis | Literature consensus | Ours today | Severity |
|---|---|---|---|
| Sync | per-tile counters, chunked producers, 1–2 µs handoff, zero global barriers | 209 global barrier phases per transformer step (`scheduler.py:511`); WAIT/SIGNAL fields reserved but unused (`vm_main.cl:172–174`) | **High** |
| Granularity | fused sub-layer instructions (norm+matmul+rope in one body) | 1 instr = 1 StableHLO op; "fusion" = same-lane adjacency, every EW op still round-trips global memory (`scheduler.py:442–469`) | **High** |
| Registers/smem | per-instruction-type budgets | one switch, max-over-11-ops budget; 8 KB `__local` declared for all programs (`vm_main.cl:71–72`); 128×128 MMA tile spills/hangs (`decisions.md:517–520`) | **High** (blocks matmul) |
| Pipelining | smem paging, warp specialization, LCSF 1→4 stages ≈ 3× on GEMM | serial decode→exec→barrier; single-buffered in-VM MMA (`mma.cl:110–141`) | Medium (intra-op), Low-Med (inter-op) |
| Scheduling | static OK small-scale; dynamic queue +14% at scale | static LPT + cost model | Low (for now) |
| Launch sizing | residency-probed persistent grid | same (probe in real kernel) | None — we match |

**Diagnosis.** Our megakernel eliminated the *cheapest* overhead in the literature's accounting —
the kernel-boundary cost (launch ≈1.3–7 µs → our barrier 1.7 µs; 209 × 1.7 µs ≈ 0.36 ms of the
9.7 ms base step, consistent with our flat lane-sweep) — while keeping kernel-boundary *semantics*:
a global join between every dependent op group, one-op instruction granularity, and a shared
register budget. The literature's actual wins come from what those semantics forbid:

1. fused instruction bodies (P2) → fewer global-memory round-trips — our profile says 65% of
   transformer time is non-matmul, i.e. exactly unfused EW/norm traffic;
2. counter-based overlap (P1) → no straggler tails, producer-consumer chunk overlap;
3. per-type register budgets (P4) → big pipelined matmul tiles *inside* the megakernel — ours is
   capped at 64×64 single-buffered, ~10% of cuBLAS.

In other words: we built a persistent kernel, but not yet a megakernel in the literature's sense.
The good news: the bytecode format already reserved the fields (wait/signal flags), the scheduler
already owns a dataflow graph with per-tile ranges, and the barrier fix already proved
device-scope coherent-load spinning works on our targets — the ingredients for P1 exist.

---

## 4. Ranked, tangible changes

### R1. Replace barrier phases with a dependency scoreboard (P1) — structural, highest leverage

Activate the reserved per-entry WAIT/SIGNAL mechanism: each entry carries
`{wait_ctr, wait_target, signal_ctr}`; a lane starting an entry spins on `ctr[wait] >= target`
using the same coherent device-scope `atomic_load` that fixed the barrier (`vm_common.cl:152–162`);
each finished tile does one `atomic_inc(signal)`. The scheduler already knows producers, consumers
and tile counts per entry — targets are just tile counts. Chunk large producers (Hazy's 4-chunk
trick) so consumers start early: signal per tile-range quartile instead of per entry.

- Portability: needs exactly the primitives today's barrier needs (device-scope acq/rel + relaxed
  loads); fence-less devices keep the host-dispatch engine, whose `clFinish` boundary *is* a
  scoreboard with one counter. PoCL unaffected (host-dispatch stays primary there).
- Keep `PJRT_OCL_SYNC=barrier|counter` for A/B; barrier engine remains the debug baseline.
- Expected effect: eliminates the phase-max serialization (LPT imbalance tail at 209 joins),
  enables producer-consumer overlap, and unlocks R3's heterogeneous lanes and the designed-but-
  unbuilt "barrier elision for lane-diagonal loops" (`decisions.md:494–497`) as a special case.
- Stage 2 (only if imbalance measured after stage 1): dynamic tile queue via atomic ticket
  (TP-70B +14%; Aila two-level fetch to avoid atomic serialization, important on PoCL).

### R2. True fusion into instruction bodies (P2) — attacks the 65% non-matmul time

- **R2a — EW micro-programs:** a `TOP_EW_FUSED` tile-op whose payload is a short sub-op program
  (the existing `SUB_*` vocabulary, `vm_common.cl:70–88`) interpreted **per element in
  registers**: N-op chain = 1 global read per input + 1 write, instead of N round-trips. This is
  the portable, generic analogue of Hazy's fused instructions — fusion by interpretation, no
  runtime compilation, consistent with the "compiler never on the hot path" rule. The scheduler's
  existing `_map_chains` component analysis already finds exactly these chains; emit one entry
  instead of a same-lane sequence.
- **R2b — fused reduce+broadcast idioms** (layernorm/softmax as single-pass segmented ops):
  already begun per decisions §19 / `TOP_RED_SEG`; finish the catalog (mean/var+normalize,
  max+exp+sum+div).
- **R2c — matmul epilogue fusion:** fold bias/residual-add/activation and output views into
  `TOP_MMA`'s store (it already composes input views, `mma.cl:95–128`), removing the EW op after
  every matmul.

### R3. Split the VM by register class (P4) — unlocks matmul inside the megakernel

Precompile 2–3 `vm2` variants at init (still once per device, cached): `vm2_light` (EW/reduce/
gather/scatter; no MMA `__local` arrays → smaller footprint, more resident lanes) and `vm2_heavy`
(MMA + the ops matmul programs need; budget freed for the standalone `mm2`-class 128×64
double-buffered tile that currently can't live in the shared switch, `decisions.md:517–521`).
Select per program at compile time — the bytecode's op census is known. This turns the
"hybrid split" idea (`transformer-perf.md`) into a zero-relaunch design: with R1, matmul-heavy and
light phases are just different *programs*, not interleaved launches paying ~0.11 ms per boundary.

### R4. Inter-instruction pipelining (P3) — after R1–R3

Literature value is real but modest end-to-end (2.5–6.6%): with the scoreboard, a lane that knows
its next entry can issue the next MMA tile's first K-panel load while storing the current tile
(split our `__local` into 2 panels = paging-lite). Intra-op pipelining depth (TK's 3× on GEMM)
matters more but is gated on R3's register headroom — revisit the double-buffered K-loop there
(it was "a wash" only under the shared budget).

### R5. Non-goals for now (explicitly)

- Warp/subgroup specialization: needs subgroup guarantees that are shaky across OpenCL vendors;
  reconsider per-vendor behind the kernel-table override after R1–R4.
- MPK-style in-kernel scheduler warps: R1's static-stream scoreboard gets most of the benefit
  without a scheduler/worker split; revisit only if dynamic-queue stage 2 shows scheduling cost.

### Measurement plan (exit criteria per change)

- Instrument lane-idle time at joins (histogram) before/after R1; expect the base-config
  (22×-gap, overhead-bound) case to improve most.
- R2: track bytes moved per transformer step (arena traffic counter) — expect ≥2× reduction in
  non-matmul global traffic; base 9.7 ms target <5 ms.
- R3: in-megakernel matmul TFLOP/s @2048 vs the 21.1 standalone `mm2` figure — the gap between
  them (17→21) is the price of the shared switch today.

---

## 5. Source index

Verified primary sources: Hazy 1B blog (no-bubbles, 2025-05-27) · Hazy TP-70B blog (2025-09-28) ·
HazyResearch/Megakernels repo · MPK arXiv 2512.22219 + Zhihao Jia Medium post + mirage-project
repo · ThunderKittens arXiv 2410.20399 · Triton tutorial 09 (persistent matmul) · PyTorch
warp-specialization blog · Gupta/Stuart/Owens InPar'12 + GTC'12 slides · Aila & Laine HPG'09 (+
addendum) · Xiao & Feng IPDPS'10 · Sorensen & Donaldson OOPSLA'16 (+ artifact repo).
Claims were 3-vote adversarially verified except the MPK Medium-post numbers (verified by direct
quote fetch) and the TK persistent-grid GEMM delta (failed verification; flagged inline).
