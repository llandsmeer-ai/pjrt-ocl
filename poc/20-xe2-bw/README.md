# poc/20 — Xe2 achievable streaming bandwidth (what is the real ceiling?)

**Question.** Our elementwise ops and `dynamic_slice` on Xe2 all plateau around
75–94 GB/s while Lunar Lake's LPDDR5X has ~136 GB/s of *theoretical* peak. Is
the VM leaving 45% on the table, or is ~136 simply unreachable? Tuning without
knowing the achievable ceiling is guesswork.

**Method.** A bare grid-stride triad (`c = a + b`), a pure copy, and a read-only
reduction — swept over vector width (`float1/2/4/8/16`) and work-group size
(64/128/256/512). No VM, no plugin. 256 MiB arrays (far past any cache).

```bash
cc -O2 -o bw20 bw20.c -lOpenCL
./bw20 256 Intel            # [MiB] [platform-substr]
BW_VERBOSE=1 ./bw20 256 Intel   # every (vw, lsz) cell
```

## Result (Intel Arc 140V, 64 CUs, 2050 MHz)

| kernel | best config | GB/s |
|---|---|---|
| triad (2 read + 1 write) | vw=1, lsz=512 | **109.5** |
| copy (1 read + 1 write) | vw=2, lsz=512 | 108.4 |
| read-only reduction | vw=1, lsz=512 | 108.5 |

Vector-width sweep for triad (GB/s):

| vw | lsz=64 | 128 | 256 | 512 |
|---|---|---|---|---|
| 1 | 104.7 | 106.2 | 108.0 | **109.5** |
| 2 | 105.8 | 106.6 | 107.7 | 108.3 |
| 4 | 106.8 | 107.3 | 107.6 | 106.3 |
| **8** | 79.5 | 79.5 | 79.7 | 79.8 |
| **16** | 79.6 | 79.9 | 79.9 | 80.0 |

## Findings

1. **The achievable ceiling is ~109 GB/s, not 136** — 80% of theoretical, which
   is the normal ratio. Read, copy and triad all converge on the same number, so
   this is a pure bandwidth wall, symmetric in reads and writes.
2. **`float8`/`float16` are a 27% PENALTY on Xe2** (80 vs 108 GB/s), while
   `float1/2/4` are all equivalent. This is the opposite of the CPU result
   (poc/09: explicit `float8` was the whole point on PoCL, 5 → 46 GB/s). Our
   `float8` tile bodies are gated to `VMO_CPU_TILES` and GPUs take a `float4`
   path — correct by construction here, but worth knowing before anyone widens
   the GPU vectors "for free speed".
3. **Work-group size barely matters** (104.7 → 109.5 across 64→512, ~5%).

## Consequence for the VM (measured through the plugin)

Our elementwise `a+b` at 16M reaches **~94 GB/s = 86% of the 109 achievable**.
The residual ~14% is bytecode-VM interpretation: per tile the kernel reads a
task descriptor, resolves operand handles (`AP`/`VMO_BASE`), and branches on the
sub-opcode, where the POC's loop is bare. That is the cost of the "one generic
kernel, never recompile per dispatch" design, not a fixable kernel bug.

**Tile size `EW_TS` was re-checked and left alone.** Sweeping it on Xe2:

| EW_TS | n=64K | n=256K | n=1M | n=16M |
|---|---|---|---|---|
| **4096 (shipped)** | **0.065 ms** | **0.115 ms** | **0.221 ms** | 2.15 ms (94 GB/s) |
| 65536 | 0.093 ms | 0.130 ms | 0.239 ms | 2.15 ms (94 GB/s) |

A larger tile buys **nothing** at 16M and costs **44%** at 64K (too few tiles →
too few work-groups → idle XVEs). The NVIDIA-derived 4096 default is right for
Xe2 as well. **No change shipped from this PoC** — its value is the ceiling
number and the `float8` trap.
