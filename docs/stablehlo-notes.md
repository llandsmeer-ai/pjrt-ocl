# StableHLO notes

Source: https://openxla.org/stablehlo/spec (checked 2026-07-14).

## Control flow: structured only — no jumps

The spec is explicit: *"StableHLO doesn't have jump ops, so the corresponding part of MLIR syntax
is unused."* All control flow is expressed as **regions** (anonymous functions) attached to ops:

| Op | Regions |
|---|---|
| `while` | cond region, body region |
| `if` | then region, else region |
| `case` | N branch regions, selected by an index operand |
| `reduce`, `reduce_window` | reduction body |
| `map` | per-element computation |
| `select_and_scatter` | select + scatter computations |
| `sort` | comparator |
| `scatter` | update computation |

Consequence for our bytecode: it can stay **strictly linear** (no pc manipulation). Each
region-carrying op lowers to one instruction that references nested (also linear) instruction
lists; the VM drives cond/body sub-lists itself (e.g. `while`: run cond list → read scalar →
maybe run body list → repeat). Region bodies are themselves pure StableHLO, so lowering recurses
with the same code path as the top-level function.

Note: many region-carrying ops (`reduce`, `sort` with the common comparators, `map` over simple
lambdas) should be **pattern-matched to dedicated fused kernels** first (a plain `reduce_sum_f32`
kernel), with the generic interpret-the-region path as fallback.
