# poc/12 — fixed-trip while: OP_FOR construct vs bytecode unroll vs OP_WHILE

Nearly every `stablehlo.while` JAX emits is a **counted loop** (`lax.scan`,
`fori_loop`): carry k starts at a constant, cond is exactly `arg_k < const`,
body returns `arg_k + const_step`. Nobody hand-writes data-dependent whiles in
practice. If the trip count is known at compile time, the cond sub-list never
needs to execute and — critically — nothing has to be *read back* to decide
whether to continue:

| path       | per-iteration cost (persistent GPU engine)         | (host-dispatch CPU engine) |
|------------|-----------------------------------------------------|-----------------------------|
| OP_WHILE   | cond phases + 2 global barriers + per-lane atomic cond read | blocking host read of the cond flag (queue drain) |
| OP_FOR     | 1 global barrier                                    | nothing — the whole loop streams into the enqueue ring |
| unroll     | root-list phases (cross-iteration fusion applies)   | same, and phases batch across iterations |

The unroll path additionally binds the counter to a per-iteration const-pool
scalar, so the counter add-chain is DCE'd and `_compose_affines` can collapse
an affine recurrence across iterations (a 10-step `x*a+b` scalar recurrence
becomes ONE instruction).

## Run

```
.venv/bin/python poc/12-fixed-trip/bench.py --device Portable   # PoCL
.venv/bin/python poc/12-fixed-trip/bench.py --device NVIDIA
```

Workloads (all f32, checked bit-comparable across modes via the `sig` column):
- `fori-ew`: `x = x*a + b` with **vector** a, b (not scalar-foldable — the
  affine-composition collapse is deliberately off the table) for T steps.
- `scan-rnn`: `c = c*0.9 + xs[t]`, stacking `ys` — dynamic_slice +
  dynamic_update_slice per step, the canonical scan shape.

`PJRT_OCL_WHILE=while|for|unroll` pins the path per subprocess; `xla-cpu` rows
are the JAX CPU backend on the same machine.

## Results

See `results_pocl.csv` / `results_nvidia.csv` (best-of-5 wall ms per execute;
`compile_s` includes the first execute). Summary and the chosen `auto`
threshold: docs/decisions.md §14.
