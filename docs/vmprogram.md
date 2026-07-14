# VMProgram v2 — binary format spec

v2 (2026-07-14, M3): adds the **aux pool** (per-instruction shape/stride metadata), shaped
opcodes (gather/reduce/dot/iota-dim), an expanded elementwise set, `IF`, and the
**ints-as-f32 policy**. v1 kept below for history is IDENTICAL except: header had no
`n_aux`/`aux` section and only opcodes 0–7 existed. Version field distinguishes them; the
executor rejects v1 (producer and consumer live in one repo).

## v2 deltas

- **Header** grows to 48 bytes: after `n_outputs` insert `n_aux u32, pad u32` (then
  `arena_bytes u64` as before). `n_aux` = number of u32 words in the aux pool.
- **Aux pool section** (after IO shapes, before const pool): `n_aux × u32`, 8B-aligned at end.
  Instructions reference it by word offset in `pad0` (renamed `aux`); `pad1` stays reserved.
  Aux words are u32; strides are stored as i32 (two's complement, may be negative).
- **Dtype policy**: everything is f32 in the arena. Lowering maps i32/i64/pred/bool values to
  f32 (exact for |int| ≤ 2^24 — documented caveat, revisit for real int dtypes in v3).
  Compare ops write exactly 1.0f/0.0f.
- **New opcodes** (dst/a/b = buffer ids as before; `aux` = aux-pool word offset):

| op | name | semantics |
|---|---|---|
| 8 | DIV_F32 | `dst[i] = a[i] / b[i]` |
| 9 | MAX_F32 | elementwise max |
| 10 | MIN_F32 | elementwise min |
| 11 | POW_F32 | `pow(a[i], b[i])` |
| 12 | COPY_F32 | `dst[i] = a[i]` |
| 13 | NEG_F32 | unary; b ignored |
| 14 | EXP_F32 | |
| 15 | LOG_F32 | |
| 16 | SQRT_F32 | |
| 17 | RSQRT_F32 | |
| 18 | TANH_F32 | |
| 19 | ABS_F32 | |
| 20 | FLOOR_F32 | |
| 21 | CEIL_F32 | |
| 22 | SIGN_F32 | |
| 23 | CMP_F32 | `imm` = predicate: 0 EQ, 1 NE, 2 LT, 3 LE, 4 GT, 5 GE; writes 1.0/0.0 |
| 24 | SELECT_F32 | `dst[i] = pred[i]!=0 ? a[i] : b[i]`; pred buffer id in `imm` |
| 25 | GATHER_STRIDED | `dst[i] = a[src_off + Σ_d idx_d(i)*stride_d]`; aux: `rank u32, out_dims i32[rank], in_strides i32[rank], src_off i32` (elements). Covers broadcast_in_dim (stride 0), transpose (permuted strides), slice (src_off + strides), reverse (negative strides + src_off). |
| 26 | REDUCE | aux: `kind u32 (0 sum, 1 max, 2 min, 3 prod), out_rank u32, out_dims i32[out_rank], in_strides_kept i32[out_rank], red_rank u32, red_dims i32[red_rank], red_strides i32[red_rank], src_off i32`. Each out elem serially reduces its red-space. `n` = out elem count. |
| 27 | DOT | aux: `M u32, N u32, K u32` — `dst[MxN] = a[MxK] @ b[KxN]` row-major dense. `n` = M*N. |
| 28 | IOTA_DIM | aux: `rank u32, out_dims i32[rank], dim u32` — `dst[i] = idx_dim(i)` |
| 29 | IF | like WHILE: cond scalar = buffer `dst` (read atomically); then = instrs `[a, a+b)`, else = `[n, n+imm)` (either may be empty) |

Opcodes 0–7 unchanged from v1. WHILE/IF sub-list rules per v1 (linear, nested, frame stack —
shared depth ≤ 8 budget).

# VMProgram v1 — binary format spec (historical)

The contract between Python lowering (`python/pjrt_ocl/lowering`, producer) and the C++ executor
(`pjrt_plugin/runtime`, consumer). Produced once per PJRT compile; executed many times. Strictly
linear instruction lists (no jumps); region ops reference nested lists by (start, len) ranges.

All integers little-endian. File = header, then sections in order, each 8-byte aligned.

## Header (40 bytes)

| field | type | notes |
|---|---|---|
| magic | u32 | `0x314D5056` ("VPM1") |
| version | u32 | 1 |
| n_buffers | u32 | buffer-table entries |
| n_instrs | u32 | total instructions (root + sub-lists) |
| n_consts | u32 | const-pool entries |
| main_len | u32 | root list = instrs `[0, main_len)` |
| n_inputs | u32 | |
| n_outputs | u32 | |
| arena_bytes | u64 | total arena size (executor allocates one buffer this big) |

## Buffer table (n_buffers × 24 bytes)

`{ arena_byte_offset u64, size_bytes u64, dtype u32, pad u32 }`

- Lowering performs arena layout (offsets 64-byte aligned). dtype: `0 = f32` (v1 is f32-only;
  the field exists so v2 can add dtypes without a format break).
- Instructions reference buffers by **table index**; the executor resolves indices to element
  offsets when loading the program (arena relayout never touches instruction encoding).

## IO maps

`inputs: n_inputs × u32` buffer ids in PJRT argument order, then
`outputs: n_outputs × u32` buffer ids in result order (padded to 8B after each array).

## IO shapes (added before first implementation — still "v1")

For each IO buffer (inputs in order, then outputs in order):
`{ rank u32, pad u32, dims u64[rank] }`, each entry 8B aligned.
The executor needs these to answer `PJRT_Buffer_ElementType/Dimensions` on result buffers;
dtype comes from the buffer table.

## Const pool (n_consts entries)

`{ buffer_id u32, byte_len u32, data[byte_len] }`, each entry padded to 8B. Uploaded into the
arena once at program-load time.

## Instructions (n_instrs × 32 bytes)

`{ op u32, dst u32, a u32, b u32, n u32, imm u32, pad u32, pad u32 }`

`dst/a/b` = buffer-table indices; `n` = element count; `imm` = f32 bits for FILL.

| op | name | semantics |
|---|---|---|
| 0 | NOP | |
| 1 | ADD_F32 | `dst[i] = a[i] + b[i]`, i < n |
| 2 | MUL_F32 | `dst[i] = a[i] * b[i]` |
| 3 | SUB_F32 | `dst[i] = a[i] - b[i]` |
| 4 | FILL_F32 | `dst[i] = as_f32(imm)` |
| 5 | IOTA_F32 | `dst[i] = (f32)i` |
| 6 | LTS_F32 | `dst[0] = a[0] < b[0] ? 1.0 : 0.0` (n = 1) |
| 7 | WHILE | cond list = instrs `[a, a+b)`, body = `[n, n+imm)`, loop while `dst[0] != 0` |

WHILE rules (validated in poc/01): sub-lists are linear; nesting via further WHILEs (executor
frame stack, depth ≤ 8); the cond scalar is read atomically by the VM (see poc/01/NOTES.md);
cond producers write exactly 1.0f/0.0f.

## Executor contract

- Load once per compile: allocate arena, upload consts, resolve buffer ids → element offsets,
  upload instruction array.
- Execute: write input host buffers into their arena regions, launch the VM megakernel
  (root list), read outputs out. v1 is synchronous.
- Reject unknown magic/version/opcode/dtype loudly (PJRT INVALID_ARGUMENT), never skip.

## Versioning

Format changes bump `version` and this doc. Producer and consumer live in one repo — no
cross-version compatibility promises; the check exists to fail loudly on skew.
