# VMProgram v1 — binary format spec

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
