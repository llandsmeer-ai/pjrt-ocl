/* Elementwise tile op (TOP_EW). subop in task.p0; n in p1; cmp pred / fill bits
 * in p2; select pred BYTE offset in p3. dst/a/b are BYTE offsets.
 * Dispatched by vmo_exec_tiles on result dtype `dt` and operand dtype `adt`
 * (compare: operands adt -> bool result; select: bool pred -> operands dt).
 * bool is 1-byte (uchar 0/1), matching jax PRED. */

static float vmo_ew_bin(const uint sub, const float x, const float y)
{
    switch (sub) {
    case SUB_ADD: return x + y;   case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;   case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y); case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y);
    case SUB_ATAN2: return atan2(x, y);
    case SUB_REMAINDER: return fmod(x, y);   /* C fmod: sign of dividend x */
    default: return 0.0f;
    }
}
static float vmo_ew_un(const uint sub, const float x)
{
    switch (sub) {
    case SUB_COPY: return x;      case SUB_NEG: return -x;
    case SUB_EXP: return exp(x);  case SUB_LOG: return log(x);
    case SUB_SQRT: return sqrt(x); case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH: return tanh(x); case SUB_ABS: return fabs(x);
    case SUB_FLOOR: return floor(x); case SUB_CEIL: return ceil(x);
    case SUB_SIGN: return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x);
    case SUB_LOG1P: return log1p(x);
    case SUB_EXPM1: return expm1(x);
    case SUB_CBRT: return cbrt(x);
    case SUB_SIN: return sin(x);
    case SUB_COS: return cos(x);
    case SUB_TAN: return tan(x);
    case SUB_RINT: return rint(x);    /* round to nearest, ties to even */
    case SUB_ROUND: return round(x);  /* round to nearest, ties away from 0 */
    default: return 0.0f;
    }
}
/* Binary/unary EW dispatch predicates. New subops keep ADD..POW / COPY..SIGN
 * contiguous with their new siblings (ATAN2..REMAINDER / LOG1P..ROUND) so
 * this stays a pair of range checks — see the vm_common.cl enum comment. */
static int vmo_ew_is_bin(const uint sub)
{
    return sub <= SUB_POW || (sub >= SUB_ATAN2 && sub <= SUB_REMAINDER);
}
static int vmo_ew_is_un(const uint sub)
{
    return (sub >= SUB_COPY && sub <= SUB_SIGN) ||
           (sub >= SUB_LOG1P && sub <= SUB_ROUND);
}
#define CMP(p, x, y) ((p)==0?(x)==(y):(p)==1?(x)!=(y):(p)==2?(x)<(y): \
                      (p)==3?(x)<=(y):(p)==4?(x)>(y):(x)>=(y))

static float vmo_load_f(__global uchar *arena, __global uchar **iop, uint base, uint dt, uint i);  /* fwd */

/* compare: operands read as adt, result written as 1-byte bool (uchar 0/1). */
static void vmo_cmp_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                     uint adt, uint lid, uint lsz)
{
    const uint n = t.p1, p = t.p2;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
    if (adt == DT_I32 || adt == DT_U32 || adt == DT_BOOL) {
        __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
#ifdef cl_khr_fp64
    } else if (adt == DT_F64) {
        __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
#endif
    } else if (adt == DT_F16 || adt == DT_BF16) {
        for (uint i = lo + lid; i < hi; i += lsz) {
            float x = vmo_load_f(arena, iop, t.a, adt, i), y = vmo_load_f(arena, iop, t.b, adt, i);
            d[i] = CMP(p, x, y) ? 1 : 0;
        }
    } else {
        __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
    }
}

/* select: pred (p3) is 1-byte bool; a/b/dst read/written as dt. */
static void vmo_select_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint dt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global const uchar *p = AP(const uchar, t.p3);
    if (dt == DT_I32 || dt == DT_U32) {
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
    } else if (dt == DT_BOOL) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a), *b = AP(const uchar, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
    } else if (dt == DT_F16 || dt == DT_BF16) {
        /* 2-byte select = copy the chosen 2-byte element (no arithmetic) */
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a), *b = AP(const ushort, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
#ifdef cl_khr_fp64
    } else if (dt == DT_F64) {
        __global double *d = AP(double, t.dst);
        __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
#endif
    } else {
        __global float *d = AP(float, t.dst);
        __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
    }
}

static void vmo_ew_tile_f32(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global float *d = AP(float, t.dst);
    __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
    if (vmo_ew_is_bin(sub))
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = vmo_ew_bin(sub, a[i], b[i]);
    else if (vmo_ew_is_un(sub))
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = vmo_ew_un(sub, a[i]);
    else if (sub == SUB_AFFINE) {
        /* d[i] = a[i]*s + t. a may alias d (in-place carry update): each work
         * item reads then writes the same element, so aliasing is safe. */
        const float s = as_float(t.p2), tt = as_float(t.p3);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = mad(a[i], s, tt);
    }
    else if (sub == SUB_FILL)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = as_float(t.p2);
    else if (sub == SUB_IOTA_FLAT)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (float)i;
    else if (sub == SUB_LTS) { if (lid == 0 && lo == 0) d[0] = (a[0] < b[0]) ? 1.0f : 0.0f; }
}

#ifdef cl_khr_fp64
static void vmo_ew_tile_f64(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global double *d = AP(double, t.dst);
    __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
    for (uint i = lo + lid; i < hi; i += lsz) {
        const double x = a[i], y = vmo_ew_is_bin(sub) ? b[i] : 0.0;
        double r;
        switch (sub) {
        case SUB_ADD: r = x + y; break;   case SUB_MUL: r = x * y; break;
        case SUB_SUB: r = x - y; break;   case SUB_DIV: r = x / y; break;
        case SUB_MAX: r = fmax(x, y); break; case SUB_MIN: r = fmin(x, y); break;
        case SUB_POW: r = pow(x, y); break;  case SUB_COPY: r = x; break;
        case SUB_NEG: r = -x; break;      case SUB_EXP: r = exp(x); break;
        case SUB_LOG: r = log(x); break;  case SUB_SQRT: r = sqrt(x); break;
        case SUB_RSQRT: r = rsqrt(x); break; case SUB_TANH: r = tanh(x); break;
        case SUB_ABS: r = fabs(x); break; case SUB_FLOOR: r = floor(x); break;
        case SUB_CEIL: r = ceil(x); break;
        case SUB_SIGN: r = x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : x); break;
        case SUB_ATAN2: r = atan2(x, y); break;
        case SUB_REMAINDER: r = fmod(x, y); break;
        case SUB_LOG1P: r = log1p(x); break;
        case SUB_EXPM1: r = expm1(x); break;
        case SUB_CBRT: r = cbrt(x); break;
        case SUB_SIN: r = sin(x); break;
        case SUB_COS: r = cos(x); break;
        case SUB_TAN: r = tan(x); break;
        case SUB_RINT: r = rint(x); break;
        case SUB_ROUND: r = round(x); break;
        default: r = x; break;
        }
        d[i] = r;
    }
}
#endif

static void vmo_ew_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global int *d = AP(int, t.dst);
    __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
    if (sub == SUB_FILL) { for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)t.p2; return; }
    if (sub == SUB_IOTA_FLAT) { for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)i; return; }
    const int needs_y = (sub <= SUB_POW) || sub == SUB_AND || sub == SUB_OR || sub == SUB_XOR;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const int x = a[i], y = needs_y ? b[i] : 0;
        int r;
        switch (sub) {
        case SUB_ADD: r = x + y; break;   case SUB_MUL: r = x * y; break;
        case SUB_SUB: r = x - y; break;   case SUB_DIV: r = y ? x / y : 0; break;
        case SUB_MAX: r = max(x, y); break; case SUB_MIN: r = min(x, y); break;
        case SUB_COPY: r = x; break;      case SUB_NEG: r = -x; break;
        case SUB_ABS: r = abs(x); break;
        case SUB_SIGN: r = x > 0 ? 1 : (x < 0 ? -1 : 0); break;
        case SUB_AND: r = x & y; break;   case SUB_OR: r = x | y; break;
        case SUB_XOR: r = x ^ y; break;   case SUB_NOT: r = ~x; break;
        default: r = x; break;
        }
        d[i] = r;
    }
}

/* bool (1-byte) elementwise: copy/fill/iota + and/or/xor/not (jax logical_*
 * on bool -> stablehlo.and/or/xor/not). Bool is stored as uchar 0/1, so
 * AND/OR/XOR are plain bitwise ops on the byte (only bit 0 is ever set), but
 * NOT must flip 0<->1 rather than bitwise-complement the whole byte. */
static void vmo_ew_tile_bool(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                         uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
    __global const uchar *a = AP(const uchar, t.a);
    if (sub == SUB_FILL) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (uchar)(t.p2 != 0);
    } else if (sub == SUB_NOT) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (uchar)(a[i] == 0);
    } else if (sub == SUB_AND || sub == SUB_OR || sub == SUB_XOR) {
        __global const uchar *b = AP(const uchar, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) {
            const uchar x = a[i] != 0, y = b[i] != 0;
            d[i] = (sub == SUB_AND) ? (uchar)(x & y)
                 : (sub == SUB_OR)  ? (uchar)(x | y) : (uchar)(x ^ y);
        }
    } else {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];  /* copy */
    }
}

/* f16/bf16: 2-byte storage, f32 arithmetic. LOAD/STORE are the half or bf16
 * accessors; only the floating subops apply (jax never emits int ops on them).*/
#define EW_HALF_TILE(NAME, LOAD, STORE)                                        \
static void NAME(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,             \
                 uint lid, uint lsz) {                                         \
    const uint sub = t.p0, n = t.p1;                                           \
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);                     \
    if (vmo_ew_is_bin(sub))                                                        \
        for (uint i = lo + lid; i < hi; i += lsz)                             \
            STORE(t.dst, i, vmo_ew_bin(sub, LOAD(t.a, i), LOAD(t.b, i)));          \
    else if (vmo_ew_is_un(sub))                                                    \
        for (uint i = lo + lid; i < hi; i += lsz)                             \
            STORE(t.dst, i, vmo_ew_un(sub, LOAD(t.a, i)));                         \
    else if (sub == SUB_FILL)                                                  \
        for (uint i = lo + lid; i < hi; i += lsz) STORE(t.dst, i, as_float(t.p2)); \
    else if (sub == SUB_IOTA_FLAT)                                             \
        for (uint i = lo + lid; i < hi; i += lsz) STORE(t.dst, i, (float)i);   \
}
EW_HALF_TILE(vmo_ew_tile_f16, LDH, STH)
EW_HALF_TILE(vmo_ew_tile_bf16, LDB, STB)

/* read element i of buffer `base` as a float, for any float-domain dtype. */
static float vmo_load_f(__global uchar *arena, __global uchar **iop, uint base, uint dt, uint i)
{
    switch (dt) {
    case DT_F16:  return LDH(base, i);
    case DT_BF16: return LDB(base, i);
    case DT_I32: case DT_U32: return (float)AP(const int, base)[i];
    case DT_BOOL: return (float)AP(const uchar, base)[i];
#ifdef cl_khr_fp64
    case DT_F64:  return (float)AP(const double, base)[i];
#endif
    default:      return AP(const float, base)[i];
    }
}

/* convert: read a[i] as adt, write dst[i] as dt with a C cast (float->int
 * truncates toward zero, matching stablehlo). Via a double intermediate where
 * fp64 exists (exact to 2^53); a float intermediate otherwise (4-byte types).
 * i64 beyond 2^53 loses precision — acceptable for now. */
static void vmo_convert_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                         uint dt, uint adt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    for (uint i = lo + lid; i < hi; i += lsz) {
#ifdef cl_khr_fp64
        double v;
        switch (adt) {
        case DT_I32: case DT_U32: v = (double)AP(const int, t.a)[i]; break;
        case DT_BOOL: v = (double)AP(const uchar, t.a)[i]; break;
        case DT_I64: v = (double)AP(const long, t.a)[i]; break;
        case DT_F64: v = AP(const double, t.a)[i]; break;
        case DT_F16: v = (double)LDH(t.a, i); break;
        case DT_BF16: v = (double)LDB(t.a, i); break;
        default: v = (double)AP(const float, t.a)[i]; break;
        }
        switch (dt) {
        case DT_I32: case DT_U32: AP(int, t.dst)[i] = (int)v; break;
        case DT_BOOL: AP(uchar, t.dst)[i] = (uchar)(v != 0.0); break;
        case DT_I64: AP(long, t.dst)[i] = (long)v; break;
        case DT_F64: AP(double, t.dst)[i] = v; break;
        case DT_F16: STH(t.dst, i, (float)v); break;
        case DT_BF16: STB(t.dst, i, (float)v); break;
        default: AP(float, t.dst)[i] = (float)v; break;
        }
#else
        float v = vmo_load_f(arena, iop, t.a, adt, i);
        switch (dt) {
        case DT_I32: case DT_U32: AP(int, t.dst)[i] = (int)v; break;
        case DT_BOOL: AP(uchar, t.dst)[i] = (uchar)(v != 0.0f); break;
        case DT_F16: STH(t.dst, i, v); break;
        case DT_BF16: STB(t.dst, i, v); break;
        default: AP(float, t.dst)[i] = v; break;
        }
#endif
    }
}

/* bitcast_convert: reinterpret the BITS of each element as the result dtype
 * (same element size). A typed memcpy of the 2/4/8-byte word — NOT a numeric
 * conversion (f32<->i32<->u32; f64<->i64). Width comes from the result dtype;
 * source and dest share the byte width by construction (checked in lowering). */
static void vmo_bitcast_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                         uint dt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    if (dt == DT_I64 || dt == DT_F64) {
        __global long *d = AP(long, t.dst);
        __global const long *a = AP(const long, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];
    } else if (dt == DT_F16 || dt == DT_BF16) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];
    } else {   /* 4-byte: f32/i32/u32 */
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];
    }
}

/* is_finite: operand read as adt (float-domain), result written as 1-byte
 * bool (uchar 0/1) — same mixed-dtype shape as vmo_cmp_tile. */
static void vmo_isfinite_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                          uint adt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
#ifdef cl_khr_fp64
    if (adt == DT_F64) {
        __global const double *a = AP(const double, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = isfinite(a[i]) ? 1 : 0;
        return;
    }
#endif
    if (adt == DT_F16 || adt == DT_BF16) {
        for (uint i = lo + lid; i < hi; i += lsz)
            d[i] = isfinite(vmo_load_f(arena, iop, t.a, adt, i)) ? 1 : 0;
        return;
    }
    __global const float *a = AP(const float, t.a);
    for (uint i = lo + lid; i < hi; i += lsz) d[i] = isfinite(a[i]) ? 1 : 0;
}

static void vmo_ew_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                    uint dt, uint adt, uint lid, uint lsz)
{
    if (t.p0 == SUB_CMP) { vmo_cmp_tile(arena, iop, t, tile, adt, lid, lsz); return; }
    if (t.p0 == SUB_SELECT) { vmo_select_tile(arena, iop, t, tile, dt, lid, lsz); return; }
    if (t.p0 == SUB_CONVERT) { vmo_convert_tile(arena, iop, t, tile, dt, adt, lid, lsz); return; }
    if (t.p0 == SUB_BITCAST) { vmo_bitcast_tile(arena, iop, t, tile, dt, lid, lsz); return; }
    if (t.p0 == SUB_ISFINITE) { vmo_isfinite_tile(arena, iop, t, tile, adt, lid, lsz); return; }
    switch (dt) {
    case DT_I32: case DT_U32: vmo_ew_tile_i32(arena, iop, t, tile, lid, lsz); break;
    case DT_BOOL:             vmo_ew_tile_bool(arena, iop, t, tile, lid, lsz); break;
    case DT_F16:              vmo_ew_tile_f16(arena, iop, t, tile, lid, lsz); break;
    case DT_BF16:             vmo_ew_tile_bf16(arena, iop, t, tile, lid, lsz); break;
#ifdef cl_khr_fp64
    case DT_F64:              vmo_ew_tile_f64(arena, iop, t, tile, lid, lsz); break;
#endif
    default:                  vmo_ew_tile_f32(arena, iop, t, tile, lid, lsz); break;
    }
}
