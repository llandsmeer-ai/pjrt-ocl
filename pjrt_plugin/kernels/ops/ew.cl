/* Elementwise tile op (TOP_EW). subop in task.p0; n in p1; cmp pred / fill bits
 * in p2; select pred BYTE offset in p3. dst/a/b are BYTE offsets.
 * Dispatched by exec_tiles on result dtype `dt` and operand dtype `adt`
 * (compare: operands adt -> bool result; select: bool pred -> operands dt).
 * bool is 1-byte (uchar 0/1), matching jax PRED. */

static float ew_bin(const uint sub, const float x, const float y)
{
    switch (sub) {
    case SUB_ADD: return x + y;   case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;   case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y); case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y); default: return 0.0f;
    }
}
static float ew_un(const uint sub, const float x)
{
    switch (sub) {
    case SUB_COPY: return x;      case SUB_NEG: return -x;
    case SUB_EXP: return exp(x);  case SUB_LOG: return log(x);
    case SUB_SQRT: return sqrt(x); case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH: return tanh(x); case SUB_ABS: return fabs(x);
    case SUB_FLOOR: return floor(x); case SUB_CEIL: return ceil(x);
    case SUB_SIGN: return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x);
    default: return 0.0f;
    }
}
#define CMP(p, x, y) ((p)==0?(x)==(y):(p)==1?(x)!=(y):(p)==2?(x)<(y): \
                      (p)==3?(x)<=(y):(p)==4?(x)>(y):(x)>=(y))

static float load_f(__global uchar *arena, uint base, uint dt, uint i);  /* fwd */

/* compare: operands read as adt, result written as 1-byte bool (uchar 0/1). */
static void cmp_tile(__global uchar *arena, const task_t t, uint tile,
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
            float x = load_f(arena, t.a, adt, i), y = load_f(arena, t.b, adt, i);
            d[i] = CMP(p, x, y) ? 1 : 0;
        }
    } else {
        __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
    }
}

/* select: pred (p3) is 1-byte bool; a/b/dst read/written as dt. */
static void select_tile(__global uchar *arena, const task_t t, uint tile,
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

static void ew_tile_f32(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global float *d = AP(float, t.dst);
    __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
    if (sub <= SUB_POW)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = ew_bin(sub, a[i], b[i]);
    else if (sub <= SUB_SIGN)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = ew_un(sub, a[i]);
    else if (sub == SUB_FILL)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = as_float(t.p2);
    else if (sub == SUB_IOTA_FLAT)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (float)i;
    else if (sub == SUB_LTS) { if (lid == 0 && lo == 0) d[0] = (a[0] < b[0]) ? 1.0f : 0.0f; }
}

#ifdef cl_khr_fp64
static void ew_tile_f64(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global double *d = AP(double, t.dst);
    __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
    for (uint i = lo + lid; i < hi; i += lsz) {
        const double x = a[i], y = (sub <= SUB_POW) ? b[i] : 0.0;
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
        default: r = x; break;
        }
        d[i] = r;
    }
}
#endif

static void ew_tile_i32(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global int *d = AP(int, t.dst);
    __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
    if (sub == SUB_FILL) { for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)t.p2; return; }
    if (sub == SUB_IOTA_FLAT) { for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)i; return; }
    for (uint i = lo + lid; i < hi; i += lsz) {
        const int x = a[i], y = (sub <= SUB_POW) ? b[i] : 0;
        int r;
        switch (sub) {
        case SUB_ADD: r = x + y; break;   case SUB_MUL: r = x * y; break;
        case SUB_SUB: r = x - y; break;   case SUB_DIV: r = y ? x / y : 0; break;
        case SUB_MAX: r = max(x, y); break; case SUB_MIN: r = min(x, y); break;
        case SUB_COPY: r = x; break;      case SUB_NEG: r = -x; break;
        case SUB_ABS: r = abs(x); break;
        case SUB_SIGN: r = x > 0 ? 1 : (x < 0 ? -1 : 0); break;
        default: r = x; break;
        }
        d[i] = r;
    }
}

/* bool (1-byte) elementwise: copy/fill/iota (logical ops arrive as and/or/xor,
 * added when those ops land). */
static void ew_tile_bool(__global uchar *arena, const task_t t, uint tile,
                         uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
    __global const uchar *a = AP(const uchar, t.a);
    if (sub == SUB_FILL) for (uint i = lo + lid; i < hi; i += lsz) d[i] = (uchar)(t.p2 != 0);
    else for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];  /* copy */
}

/* f16/bf16: 2-byte storage, f32 arithmetic. LOAD/STORE are the half or bf16
 * accessors; only the floating subops apply (jax never emits int ops on them).*/
#define EW_HALF_TILE(NAME, LOAD, STORE)                                        \
static void NAME(__global uchar *arena, const task_t t, uint tile,             \
                 uint lid, uint lsz) {                                         \
    const uint sub = t.p0, n = t.p1;                                           \
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);                     \
    if (sub <= SUB_POW)                                                        \
        for (uint i = lo + lid; i < hi; i += lsz)                             \
            STORE(t.dst, i, ew_bin(sub, LOAD(t.a, i), LOAD(t.b, i)));          \
    else if (sub <= SUB_SIGN)                                                  \
        for (uint i = lo + lid; i < hi; i += lsz)                             \
            STORE(t.dst, i, ew_un(sub, LOAD(t.a, i)));                         \
    else if (sub == SUB_FILL)                                                  \
        for (uint i = lo + lid; i < hi; i += lsz) STORE(t.dst, i, as_float(t.p2)); \
    else if (sub == SUB_IOTA_FLAT)                                             \
        for (uint i = lo + lid; i < hi; i += lsz) STORE(t.dst, i, (float)i);   \
}
EW_HALF_TILE(ew_tile_f16, LDH, STH)
EW_HALF_TILE(ew_tile_bf16, LDB, STB)

/* read element i of buffer `base` as a float, for any float-domain dtype. */
static float load_f(__global uchar *arena, uint base, uint dt, uint i)
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
static void convert_tile(__global uchar *arena, const task_t t, uint tile,
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
        float v = load_f(arena, t.a, adt, i);
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

static void ew_tile(__global uchar *arena, const task_t t, uint tile,
                    uint dt, uint adt, uint lid, uint lsz)
{
    if (t.p0 == SUB_CMP) { cmp_tile(arena, t, tile, adt, lid, lsz); return; }
    if (t.p0 == SUB_SELECT) { select_tile(arena, t, tile, dt, lid, lsz); return; }
    if (t.p0 == SUB_CONVERT) { convert_tile(arena, t, tile, dt, adt, lid, lsz); return; }
    switch (dt) {
    case DT_I32: case DT_U32: ew_tile_i32(arena, t, tile, lid, lsz); break;
    case DT_BOOL:             ew_tile_bool(arena, t, tile, lid, lsz); break;
    case DT_F16:              ew_tile_f16(arena, t, tile, lid, lsz); break;
    case DT_BF16:             ew_tile_bf16(arena, t, tile, lid, lsz); break;
#ifdef cl_khr_fp64
    case DT_F64:              ew_tile_f64(arena, t, tile, lid, lsz); break;
#endif
    default:                  ew_tile_f32(arena, t, tile, lid, lsz); break;
    }
}
