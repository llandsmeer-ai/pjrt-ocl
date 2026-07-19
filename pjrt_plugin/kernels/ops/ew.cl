/* Elementwise tile op (TOP_EW). subop in task.p0; n in p1; cmp pred / fill bits
 * in p2; select pred BYTE offset in p3. dst/a/b are BYTE offsets.
 * Dispatched by vmo_exec_tiles on result dtype `dt` and operand dtype `adt`
 * (compare: operands adt -> bool result; select: bool pred -> operands dt).
 * bool is 1-byte (uchar 0/1), matching jax PRED. */

/* GELU tanh-approx (§19b/§24): 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3))).
 * One scalar + float8/float4 twin so it rides all three EW paths (scalar tail,
 * CPU float8 chunk, GPU float4 fast path). Matches python _gelu_np exactly. */
#define VMO_GELU_BODY(x) \
    ((float)0.5f * (x) * ((float)1.0f + \
        tanh((float)0.7978845608f * ((x) + (float)0.044715f * (x) * (x) * (x)))))
static float  vmo_gelu1(const float  x) { return VMO_GELU_BODY(x); }
static float8 vmo_gelu8(const float8 x)
{ return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }
static float4 vmo_gelu4(const float4 x)
{ return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

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
    case SUB_GELU: return vmo_gelu1(x);
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
           (sub >= SUB_LOG1P && sub <= SUB_ROUND) ||
           sub == SUB_GELU;
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

#ifdef VMO_CPU_TILES
/* float8 twins of vmo_ew_bin/un — every builtin used has a vector overload.
 * SUB_SIGN is spelled with select() (vector-safe ternary). */
static float8 vmo_ew_bin8(const uint sub, const float8 x, const float8 y)
{
    switch (sub) {
    case SUB_ADD: return x + y;   case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;   case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y); case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y);
    case SUB_ATAN2: return atan2(x, y);
    case SUB_REMAINDER: return fmod(x, y);
    default: return (float8)(0.0f);
    }
}
static float8 vmo_ew_un8(const uint sub, const float8 x)
{
    switch (sub) {
    case SUB_COPY: return x;      case SUB_NEG: return -x;
    case SUB_EXP: return exp(x);  case SUB_LOG: return log(x);
    case SUB_SQRT: return sqrt(x); case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH: return tanh(x); case SUB_ABS: return fabs(x);
    case SUB_FLOOR: return floor(x); case SUB_CEIL: return ceil(x);
    case SUB_SIGN: return select(select((float8)(1.0f), (float8)(-1.0f),
                                        x < (float8)(0.0f)),
                                 x, x == (float8)(0.0f) | isnan(x));
    case SUB_LOG1P: return log1p(x);
    case SUB_EXPM1: return expm1(x);
    case SUB_CBRT: return cbrt(x);
    case SUB_SIN: return sin(x);
    case SUB_COS: return cos(x);
    case SUB_TAN: return tan(x);
    case SUB_RINT: return rint(x);
    case SUB_ROUND: return round(x);
    case SUB_GELU: return vmo_gelu8(x);
    default: return (float8)(0.0f);
    }
}
#endif

/* Strided VIEW index: output element i -> source flat index, via an aux
 * descriptor {rank, out_dims[rank], in_strides[rank], src_off} — the SAME map
 * gather uses. Lets an elementwise op read a folded broadcast/transpose/slice
 * operand in place (docs/decisions.md: access-map fusion) instead of the
 * producer materializing a whole tensor. */
static uint vmo_view_idx(__global const int *aux, uint off, uint i)
{
    __global const int *x = aux + off;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    int rem = (int)i, o = x[1 + 2 * rank];
    for (int e = rank - 1; e >= 0; --e) {
        o += (rem % dims[e]) * strides[e];
        rem /= dims[e];
    }
    return (uint)o;
}

/* float4 twins of vmo_ew_bin/un for the GPU vector fast path — every builtin
 * used has a vector overload; SUB_SIGN spelled with select() as in the float8
 * CPU twins. */
static float4 vmo_ew_bin4(const uint sub, const float4 x, const float4 y)
{
    switch (sub) {
    case SUB_ADD: return x + y;   case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;   case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y); case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y);
    case SUB_ATAN2: return atan2(x, y);
    case SUB_REMAINDER: return fmod(x, y);
    default: return (float4)(0.0f);
    }
}
static float4 vmo_ew_un4(const uint sub, const float4 x)
{
    switch (sub) {
    case SUB_COPY: return x;      case SUB_NEG: return -x;
    case SUB_EXP: return exp(x);  case SUB_LOG: return log(x);
    case SUB_SQRT: return sqrt(x); case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH: return tanh(x); case SUB_ABS: return fabs(x);
    case SUB_FLOOR: return floor(x); case SUB_CEIL: return ceil(x);
    case SUB_SIGN: return select(select((float4)(1.0f), (float4)(-1.0f),
                                        x < (float4)(0.0f)),
                                 x, x == (float4)(0.0f) | isnan(x));
    case SUB_LOG1P: return log1p(x);
    case SUB_EXPM1: return expm1(x);
    case SUB_CBRT: return cbrt(x);
    case SUB_SIN: return sin(x);
    case SUB_COS: return cos(x);
    case SUB_TAN: return tan(x);
    case SUB_RINT: return rint(x);
    case SUB_ROUND: return round(x);
    case SUB_GELU: return vmo_gelu4(x);
    default: return (float4)(0.0f);
    }
}

static void vmo_ew_tile_f32(__global uchar *arena, __global uchar **iop,
                        __global const int *aux, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global float *d = AP(float, t.dst);
    __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
    /* Viewed operands: only plain float binary/unary subops carry a/b view
     * aux-offsets in p2/p3 (cmp/select/affine/fill reuse those fields). */
    const uint av = (vmo_ew_is_bin(sub) || vmo_ew_is_un(sub)) ? t.p2 : 0u;
    const uint bv = (vmo_ew_is_bin(sub) || vmo_ew_is_un(sub)) ? t.p3 : 0u;
    if (av || bv) {          /* strided (gathered) read; not vectorizable */
        const int isbin = vmo_ew_is_bin(sub);
        for (uint i = lo + lid; i < hi; i += lsz) {
            const float x = av ? a[vmo_view_idx(aux, av - 1u, i)] : a[i];
            d[i] = isbin ? vmo_ew_bin(sub, x,
                             bv ? b[vmo_view_idx(aux, bv - 1u, i)] : b[i])
                         : vmo_ew_un(sub, x);
        }
        return;
    }
#ifdef VMO_CPU_TILES
    /* CPU shape (poc/09 a4, decisions.md #11): a contiguous chunk per WI,
     * explicit float8 body + scalar tail. PoCL's work-group vectorizer cannot
     * vectorize ANY explicit in-kernel loop (measured 5 GB/s scalar vs
     * 46 GB/s this shape); on GPUs this define is never set — they keep the
     * coalesced stride-lsz loop below. Chunking preserves the aliasing
     * guarantee for SUB_AFFINE (each WI reads/writes only its own elements). */
    const uint chunk = (hi - lo + lsz - 1) / lsz;
    const uint clo = min(lo + lid * chunk, hi), chi = min(clo + chunk, hi);
    uint i = clo;
    if (vmo_ew_is_bin(sub)) {
        for (; i + 8 <= chi; i += 8)
            vstore8(vmo_ew_bin8(sub, vload8(0, a + i), vload8(0, b + i)), 0, d + i);
        for (; i < chi; ++i) d[i] = vmo_ew_bin(sub, a[i], b[i]);
    } else if (vmo_ew_is_un(sub)) {
        for (; i + 8 <= chi; i += 8)
            vstore8(vmo_ew_un8(sub, vload8(0, a + i)), 0, d + i);
        for (; i < chi; ++i) d[i] = vmo_ew_un(sub, a[i]);
    } else if (sub == SUB_AFFINE) {
        const float s = as_float(t.p2), tt = as_float(t.p3);
        for (; i + 8 <= chi; i += 8)
            vstore8(mad(vload8(0, a + i), (float8)(s), (float8)(tt)), 0, d + i);
        for (; i < chi; ++i) d[i] = mad(a[i], s, tt);
    } else if (sub == SUB_FILL) {
        const float v = as_float(t.p2);
        for (; i + 8 <= chi; i += 8) vstore8((float8)(v), 0, d + i);
        for (; i < chi; ++i) d[i] = v;
    } else if (sub == SUB_IOTA_FLAT) {
        for (; i < chi; ++i) d[i] = (float)i;
    }
    else if (sub == SUB_LTS) { if (lid == 0 && lo == 0) d[0] = (a[0] < b[0]) ? 1.0f : 0.0f; }
#else
    /* GPU shape: the plain stride-lsz scalar loop is LATENCY-bound — a 16K
     * tile = 64 dependent iterations/thread ~= 15 us on Blackwell (13 GB/s
     * per lane). float4 lanes + 2x manual unroll cut that to 8 wider, more
     * independent memory round trips per thread. Applies when all resolved
     * operand pointers are 16B-aligned (arena allocs and IO ports are; tile
     * origin lo is a multiple of EW_TS); the last tile's non-multiple-of-4
     * remainder runs the scalar tail below. Aliasing (SUB_AFFINE in-place
     * carry) stays safe: each work item reads then writes only its own
     * elements. */
    const int isb = vmo_ew_is_bin(sub), isu = vmo_ew_is_un(sub);
    const uintptr_t amask = (uintptr_t)d |
        (sub == SUB_FILL ? (uintptr_t)0 : (uintptr_t)a) |
        (isb ? (uintptr_t)b : (uintptr_t)0);
    const int vec4 = (isb | isu | (sub == SUB_AFFINE) | (sub == SUB_FILL)) &&
        !(amask & (uintptr_t)15);
    if (vec4) {
        __global float4 *d4 = (__global float4 *)d;
        __global const float4 *a4 = (__global const float4 *)a;
        __global const float4 *b4 = (__global const float4 *)b;
        const uint lo4 = lo >> 2, hi4 = lo4 + ((hi - lo) >> 2);
        uint i = lo4 + lid;
        if (isb) {
            for (; i + lsz < hi4; i += 2u * lsz) {
                const float4 x0 = a4[i], y0 = b4[i];
                const float4 x1 = a4[i + lsz], y1 = b4[i + lsz];
                d4[i] = vmo_ew_bin4(sub, x0, y0);
                d4[i + lsz] = vmo_ew_bin4(sub, x1, y1);
            }
            if (i < hi4) d4[i] = vmo_ew_bin4(sub, a4[i], b4[i]);
        } else if (isu) {
            for (; i + lsz < hi4; i += 2u * lsz) {
                const float4 x0 = a4[i], x1 = a4[i + lsz];
                d4[i] = vmo_ew_un4(sub, x0);
                d4[i + lsz] = vmo_ew_un4(sub, x1);
            }
            if (i < hi4) d4[i] = vmo_ew_un4(sub, a4[i]);
        } else if (sub == SUB_AFFINE) {
            const float4 s4 = (float4)(as_float(t.p2));
            const float4 t4 = (float4)(as_float(t.p3));
            for (; i + lsz < hi4; i += 2u * lsz) {
                const float4 x0 = a4[i], x1 = a4[i + lsz];
                d4[i] = mad(x0, s4, t4);
                d4[i + lsz] = mad(x1, s4, t4);
            }
            if (i < hi4) d4[i] = mad(a4[i], s4, t4);
        } else { /* SUB_FILL */
            const float4 v4 = (float4)(as_float(t.p2));
            for (; i < hi4; i += lsz) d4[i] = v4;
        }
        /* scalar tail: elements past the last full float4 of this tile */
        for (uint j = lo + ((hi - lo) & ~3u) + lid; j < hi; j += lsz)
            d[j] = isb ? vmo_ew_bin(sub, a[j], b[j])
                 : isu ? vmo_ew_un(sub, a[j])
                 : sub == SUB_AFFINE ? mad(a[j], as_float(t.p2), as_float(t.p3))
                 : as_float(t.p2);
        return;
    }
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
#endif
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

static void vmo_ew_tile(__global uchar *arena, __global uchar **iop,
                    __global const int *aux, const task_t t, uint tile,
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
    default:                  vmo_ew_tile_f32(arena, iop, aux, t, tile, lid, lsz); break;
    }
}
