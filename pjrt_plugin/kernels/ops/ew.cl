/* Elementwise tile op (TOP_EW). subop in task.p0; n in p1; cmp pred / fill bits
 * in p2; select pred BYTE offset in p3. dst/a/b are BYTE offsets. Dispatched on
 * dtype `dt` by exec_tiles: f32 / f64 (floating) and i32 / u32 / bool (integer,
 * bool stored as i32 0/1). */

static float ew_bin(const uint sub, const float x, const float y)
{
    switch (sub) {
    case SUB_ADD: return x + y;
    case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;
    case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y);
    case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y);
    default:      return 0.0f;
    }
}

static float ew_un(const uint sub, const float x)
{
    switch (sub) {
    case SUB_COPY:  return x;
    case SUB_NEG:   return -x;
    case SUB_EXP:   return exp(x);
    case SUB_LOG:   return log(x);
    case SUB_SQRT:  return sqrt(x);
    case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH:  return tanh(x);
    case SUB_ABS:   return fabs(x);
    case SUB_FLOOR: return floor(x);
    case SUB_CEIL:  return ceil(x);
    case SUB_SIGN:  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x);
    default:        return 0.0f;
    }
}

static int cmp_pred(const uint p, const float x, const float y)
{
    switch (p) {
    case 0:  return x == y;
    case 1:  return x != y;
    case 2:  return x < y;
    case 3:  return x <= y;
    case 4:  return x > y;
    default: return x >= y;
    }
}

static void ew_tile_f32(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global float *d = AP(float, t.dst);
    __global const float *a = AP(const float, t.a);
    __global const float *b = AP(const float, t.b);
    if (sub <= SUB_POW) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = ew_bin(sub, a[i], b[i]);
    } else if (sub <= SUB_SIGN) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = ew_un(sub, a[i]);
    } else if (sub == SUB_FILL) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = as_float(t.p2);
    } else if (sub == SUB_IOTA_FLAT) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (float)i;
    } else if (sub == SUB_CMP) {
        for (uint i = lo + lid; i < hi; i += lsz)
            d[i] = cmp_pred(t.p2, a[i], b[i]) ? 1.0f : 0.0f;
    } else if (sub == SUB_SELECT) {
        __global const float *p = AP(const float, t.p3);
        for (uint i = lo + lid; i < hi; i += lsz)
            d[i] = p[i] != 0.0f ? a[i] : b[i];
    } else if (sub == SUB_LTS) {
        if (lid == 0 && lo == 0) d[0] = (a[0] < b[0]) ? 1.0f : 0.0f;
    }
}

#ifdef cl_khr_fp64
static void ew_tile_f64(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global double *d = AP(double, t.dst);
    __global const double *a = AP(const double, t.a);
    __global const double *b = AP(const double, t.b);
    if (sub == SUB_CMP) {
        /* cmp writes bool (i32 0/1), not f64 */
        __global int *di = AP(int, t.dst);
        for (uint i = lo + lid; i < hi; i += lsz) {
            const double x = a[i], y = b[i];
            int r;
            switch (t.p2) {
            case 0:  r = x == y; break; case 1: r = x != y; break;
            case 2:  r = x < y;  break; case 3: r = x <= y; break;
            case 4:  r = x > y;  break; default: r = x >= y; break;
            }
            di[i] = r;
        }
        return;
    }
    for (uint i = lo + lid; i < hi; i += lsz) {
        const double x = a[i], y = (sub <= SUB_POW || sub == SUB_SELECT) ? b[i] : 0.0;
        double r = 0.0;
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
        case SUB_SELECT: {
            __global const double *p = AP(const double, t.p3);
            r = p[i] != 0.0 ? x : y; break;
        }
        default: r = x; break;
        }
        d[i] = r;
    }
}
#endif

/* Integer elementwise (i32/u32/bool). bool is i32 0/1. Transcendentals don't
 * apply to integers (jax never emits them on int types). */
static void ew_tile_i32(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global int *d = AP(int, t.dst);
    __global const int *a = AP(const int, t.a);
    __global const int *b = AP(const int, t.b);
    if (sub == SUB_SELECT) {
        __global const int *p = AP(const int, t.p3);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] != 0 ? a[i] : b[i];
        return;
    }
    if (sub == SUB_FILL) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)t.p2;
        return;
    }
    if (sub == SUB_IOTA_FLAT) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)i;
        return;
    }
    if (sub == SUB_CMP) {
        for (uint i = lo + lid; i < hi; i += lsz) {
            const int x = a[i], y = b[i];
            int r;
            switch (t.p2) {
            case 0:  r = x == y; break; case 1: r = x != y; break;
            case 2:  r = x < y;  break; case 3: r = x <= y; break;
            case 4:  r = x > y;  break; default: r = x >= y; break;
            }
            d[i] = r;
        }
        return;
    }
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

static void ew_tile(__global uchar *arena, const task_t t, uint tile,
                    uint dt, uint lid, uint lsz)
{
    switch (dt) {
    case DT_I32: case DT_U32: case DT_BOOL:
        ew_tile_i32(arena, t, tile, lid, lsz); break;
#ifdef cl_khr_fp64
    case DT_F64:
        ew_tile_f64(arena, t, tile, lid, lsz); break;
#endif
    default:
        ew_tile_f32(arena, t, tile, lid, lsz); break;
    }
}
