/* Elementwise tile op (TOP_EW). subop in task.p0; n in p1; cmp pred / fill bits
 * in p2; select pred offset in p3. One tile = EW_TS elements. */

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

/* f32 elementwise. dt selects the arithmetic type (Tier 1 + f64); dst/a/b/p3
 * are BYTE offsets into the arena. */
static void ew_tile_f32(__global uchar *arena, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS;
    const uint hi = min(lo + EW_TS, n);
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
        for (uint i = lo + lid; i < hi; i += lsz) {
            const float x = a[i], y = b[i];
            int r;
            switch (t.p2) {
            case 0:  r = x == y; break;
            case 1:  r = x != y; break;
            case 2:  r = x < y;  break;
            case 3:  r = x <= y; break;
            case 4:  r = x > y;  break;
            default: r = x >= y; break;
            }
            d[i] = r ? 1.0f : 0.0f;
        }
    } else if (sub == SUB_SELECT) {
        __global const float *p = AP(const float, t.p3);
        for (uint i = lo + lid; i < hi; i += lsz)
            d[i] = p[i] != 0.0f ? a[i] : b[i];
    } else if (sub == SUB_LTS) {
        if (lid == 0 && lo == 0) d[0] = (a[0] < b[0]) ? 1.0f : 0.0f;
    }
}

static void ew_tile(__global uchar *arena, const task_t t, uint tile,
                    uint dt, uint lid, uint lsz)
{
    ew_tile_f32(arena, t, tile, lid, lsz);
}
