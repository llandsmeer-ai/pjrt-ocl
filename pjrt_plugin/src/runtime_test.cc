// Standalone runtime test: hand-encodes VMProgram v3 blobs (docs/vmprogram.md
// v2.1) and executes them on the VLIW engine — no PJRT/python involved.
//
// A: c = (a+b)*a across 4 lanes, two levels with barriers.
// B: while-loop doubling (scalar cond, on-device control entries), 2 lanes.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "runtime.h"

using namespace pjrt_ocl;

namespace {

constexpr uint32_t NOPE = 0xFFFFFFFFu, BARR = 0xFFFFFFFEu, WHIL = 0xFFFFFFFDu;
constexpr uint32_t FLAGN = 0xFFFFFFFFu;

struct Blob {
  std::vector<uint8_t> b;
  void U32(uint32_t v) { Raw(&v, 4); }
  void U64(uint64_t v) { Raw(&v, 8); }
  void Raw(const void* p, size_t n) {
    const auto* c = static_cast<const uint8_t*>(p);
    b.insert(b.end(), c, c + n);
  }
  void Align8() { while (b.size() % 8) b.push_back(0); }
};

struct Builder {
  uint32_t n_lanes;
  std::vector<std::pair<uint64_t, uint64_t>> bufs;  // offset,size
  uint64_t arena = 0;
  std::vector<uint32_t> inputs, outputs;
  std::vector<std::vector<uint64_t>> in_dims, out_dims;
  std::vector<std::pair<uint32_t, std::vector<float>>> consts;
  std::vector<int32_t> aux;
  std::vector<VmTask> tasks;
  std::vector<std::vector<VmEntry>> lanes;
  std::vector<uint32_t> root_lens;  // 0 = whole stream (no control flow)

  explicit Builder(uint32_t nl) : n_lanes(nl) {
    lanes.resize(nl);
    root_lens.assign(nl, 0);
  }
  uint32_t buf(uint64_t elems) {
    bufs.push_back({arena, elems * 4});
    arena = (arena + elems * 4 + 63) & ~uint64_t{63};
    return bufs.size() - 1;
  }
  uint32_t task(VmTask t) { tasks.push_back(t); return tasks.size() - 1; }
  void ent(uint32_t lane, VmEntry e) { lanes[lane].push_back(e); }
  void barrier_all() {
    for (auto& l : lanes)
      l.push_back({BARR, 0, 0, FLAGN, 0, FLAGN, 0, 0});
  }
  std::vector<uint8_t> Serialize() {
    Blob p;
    uint32_t n_entries = 0;
    for (auto& l : lanes) n_entries += l.size();
    p.U32(0x314D5056u); p.U32(3);
    p.U32(bufs.size()); p.U32(0); p.U32(consts.size()); p.U32(0);
    p.U32(inputs.size()); p.U32(outputs.size());
    p.U32(aux.size()); p.U32(0);     // n_aux, pad
    p.U64(arena);
    for (auto [o, s] : bufs) { p.U64(o); p.U64(s); p.U32(0); p.U32(0); }
    for (uint32_t i : inputs) p.U32(i);
    p.Align8();
    for (uint32_t i : outputs) p.U32(i);
    p.Align8();
    for (auto& d : in_dims) {
      p.U32(d.size()); p.U32(0);
      for (uint64_t v : d) p.U64(v);
      p.Align8();
    }
    for (auto& d : out_dims) {
      p.U32(d.size()); p.U32(0);
      for (uint64_t v : d) p.U64(v);
      p.Align8();
    }
    if (!aux.empty()) p.Raw(aux.data(), aux.size() * 4);
    p.Align8();
    for (auto& [id, vals] : consts) {
      p.U32(id); p.U32(vals.size() * 4);
      p.Raw(vals.data(), vals.size() * 4);
      p.Align8();
    }
    // no instrs; sched sections
    p.U32(tasks.size()); p.U32(n_entries); p.U32(0); p.U32(n_lanes);
    p.Raw(tasks.data(), tasks.size() * sizeof(VmTask));
    uint32_t off = 0;
    for (size_t l = 0; l < lanes.size(); ++l) {
      p.U32(off); p.U32(lanes[l].size());
      p.U32(root_lens[l] ? root_lens[l] : (uint32_t)lanes[l].size());
      p.U32(0);
      off += lanes[l].size();
    }
    p.Align8();
    for (auto& l : lanes) p.Raw(l.data(), l.size() * sizeof(VmEntry));
    return p.b;
  }
};

int RunProg(OclRuntime* rt, Builder& bld,
            const std::vector<const void*>& ins,
            std::vector<std::vector<uint8_t>>* outs) {
  std::string err;
  auto bytes = bld.Serialize();
  VmProgram prog;
  if (!VmProgram::Parse(bytes.data(), bytes.size(), &prog, &err)) {
    std::fprintf(stderr, "FAIL parse: %s\n", err.c_str());
    return 1;
  }
  auto lp = LoadedProgram::Load(rt, std::move(prog), &err);
  if (!lp) { std::fprintf(stderr, "FAIL load: %s\n", err.c_str()); return 1; }
  if (!lp->Execute(ins, outs, &err)) {
    std::fprintf(stderr, "FAIL execute: %s\n", err.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  std::string err;
  auto rt = OclRuntime::Create(&err);
  if (!rt) { std::fprintf(stderr, "FAIL runtime: %s\n", err.c_str()); return 1; }
  std::printf("device: %s / %s (%u lanes advertised)\n",
              rt->info().platform_name.c_str(), rt->info().device_name.c_str(),
              rt->ngroups());
  int fails = 0;

  {  // ---- A: c = (a+b)*a, 4 lanes, 2 levels ----
    // Size N to exactly LANES tiles at the device's ACTUAL EW tile size
    // (GPU 4096, CPU 16384 since decisions.md §22) — one tile per lane. A
    // hardcoded 16384 here left 3/4 of N uncovered on GPUs (EW_TS=4096).
    const uint32_t LANES = 4;
    const uint32_t EW_TS = rt->ew_ts();
    const uint32_t N = LANES * EW_TS;
    const uint32_t TILES = LANES;
    Builder b(LANES);
    uint32_t ba = b.buf(N), bb = b.buf(N), bt = b.buf(N), bc = b.buf(N);
    b.inputs = {ba, bb};
    b.outputs = {bc};
    b.in_dims = {{N}, {N}};
    b.out_dims = {{N}};
    uint32_t t0 = b.task({kTopEw, bt, ba, bb, 0 /*add*/, N, 0, 0});
    uint32_t t1 = b.task({kTopEw, bc, bt, ba, 1 /*mul*/, N, 0, 0});
    for (uint32_t l = 0; l < LANES && l < TILES; ++l)
      b.ent(l, {t0, l, l + 1, FLAGN, 0, FLAGN, 0, 0});
    b.barrier_all();
    for (uint32_t l = 0; l < LANES && l < TILES; ++l)
      b.ent(l, {t1, l, l + 1, FLAGN, 0, FLAGN, 0, 0});
    b.barrier_all();

    std::vector<float> a(N), bb_h(N);
    for (uint32_t i = 0; i < N; ++i) { a[i] = i % 251; bb_h[i] = (i % 83) + 1; }
    std::vector<std::vector<uint8_t>> outs;
    if (RunProg(rt.get(), b, {a.data(), bb_h.data()}, &outs)) return 1;
    const float* c = reinterpret_cast<const float*>(outs[0].data());
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i)
      if (c[i] != (a[i] + bb_h[i]) * a[i]) bad++;
    std::printf("A two-level EW (4 lanes): %s (%d bad)\n",
                bad ? "FAIL" : "PASS", bad);
    fails += bad != 0;
  }

  {  // ---- B: while (i < 10) { x += x; i += 1 } on 2 lanes ----
    // N = LANES tiles at the device's actual EW tile size (see block A).
    const uint32_t LANES = 2;
    const uint32_t N = LANES * rt->ew_ts();  // 2 tiles, one per lane
    Builder b(LANES);
    uint32_t bx = b.buf(N), bi = b.buf(1), bk = b.buf(1), bone = b.buf(1),
             bcond = b.buf(1);
    b.inputs = {bx};
    b.outputs = {bx};
    b.in_dims = {{N}};
    b.out_dims = {{N}};
    b.consts = {{bi, {0.0f}}, {bk, {10.0f}}, {bone, {1.0f}}};
    uint32_t tdbl = b.task({kTopEw, bx, bx, bx, 0 /*add*/, N, 0, 0});
    uint32_t tinc = b.task({kTopEw, bi, bi, bone, 0 /*add*/, 1, 0, 0});
    uint32_t tlts = b.task({kTopEw, bcond, bi, bk, 22 /*lts*/, 1, 0, 0});
    // Per-lane streams: [0]=WHILE, [1]=cond entry, [2..4)=body entries.
    // lane 0: cond = LTS; body = dbl tile 0 + inc.  lane 1: cond = NOP;
    // body = dbl tile 1 + NOP (uniform lengths keep ranges simple).
    VmEntry wh = {WHIL, /*cond_start=*/1, /*cond_len=*/1,
                  /*body_start=*/2, /*body_len=*/2, /*cond buf id=*/bcond,
                  0, 0};
    b.ent(0, wh);
    b.ent(0, {tlts, 0, 1, FLAGN, 0, FLAGN, 0, 0});
    b.ent(0, {tdbl, 0, 1, FLAGN, 0, FLAGN, 0, 0});
    b.ent(0, {tinc, 0, 1, FLAGN, 0, FLAGN, 0, 0});
    b.ent(1, wh);
    b.ent(1, {NOPE, 0, 0, FLAGN, 0, FLAGN, 0, 0});
    b.ent(1, {tdbl, 1, 2, FLAGN, 0, FLAGN, 0, 0});
    b.ent(1, {NOPE, 0, 0, FLAGN, 0, FLAGN, 0, 0});
    b.root_lens[0] = 1;  // root = just the WHILE; cond/body live beyond it
    b.root_lens[1] = 1;

    std::vector<float> x(N);
    for (uint32_t i = 0; i < N; ++i) x[i] = (i % 13) + 1;
    std::vector<std::vector<uint8_t>> outs;
    if (RunProg(rt.get(), b, {x.data()}, &outs)) return 1;
    const float* xo = reinterpret_cast<const float*>(outs[0].data());
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i)
      if (xo[i] != x[i] * 1024.0f) bad++;
    std::printf("B on-device while (2 lanes): %s (%d bad)\n",
                bad ? "FAIL" : "PASS", bad);
    fails += bad != 0;
  }

  if (std::getenv("REGION_POC")) {  // ---- C: §27/§28 register-resident GELU region ----
    // The TOP_MAP_REGION case ships in the default build (§28: recognizer-driven
    // OP_MAP_REGION). This device test hand-emits a GELU micro-program to check
    // the interpreter end-to-end; kept behind REGION_POC=1 to keep the default
    // runtime_test set small.
    auto fb = [](float f) { int32_t i; std::memcpy(&i, &f, 4); return i; };
    const uint32_t LANES = 8;
    const uint32_t EW_TS = rt->ew_ts();
    const uint32_t N = LANES * EW_TS;
    Builder b(LANES);
    uint32_t bx = b.buf(N), by = b.buf(N);
    b.inputs = {bx};
    b.outputs = {by};
    b.in_dims = {{N}};
    b.out_dims = {{N}};
    // GELU tail as a pure-map micro-DAG over float4 slots (slot0=x reused 4×):
    //   {kind, dst, a, b, s_bits, t_bits}; AFFINE = a*s+t.
    const int32_t MUL = 1, ADD = 0, TANH = 13, AFF = 40;
    b.aux = {
        /*in0*/ 0, /*in1*/ (int32_t)0xFFFF, /*n_micro*/ 8, /*out*/ 0,
        MUL, 1, 0, 0, 0, 0,                          // s1 = x*x
        MUL, 1, 1, 0, 0, 0,                          // s1 = x^3
        AFF, 1, 1, 0, fb(0.044715f), fb(0.0f),       // s1 = 0.044715*x^3
        ADD, 1, 1, 0, 0, 0,                          // s1 = x + 0.044715*x^3
        AFF, 1, 1, 0, fb(0.7978845608f), fb(0.0f),   // s1 = 0.79788*(...)
        TANH, 1, 1, 0, 0, 0,                         // s1 = tanh(...)
        AFF, 1, 1, 0, fb(0.5f), fb(0.5f),            // s1 = 0.5*(1+tanh)
        MUL, 0, 0, 1, 0, 0,                          // s0 = x * 0.5*(1+tanh)
    };
    uint32_t treg = b.task({13 /*kTopMapRegion*/, by, bx, bx, 0 /*desc off*/, N, 0, 0});
    for (uint32_t l = 0; l < LANES; ++l)
      b.ent(l, {treg, l, l + 1, FLAGN, 0, FLAGN, 0, 0});

    std::vector<float> x(N);
    for (uint32_t i = 0; i < N; ++i) x[i] = -4.0f + 8.0f * (float)(i % 101) / 100.0f;
    std::vector<std::vector<uint8_t>> outs;
    if (RunProg(rt.get(), b, {x.data()}, &outs)) return 1;
    const float* yo = reinterpret_cast<const float*>(outs[0].data());
    int bad = 0; float maxerr = 0;
    for (uint32_t i = 0; i < N; ++i) {
      const float xx = x[i];
      const float g = 0.5f * xx * (1.0f + std::tanh(0.7978845608f *
                        (xx + 0.044715f * xx * xx * xx)));
      const float e = std::fabs(yo[i] - g);
      if (e > maxerr) maxerr = e;
      if (e > 1e-5f) bad++;
    }
    std::printf("C region GELU (8 lanes, %u elems): %s (%d bad, maxerr %.2e)\n",
                N, bad ? "FAIL" : "PASS", bad, maxerr);
    fails += bad != 0;
  }

  std::printf("%s\n", fails ? "SOME FAILED" : "runtime_test: PASS");
  return fails ? 1 : 0;
}
