// Standalone runtime test: hand-encodes a VMProgram v1 blob (docs/vmprogram.md)
// and executes it — validates parser + loader + VM with no PJRT/python.
#include <cstdio>
#include <cstring>
#include <vector>

#include "runtime.h"

using namespace pjrt_ocl;

namespace {

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

}  // namespace

int main() {
  // Program: out = (a + b) * a  on f32[8], plus a while-loop doubling:
  //   i = 0; while (i < 3) { out = out + out; i = i + 1 }
  // Buffers: 0=a(in) 1=b(in) 2=out 3=i 4=three 5=one 6=cond  (64B-aligned)
  const uint32_t N = 8;
  Blob p;
  const uint32_t n_buffers = 7, n_consts = 2, n_inputs = 2, n_outputs = 1;
  const uint32_t main_len = 4;   // add, mul, fill i=0, while
  const uint32_t n_instrs = 7;   // + cond lts, body add, body add-one
  p.U32(0x314D5056u); p.U32(1);
  p.U32(n_buffers); p.U32(n_instrs); p.U32(n_consts); p.U32(main_len);
  p.U32(n_inputs); p.U32(n_outputs);
  uint64_t off = 0;
  std::vector<std::pair<uint64_t, uint64_t>> bufs;
  for (uint64_t size : std::vector<uint64_t>{N * 4, N * 4, N * 4, 4, 4, 4, 4}) {
    bufs.push_back({off, size});
    off = (off + size + 63) & ~uint64_t{63};
  }
  p.U64(off);  // arena_bytes
  for (auto [o, s] : bufs) { p.U64(o); p.U64(s); p.U32(0); p.U32(0); }
  p.U32(0); p.U32(1); p.Align8();          // inputs
  p.U32(2); p.Align8();                    // outputs
  for (int i = 0; i < 2; ++i) { p.U32(1); p.U32(0); p.U64(N); p.Align8(); }  // in shapes
  p.U32(1); p.U32(0); p.U64(N); p.Align8();                                  // out shape
  const float three = 3.0f, one = 1.0f;
  p.U32(4); p.U32(4); p.Raw(&three, 4); p.Align8();   // const: three
  p.U32(5); p.U32(4); p.Raw(&one, 4); p.Align8();     // const: one
  auto instr = [&](uint32_t op, uint32_t dst, uint32_t a, uint32_t b,
                   uint32_t n, uint32_t imm) {
    p.U32(op); p.U32(dst); p.U32(a); p.U32(b); p.U32(n); p.U32(imm);
    p.U32(0); p.U32(0);
  };
  // main [0,4)
  instr(kAddF32, 2, 0, 1, N, 0);
  instr(kMulF32, 2, 2, 0, N, 0);
  instr(kFillF32, 3, 0, 0, 1, 0);          // i = 0.0f (bits of 0.0 = 0)
  instr(kWhile, 6, 4, 1, 5, 2);            // cond=[4,5) body=[5,7)
  // cond [4]
  instr(kLtsF32, 6, 3, 4, 1, 0);
  // body [5,7)
  instr(kAddF32, 2, 2, 2, N, 0);
  instr(kAddF32, 3, 3, 5, 1, 0);

  std::string err;
  VmProgram prog;
  if (!VmProgram::Parse(p.b.data(), p.b.size(), &prog, &err)) {
    std::fprintf(stderr, "FAIL parse: %s\n", err.c_str());
    return 1;
  }
  auto rt = OclRuntime::Create(&err);
  if (!rt) { std::fprintf(stderr, "FAIL runtime: %s\n", err.c_str()); return 1; }
  std::printf("device: %s / %s\n", rt->info().platform_name.c_str(),
              rt->info().device_name.c_str());
  auto lp = LoadedProgram::Load(rt.get(), std::move(prog), &err);
  if (!lp) { std::fprintf(stderr, "FAIL load: %s\n", err.c_str()); return 1; }

  float a[N], b[N];
  for (uint32_t i = 0; i < N; ++i) { a[i] = i + 1.0f; b[i] = 2.0f * i; }
  std::vector<std::vector<uint8_t>> outputs;
  if (!lp->Execute({a, b}, &outputs, &err)) {
    std::fprintf(stderr, "FAIL execute: %s\n", err.c_str());
    return 1;
  }
  int bad = 0;
  const float* out = reinterpret_cast<const float*>(outputs[0].data());
  for (uint32_t i = 0; i < N; ++i) {
    float want = (a[i] + b[i]) * a[i] * 8.0f;  // 2^3 from the while loop
    if (out[i] != want) {
      ++bad;
      std::fprintf(stderr, "  out[%u]=%g want %g\n", i, out[i], want);
    }
  }
  std::printf("runtime_test: %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
