#include "runtime.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "vm_cl_source.h"  // generated: kVmClSource

namespace pjrt_ocl {
namespace {

std::string ClInfoStr(cl_device_id dev, cl_device_info param) {
  size_t size = 0;
  if (clGetDeviceInfo(dev, param, 0, nullptr, &size) != CL_SUCCESS || !size)
    return "";
  std::string s(size, '\0');
  clGetDeviceInfo(dev, param, size, s.data(), nullptr);
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

}  // namespace

// ---- VmProgram::Parse -------------------------------------------------------

namespace {

class Reader {
 public:
  Reader(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  bool U32(uint32_t* v) { return Raw(v, 4); }
  bool U64(uint64_t* v) { return Raw(v, 8); }
  bool Bytes(void* dst, size_t n) { return Raw(dst, n); }
  bool Align8() {
    size_t target = (pos_ + 7) & ~size_t{7};
    if (target > n_) return false;
    pos_ = target;
    return true;
  }
  size_t remaining() const { return n_ - pos_; }

 private:
  bool Raw(void* dst, size_t n) {
    if (pos_ + n > n_) return false;
    std::memcpy(dst, d_ + pos_, n);
    pos_ += n;
    return true;
  }
  const uint8_t* d_;
  size_t n_, pos_ = 0;
};

}  // namespace

bool VmProgram::Parse(const uint8_t* data, size_t len, VmProgram* out,
                      std::string* err) {
  Reader r(data, len);
  auto fail = [&](const std::string& m) {
    *err = "VMProgram parse error: " + m;
    return false;
  };

  uint32_t magic, version, n_buffers, n_instrs, n_consts, main_len, n_inputs,
      n_outputs, n_aux, hpad;
  if (!r.U32(&magic) || !r.U32(&version)) return fail("truncated header");
  if (magic != 0x314D5056u) return fail("bad magic");
  if (version != 3)
    return fail("unsupported version " + std::to_string(version) +
                " (executor expects 3)");
  if (!r.U32(&n_buffers) || !r.U32(&n_instrs) || !r.U32(&n_consts) ||
      !r.U32(&main_len) || !r.U32(&n_inputs) || !r.U32(&n_outputs) ||
      !r.U32(&n_aux) || !r.U32(&hpad) || !r.U64(&out->arena_bytes))
    return fail("truncated header");
  out->main_len = main_len;

  out->buffers.resize(n_buffers);
  for (auto& b : out->buffers) {
    uint32_t pad;
    if (!r.U64(&b.arena_byte_offset) || !r.U64(&b.size_bytes) ||
        !r.U32(&b.dtype) || !r.U32(&pad))
      return fail("truncated buffer table");
    if (b.dtype > kDtMax) return fail("unsupported dtype " + std::to_string(b.dtype));
    if (b.arena_byte_offset % 4 || b.arena_byte_offset + b.size_bytes > out->arena_bytes)
      return fail("buffer outside arena / misaligned");
  }

  auto read_ids = [&](uint32_t n, std::vector<uint32_t>* v) {
    v->resize(n);
    for (auto& id : *v)
      if (!r.U32(&id) || id >= n_buffers) return false;
    return r.Align8();
  };
  if (!read_ids(n_inputs, &out->inputs)) return fail("bad input map");
  if (!read_ids(n_outputs, &out->outputs)) return fail("bad output map");

  auto read_shapes = [&](size_t n, std::vector<std::vector<int64_t>>* v) {
    v->resize(n);
    for (auto& dims : *v) {
      uint32_t rank, pad;
      if (!r.U32(&rank) || !r.U32(&pad) || rank > 16) return false;
      dims.resize(rank);
      for (auto& d : dims) {
        uint64_t u;
        if (!r.U64(&u)) return false;
        dims[&d - dims.data()] = static_cast<int64_t>(u);
      }
      if (!r.Align8()) return false;
    }
    return true;
  };
  if (!read_shapes(n_inputs, &out->input_dims)) return fail("bad input shapes");
  if (!read_shapes(n_outputs, &out->output_dims)) return fail("bad output shapes");

  out->aux.resize(n_aux);
  if (n_aux && !r.Bytes(out->aux.data(), n_aux * 4)) return fail("truncated aux");
  if (!r.Align8()) return fail("aux align");

  out->consts.resize(n_consts);
  for (auto& [id, bytes] : out->consts) {
    uint32_t byte_len;
    if (!r.U32(&id) || !r.U32(&byte_len) || id >= n_buffers)
      return fail("bad const entry");
    if (byte_len > out->buffers[id].size_bytes) return fail("const too big");
    bytes.resize(byte_len);
    if (!r.Bytes(bytes.data(), byte_len) || !r.Align8())
      return fail("truncated const data");
  }

  if (main_len > n_instrs) return fail("main_len > n_instrs");
  out->instrs.resize(n_instrs);
  if (n_instrs && !r.Bytes(out->instrs.data(), n_instrs * sizeof(VmInstr)))
    return fail("truncated instructions");
  if (!r.Align8()) return fail("instr align");

  // ---- v2.1 schedule sections: what the VLIW engine actually executes ----
  uint32_t n_tasks, n_entries;
  if (!r.U32(&n_tasks) || !r.U32(&n_entries) || !r.U32(&out->n_flags) ||
      !r.U32(&out->n_lanes))
    return fail("truncated sched header");
  if (out->n_lanes == 0 || out->n_lanes > 4096) return fail("bad n_lanes");

  out->tasks.resize(n_tasks);
  if (n_tasks && !r.Bytes(out->tasks.data(), n_tasks * sizeof(VmTask)))
    return fail("truncated tasks");
  out->lane_tab.resize(out->n_lanes);
  for (auto& l : out->lane_tab)
    if (!r.U32(&l.off) || !r.U32(&l.count) || !r.U32(&l.root_len) ||
        !r.U32(&l.pad) || l.root_len > l.count)
      return fail("truncated/bad lane tab");
  if (!r.Align8()) return fail("lane tab align");
  out->entries.resize(n_entries);
  if (n_entries && !r.Bytes(out->entries.data(), n_entries * sizeof(VmEntry)))
    return fail("truncated entries");

  // Validate schedule. tile_op packs the base op (low byte) + dtype (bits 8-15).
  for (const VmTask& t : out->tasks) {
    const uint32_t base_op = t.tile_op & 0xFFu;
    const uint32_t dt = (t.tile_op >> 8) & 0xFFu;
    if (base_op > kMaxTileOp)
      return fail("unknown tile_op " + std::to_string(base_op));
    if (dt > kDtMax) return fail("unknown dtype " + std::to_string(dt));
    for (uint32_t id : {t.dst, t.a, t.b})
      if (id >= n_buffers) return fail("task buffer id out of range");
    if (base_op == kTopEw && t.p0 == kEwSubSelect && t.p3 >= n_buffers)
      return fail("select pred id out of range");
    if ((base_op == kTopGather || base_op == kTopIotaDim) && t.p0 >= n_aux)
      return fail("task aux offset out of range");
  }
  uint32_t barrier_count_ref = 0;
  bool first_lane = true;
  for (auto& l : out->lane_tab) {
    const uint32_t off = l.off, count = l.count;
    if (uint64_t(off) + count > n_entries) return fail("lane range oob");
    uint32_t barriers = 0;
    for (uint32_t e = off; e < off + count; ++e) {
      const VmEntry& en = out->entries[e];
      if (en.task == kEntBarrier) barriers++;
      else if (en.task == kEntWhile || en.task == kEntIf) {
        if (en.signal_flag >= n_buffers)
          return fail("control cond buffer id out of range");
        if (uint64_t(en.tile_lo) + en.tile_hi > count ||
            uint64_t(en.wait_flag) + en.wait_count > count)
          return fail("control sub-range out of lane stream");
      } else if (en.task != kEntNop && en.task >= n_tasks) {
        return fail("entry task index out of range");
      }
    }
    if (first_lane) { barrier_count_ref = barriers; first_lane = false; }
    else if (barriers != barrier_count_ref)
      return fail("non-uniform top-level barrier counts across lanes");
    if (barriers > out->n_barriers) out->n_barriers = barriers;
  }
  // Loops multiply barrier count at runtime; reserve generous stats space.
  out->n_barriers = out->n_barriers ? out->n_barriers : 1;
  return true;
}

// ---- OclRuntime -------------------------------------------------------------

std::unique_ptr<OclRuntime> OclRuntime::Create(std::string* err) {
  auto fail = [&](const std::string& m) {
    *err = "OclRuntime: " + m;
    return nullptr;
  };

  cl_uint num_platforms = 0;
  if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS ||
      num_platforms == 0)
    return fail("no OpenCL platforms");
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  std::string want_platform;
  int want_index = -1;
  if (const char* env = std::getenv("PJRT_OCL_DEVICE"); env && env[0]) {
    std::string spec(env);
    if (auto colon = spec.rfind(':'); colon != std::string::npos) {
      want_platform = spec.substr(0, colon);
      want_index = std::atoi(spec.c_str() + colon + 1);
    } else {
      want_platform = spec;
    }
  }

  struct Candidate {
    cl_device_id device;
    std::string platform_name;
    bool is_gpu;
  };
  std::vector<Candidate> candidates;
  for (cl_platform_id p : platforms) {
    size_t name_size = 0;
    clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, nullptr, &name_size);
    std::string pname(name_size, '\0');
    clGetPlatformInfo(p, CL_PLATFORM_NAME, name_size, pname.data(), nullptr);
    while (!pname.empty() && pname.back() == '\0') pname.pop_back();
    if (!want_platform.empty() && pname.find(want_platform) == std::string::npos)
      continue;
    cl_uint num_devices = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices) !=
            CL_SUCCESS ||
        !num_devices)
      continue;
    std::vector<cl_device_id> devs(num_devices);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, num_devices, devs.data(), nullptr);
    for (cl_uint i = 0; i < num_devices; ++i) {
      if (want_index >= 0 && static_cast<int>(i) != want_index) continue;
      cl_device_type type = 0;
      clGetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
      candidates.push_back({devs[i], pname, (type & CL_DEVICE_TYPE_GPU) != 0});
    }
  }
  if (candidates.empty()) return fail("no OpenCL device matched selection");

  const Candidate* chosen = nullptr;
  if (want_platform.empty() && want_index < 0)
    for (const auto& c : candidates)
      if (c.is_gpu) { chosen = &c; break; }
  if (!chosen) chosen = &candidates.front();

  auto rt = std::unique_ptr<OclRuntime>(new OclRuntime());
  rt->dev_ = chosen->device;
  rt->info_.platform_name = chosen->platform_name;
  rt->info_.device_name = ClInfoStr(chosen->device, CL_DEVICE_NAME);
  rt->info_.driver_version = ClInfoStr(chosen->device, CL_DRIVER_VERSION);
  rt->info_.cl_version = ClInfoStr(chosen->device, CL_DEVICE_VERSION);
  rt->info_.is_gpu = chosen->is_gpu;
  rt->info_.has_fp64 =
      ClInfoStr(chosen->device, CL_DEVICE_EXTENSIONS).find("cl_khr_fp64") !=
      std::string::npos;
  clGetDeviceInfo(rt->dev_, CL_DEVICE_MAX_COMPUTE_UNITS,
                  sizeof(rt->info_.compute_units), &rt->info_.compute_units,
                  nullptr);

  cl_int cerr;
  rt->ctx_ = clCreateContext(nullptr, 1, &rt->dev_, nullptr, nullptr, &cerr);
  if (cerr != CL_SUCCESS) return fail("clCreateContext: " + std::to_string(cerr));
  rt->queue_ = clCreateCommandQueue(rt->ctx_, rt->dev_, 0, &cerr);
  if (cerr != CL_SUCCESS)
    return fail("clCreateCommandQueue: " + std::to_string(cerr));

  const char* src = kVmClSource;
  size_t src_len = std::strlen(src);
  rt->program_ = clCreateProgramWithSource(rt->ctx_, 1, &src, &src_len, &cerr);
  if (cerr != CL_SUCCESS)
    return fail("clCreateProgramWithSource: " + std::to_string(cerr));
  if (clBuildProgram(rt->program_, 1, &rt->dev_, "", nullptr, nullptr) !=
      CL_SUCCESS) {
    std::string log(1 << 16, '\0');
    size_t log_size = 0;
    clGetProgramBuildInfo(rt->program_, rt->dev_, CL_PROGRAM_BUILD_LOG,
                          log.size(), log.data(), &log_size);
    log.resize(std::min(log_size, log.size()));
    return fail("vm.cl build failed:\n" + log);
  }
  rt->vm_kernel_ = clCreateKernel(rt->program_, "vm2", &cerr);
  if (cerr != CL_SUCCESS) return fail("clCreateKernel: " + std::to_string(cerr));

  // Lanes advertised to the python scheduler (PJRT_OCL_NLANES). Validated
  // co-residency regime: <= 3x CUs at 256 threads (poc/01, poc/04); GPUs get
  // 2 lanes/CU, CPUs 1/CU.
  cl_uint cu = rt->info_.compute_units ? rt->info_.compute_units : 1;
  rt->ngroups_ = rt->info_.is_gpu ? 2 * cu : cu;
  if (const char* g = std::getenv("PJRT_OCL_VM_LANES"); g && g[0])
    rt->ngroups_ = std::max(1, std::atoi(g));
  rt->local_size_ = 256;
  return rt;
}

OclRuntime::~OclRuntime() {
  if (vm_kernel_) clReleaseKernel(vm_kernel_);
  if (program_) clReleaseProgram(program_);
  if (queue_) clReleaseCommandQueue(queue_);
  if (ctx_) clReleaseContext(ctx_);
}

// ---- LoadedProgram ----------------------------------------------------------

std::unique_ptr<LoadedProgram> LoadedProgram::Load(OclRuntime* rt,
                                                   VmProgram prog,
                                                   std::string* err) {
  auto fail = [&](const std::string& m) {
    *err = "LoadedProgram: " + m;
    return nullptr;
  };
  auto lp = std::unique_ptr<LoadedProgram>(new LoadedProgram());
  lp->rt_ = rt;
  lp->prog_ = std::move(prog);
  const VmProgram& p = lp->prog_;

  // f64 gate: the VM's f64 path is compiled only under cl_khr_fp64, so on a
  // device without it an f64 program would silently fall back to f32. Refuse.
  if (!rt->info().has_fp64) {
    for (const auto& b : p.buffers)
      if (b.dtype == kDtF64)
        return fail("program uses f64 but device lacks cl_khr_fp64 "
                    "(" + rt->info().device_name + ")");
  }

  cl_int cerr;
  lp->arena_ = clCreateBuffer(rt->ctx(), CL_MEM_READ_WRITE,
                              std::max<uint64_t>(p.arena_bytes, 4), nullptr,
                              &cerr);
  if (cerr != CL_SUCCESS) return fail("arena alloc: " + std::to_string(cerr));

  // Byte offset into the arena (the VM is byte-addressed; each op casts a
  // typed pointer at this base). Was f32-element (÷4) before dtypes.
  auto elem_off = [&](uint32_t id) {
    return static_cast<uint32_t>(p.buffers[id].arena_byte_offset);
  };
  auto make_buf = [&](const void* data, size_t bytes, const char* what) {
    static const uint64_t kZero = 0;
    cl_mem m = clCreateBuffer(
        rt->ctx(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes ? bytes : 8, bytes ? const_cast<void*>(data)
                                 : const_cast<uint64_t*>(&kZero), &cerr);
    if (cerr != CL_SUCCESS)
      *err = std::string("LoadedProgram: ") + what + " alloc: " +
             std::to_string(cerr);
    return m;
  };

  // Patch task buffer ids -> f32 element offsets (+ select pred in p3).
  std::vector<VmTask> tasks = p.tasks;
  for (VmTask& t : tasks) {
    t.dst = elem_off(t.dst);
    t.a = elem_off(t.a);
    t.b = elem_off(t.b);
    if ((t.tile_op & 0xFFu) == kTopEw && t.p0 == kEwSubSelect)
      t.p3 = elem_off(t.p3);
  }
  // Patch control-entry cond buffer ids.
  std::vector<VmEntry> entries = p.entries;
  for (VmEntry& en : entries)
    if (en.task == kEntWhile || en.task == kEntIf)
      en.signal_flag = elem_off(en.signal_flag);
  lp->tasks_buf_ = make_buf(tasks.data(), tasks.size() * sizeof(VmTask), "tasks");
  if (!lp->tasks_buf_) return nullptr;
  lp->entries_buf_ =
      make_buf(entries.data(), entries.size() * sizeof(VmEntry), "entries");
  if (!lp->entries_buf_) return nullptr;
  lp->lane_tab_buf_ = make_buf(p.lane_tab.data(),
                               p.lane_tab.size() * sizeof(VmProgram::Lane),
                               "lane_tab");
  if (!lp->lane_tab_buf_) return nullptr;
  lp->aux_buf_ = make_buf(p.aux.data(), p.aux.size() * 4, "aux");
  if (!lp->aux_buf_) return nullptr;

  uint32_t barinit[3] = {0, 0, 0};
  lp->bar_buf_ = clCreateBuffer(rt->ctx(),
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(barinit), barinit, &cerr);
  if (cerr != CL_SUCCESS) return fail("bar alloc: " + std::to_string(cerr));
  lp->stats_buf_ = clCreateBuffer(rt->ctx(), CL_MEM_READ_WRITE,
                                  4096u * p.n_lanes * 4, nullptr, &cerr);
  if (cerr != CL_SUCCESS) return fail("stats alloc: " + std::to_string(cerr));

  // Upload constants once.
  for (const auto& [id, bytes] : p.consts) {
    if (bytes.empty()) continue;
    if (clEnqueueWriteBuffer(rt->queue(), lp->arena_, CL_TRUE,
                             p.buffers[id].arena_byte_offset, bytes.size(),
                             bytes.data(), 0, nullptr, nullptr) != CL_SUCCESS)
      return fail("const upload failed");
  }
  return lp;
}

LoadedProgram::~LoadedProgram() {
  for (cl_mem m : {arena_, aux_buf_, tasks_buf_, entries_buf_, lane_tab_buf_,
                   bar_buf_, stats_buf_})
    if (m) clReleaseMemObject(m);
}

bool LoadedProgram::Execute(const std::vector<const void*>& inputs,
                            std::vector<std::vector<uint8_t>>* outputs,
                            std::string* err) {
  const VmProgram& p = prog_;
  if (inputs.size() != p.inputs.size()) {
    *err = "Execute: got " + std::to_string(inputs.size()) + " args, want " +
           std::to_string(p.inputs.size());
    return false;
  }
  std::lock_guard<std::mutex> lock(rt_->mu());
  cl_command_queue q = rt_->queue();

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& b = p.buffers[p.inputs[i]];
    if (b.size_bytes &&
        clEnqueueWriteBuffer(q, arena_, CL_FALSE, b.arena_byte_offset,
                             b.size_bytes, inputs[i], 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      *err = "Execute: input write failed";
      return false;
    }
  }

  if (!LaunchKernel(q, err)) return false;

  outputs->resize(p.outputs.size());
  for (size_t i = 0; i < p.outputs.size(); ++i) {
    const auto& b = p.buffers[p.outputs[i]];
    (*outputs)[i].resize(b.size_bytes);
    if (b.size_bytes &&
        clEnqueueReadBuffer(q, arena_, CL_FALSE, b.arena_byte_offset,
                            b.size_bytes, (*outputs)[i].data(), 0, nullptr,
                            nullptr) != CL_SUCCESS) {
      *err = "Execute: output read failed";
      return false;
    }
  }
  if (clFinish(q) != CL_SUCCESS) {
    *err = "Execute: clFinish failed";
    return false;
  }
  return true;
}

bool LoadedProgram::LaunchKernel(cl_command_queue q, std::string* err) {
  const VmProgram& p = prog_;
  if (p.entries.empty()) return true;
  const uint32_t barinit[3] = {0, 0, 0};
  if (clEnqueueWriteBuffer(q, bar_buf_, CL_FALSE, 0, sizeof(barinit), barinit,
                           0, nullptr, nullptr) != CL_SUCCESS) {
    *err = "Execute: bar reset failed";
    return false;
  }
  cl_kernel k = rt_->vm_kernel();
  cl_uint nlanes = p.n_lanes;
  size_t lsz = 256, gsz = size_t{nlanes} * lsz;
  clSetKernelArg(k, 0, sizeof(arena_), &arena_);
  clSetKernelArg(k, 1, sizeof(aux_buf_), &aux_buf_);
  clSetKernelArg(k, 2, sizeof(tasks_buf_), &tasks_buf_);
  clSetKernelArg(k, 3, sizeof(entries_buf_), &entries_buf_);
  clSetKernelArg(k, 4, sizeof(lane_tab_buf_), &lane_tab_buf_);
  clSetKernelArg(k, 5, sizeof(bar_buf_), &bar_buf_);
  clSetKernelArg(k, 6, sizeof(nlanes), &nlanes);
  clSetKernelArg(k, 7, sizeof(stats_buf_), &stats_buf_);
  if (clEnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, &lsz, 0, nullptr,
                             nullptr) != CL_SUCCESS) {
    *err = "Execute: kernel launch failed";
    return false;
  }
  return true;
}

bool LoadedProgram::ExecuteDevice(const std::vector<cl_mem>& inputs,
                                  std::vector<cl_mem>* outputs,
                                  std::string* err) {
  const VmProgram& p = prog_;
  if (inputs.size() != p.inputs.size()) {
    *err = "ExecuteDevice: got " + std::to_string(inputs.size()) +
           " args, want " + std::to_string(p.inputs.size());
    return false;
  }
  std::lock_guard<std::mutex> lock(rt_->mu());
  cl_command_queue q = rt_->queue();

  // Device->device copy each input into its arena region (on-device bandwidth,
  // no host round-trip).
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& b = p.buffers[p.inputs[i]];
    if (b.size_bytes &&
        clEnqueueCopyBuffer(q, inputs[i], arena_, 0, b.arena_byte_offset,
                            b.size_bytes, 0, nullptr, nullptr) != CL_SUCCESS) {
      *err = "ExecuteDevice: input copy failed";
      return false;
    }
  }

  if (!LaunchKernel(q, err)) return false;

  // Each output stays on device: fresh cl_mem, device->device copy from arena.
  outputs->assign(p.outputs.size(), nullptr);
  cl_int cerr;
  for (size_t i = 0; i < p.outputs.size(); ++i) {
    const auto& b = p.buffers[p.outputs[i]];
    cl_mem out = clCreateBuffer(rt_->ctx(), CL_MEM_READ_WRITE,
                                std::max<size_t>(b.size_bytes, 4), nullptr,
                                &cerr);
    if (cerr != CL_SUCCESS) {
      *err = "ExecuteDevice: output alloc failed";
      for (cl_mem m : *outputs) if (m) clReleaseMemObject(m);
      outputs->clear();
      return false;
    }
    if (b.size_bytes &&
        clEnqueueCopyBuffer(q, arena_, out, b.arena_byte_offset, 0,
                            b.size_bytes, 0, nullptr, nullptr) != CL_SUCCESS) {
      *err = "ExecuteDevice: output copy failed";
      clReleaseMemObject(out);
      for (cl_mem m : *outputs) if (m) clReleaseMemObject(m);
      outputs->clear();
      return false;
    }
    (*outputs)[i] = out;
  }
  if (cl_int e = clFinish(q); e != CL_SUCCESS) {
    *err = "ExecuteDevice: clFinish failed (" + std::to_string(e) +
           "; kernel execution error — likely the cross-workgroup barrier not "
           "co-residing on this device, or resource limits)";
    return false;
  }
  return true;
}

cl_mem OclRuntime::AllocDevice(size_t bytes, std::string* err) {
  cl_int cerr;
  cl_mem m = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, std::max<size_t>(bytes, 4),
                            nullptr, &cerr);
  if (cerr != CL_SUCCESS) {
    *err = "AllocDevice failed: " + std::to_string(cerr);
    return nullptr;
  }
  return m;
}

bool OclRuntime::WriteToDevice(cl_mem dst, const void* host, size_t bytes,
                               std::string* err) {
  if (!bytes) return true;
  std::lock_guard<std::mutex> lock(mu_);
  if (clEnqueueWriteBuffer(queue_, dst, CL_TRUE, 0, bytes, host, 0, nullptr,
                           nullptr) != CL_SUCCESS) {
    *err = "WriteToDevice failed";
    return false;
  }
  return true;
}

bool OclRuntime::ReadFromDevice(cl_mem src, void* host, size_t bytes,
                                std::string* err) {
  if (!bytes) return true;
  std::lock_guard<std::mutex> lock(mu_);
  if (clEnqueueReadBuffer(queue_, src, CL_TRUE, 0, bytes, host, 0, nullptr,
                          nullptr) != CL_SUCCESS) {
    *err = "ReadFromDevice failed";
    return false;
  }
  return true;
}

// ---- Lowering subprocess ----------------------------------------------------

bool RunLoweringSubprocess(
    const std::string& python_exe, const std::string& lower_service_path,
    const std::vector<uint8_t>& input,
    const std::vector<std::pair<std::string, std::string>>& env,
    std::vector<uint8_t>* output, std::string* err, bool* unsupported) {
  *unsupported = false;
  int in_pipe[2], out_pipe[2], err_pipe[2];
  if (pipe(in_pipe) || pipe(out_pipe) || pipe(err_pipe)) {
    *err = "pipe() failed";
    return false;
  }
  pid_t pid = fork();
  if (pid < 0) {
    *err = "fork() failed";
    return false;
  }
  if (pid == 0) {
    for (const auto& [k, v] : env) setenv(k.c_str(), v.c_str(), 1);
    dup2(in_pipe[0], 0);
    dup2(out_pipe[1], 1);
    dup2(err_pipe[1], 2);
    for (int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1],
                   err_pipe[0], err_pipe[1]})
      close(fd);
    execlp(python_exe.c_str(), python_exe.c_str(), lower_service_path.c_str(),
           (char*)nullptr);
    std::fprintf(stderr, "{\"error\":\"ExecFailed\",\"message\":\"execlp %s\"}",
                 python_exe.c_str());
    _exit(127);
  }
  close(in_pipe[0]);
  close(out_pipe[1]);
  close(err_pipe[1]);

  // poll-driven write-stdin / read-stdout+stderr to avoid pipe deadlocks.
  fcntl(in_pipe[1], F_SETFL, O_NONBLOCK);
  std::string err_text;
  size_t written = 0;
  bool in_open = true, out_open = true, eo_open = true;
  output->clear();
  while (in_open || out_open || eo_open) {
    struct pollfd fds[3];
    int n = 0;
    int in_idx = -1, out_idx = -1, eo_idx = -1;
    if (in_open) { fds[n] = {in_pipe[1], POLLOUT, 0}; in_idx = n++; }
    if (out_open) { fds[n] = {out_pipe[0], POLLIN, 0}; out_idx = n++; }
    if (eo_open) { fds[n] = {err_pipe[0], POLLIN, 0}; eo_idx = n++; }
    if (poll(fds, n, 60000) <= 0) {
      *err = "lowering subprocess timeout/poll error";
      kill(pid, SIGKILL);
      break;
    }
    char buf[65536];
    if (in_idx >= 0 && (fds[in_idx].revents & (POLLOUT | POLLERR | POLLHUP))) {
      if (written < input.size() && !(fds[in_idx].revents & (POLLERR | POLLHUP))) {
        ssize_t w = write(in_pipe[1], input.data() + written,
                          std::min<size_t>(input.size() - written, 65536));
        if (w > 0) written += w;
        else if (w < 0 && errno != EAGAIN) written = input.size();
      }
      if (written >= input.size() || (fds[in_idx].revents & (POLLERR | POLLHUP))) {
        close(in_pipe[1]);
        in_open = false;
      }
    }
    if (out_idx >= 0 && (fds[out_idx].revents & (POLLIN | POLLHUP))) {
      ssize_t r = read(out_pipe[0], buf, sizeof buf);
      if (r > 0) output->insert(output->end(), buf, buf + r);
      else { close(out_pipe[0]); out_open = false; }
    }
    if (eo_idx >= 0 && (fds[eo_idx].revents & (POLLIN | POLLHUP))) {
      ssize_t r = read(err_pipe[0], buf, sizeof buf);
      if (r > 0) err_text.append(buf, r);
      else { close(err_pipe[0]); eo_open = false; }
    }
  }
  if (in_open) close(in_pipe[1]);
  if (out_open) close(out_pipe[0]);
  if (eo_open) close(err_pipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  if (code == 0) return true;
  *unsupported = (code == 2);
  *err = "lowering subprocess exit " + std::to_string(code) + ": " +
         (err_text.empty() ? "(no stderr)" : err_text);
  return false;
}

}  // namespace pjrt_ocl
