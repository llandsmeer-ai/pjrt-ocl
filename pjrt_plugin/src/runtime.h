// pjrt-ocl runtime: OpenCL device management + VMProgram v1 loading/execution.
// Pure executor — knows nothing about PJRT or StableHLO (docs/vmprogram.md).
#ifndef PJRT_OCL_RUNTIME_H_
#define PJRT_OCL_RUNTIME_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace pjrt_ocl {

// ---- VMProgram v1 ----------------------------------------------------------

struct VmInstr {
  uint32_t op, dst, a, b, n, imm, pad0, pad1;
};
static_assert(sizeof(VmInstr) == 32);

enum VmOp : uint32_t {
  kNop = 0, kAddF32 = 1, kMulF32 = 2, kSubF32 = 3,
  kFillF32 = 4, kIotaF32 = 5, kLtsF32 = 6, kWhile = 7,
  kMaxOp = kWhile,
};

// Schedule sections (v2.1 spec in docs/vmprogram.md).
struct VmTask {
  uint32_t tile_op, dst, a, b, p0, p1, p2, p3;
};
static_assert(sizeof(VmTask) == 32);

struct VmEntry {
  uint32_t task, tile_lo, tile_hi, wait_flag, wait_count, signal_flag,
      slots, pad;
};
static_assert(sizeof(VmEntry) == 32);

enum TileOp : uint32_t {
  kTopEw = 0, kTopMma = 1, kTopGather = 2, kTopRedPart = 3, kTopRedComb = 4,
  kTopIotaDim = 5, kTopScatter = 6, kTopDynGather = 7, kTopDynScatter = 8,
  kTopRedWindow = 9, kMaxTileOp = kTopRedWindow,
};
// dtype packed in tile_op bits 8-15 (matches python DT_* / vm_common.cl).
enum VmDtype : uint32_t {
  kDtF32 = 0, kDtI32 = 1, kDtU32 = 2, kDtBool = 3, kDtI64 = 4, kDtF64 = 5,
  kDtF16 = 6, kDtBf16 = 7,
  kDtMax = kDtBf16,
};
enum EntSentinel : uint32_t {
  kEntNop = 0xFFFFFFFFu, kEntBarrier = 0xFFFFFFFEu,
  kEntWhile = 0xFFFFFFFDu, kEntIf = 0xFFFFFFFCu,
};
constexpr uint32_t kEwSubSelect = 21;

struct VmProgram {
  struct Buffer {
    uint64_t arena_byte_offset = 0;
    uint64_t size_bytes = 0;
    uint32_t dtype = 0;  // 0 = f32
  };
  uint64_t arena_bytes = 0;
  uint32_t main_len = 0;
  std::vector<Buffer> buffers;
  std::vector<uint32_t> inputs;   // buffer ids, argument order
  std::vector<uint32_t> outputs;  // buffer ids, result order
  std::vector<std::vector<int64_t>> input_dims;   // parallel to inputs
  std::vector<std::vector<int64_t>> output_dims;  // parallel to outputs
  std::vector<std::pair<uint32_t, std::vector<uint8_t>>> consts;
  std::vector<int32_t> aux;      // v2: shape/stride metadata pool
  std::vector<VmInstr> instrs;   // tensor ISA (reference semantics)
  // v2.1 schedule sections — what the VLIW engine executes.
  uint32_t n_flags = 0;
  uint32_t n_lanes = 0;
  std::vector<VmTask> tasks;
  struct Lane { uint32_t off, count, root_len, pad; };
  std::vector<Lane> lane_tab;
  std::vector<VmEntry> entries;
  uint32_t n_barriers = 0;  // max barrier count (stats sizing), derived

  // Parses + validates a serialized VMProgram (version 3). Returns false
  // with *err set.
  static bool Parse(const uint8_t* data, size_t len, VmProgram* out,
                    std::string* err);
};

// ---- OpenCL runtime (one per PJRT client) ----------------------------------

struct DeviceInfo {
  std::string platform_name, device_name, driver_version, cl_version;
  bool is_gpu = false;
  bool has_fp64 = false;   // cl_khr_fp64 — gates f64 programs
  cl_uint compute_units = 0;
};

class OclRuntime {
 public:
  // Selects a device per PJRT_OCL_DEVICE ("<platform substr>[:<idx>]"),
  // creates context/queue and builds the VM kernel. nullptr + *err on failure.
  static std::unique_ptr<OclRuntime> Create(std::string* err);
  ~OclRuntime();

  const DeviceInfo& info() const { return info_; }

  cl_context ctx() const { return ctx_; }
  cl_command_queue queue() const { return queue_; }
  cl_kernel vm_kernel() const { return vm_kernel_; }
  cl_uint ngroups() const { return ngroups_; }
  size_t local_size() const { return local_size_; }
  std::mutex& mu() { return mu_; }

  // Device-resident buffer helpers (for device-resident PJRT_Buffers). Each
  // locks the runtime mutex (single in-order queue). AllocDevice returns a
  // cl_mem the caller owns (clReleaseMemObject); nullptr + *err on failure.
  cl_mem AllocDevice(size_t bytes, std::string* err);
  bool WriteToDevice(cl_mem dst, const void* host, size_t bytes,
                     std::string* err);
  bool ReadFromDevice(cl_mem src, void* host, size_t bytes, std::string* err);

 private:
  OclRuntime() = default;
  DeviceInfo info_;
  cl_device_id dev_ = nullptr;
  cl_context ctx_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel vm_kernel_ = nullptr;
  cl_uint ngroups_ = 0;    // co-resident workgroups (poc/01 rule: <= CUs)
  size_t local_size_ = 64;
  std::mutex mu_;  // serializes execute (single in-order queue)
};

// ---- Loaded program (one per PJRT executable) ------------------------------

class LoadedProgram {
 public:
  // Uploads consts + id->offset-patched instructions; allocates the arena.
  static std::unique_ptr<LoadedProgram> Load(OclRuntime* rt, VmProgram prog,
                                             std::string* err);
  ~LoadedProgram();

  const VmProgram& prog() const { return prog_; }

  // Synchronous execute from HOST inputs to HOST outputs (H2D + D2H). Used by
  // runtime_test. inputs[i] must hold buffers[inputs[i]].size_bytes bytes.
  bool Execute(const std::vector<const void*>& inputs,
               std::vector<std::vector<uint8_t>>* outputs, std::string* err);

  // Synchronous execute keeping data ON DEVICE: inputs are device cl_mems
  // (device->device copied into the arena), outputs are freshly-allocated
  // device cl_mems (caller owns; released via clReleaseMemObject). No host
  // round-trip — this is the plugin's hot path. Thread-safe (runtime mutex).
  bool ExecuteDevice(const std::vector<cl_mem>& inputs,
                     std::vector<cl_mem>* outputs, std::string* err);

 private:
  LoadedProgram() = default;
  // Barrier reset + kernel launch on the arena (already populated). Caller
  // holds the runtime mutex. Does not clFinish.
  bool LaunchKernel(cl_command_queue q, std::string* err);
  OclRuntime* rt_ = nullptr;  // borrowed; client outlives executables
  VmProgram prog_;
  cl_mem arena_ = nullptr;
  cl_mem aux_buf_ = nullptr;
  cl_mem tasks_buf_ = nullptr;
  cl_mem entries_buf_ = nullptr;
  cl_mem lane_tab_buf_ = nullptr;
  cl_mem bar_buf_ = nullptr;
  cl_mem stats_buf_ = nullptr;
};

// ---- Lowering subprocess ----------------------------------------------------

// Runs `python_exe lower_service_path` with `input` on stdin, with `env`
// key=value pairs added to the child environment (device config for the
// scheduler: PJRT_OCL_NLANES, PJRT_OCL_COST_TABLE). Returns true and fills
// *output (stdout bytes) on exit 0; else false with *err set from stderr
// (JSON {error,message} passed through) and *unsupported=true on exit 2.
bool RunLoweringSubprocess(
    const std::string& python_exe, const std::string& lower_service_path,
    const std::vector<uint8_t>& input,
    const std::vector<std::pair<std::string, std::string>>& env,
    std::vector<uint8_t>* output, std::string* err, bool* unsupported);

}  // namespace pjrt_ocl

#endif  // PJRT_OCL_RUNTIME_H_
