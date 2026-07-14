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
  std::vector<VmInstr> instrs;

  // Parses + validates a serialized VMProgram. Returns false with *err set.
  static bool Parse(const uint8_t* data, size_t len, VmProgram* out,
                    std::string* err);
};

// ---- OpenCL runtime (one per PJRT client) ----------------------------------

struct DeviceInfo {
  std::string platform_name, device_name, driver_version, cl_version;
  bool is_gpu = false;
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

  // Synchronous execute: writes inputs into the arena, runs the VM, reads
  // outputs. inputs[i] must hold prog().buffers[prog().inputs[i]].size_bytes
  // bytes; outputs[i] is resized accordingly. Thread-safe (runtime mutex).
  bool Execute(const std::vector<const void*>& inputs,
               std::vector<std::vector<uint8_t>>* outputs, std::string* err);

 private:
  LoadedProgram() = default;
  OclRuntime* rt_ = nullptr;  // borrowed; client outlives executables
  VmProgram prog_;
  cl_mem arena_ = nullptr;
  cl_mem instr_buf_ = nullptr;
  cl_mem bar_buf_ = nullptr;
};

// ---- Lowering subprocess ----------------------------------------------------

// Runs `python_exe lower_service_path` with `input` on stdin. Returns true and
// fills *output (stdout bytes) on exit 0; else false with *err set from stderr
// (JSON {error,message} passed through) and *unsupported=true on exit 2.
bool RunLoweringSubprocess(const std::string& python_exe,
                           const std::string& lower_service_path,
                           const std::vector<uint8_t>& input,
                           std::vector<uint8_t>* output, std::string* err,
                           bool* unsupported);

}  // namespace pjrt_ocl

#endif  // PJRT_OCL_RUNTIME_H_
