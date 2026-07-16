// pjrt-ocl runtime: OpenCL device management + VMProgram v1 loading/execution.
// Pure executor — knows nothing about PJRT or StableHLO (docs/vmprogram.md).
#ifndef PJRT_OCL_RUNTIME_H_
#define PJRT_OCL_RUNTIME_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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
  // p4/p5: MMA operand VIEW aux-offsets (+1; 0 = contiguous), for shape-op
  // (transpose/reshape/broadcast) fold into the matmul operand read (§13).
  uint32_t tile_op, dst, a, b, p0, p1, p2, p3, p4, p5;
};
static_assert(sizeof(VmTask) == 40);

struct VmEntry {
  uint32_t task, tile_lo, tile_hi, wait_flag, wait_count, signal_flag,
      slots, pad;
};
static_assert(sizeof(VmEntry) == 32);

enum TileOp : uint32_t {
  kTopEw = 0, kTopMma = 1, kTopGather = 2, kTopRedPart = 3, kTopRedComb = 4,
  kTopIotaDim = 5, kTopScatter = 6, kTopDynGather = 7, kTopDynScatter = 8,
  kTopRedWindow = 9, kTopRedSeg = 10, kMaxTileOp = kTopRedSeg,
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
  // fixed-trip loop: body range in tile_lo/tile_hi, trip count in wait_flag
  kEntFor = 0xFFFFFFFBu,
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
  std::string build_opts;  // clBuildProgram options that vm.cl was built with
  bool is_gpu = false;
  bool has_fp64 = false;   // cl_khr_fp64 — gates f64 programs
  // Device-scope acq/rel fences compiled into vmo_barrier (OpenCL C 2.0+).
  // Without them the vm2 spin-barrier is a data race (poc/07), so the
  // megakernel engine is refused and host-dispatch is forced.
  bool has_device_fence = false;
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
  cl_device_id dev() const { return dev_; }
  cl_command_queue queue() const { return queue_; }
  cl_kernel vm_kernel() const { return vm_kernel_; }
  // Megakernel actually launched by the persistent engine: the NVIDIA TF32
  // tensor-core variant (vm_tc_kernel_, built with -DVMO_NV_PTX) when it
  // compiled, else the portable vm_kernel_. Both share the vm2 ABI; only
  // vmo_mma_tile's body differs (docs/decisions.md §10b). PJRT_OCL_MEGA_TC=0
  // forces the portable one (for A/B occupancy measurement).
  cl_kernel vm_exec_kernel() const {
    return vm_tc_kernel_ ? vm_tc_kernel_ : vm_kernel_;
  }
  cl_kernel vm_seg_kernel() const { return vm_seg_kernel_; }
  cl_kernel vm_one_kernel() const { return vm_one_kernel_; }
  cl_kernel mm_kernel() const { return mm_kernel_; }
  cl_kernel gemv_kernel() const { return gemv_kernel_; }
  // CPU-only (VMO_CPU_TILES builds; null on GPU): packed+blocked SGEMM.
  cl_kernel mm_pack_kernel() const { return mm_pack_kernel_; }
  cl_kernel mm_packed_kernel() const { return mm_packed_kernel_; }
  // Execution trace (PJRT_OCL_VM_TRACE=<path>): host-dispatch is forced and
  // every schedule entry runs as its own single-workgroup launch on a per-lane
  // profiling queue; per-entry device timestamps are appended to <path> as one
  // JSON line per Execute. For engineering timeline plots (tools/
  // plot_schedule.py) — per-entry launches add overhead, don't benchmark it.
  const std::string& trace_path() const { return trace_path_; }
  // True while the first-run cost calibration executes its µbenchmark
  // programs: they must run on the UNTRACED path (per-entry trace launches
  // would distort the measured per-tile costs and pollute the trace file).
  bool trace_suppressed() const { return trace_suppressed_; }
  // Measured per-tile-op cost model (docs/decisions.md #1): Create() runs a
  // first-run µbenchmark per tile-op family (slope over two tile counts, so
  // launch overhead cancels) and caches the result as JSON keyed by
  // (platform, device, driver). Empty when calibration is disabled
  // (PJRT_OCL_CALIBRATE=0), superseded by a user PJRT_OCL_COST_TABLE, or
  // failed. PJRT_OCL_CALIBRATE=1 forces re-measurement past the cache.
  const std::string& cost_table_path() const { return cost_table_path_; }
  cl_uint ngroups() const { return ngroups_; }
  size_t local_size() const { return local_size_; }
  // Host-dispatch engine: the host drives control flow and enforces the
  // cross-workgroup barrier via clFinish between per-phase launches (no
  // in-kernel spin-barrier). Default ON for non-GPU (CPU) devices, where the
  // persistent spin-barrier deadlocks (imbalance-starvation, docs/decisions.md
  // #1 / poc/07); OFF for GPUs. Overridable via PJRT_OCL_ENGINE=host|mega|auto.
  bool host_dispatch() const { return host_dispatch_; }
  // True for OpenCL GPU devices. Matmul launch GEOMETRY keys on this (not on
  // host_dispatch): the mm2 kernel is correct only under its GPU tiled
  // geometry on a GPU, so a fence-less GPU that runs the host EW engine must
  // still launch matmul with GPU geometry — the CPU packed/register geometry
  // silently produces wrong results on a GPU.
  bool is_gpu() const { return info_.is_gpu; }
  std::mutex& mu() { return mu_; }

  // Device-resident buffer helpers (for device-resident PJRT_Buffers). Each
  // locks the runtime mutex (single in-order queue). AllocDevice returns a
  // cl_mem the caller owns (clReleaseMemObject); nullptr + *err on failure.
  cl_mem AllocDevice(size_t bytes, std::string* err);
  bool WriteToDevice(cl_mem dst, const void* host, size_t bytes,
                     std::string* err);
  bool ReadFromDevice(cl_mem src, void* host, size_t bytes, std::string* err);

  // Size-keyed cl_mem pool. Fresh device allocations of >=2 MB hit a slow
  // driver path (measured: ~0.1 ms on NVIDIA, done lazily on first kernel
  // write) — costly when a fresh output buffer is allocated every execute.
  // PoolAlloc reuses a same-size buffer if one was recently freed (PoolFree),
  // else allocates. Thread-safe (own mutex, independent of the execute mutex).
  cl_mem PoolAlloc(size_t bytes, std::string* err);
  void PoolFree(cl_mem m, size_t bytes);

  // 1-byte placeholder bound to unused kernel I/O ports (never dereferenced).
  cl_mem dummy_buf() const { return dummy_buf_; }

 private:
  OclRuntime() = default;
  // Launches vm2 in probe mode (nlanes=0 sentinel) with an oversized grid and
  // returns the measured co-resident workgroup count for the REAL vm2 kernel
  // (poc/08 discovery protocol — deadlock-free for any launch size). 0 on any
  // CL error (caller falls back to the heuristic). Runs before
  // CalibrateCosts so lane sizing is final before any program executes
  // (calibration programs are n_lanes=1 today, but keep the invariant).
  cl_uint ProbeResidency();
  // First-run µbenchmark: measure per-tile costs for the tile-op families the
  // scheduler's cost model keys on, write/reuse the cached JSON, set
  // cost_table_path_. Never fails the client — on any error the path stays
  // empty (scheduler falls back to unit costs).
  void CalibrateCosts();
  DeviceInfo info_;
  cl_device_id dev_ = nullptr;
  cl_context ctx_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_program tc_mega_program_ = nullptr;  // NVIDIA -DVMO_NV_PTX megakernel (inline PTX)
  cl_kernel vm_kernel_ = nullptr;
  cl_kernel vm_tc_kernel_ = nullptr;   // TF32 tensor-core vm2 (null => portable)
  cl_kernel vm_seg_kernel_ = nullptr;  // host-dispatch segment kernel
  cl_kernel vm_one_kernel_ = nullptr;  // trace mode: one entry per launch
  cl_kernel mm_kernel_ = nullptr;      // standalone SGEMM (pure-matmul fast path)
  cl_kernel gemv_kernel_ = nullptr;    // width-1 matmul (both device classes)
  cl_kernel mm_pack_kernel_ = nullptr;    // CPU only: B panel packing
  cl_kernel mm_packed_kernel_ = nullptr;  // CPU only: packed 6x16 KC-swept
  std::string trace_path_;             // empty = tracing off
  bool trace_suppressed_ = false;      // true during cost calibration
  std::string cost_table_path_;        // measured cost JSON ("" = unit costs)
  cl_mem dummy_buf_ = nullptr;         // placeholder for unused I/O ports
  cl_uint ngroups_ = 0;    // co-resident workgroups (GPUs: measured, poc/08)
  size_t local_size_ = 64;
  bool host_dispatch_ = false;
  std::mutex mu_;  // serializes execute (single in-order queue)
  std::mutex pool_mu_;
  std::unordered_map<size_t, std::vector<cl_mem>> buf_pool_;
  static constexpr size_t kPoolPerSize = 4;  // cap per size (bounds memory)
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
  // Host-dispatch execution: mirror vm2's frame-walk on the host, launching a
  // barrier-free segment kernel per phase with clFinish (= the barrier) between
  // them. Reads while-cond scalars from the arena between phases. Caller holds
  // the runtime mutex; blocks until the program completes.
  bool LaunchHostDispatch(cl_command_queue q, std::string* err);
  // Pure-matmul fast path: launch the standalone mm2 SGEMM (one workgroup per
  // 128x128 output tile) instead of the megakernel. Only taken when mm_ok_.
  bool LaunchMatmul(cl_command_queue q, std::string* err);
  // Trace mode: lazily creates the per-lane profiling queues (one per lane so
  // lanes run concurrently, like workgroups of one launch do).
  bool EnsureTraceQueues(std::string* err);
  OclRuntime* rt_ = nullptr;  // borrowed; client outlives executables
  VmProgram prog_;
  cl_mem arena_ = nullptr;
  cl_mem aux_buf_ = nullptr;
  cl_mem tasks_buf_ = nullptr;
  cl_mem entries_buf_ = nullptr;
  cl_mem lane_tab_buf_ = nullptr;
  cl_mem bar_buf_ = nullptr;
  cl_mem stats_buf_ = nullptr;
  cl_mem seg_tab_buf_ = nullptr;   // host-dispatch: per-lane {off,count} u2
  cl_mem mm_bp_ = nullptr;         // CPU packed matmul: B-panel scratch (pool)
  size_t mm_bp_bytes_ = 0;
  std::vector<cl_command_queue> trace_queues_;  // trace mode, one per lane

  // Zero-copy I/O ports (docs/decisions.md): up to kNIoPorts input/output
  // buffers are passed straight to the kernel instead of copied through the
  // arena. input_port_[i] / output_port_[o] = the port for that I/O buffer, or
  // -1 to fall back to an arena copy. io_bufs_ is the per-execute cl_mem bound
  // to each port (dummy for unused), filled by ExecuteDevice and read by the
  // Launch* helpers.
  static constexpr int kNIoPorts = 8;   // must match VMO_N_IO in vm_common.cl
  std::vector<int> input_port_;
  std::vector<int> output_port_;
  std::vector<cl_mem> io_bufs_;         // size kNIoPorts

  // Pure-matmul fast path (docs/decisions.md #9b): the program is a single
  // TILE_MMA task, no barriers/control, so ExecuteDevice dispatches the
  // standalone mm2 kernel. mm_{dst,a,b}_ are the offset/port-patched handles.
  bool mm_ok_ = false;
  uint32_t mm_M_ = 0, mm_N_ = 0, mm_K_ = 0;
  uint32_t mm_dst_ = 0, mm_a_ = 0, mm_b_ = 0;
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
