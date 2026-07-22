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
  // p6/p7 (§33 R2c matmul epilogue): p6 = epilogue descriptor aux word-offset
  // (+1; 0 = none); p7 = the epilogue's second-input (residual/bias) buffer
  // handle, loader-patched to a byte offset like dst/a/b when p6 != 0.
  uint32_t tile_op, dst, a, b, p0, p1, p2, p3, p4, p5, p6, p7;
};
static_assert(sizeof(VmTask) == 48);

struct VmEntry {
  uint32_t task, tile_lo, tile_hi, wait_flag, wait_count, signal_flag,
      slots, pad;
};
static_assert(sizeof(VmEntry) == 32);

enum TileOp : uint32_t {
  kTopEw = 0, kTopMma = 1, kTopGather = 2, kTopRedPart = 3, kTopRedComb = 4,
  kTopIotaDim = 5, kTopScatter = 6, kTopDynGather = 7, kTopDynScatter = 8,
  kTopRedWindow = 9, kTopRedSeg = 10,
  kTopSoftmaxSeg = 11, kTopLayernormSeg = 12,  // fused segmented norms (§19)
  kTopMapRegion = 13,  // §27 register-resident map-region (PoC; -DVMO_REGION_POC)
  kTopFlashAttn = 14,  // §34 fused flash-attention (online softmax; p0 = V buffer)
  kTopRedStrided = 15, // partial-axis reduce over interior/prefix axis block
  kTopGatherIndex = 16,  // §38 general data-dependent gather (stablehlo.gather)
  kTopConv = 17,       // §39 direct N-D convolution (NHWC input / HWIO kernel)
  kTopScatterIndex = 18,  // §42 general data-dependent scatter (stablehlo.scatter)
  kMaxTileOp = kTopScatterIndex,
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
  // §36 standalone TF32 tensor-core SGEMM (poc/17); null unless the NVIDIA
  // VMO_NV_PTX program built. Preferred over mm_kernel_ (SGEMM) on GPU/TF32.
  cl_kernel mm_tc_kernel() const { return mm_tc_kernel_; }
  // §38 standalone fp16 tensor-core SGEMM (poc/17 mma17_hp.cl); null unless the
  // VMO_NV_PTX program built. ~72/92 TF/s @2048/4096 vs mm_tc's ~47/57 (2x
  // tensor rate, tf32-equal 10-bit mantissa). Opt-in via PJRT_OCL_MM_FP16=1
  // (mm_fp16()) — it narrows the staged A/B inputs to fp16 (max_normal ~65504),
  // fine for normalized activations; the f32 accumulator is unchanged.
  cl_kernel mm_tc_fp16_kernel() const { return mm_tc_fp16_kernel_; }
  // §39 fp16-INPUT tile: reads A/B pre-packed as fp16 from device scratch (half
  // the staging bytes of the f32-arena mm_tc_fp16) + its f32->fp16 pack kernel.
  // ~107/76 TF/s @4096/2048 vs mm_tc_fp16's ~85/67. Same NV_PTX program.
  cl_kernel mm_tc_fp16p_kernel() const { return mm_tc_fp16p_kernel_; }
  cl_kernel pack_f16_kernel() const { return pack_f16_kernel_; }
  // True when PJRT_OCL_MM_FP16=1 AND the fp16 WMMA kernel built (GPU/NV_PTX).
  bool mm_fp16() const { return mm_fp16_ && mm_tc_fp16_kernel_; }
  // True when the §39 packed fp16 path (pack + fp16-input tile) should be used:
  // mm_fp16() enabled, both kernels built, and not disabled via
  // PJRT_OCL_MM_FP16_PACK=0 (kept as an A/B escape hatch to the f32-arena tile).
  bool mm_fp16_pack() const {
    return mm_fp16() && mm_tc_fp16p_kernel_ && pack_f16_kernel_ &&
           mm_fp16_pack_;
  }
  cl_kernel gemv_kernel() const { return gemv_kernel_; }
  // CPU-only (VMO_CPU_TILES builds; null on GPU): packed+blocked SGEMM.
  cl_kernel mm_pack_kernel() const { return mm_pack_kernel_; }
  cl_kernel mm_packed_kernel() const { return mm_packed_kernel_; }
  cl_kernel mm_packed_epi_kernel() const { return mm_packed_epi_kernel_; }
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
  // EW tile size (elements): compiled into the kernels (-DEW_TS) and
  // advertised to the python scheduler (PJRT_OCL_EW_TS); the two MUST agree.
  cl_uint ew_ts() const { return ew_ts_; }
  // MMA output-tile edge (elements): compiled into the TF32 megakernel
  // (-DVMO_MEGA_BIGTILE => 128) and advertised to the scheduler
  // (PJRT_OCL_MMA_T); the two MUST agree so tile counts match. Default 64.
  cl_uint mma_t() const { return mma_t_; }
  size_t local_size() const { return local_size_; }
  // Host-dispatch (CPU) work-group size for the vm2_seg tile launches. On PoCL
  // a work-group is ONE CPU thread iterating its work-items as a serial loop
  // with barriers implemented by loop-splitting, so the collaborative reduce
  // trees (softmax/layernorm/redseg) are pure overhead. lsz=1 makes every
  // work-group a single work-item (no barriers, no tree; each WI does a whole
  // tile / segment serially — thread-per-segment for the norm tiles), which the
  // VMO_CPU_TILES float8 bodies already vectorize. Parallelism comes from the
  // n_lanes work-groups (= compute units). Default 1 on CPU; PJRT_OCL_CPU_LSZ
  // overrides (256 restores the old collaborative geometry for A/B).
  size_t seg_lsz() const { return seg_lsz_; }
  // Host-dispatch engine: the host drives control flow and enforces the
  // cross-workgroup barrier via clFinish between per-phase launches (no
  // in-kernel spin-barrier). Default ON for non-GPU (CPU) devices, where the
  // persistent spin-barrier deadlocks (imbalance-starvation, docs/decisions.md
  // #1 / poc/07); OFF for GPUs. Overridable via PJRT_OCL_ENGINE=host|mega|auto.
  bool host_dispatch() const { return host_dispatch_; }
  // True when host-dispatch was forced SOLELY by PJRT_OCL_MM_HYBRID on a GPU
  // that can also run the megakernel (fence + vm kernel). Such programs may fall
  // back to the (faster on small/latency-bound work) megakernel per-program when
  // they contain no routable big-matmul phase — otherwise HYBRID's host-dispatch
  // penalty regresses matmul-light programs (e.g. transformer base 5.4->8.3 ms).
  bool hybrid_forces_hd() const { return hybrid_forces_hd_; }
  bool can_use_megakernel() const {
    return info_.is_gpu && info_.has_device_fence &&
           (vm_tc_kernel_ || vm_kernel_);
  }
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
  cl_kernel mm_tc_kernel_ = nullptr;   // §36 standalone TF32 WMMA (NV_PTX only)
  cl_kernel mm_tc_fp16_kernel_ = nullptr;  // §38 standalone fp16 WMMA (NV_PTX only)
  cl_kernel mm_tc_fp16p_kernel_ = nullptr; // §39 fp16-INPUT WMMA (packed scratch)
  cl_kernel pack_f16_kernel_ = nullptr;    // §39 f32->fp16 pack for the above
  bool mm_fp16_ = false;               // PJRT_OCL_MM_FP16=1: prefer fp16 WMMA
  bool mm_fp16_pack_ = false;          // PJRT_OCL_MM_FP16_PACK=1 enables §39 (wash)
  cl_kernel gemv_kernel_ = nullptr;    // width-1 matmul (both device classes)
  cl_kernel mm_pack_kernel_ = nullptr;    // CPU only: B panel packing
  cl_kernel mm_packed_kernel_ = nullptr;  // CPU only: packed 6x16 KC-swept
  cl_kernel mm_packed_epi_kernel_ = nullptr;  // CPU only: packed 6x16 + epilogue
  std::string trace_path_;             // empty = tracing off
  bool trace_suppressed_ = false;      // true during cost calibration
  std::string cost_table_path_;        // measured cost JSON ("" = unit costs)
  cl_mem dummy_buf_ = nullptr;         // placeholder for unused I/O ports
  cl_uint ngroups_ = 0;    // co-resident workgroups (GPUs: measured, poc/08)
  cl_uint ew_ts_ = 16384;  // EW tile elements (GPU: 4096; see ew_ts())
  cl_uint mma_t_ = 64;     // MMA tile edge (128 when TF32 big-tile built; mma_t())
  size_t local_size_ = 64;
  size_t seg_lsz_ = 256;   // host-dispatch tile work-group size (CPU: see seg_lsz())
  bool host_dispatch_ = false;
  bool hybrid_forces_hd_ = false;  // host-dispatch forced only by MM_HYBRID
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
  // Host-dispatch tile work-group size for THIS program (see OclRuntime::seg_lsz).
  // Computed at Load: the runtime's CPU preference (default 1) is used only when
  // every tile op is correct at lsz=1; a program containing a collaborative op
  // (matmul / flash-attention / conv — barriered workgroup reductions PoCL
  // miscompiles or that assume lsz>1 lanes) is pinned to 256.
  size_t seg_lsz() const { return seg_lsz_; }

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
  // Enqueue the packed CPU SGEMM (mm2_pack + mm2p KC-sweeps) for one matmul,
  // writing into scratch `bp`. Shared by the pure-matmul fast path and the CPU
  // in-program hybrid routing (both need the identical pack+sweep sequence).
  bool EnqueuePackedMM(cl_command_queue q, uint32_t M, uint32_t N, uint32_t K,
                       uint32_t dst, uint32_t a, uint32_t bh, cl_mem bp,
                       uint32_t p6, uint32_t p7, std::string* err);
  // §39 fp16-INPUT matmul: pack A(MxK)/B(KxN) f32->fp16 into device scratch with
  // padded leading dims (mm_f16a_/mm_f16b_, grown lazily), then run the
  // fp16-input WMMA tile. Halves staging bytes (~107 vs ~85 TF/s @4096) and pads
  // to break the power-of-2 (K=2048) cache aliasing that dips the f32-arena tile.
  bool EnqueueFp16Matmul(cl_command_queue q, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t dst, uint32_t ah, uint32_t bh,
                         std::string* err);
  // Trace mode: lazily creates the per-lane profiling queues (one per lane so
  // lanes run concurrently, like workgroups of one launch do).
  bool EnsureTraceQueues(std::string* err);
  OclRuntime* rt_ = nullptr;  // borrowed; client outlives executables
  VmProgram prog_;
  // Per-program engine: normally == rt_->host_dispatch(), but a program that
  // HYBRID forced onto host-dispatch yet has NO routable big-matmul phase falls
  // back to the megakernel (host-dispatch's per-phase penalty would otherwise
  // regress matmul-light programs). Computed at load; see the load body.
  bool host_dispatch_ = false;
  size_t seg_lsz_ = 256;      // per-program host-dispatch lsz (see seg_lsz())
  // §36 hybrid: tasks with dst/a/b patched to arena byte-offsets / port handles
  // (prog_.tasks keeps raw buffer ids). The mm_tc dispatch needs the patched
  // handles; the persistent VM reads them from tasks_buf_ instead.
  std::vector<VmTask> tasks_patched_;
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
  // CPU in-program matmul hybrid: B-panel scratch reused across every routed
  // matmul in the program (in-order queue serializes pack/sweep so one buffer
  // is safe). Sized at load time to the largest routable matmul's K*N.
  cl_mem hy_bp_ = nullptr;
  size_t hy_bp_bytes_ = 0;
  // §39 fp16-input matmul scratch: A packed as fp16 (M x lda) and B (K x ldb).
  // Grown lazily to the largest matmul seen; reused across the program (in-order
  // queue serializes pack -> tile -> next). Released with the program.
  cl_mem mm_f16a_ = nullptr;
  size_t mm_f16a_bytes_ = 0;
  cl_mem mm_f16b_ = nullptr;
  size_t mm_f16b_bytes_ = 0;
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
