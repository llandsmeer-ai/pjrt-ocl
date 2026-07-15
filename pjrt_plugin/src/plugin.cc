// pjrt-ocl: hand-rolled PJRT C API plugin executing StableHLO on OpenCL via
// the VMProgram bytecode VM (docs/vmprogram.md, docs/decisions.md).
//
// Skeleton validated in poc/02 (device enumeration, error/memory vtables,
// self-diagnosing UNIMPLEMENTED stubs). This file adds the M2 surface:
// Compile (python lowering subprocess), buffers (host-staging v1 — see
// docs/memory.md for the M3 device-resident plan), synchronous Execute,
// immediately-ready events.
//
// Vendored header: vendor/pjrt_c_api.h from openxla/xla @
// 5a9e73cbd92530cac2ac36f4736a774b2412afe2 (jax v0.10.2) => PJRT C API 0.112.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "vendor/pjrt_c_api.h"
#include "pjrt_api_fn_list.h"

using pjrt_ocl::LoadedProgram;
using pjrt_ocl::OclRuntime;
using pjrt_ocl::VmProgram;

namespace {

bool LogEnabled() {
  static bool enabled = [] {
    const char* v = std::getenv("PJRT_OCL_LOG");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

void Log(const std::string& msg) {
  if (LogEnabled()) std::fprintf(stderr, "[pjrt-ocl] %s\n", msg.c_str());
}

}  // namespace

// ---------------------------------------------------------------------------
// Error object (vtable'd in API 0.112; see poc/02 NOTES).
// ---------------------------------------------------------------------------

struct OclError {
  PJRT_Error base;  // must be first (vtable)
  PJRT_Error_Code code;
  std::string message;
};

static void ErrorVt_Destroy(PJRT_Error* error) {
  delete reinterpret_cast<OclError*>(error);
}
static void ErrorVt_Message(const PJRT_Error* error, const char** message,
                            size_t* message_size) {
  const auto* e = reinterpret_cast<const OclError*>(error);
  *message = e->message.c_str();
  *message_size = e->message.size();
}
static PJRT_Error_Code ErrorVt_GetCode(const PJRT_Error* error) {
  return reinterpret_cast<const OclError*>(error)->code;
}
static void ErrorVt_ForEachPayload(const PJRT_Error*, PJRT_Error_PayloadVisitor,
                                   void*) {}

static const PJRT_Error_FunctionTable kErrorVtable = {
    PJRT_Error_FunctionTable_STRUCT_SIZE,
    sizeof(OclError),
    nullptr,
    ErrorVt_Destroy,
    ErrorVt_Message,
    ErrorVt_GetCode,
    ErrorVt_ForEachPayload,
};

static PJRT_Error* MakeError(PJRT_Error_Code code, std::string message) {
  Log("error(" + std::to_string(code) + "): " + message);
  auto* e = new OclError{};
  e->base.vtable = &kErrorVtable;
  e->code = code;
  e->message = std::move(message);
  return &e->base;
}

// ---------------------------------------------------------------------------
// Stubs: self-diagnosing UNIMPLEMENTED (poc/02 trick — keep permanently).
// ---------------------------------------------------------------------------

#define DEFINE_STUB(fn)                                                  \
  static PJRT_Error* Stub_##fn(fn##_Args* args) {                       \
    (void)args;                                                          \
    return MakeError(PJRT_Error_Code_UNIMPLEMENTED,                     \
                     "pjrt-ocl: unimplemented callback: " #fn);          \
  }
PJRT_OCL_ERR_FN_LIST(DEFINE_STUB)
#undef DEFINE_STUB

// ---------------------------------------------------------------------------
// Objects.
// ---------------------------------------------------------------------------

struct PJRT_DeviceDescription {
  int id = 0;
  int process_index = 0;
  std::string kind;
  std::string debug_string;
  std::string to_string;
  std::vector<PJRT_NamedValue> attributes;
};

struct OclMemory {
  PJRT_Memory base;  // must be first (vtable)
  int id = 0;
  std::string kind = "device";
  std::string debug_string;
  std::string to_string;
  PJRT_Client* client = nullptr;
  std::vector<PJRT_Device*> devices;
  std::map<const void*, std::pair<void*, void (*)(void*)>> user_data;
  ~OclMemory() {
    for (auto& [key, value] : user_data)
      if (value.second) value.second(value.first);
  }
};

static void* MemoryVt_GetUserData(PJRT_Memory* memory, const void* key) {
  auto* m = reinterpret_cast<OclMemory*>(memory);
  auto it = m->user_data.find(key);
  return it == m->user_data.end() ? nullptr : it->second.first;
}
static void MemoryVt_SetUserData(PJRT_Memory* memory, const void* key,
                                 void* data, void (*dtor)(void*)) {
  auto* m = reinterpret_cast<OclMemory*>(memory);
  auto it = m->user_data.find(key);
  if (it != m->user_data.end() && it->second.second)
    it->second.second(it->second.first);
  m->user_data[key] = {data, dtor};
}

static const PJRT_Memory_FunctionTable kMemoryVtable = {
    PJRT_Memory_FunctionTable_STRUCT_SIZE,
    nullptr,
    sizeof(OclMemory),
    MemoryVt_GetUserData,
    MemoryVt_SetUserData,
};

struct PJRT_Device {
  PJRT_Client* client = nullptr;
  PJRT_DeviceDescription description;
  int local_hardware_id = 0;
  std::vector<PJRT_Memory*> memories;
  PJRT_Memory* default_memory = nullptr;
};

struct PJRT_Client {
  std::string platform_name = "opencl";
  std::string platform_version = "pjrt-ocl 0.1";
  int process_index = 0;
  std::vector<std::unique_ptr<PJRT_Device>> devices_storage;
  std::vector<std::unique_ptr<OclMemory>> memories_storage;
  std::vector<PJRT_Device*> devices;
  std::vector<PJRT_Memory*> memories;
  std::unique_ptr<OclRuntime> runtime;
  std::string python_exe;      // from create_options / env
  std::string lower_service;   // from create_options / env
};

// Events: v1 is fully synchronous — every event is born ready, optionally
// carrying an error that Await/OnReady hand over (transferring ownership).
struct PJRT_Event {
  PJRT_Error_Code code = PJRT_Error_Code_OK;  // OK => no error
  std::string message;
};

static PJRT_Error* EventError(const PJRT_Event& e) {
  return e.code == PJRT_Error_Code_OK ? nullptr : MakeError(e.code, e.message);
}

// Buffers: v1 host-staging (docs/memory.md §2). Raw bytes + type + dims.
// Device-resident buffer: the data lives in a device cl_mem and only touches
// host memory when the framework calls ToHostBuffer. This keeps intermediate
// results of chained jit calls on the device (no PCIe round-trip per op).
struct PJRT_Buffer {
  PJRT_Client* client = nullptr;
  PJRT_Device* device = nullptr;
  PJRT_Buffer_Type type = PJRT_Buffer_Type_F32;
  std::vector<int64_t> dims;
  cl_mem mem = nullptr;       // device-resident bytes (owned)
  size_t size_bytes = 0;
  bool deleted = false;
};

// Executable metadata, shared between PJRT_LoadedExecutable and the
// PJRT_Executable views handed to the framework (independent lifetimes).
struct ExeMeta {
  std::string name = "vmprogram";
  std::string fingerprint;
  size_t num_outputs = 0;
  size_t code_bytes = 0;
  std::vector<PJRT_Buffer_Type> output_types;
  std::vector<int64_t> output_dims_flat;
  std::vector<size_t> output_dim_sizes;
  std::vector<const char*> output_memory_kinds;
  std::vector<size_t> output_memory_kind_sizes;
};

struct PJRT_Executable {
  std::shared_ptr<ExeMeta> meta;
};

struct PJRT_LoadedExecutable {
  PJRT_Client* client = nullptr;
  std::shared_ptr<ExeMeta> meta;
  std::unique_ptr<LoadedProgram> lp;
  bool deleted = false;
};

// Our VM dtype enum (runtime.h VmDtype) -> PJRT buffer type, for reporting
// output element types back to the framework.
static PJRT_Buffer_Type VmDtypeToPjrt(uint32_t dt) {
  switch (dt) {
    case pjrt_ocl::kDtI32:  return PJRT_Buffer_Type_S32;
    case pjrt_ocl::kDtU32:  return PJRT_Buffer_Type_U32;
    case pjrt_ocl::kDtBool: return PJRT_Buffer_Type_PRED;
    case pjrt_ocl::kDtI64:  return PJRT_Buffer_Type_S64;
    case pjrt_ocl::kDtF64:  return PJRT_Buffer_Type_F64;
    case pjrt_ocl::kDtF16:  return PJRT_Buffer_Type_F16;
    case pjrt_ocl::kDtBf16: return PJRT_Buffer_Type_BF16;
    default:                return PJRT_Buffer_Type_F32;
  }
}

static size_t DtypeSize(PJRT_Buffer_Type t) {
  switch (t) {
    case PJRT_Buffer_Type_PRED: case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8: case PJRT_Buffer_Type_F8E5M2:
    case PJRT_Buffer_Type_F8E4M3FN: case PJRT_Buffer_Type_F8E4M3B11FNUZ:
    case PJRT_Buffer_Type_F8E5M2FNUZ: case PJRT_Buffer_Type_F8E4M3FNUZ:
      return 1;
    case PJRT_Buffer_Type_S16: case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16: case PJRT_Buffer_Type_BF16:
      return 2;
    case PJRT_Buffer_Type_S32: case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_S64: case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64: case PJRT_Buffer_Type_C64:
      return 8;
    case PJRT_Buffer_Type_C128:
      return 16;
    default:
      return 0;  // unsupported (S4/U4, INVALID, ...)
  }
}

// ---------------------------------------------------------------------------
// Implemented callbacks: errors, plugin, client, device, memory (poc/02).
// ---------------------------------------------------------------------------

static void Impl_PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  if (args->error) ErrorVt_Destroy(args->error);
}

static void Impl_PJRT_Error_Message(PJRT_Error_Message_Args* args) {
  ErrorVt_Message(args->error, &args->message, &args->message_size);
}

static PJRT_Error* Impl_PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args) {
  args->code = ErrorVt_GetCode(args->error);
  return nullptr;
}

// Must never be a stub: the framework calls it on every error (poc/02).
static PJRT_Error* Impl_PJRT_Error_ForEachPayload(
    PJRT_Error_ForEachPayload_Args* args) {
  (void)args;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Plugin_Initialize(
    PJRT_Plugin_Initialize_Args* args) {
  (void)args;
  return nullptr;
}

// Advertise stablehlo_current_version so jaxlib serializes artifacts at a
// version our jaxlib-bundled deserializer understands (poc/03 finding).
static PJRT_Error* Impl_PJRT_Plugin_Attributes(
    PJRT_Plugin_Attributes_Args* args) {
  static const int64_t kStablehloVersion[3] = {1, 17, 0};
  static const PJRT_NamedValue kAttrs[] = {{
      PJRT_NamedValue_STRUCT_SIZE, nullptr, "stablehlo_current_version", 25,
      PJRT_NamedValue_kInt64List, {.int64_array_value = kStablehloVersion}, 3,
  }};
  args->attributes = kAttrs;
  args->num_attributes = 1;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  Log("PJRT_Client_Create");
  auto client = std::make_unique<PJRT_Client>();

  for (size_t i = 0; i < args->num_options; ++i) {
    const PJRT_NamedValue& o = args->create_options[i];
    std::string name(o.name, o.name_size);
    if (o.type != PJRT_NamedValue_kString) continue;
    std::string value(o.string_value, o.value_size);
    if (name == "python_exe") client->python_exe = value;
    else if (name == "lower_service") client->lower_service = value;
  }
  if (const char* v = std::getenv("PJRT_OCL_PYTHON"); v && v[0])
    client->python_exe = v;
  if (const char* v = std::getenv("PJRT_OCL_LOWER_SERVICE"); v && v[0])
    client->lower_service = v;

  std::string err;
  client->runtime = OclRuntime::Create(&err);
  if (!client->runtime)
    return MakeError(PJRT_Error_Code_INTERNAL, "pjrt-ocl: " + err);
  const pjrt_ocl::DeviceInfo& info = client->runtime->info();

  auto dev = std::make_unique<PJRT_Device>();
  dev->client = client.get();
  dev->description.id = 0;
  dev->description.kind = info.device_name;
  dev->description.debug_string =
      "OclDevice(id=0, platform=\"" + info.platform_name + "\", device=\"" +
      info.device_name + "\", driver=\"" + info.driver_version + "\", \"" +
      info.cl_version + "\")";
  dev->description.to_string = "OclDevice(id=0)";

  auto mem = std::make_unique<OclMemory>();
  mem->base.vtable = &kMemoryVtable;
  mem->debug_string = "OclMemory(id=0, kind=device)";
  mem->to_string = "OclMemory(id=0)";
  mem->client = client.get();
  mem->devices.push_back(dev.get());
  dev->memories.push_back(&mem->base);
  dev->default_memory = &mem->base;

  client->devices.push_back(dev.get());
  client->memories.push_back(&mem->base);
  client->devices_storage.push_back(std::move(dev));
  client->memories_storage.push_back(std::move(mem));

  Log("client ready on " + info.platform_name + " / " + info.device_name);
  args->client = client.release();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  delete args->client;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_PlatformName(
    PJRT_Client_PlatformName_Args* args) {
  args->platform_name = args->client->platform_name.c_str();
  args->platform_name_size = args->client->platform_name.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args) {
  args->platform_version = args->client->platform_version.c_str();
  args->platform_version_size = args->client->platform_version.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_ProcessIndex(
    PJRT_Client_ProcessIndex_Args* args) {
  args->process_index = args->client->process_index;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_Devices(PJRT_Client_Devices_Args* args) {
  args->devices = args->client->devices.data();
  args->num_devices = args->client->devices.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  args->addressable_devices = args->client->devices.data();
  args->num_addressable_devices = args->client->devices.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_LookupDevice(
    PJRT_Client_LookupDevice_Args* args) {
  for (PJRT_Device* d : args->client->devices)
    if (d->description.id == args->id) {
      args->device = d;
      return nullptr;
    }
  return MakeError(PJRT_Error_Code_NOT_FOUND,
                   "pjrt-ocl: no device with id " + std::to_string(args->id));
}

static PJRT_Error* Impl_PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  for (PJRT_Device* d : args->client->devices)
    if (d->local_hardware_id == args->local_hardware_id) {
      args->addressable_device = d;
      return nullptr;
    }
  return MakeError(PJRT_Error_Code_NOT_FOUND,
                   "pjrt-ocl: no device with local_hardware_id " +
                       std::to_string(args->local_hardware_id));
}

static PJRT_Error* Impl_PJRT_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args* args) {
  args->addressable_memories = args->client->memories.data();
  args->num_addressable_memories = args->client->memories.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_DeviceDescription_Id(
    PJRT_DeviceDescription_Id_Args* args) {
  args->id = args->device_description->id;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  args->process_index = args->device_description->process_index;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  args->attributes = args->device_description->attributes.data();
  args->num_attributes = args->device_description->attributes.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_DeviceDescription_Kind(
    PJRT_DeviceDescription_Kind_Args* args) {
  args->device_kind = args->device_description->kind.c_str();
  args->device_kind_size = args->device_description->kind.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  args->debug_string = args->device_description->debug_string.c_str();
  args->debug_string_size = args->device_description->debug_string.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  args->to_string = args->device_description->to_string.c_str();
  args->to_string_size = args->device_description->to_string.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Device_GetDescription(
    PJRT_Device_GetDescription_Args* args) {
  args->device_description = &args->device->description;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Device_IsAddressable(
    PJRT_Device_IsAddressable_Args* args) {
  args->is_addressable = true;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  args->local_hardware_id = args->device->local_hardware_id;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args* args) {
  args->memories = args->device->memories.data();
  args->num_memories = args->device->memories.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Device_DefaultMemory(
    PJRT_Device_DefaultMemory_Args* args) {
  args->memory = args->device->default_memory;
  return nullptr;
}

// CHECK-crashes if it errors (poc/02) — mandatory.
struct PJRT_Device_Attributes {
  std::vector<PJRT_NamedValue> attributes;
};

static PJRT_Error* Impl_PJRT_Device_GetAttributes(
    PJRT_Device_GetAttributes_Args* args) {
  auto* holder = new PJRT_Device_Attributes{};
  holder->attributes = args->device->description.attributes;
  args->attributes = holder->attributes.data();
  args->num_attributes = holder->attributes.size();
  args->device_attributes = holder;
  args->attributes_deleter = [](PJRT_Device_Attributes* a) { delete a; };
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Memory_Id(PJRT_Memory_Id_Args* args) {
  args->id = reinterpret_cast<OclMemory*>(args->memory)->id;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Memory_Kind(PJRT_Memory_Kind_Args* args) {
  auto* m = reinterpret_cast<OclMemory*>(args->memory);
  args->kind = m->kind.c_str();
  args->kind_size = m->kind.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args) {
  args->kind_id = 0;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Memory_DebugString(
    PJRT_Memory_DebugString_Args* args) {
  auto* m = reinterpret_cast<OclMemory*>(args->memory);
  args->debug_string = m->debug_string.c_str();
  args->debug_string_size = m->debug_string.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Memory_ToString(PJRT_Memory_ToString_Args* args) {
  auto* m = reinterpret_cast<OclMemory*>(args->memory);
  args->to_string = m->to_string.c_str();
  args->to_string_size = m->to_string.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  auto* m = reinterpret_cast<OclMemory*>(args->memory);
  args->devices = m->devices.data();
  args->num_devices = m->devices.size();
  return nullptr;
}

// ---------------------------------------------------------------------------
// Events (all pre-signaled in v1).
// ---------------------------------------------------------------------------

static PJRT_Error* Impl_PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args) {
  delete args->event;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args) {
  args->is_ready = true;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Event_Error(PJRT_Event_Error_Args* args) {
  return EventError(*args->event);
}

static PJRT_Error* Impl_PJRT_Event_Await(PJRT_Event_Await_Args* args) {
  return EventError(*args->event);
}

static PJRT_Error* Impl_PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args) {
  // Already ready: invoke immediately. Callback takes ownership of the error.
  args->callback(EventError(*args->event), args->user_arg);
  return nullptr;
}

// ---------------------------------------------------------------------------
// Buffers (host-staging v1).
// ---------------------------------------------------------------------------

static PJRT_Error* Impl_PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  const size_t elem = DtypeSize(args->type);
  if (elem == 0)
    return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
                     "pjrt-ocl: unsupported buffer dtype " +
                         std::to_string(args->type));

  auto buf = std::make_unique<PJRT_Buffer>();
  buf->client = args->client;
  buf->device = args->device ? args->device : args->client->devices[0];
  buf->type = args->type;
  buf->dims.assign(args->dims, args->dims + args->num_dims);

  size_t n = 1;
  for (int64_t d : buf->dims) n *= static_cast<size_t>(d);
  buf->size_bytes = n * elem;

  // Stage host-side into a dense contiguous block, then upload once to device.
  std::vector<uint8_t> staging(buf->size_bytes);
  bool dense = true;
  if (args->num_byte_strides) {
    int64_t stride = static_cast<int64_t>(elem);
    for (size_t i = args->num_dims; i-- > 0;) {
      if (args->byte_strides[i] != stride) dense = false;
      stride *= args->dims[i];
    }
  }
  if (dense || n == 0) {
    std::memcpy(staging.data(), args->data, staging.size());
  } else {
    std::vector<size_t> idx(args->num_dims, 0);
    const auto* src = static_cast<const uint8_t*>(args->data);
    for (size_t out = 0; out < n; ++out) {
      int64_t off = 0;
      for (size_t d = 0; d < args->num_dims; ++d)
        off += static_cast<int64_t>(idx[d]) * args->byte_strides[d];
      std::memcpy(staging.data() + out * elem, src + off, elem);
      for (size_t d = args->num_dims; d-- > 0;) {
        if (++idx[d] < static_cast<size_t>(args->dims[d])) break;
        idx[d] = 0;
      }
    }
  }

  std::string err;
  buf->mem = buf->client->runtime->AllocDevice(buf->size_bytes, &err);
  if (!buf->mem)
    return MakeError(PJRT_Error_Code_RESOURCE_EXHAUSTED, "pjrt-ocl: " + err);
  if (!buf->client->runtime->WriteToDevice(buf->mem, staging.data(),
                                           buf->size_bytes, &err))
    return MakeError(PJRT_Error_Code_INTERNAL, "pjrt-ocl: " + err);

  args->buffer = buf.release();
  args->done_with_host_buffer = new PJRT_Event{};
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
  if (args->buffer && args->buffer->mem)
    clReleaseMemObject(args->buffer->mem);
  delete args->buffer;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_ElementType(
    PJRT_Buffer_ElementType_Args* args) {
  args->type = args->buffer->type;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_Dimensions(
    PJRT_Buffer_Dimensions_Args* args) {
  args->dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args) {
  args->unpadded_dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args) {
  args->dynamic_dim_indices = nullptr;
  args->num_dynamic_dims = 0;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  args->on_device_size_in_bytes = args->buffer->size_bytes;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args) {
  args->device = args->buffer->device;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
  args->memory = args->buffer->device->default_memory;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
  args->buffer->deleted = true;
  if (args->buffer->mem) {
    clReleaseMemObject(args->buffer->mem);
    args->buffer->mem = nullptr;
  }
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_IsDeleted(
    PJRT_Buffer_IsDeleted_Args* args) {
  args->is_deleted = args->buffer->deleted;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  args->is_on_cpu = false;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_ReadyEvent(
    PJRT_Buffer_ReadyEvent_Args* args) {
  args->event = new PJRT_Event{};
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Buffer_ToHostBuffer(
    PJRT_Buffer_ToHostBuffer_Args* args) {
  PJRT_Buffer* src = args->src;
  if (src->deleted)
    return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
                     "pjrt-ocl: ToHostBuffer on deleted buffer");
  if (args->dst == nullptr) {
    args->dst_size = src->size_bytes;
    return nullptr;
  }
  if (args->dst_size < src->size_bytes)
    return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
                     "pjrt-ocl: ToHostBuffer dst too small");
  // Lazy D2H: the only point data leaves the device.
  std::string err;
  if (src->mem && !src->client->runtime->ReadFromDevice(
                      src->mem, args->dst, src->size_bytes, &err))
    return MakeError(PJRT_Error_Code_INTERNAL, "pjrt-ocl: " + err);
  args->event = new PJRT_Event{};
  return nullptr;
}

// ---------------------------------------------------------------------------
// Compile & Execute.
// ---------------------------------------------------------------------------

static PJRT_Error* Impl_PJRT_Client_Compile(PJRT_Client_Compile_Args* args) {
  PJRT_Client* client = args->client;
  std::string format(args->program->format, args->program->format_size);
  Log("PJRT_Client_Compile: format=" + format +
      " code_size=" + std::to_string(args->program->code_size));
  if (format != "mlir")
    return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
                     "pjrt-ocl: unsupported program format: " + format);
  if (client->python_exe.empty() || client->lower_service.empty())
    return MakeError(
        PJRT_Error_Code_FAILED_PRECONDITION,
        "pjrt-ocl: lowering subprocess not configured (need python_exe + "
        "lower_service create_options, or PJRT_OCL_PYTHON + "
        "PJRT_OCL_LOWER_SERVICE env)");

  std::vector<uint8_t> artifact(
      reinterpret_cast<const uint8_t*>(args->program->code),
      reinterpret_cast<const uint8_t*>(args->program->code) +
          args->program->code_size);
  std::vector<uint8_t> vmp_bytes;
  std::string err;
  bool unsupported = false;
  std::vector<std::pair<std::string, std::string>> sub_env = {
      {"PJRT_OCL_NLANES", std::to_string(client->runtime->ngroups())}};
  if (const char* ct = std::getenv("PJRT_OCL_COST_TABLE"); ct && ct[0])
    sub_env.push_back({"PJRT_OCL_COST_TABLE", ct});
  if (!pjrt_ocl::RunLoweringSubprocess(client->python_exe,
                                       client->lower_service, artifact,
                                       sub_env, &vmp_bytes, &err, &unsupported))
    return MakeError(unsupported ? PJRT_Error_Code_UNIMPLEMENTED
                                 : PJRT_Error_Code_INTERNAL,
                     "pjrt-ocl lowering: " + err);

  VmProgram prog;
  if (!VmProgram::Parse(vmp_bytes.data(), vmp_bytes.size(), &prog, &err))
    return MakeError(PJRT_Error_Code_INTERNAL, "pjrt-ocl: " + err);

  auto meta = std::make_shared<ExeMeta>();
  meta->num_outputs = prog.outputs.size();
  meta->code_bytes = vmp_bytes.size();
  meta->fingerprint = "pjrt-ocl-" + std::to_string(std::hash<std::string>{}(
                          std::string(vmp_bytes.begin(), vmp_bytes.end())));
  for (size_t i = 0; i < prog.outputs.size(); ++i) {
    meta->output_types.push_back(
        VmDtypeToPjrt(prog.buffers[prog.outputs[i]].dtype));
    const auto& dims = prog.output_dims[i];
    meta->output_dim_sizes.push_back(dims.size());
    meta->output_dims_flat.insert(meta->output_dims_flat.end(), dims.begin(),
                                  dims.end());
    meta->output_memory_kinds.push_back("device");
    meta->output_memory_kind_sizes.push_back(6);
  }

  auto lp = LoadedProgram::Load(client->runtime.get(), std::move(prog), &err);
  if (!lp) return MakeError(PJRT_Error_Code_INTERNAL, "pjrt-ocl: " + err);

  auto* lexe = new PJRT_LoadedExecutable{};
  lexe->client = client;
  lexe->meta = std::move(meta);
  lexe->lp = std::move(lp);
  args->executable = lexe;
  Log("compile ok: " + std::to_string(lexe->lp->prog().instrs.size()) +
      " instrs, arena " + std::to_string(lexe->lp->prog().arena_bytes) + " B");
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  args->executable = new PJRT_Executable{args->loaded_executable->meta};
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  args->addressable_devices = args->executable->client->devices.data();
  args->num_addressable_devices = args->executable->client->devices.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_AddressableDeviceLogicalIds(
    PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args* args) {
  static PJRT_LogicalDeviceIds kIds[] = {{0, 0}};  // replica 0, partition 0
  args->addressable_device_logical_ids = kIds;
  args->num_addressable_device_logical_ids = 1;
  return nullptr;
}

// Hand-encoded xla.DeviceAssignmentProto for our fixed single-device case:
// replica_count=1 (field 1), computation_count=1 (field 2), one
// computation_devices entry (field 3) with replica_device_ids=[0] (packed).
static PJRT_Error* Impl_PJRT_LoadedExecutable_GetDeviceAssignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
  static const char kProto[] = {0x08, 0x01, 0x10, 0x01, 0x1A, 0x03,
                                0x0A, 0x01, 0x00};
  args->serialized_bytes = kProto;
  args->serialized_bytes_size = sizeof(kProto);
  args->serialized_device_assignment = nullptr;
  args->serialized_device_assignment_deleter =
      [](PJRT_DeviceAssignmentSerialized*) {};
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args) {
  args->executable->deleted = true;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args) {
  args->is_deleted = args->executable->deleted;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_Destroy(
    PJRT_Executable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_Name(PJRT_Executable_Name_Args* args) {
  args->executable_name = args->executable->meta->name.c_str();
  args->executable_name_size = args->executable->meta->name.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args) {
  args->num_replicas = 1;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args) {
  args->num_partitions = 1;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_NumOutputs(
    PJRT_Executable_NumOutputs_Args* args) {
  args->num_outputs = args->executable->meta->num_outputs;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
  args->size_in_bytes = args->executable->meta->code_bytes;
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args* args) {
  args->executable_fingerprint = args->executable->meta->fingerprint.c_str();
  args->executable_fingerprint_size = args->executable->meta->fingerprint.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args) {
  args->output_types = args->executable->meta->output_types.data();
  args->num_output_types = args->executable->meta->output_types.size();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args) {
  args->num_outputs = args->executable->meta->num_outputs;
  args->dims = args->executable->meta->output_dims_flat.data();
  args->dim_sizes = args->executable->meta->output_dim_sizes.data();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args) {
  args->num_outputs = args->executable->meta->num_outputs;
  args->memory_kinds = args->executable->meta->output_memory_kinds.data();
  args->memory_kind_sizes = args->executable->meta->output_memory_kind_sizes.data();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  PJRT_LoadedExecutable* lexe = args->executable;
  if (lexe->deleted)
    return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
                     "pjrt-ocl: Execute on deleted executable");
  if (args->num_devices != 1)
    return MakeError(PJRT_Error_Code_UNIMPLEMENTED,
                     "pjrt-ocl: multi-device execute (num_devices=" +
                         std::to_string(args->num_devices) + ")");

  const VmProgram& prog = lexe->lp->prog();
  std::vector<cl_mem> inputs;
  for (size_t i = 0; i < args->num_args; ++i) {
    PJRT_Buffer* b = args->argument_lists[0][i];
    if (b->deleted || !b->mem)
      return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
                       "pjrt-ocl: Execute with deleted input buffer");
    if (i < prog.inputs.size() &&
        b->size_bytes != prog.buffers[prog.inputs[i]].size_bytes)
      return MakeError(
          PJRT_Error_Code_INVALID_ARGUMENT,
          "pjrt-ocl: arg " + std::to_string(i) + " byte size " +
              std::to_string(b->size_bytes) + " != expected " +
              std::to_string(prog.buffers[prog.inputs[i]].size_bytes));
    inputs.push_back(b->mem);
  }

  std::vector<cl_mem> outputs;
  std::string err;
  if (!lexe->lp->ExecuteDevice(inputs, &outputs, &err))
    return MakeError(PJRT_Error_Code_INTERNAL, "pjrt-ocl: " + err);

  // Outputs stay on device; wrap each cl_mem in a device-resident buffer.
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto* out = new PJRT_Buffer{};
    out->client = lexe->client;
    out->device = lexe->client->devices[0];
    out->type = VmDtypeToPjrt(prog.buffers[prog.outputs[i]].dtype);
    out->dims = prog.output_dims[i];
    out->mem = outputs[i];
    out->size_bytes = prog.buffers[prog.outputs[i]].size_bytes;
    args->output_lists[0][i] = out;
  }
  if (args->device_complete_events)
    args->device_complete_events[0] = new PJRT_Event{};
  return nullptr;
}

// ---------------------------------------------------------------------------
// API table.
// ---------------------------------------------------------------------------

static PJRT_Api CreateApi() {
  PJRT_Api api = {};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  api.pjrt_api_version.minor_version = PJRT_API_MINOR;

#define ASSIGN_STUB(fn) api.fn = Stub_##fn;
  PJRT_OCL_ERR_FN_LIST(ASSIGN_STUB)
#undef ASSIGN_STUB

#define ASSIGN_IMPL(fn) api.fn = Impl_##fn;
  ASSIGN_IMPL(PJRT_Error_Destroy)
  ASSIGN_IMPL(PJRT_Error_Message)
  ASSIGN_IMPL(PJRT_Error_GetCode)
  ASSIGN_IMPL(PJRT_Error_ForEachPayload)
  ASSIGN_IMPL(PJRT_Plugin_Initialize)
  ASSIGN_IMPL(PJRT_Plugin_Attributes)
  ASSIGN_IMPL(PJRT_Client_Create)
  ASSIGN_IMPL(PJRT_Client_Destroy)
  ASSIGN_IMPL(PJRT_Client_PlatformName)
  ASSIGN_IMPL(PJRT_Client_PlatformVersion)
  ASSIGN_IMPL(PJRT_Client_ProcessIndex)
  ASSIGN_IMPL(PJRT_Client_Devices)
  ASSIGN_IMPL(PJRT_Client_AddressableDevices)
  ASSIGN_IMPL(PJRT_Client_LookupDevice)
  ASSIGN_IMPL(PJRT_Client_LookupAddressableDevice)
  ASSIGN_IMPL(PJRT_Client_AddressableMemories)
  ASSIGN_IMPL(PJRT_Client_Compile)
  ASSIGN_IMPL(PJRT_Client_BufferFromHostBuffer)
  ASSIGN_IMPL(PJRT_DeviceDescription_Id)
  ASSIGN_IMPL(PJRT_DeviceDescription_ProcessIndex)
  ASSIGN_IMPL(PJRT_DeviceDescription_Attributes)
  ASSIGN_IMPL(PJRT_DeviceDescription_Kind)
  ASSIGN_IMPL(PJRT_DeviceDescription_DebugString)
  ASSIGN_IMPL(PJRT_DeviceDescription_ToString)
  ASSIGN_IMPL(PJRT_Device_GetDescription)
  ASSIGN_IMPL(PJRT_Device_IsAddressable)
  ASSIGN_IMPL(PJRT_Device_LocalHardwareId)
  ASSIGN_IMPL(PJRT_Device_AddressableMemories)
  ASSIGN_IMPL(PJRT_Device_DefaultMemory)
  ASSIGN_IMPL(PJRT_Device_GetAttributes)
  ASSIGN_IMPL(PJRT_Memory_Id)
  ASSIGN_IMPL(PJRT_Memory_Kind)
  ASSIGN_IMPL(PJRT_Memory_Kind_Id)
  ASSIGN_IMPL(PJRT_Memory_DebugString)
  ASSIGN_IMPL(PJRT_Memory_ToString)
  ASSIGN_IMPL(PJRT_Memory_AddressableByDevices)
  ASSIGN_IMPL(PJRT_Event_Destroy)
  ASSIGN_IMPL(PJRT_Event_IsReady)
  ASSIGN_IMPL(PJRT_Event_Error)
  ASSIGN_IMPL(PJRT_Event_Await)
  ASSIGN_IMPL(PJRT_Event_OnReady)
  ASSIGN_IMPL(PJRT_Buffer_Destroy)
  ASSIGN_IMPL(PJRT_Buffer_ElementType)
  ASSIGN_IMPL(PJRT_Buffer_Dimensions)
  ASSIGN_IMPL(PJRT_Buffer_UnpaddedDimensions)
  ASSIGN_IMPL(PJRT_Buffer_DynamicDimensionIndices)
  ASSIGN_IMPL(PJRT_Buffer_OnDeviceSizeInBytes)
  ASSIGN_IMPL(PJRT_Buffer_Device)
  ASSIGN_IMPL(PJRT_Buffer_Memory)
  ASSIGN_IMPL(PJRT_Buffer_Delete)
  ASSIGN_IMPL(PJRT_Buffer_IsDeleted)
  ASSIGN_IMPL(PJRT_Buffer_IsOnCpu)
  ASSIGN_IMPL(PJRT_Buffer_ReadyEvent)
  ASSIGN_IMPL(PJRT_Buffer_ToHostBuffer)
  ASSIGN_IMPL(PJRT_LoadedExecutable_Destroy)
  ASSIGN_IMPL(PJRT_LoadedExecutable_GetExecutable)
  ASSIGN_IMPL(PJRT_LoadedExecutable_AddressableDevices)
  ASSIGN_IMPL(PJRT_LoadedExecutable_AddressableDeviceLogicalIds)
  ASSIGN_IMPL(PJRT_LoadedExecutable_GetDeviceAssignment)
  ASSIGN_IMPL(PJRT_LoadedExecutable_Delete)
  ASSIGN_IMPL(PJRT_LoadedExecutable_IsDeleted)
  ASSIGN_IMPL(PJRT_LoadedExecutable_Execute)
  ASSIGN_IMPL(PJRT_Executable_Destroy)
  ASSIGN_IMPL(PJRT_Executable_Name)
  ASSIGN_IMPL(PJRT_Executable_NumReplicas)
  ASSIGN_IMPL(PJRT_Executable_NumPartitions)
  ASSIGN_IMPL(PJRT_Executable_NumOutputs)
  ASSIGN_IMPL(PJRT_Executable_SizeOfGeneratedCodeInBytes)
  ASSIGN_IMPL(PJRT_Executable_Fingerprint)
  ASSIGN_IMPL(PJRT_Executable_OutputElementTypes)
  ASSIGN_IMPL(PJRT_Executable_OutputDimensions)
  ASSIGN_IMPL(PJRT_Executable_OutputMemoryKinds)
#undef ASSIGN_IMPL

  return api;
}

// jaxlib dlsym's "GetPjrtApi" — lowercase "rt" (poc/02).
extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  static PJRT_Api api = CreateApi();
  return &api;
}
