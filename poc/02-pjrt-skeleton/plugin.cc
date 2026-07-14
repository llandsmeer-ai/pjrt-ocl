// poc/02-pjrt-skeleton: minimal hand-rolled PJRT C API plugin.
// Goal: JAX_PLATFORMS=opencl python -c "import jax; print(jax.devices())"
// prints our device. Everything not needed for that is a stub returning
// UNIMPLEMENTED with the callback name in the message (so jax's error output
// tells us exactly which callback to implement next).
//
// Vendored header: vendor/pjrt_c_api.h from openxla/xla @
// 5a9e73cbd92530cac2ac36f4736a774b2412afe2 (the XLA commit pinned by
// jax v0.10.2) => PJRT C API version 0.112.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "vendor/pjrt_c_api.h"
#include "pjrt_api_fn_list.h"

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
// Error object. NOTE (PJRT C API 0.112): PJRT_Error is NOT opaque — the header
// defines `struct PJRT_Error { const PJRT_Error_FunctionTable* vtable; }`, so
// the framework may operate on errors through the vtable directly, in addition
// to the PJRT_Error_Destroy/Message/GetCode API entries. Support both.
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
                                   void*) {
  // No payloads.
}

static const PJRT_Error_FunctionTable kErrorVtable = {
    /*struct_size=*/PJRT_Error_FunctionTable_STRUCT_SIZE,
    /*instance_size=*/sizeof(OclError),
    /*extension_start=*/nullptr,
    /*destroy=*/ErrorVt_Destroy,
    /*message=*/ErrorVt_Message,
    /*get_code=*/ErrorVt_GetCode,
    /*for_each_payload=*/ErrorVt_ForEachPayload,
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
// Stubs: every PJRT_Error*-returning API function defaults to UNIMPLEMENTED
// with its own name in the message. Real implementations override specific
// entries in the API table below.
// ---------------------------------------------------------------------------

#define DEFINE_STUB(fn)                                                  \
  static PJRT_Error* Stub_##fn(fn##_Args* args) {                       \
    (void)args;                                                          \
    return MakeError(PJRT_Error_Code_UNIMPLEMENTED,                     \
                     "pjrt-ocl skeleton: unimplemented callback: " #fn); \
  }
PJRT_OCL_ERR_FN_LIST(DEFINE_STUB)
#undef DEFINE_STUB

// ---------------------------------------------------------------------------
// Objects. PJRT_Client / PJRT_Device / PJRT_DeviceDescription are opaque in
// the header (forward-declared only) so we define them here. PJRT_Memory is
// NOT opaque (has a user-data vtable as first member) so we embed it.
// ---------------------------------------------------------------------------

struct PJRT_DeviceDescription {
  int id = 0;
  int process_index = 0;
  std::string kind;          // e.g. real CL device name
  std::string debug_string;  // verbose
  std::string to_string;     // e.g. "OclDevice(id=0)"
  std::vector<PJRT_NamedValue> attributes;  // backing storage below
  std::vector<std::string> attr_string_storage;
};

struct OclMemory {
  PJRT_Memory base;  // must be first (vtable)
  int id = 0;
  std::string kind = "device";
  std::string debug_string;
  std::string to_string;
  PJRT_Client* client = nullptr;
  std::vector<PJRT_Device*> devices;  // devices that can address this memory
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
    /*struct_size=*/PJRT_Memory_FunctionTable_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*instance_struct_size=*/sizeof(OclMemory),
    /*get_user_data=*/MemoryVt_GetUserData,
    /*set_user_data=*/MemoryVt_SetUserData,
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
  std::string platform_version = "pjrt-ocl skeleton 0.1";
  int process_index = 0;
  std::vector<std::unique_ptr<PJRT_Device>> devices_storage;
  std::vector<std::unique_ptr<OclMemory>> memories_storage;
  std::vector<PJRT_Device*> devices;   // == addressable devices (single proc)
  std::vector<PJRT_Memory*> memories;  // addressable memories
};

// ---------------------------------------------------------------------------
// OpenCL enumeration (bonus): pick a real device via PJRT_OCL_DEVICE
// ("<platform substring>[:<device index>]"); default first GPU, else first
// CPU, else first device. Falls back to a stub description if OpenCL fails.
// ---------------------------------------------------------------------------

struct SelectedClDevice {
  std::string platform_name;
  std::string device_name;
  std::string driver_version;
  std::string cl_version;
  bool is_gpu = false;
  bool valid = false;
};

static std::string ClDeviceInfoStr(cl_device_id dev, cl_device_info param) {
  size_t size = 0;
  if (clGetDeviceInfo(dev, param, 0, nullptr, &size) != CL_SUCCESS || size == 0)
    return "";
  std::string s(size, '\0');
  clGetDeviceInfo(dev, param, size, s.data(), nullptr);
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

static SelectedClDevice SelectOpenClDevice() {
  SelectedClDevice result;
  cl_uint num_platforms = 0;
  if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS ||
      num_platforms == 0) {
    Log("clGetPlatformIDs found no platforms");
    return result;
  }
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
    cl_platform_id platform;
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
    if (!want_platform.empty() &&
        pname.find(want_platform) == std::string::npos)
      continue;
    cl_uint num_devices = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices) !=
            CL_SUCCESS ||
        num_devices == 0)
      continue;
    std::vector<cl_device_id> devs(num_devices);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, num_devices, devs.data(), nullptr);
    for (cl_uint i = 0; i < num_devices; ++i) {
      if (want_index >= 0 && static_cast<int>(i) != want_index) continue;
      cl_device_type type = 0;
      clGetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
      candidates.push_back({p, devs[i], pname, (type & CL_DEVICE_TYPE_GPU) != 0});
    }
  }
  if (candidates.empty()) {
    Log("no OpenCL device matched selection");
    return result;
  }
  const Candidate* chosen = nullptr;
  if (want_platform.empty() && want_index < 0) {
    for (const auto& c : candidates)
      if (c.is_gpu) { chosen = &c; break; }
  }
  if (!chosen) chosen = &candidates.front();

  result.platform_name = chosen->platform_name;
  result.device_name = ClDeviceInfoStr(chosen->device, CL_DEVICE_NAME);
  result.driver_version = ClDeviceInfoStr(chosen->device, CL_DRIVER_VERSION);
  result.cl_version = ClDeviceInfoStr(chosen->device, CL_DEVICE_VERSION);
  result.is_gpu = chosen->is_gpu;
  result.valid = true;
  return result;
}

// ---------------------------------------------------------------------------
// Implemented callbacks.
// ---------------------------------------------------------------------------

// --- Errors (void-returning; not in the stub list) ---

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

// The framework calls this on EVERY error it receives (to collect status
// payloads). Leaving it as an UNIMPLEMENTED stub made error handling recurse
// (each stub error triggered another ForEachPayload) until a core dump.
static PJRT_Error* Impl_PJRT_Error_ForEachPayload(
    PJRT_Error_ForEachPayload_Args* args) {
  (void)args;  // We never attach payloads; nothing to visit.
  return nullptr;
}

// --- Plugin ---

static PJRT_Error* Impl_PJRT_Plugin_Initialize(
    PJRT_Plugin_Initialize_Args* args) {
  (void)args;
  Log("PJRT_Plugin_Initialize");
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Plugin_Attributes(
    PJRT_Plugin_Attributes_Args* args) {
  Log("PJRT_Plugin_Attributes");
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

// --- Client ---

static PJRT_Error* Impl_PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  Log("PJRT_Client_Create");
  auto client = std::make_unique<PJRT_Client>();

  SelectedClDevice cl = SelectOpenClDevice();

  auto dev = std::make_unique<PJRT_Device>();
  dev->client = client.get();
  dev->description.id = 0;
  dev->description.process_index = 0;
  if (cl.valid) {
    dev->description.kind = cl.device_name;
    dev->description.debug_string =
        "OclDevice(id=0, platform=\"" + cl.platform_name + "\", device=\"" +
        cl.device_name + "\", driver=\"" + cl.driver_version + "\", \"" +
        cl.cl_version + "\")";
  } else {
    dev->description.kind = "OpenCL stub device";
    dev->description.debug_string = "OclDevice(id=0, stub, no OpenCL device)";
  }
  dev->description.to_string = "OclDevice(id=0)";
  dev->local_hardware_id = 0;

  auto mem = std::make_unique<OclMemory>();
  mem->base.vtable = &kMemoryVtable;
  mem->id = 0;
  mem->kind = "device";
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

  args->client = client.release();
  return nullptr;
}

static PJRT_Error* Impl_PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  Log("PJRT_Client_Destroy");
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
  for (PJRT_Device* d : args->client->devices) {
    if (d->description.id == args->id) {
      args->device = d;
      return nullptr;
    }
  }
  return MakeError(PJRT_Error_Code_NOT_FOUND,
                   "pjrt-ocl: no device with id " + std::to_string(args->id));
}

static PJRT_Error* Impl_PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  for (PJRT_Device* d : args->client->devices) {
    if (d->local_hardware_id == args->local_hardware_id) {
      args->addressable_device = d;
      return nullptr;
    }
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

// --- DeviceDescription ---

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

// --- Device ---

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

// The framework CHECK-crashes (LogFatalIfPjrtError in
// xla::PjRtCApiDevice::InitAttributes) if this returns an error, so it is
// mandatory for jax.devices(). Note: jaxlib 0.10.2 uses this newer
// PJRT_Device_GetAttributes entry, not PJRT_DeviceDescription_Attributes.
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

// --- Memory ---

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
// API table.
// ---------------------------------------------------------------------------

static PJRT_Api CreateApi() {
  PJRT_Api api = {};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.extension_start = nullptr;
  api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  api.pjrt_api_version.extension_start = nullptr;
  api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  api.pjrt_api_version.minor_version = PJRT_API_MINOR;

  // Default: everything is a named UNIMPLEMENTED stub.
#define ASSIGN_STUB(fn) api.fn = Stub_##fn;
  PJRT_OCL_ERR_FN_LIST(ASSIGN_STUB)
#undef ASSIGN_STUB

  // Real implementations.
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
#undef ASSIGN_IMPL

  return api;
}

// NOTE: the symbol jaxlib dlsym()s is "GetPjrtApi" (lowercase "rt"), not the
// "GetPjRtApi" spelling used in some openxla docs. Verified empirically:
// jaxlib 0.10.2 raised NOT_FOUND: GetPjrtApi not found in ... .so
extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  static PJRT_Api api = CreateApi();
  Log("GetPjRtApi() -> version " + std::to_string(api.pjrt_api_version.major_version) +
      "." + std::to_string(api.pjrt_api_version.minor_version));
  return &api;
}
