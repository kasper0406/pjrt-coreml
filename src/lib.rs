#![feature(vec_into_raw_parts, box_as_ptr)]

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));


#[macro_use]
extern crate lazy_static;

use std::ffi::c_void;
use std::os::raw::c_int;
use std::ffi::CString;
use std::ptr;
use std::sync::RwLock;
use std::cell::UnsafeCell;

use sysinfo::System;

static mut COREML_DEVICES_POINTERS: Vec<*mut PJRT_Device> = vec![];
static mut COREML_MEMORYS_POINTERS: Vec<*mut PJRT_Memory> = vec![];

struct Device {
    device: Box<PJRT_Device>,
    description: CString,
}

struct Memory {
    memory: Box<PJRT_Memory>,
    kind: CString,
}

struct Executable {
    memory_kinds: Vec<CString>,

    // C pointers
    memory_kinds_ptr: Vec<*const i8>,
    memory_kinds_sizes: Vec<usize>,
}

impl Executable {
    fn new() -> Self {
        let memory_kinds = vec![CString::new("CoreML").unwrap()];
        let memory_kinds_ptr: Vec<*const i8> = memory_kinds.iter()
            .map(|memory_kind| memory_kind.as_ptr())
            .collect();
        let memory_kinds_sizes = memory_kinds.iter()
            .map(|memory_kind| memory_kind.count_bytes())
            .collect();

        Executable {
            memory_kinds,
            memory_kinds_ptr,
            memory_kinds_sizes,
        }
    }
}

lazy_static! {
    static ref COREML_DEVICES: Vec<Device> = {
        let devices = vec![Device {
            device: Box::new(PJRT_Device { _unused: [0; 0] }),
            description: CString::new("CoreML").unwrap(),
        }];

        let device_pointers: Vec<*mut PJRT_Device> = devices.iter()
            .map(|device| Box::as_ptr(&device.device) as *mut PJRT_Device)
            .collect();
    
        unsafe { COREML_DEVICES_POINTERS = device_pointers };

        devices
    };
    static ref COREML_MEMORYS: Vec<Memory> = {
        let memories = vec![Memory {
            memory: Box::new(PJRT_Memory { _unused: [0; 0] }),
            kind: CString::new("CoreML Unified Memory").unwrap(),
        }];

        let memory_pointers: Vec<*mut PJRT_Memory> = memories.iter()
            .map(|memory| Box::as_ptr(&memory.memory) as *mut PJRT_Memory)
            .collect();
    
        unsafe { COREML_MEMORYS_POINTERS = memory_pointers };

        memories
    };

    static ref PLATFORM_NAME: CString = CString::new("CoreML").unwrap();
    static ref VERSION: CString = CString::new("0.0.0").unwrap();

    static ref TEST_EXECUTABLE: Executable = Executable::new();
}

#[no_mangle]
pub unsafe extern "C" fn PluginInitialize(arg_ptr: *mut PJRT_Plugin_Initialize_Args) -> *mut PJRT_Error {
    println!("PluginInitialize was called...");
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn PluginAttributes(arg_ptr: *mut PJRT_Plugin_Attributes_Args) -> *mut PJRT_Error {
    println!("PluginAttributes was called...");

    (*arg_ptr).num_attributes = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientCreate(arg_ptr: *mut PJRT_Client_Create_Args) -> *mut PJRT_Error {
    println!("ClientCreate was called...");

    let client = Box::new(PJRT_Client {
        _unused: [0; 0],
    });

    (*arg_ptr).client = Box::into_raw(client) as *mut PJRT_Client;
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientDestroy(arg_ptr: *mut PJRT_Client_Destroy_Args) -> *mut PJRT_Error {
    println!("ClientDestroy was called...");

    if !(*arg_ptr).client.is_null() {
        // Free the client memory
        unsafe { drop(Box::from_raw((*arg_ptr).client)) };
    }
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientPlatformName(arg_ptr: *mut PJRT_Client_PlatformName_Args) -> *mut PJRT_Error {
    println!("ClientPlatformName was called...");

    (*arg_ptr).platform_name_size = PLATFORM_NAME.count_bytes();
    (*arg_ptr).platform_name = PLATFORM_NAME.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientProcessIndex(arg_ptr: *mut PJRT_Client_ProcessIndex_Args) -> *mut PJRT_Error {
    println!("ClientProcessIndex was called...");

    (*arg_ptr).process_index = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientPlatformVersion(arg_ptr: *mut PJRT_Client_PlatformVersion_Args) -> *mut PJRT_Error {
    println!("ClientPlatformVersion was called...");

    (*arg_ptr).platform_version_size = VERSION.count_bytes();
    (*arg_ptr).platform_version = VERSION.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientDevices(arg_ptr: *mut PJRT_Client_Devices_Args) -> *mut PJRT_Error {
    println!("ClientDevices was called...");

    (*arg_ptr).num_devices = COREML_DEVICES.len();
    (*arg_ptr).devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceGetDescription(arg_ptr: *mut PJRT_Device_GetDescription_Args) -> *mut PJRT_Error {
    println!("DeviceGetDescription was called...");

    let description = Box::new(PJRT_DeviceDescription {
        _unused: [0; 0],
    });
    (*arg_ptr).device_description = Box::into_raw(description) as *mut PJRT_DeviceDescription;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceIsAddressable(arg_ptr: *mut PJRT_Device_IsAddressable_Args) -> *mut PJRT_Error {
    println!("DeviceIsAddressable was called...");

    (*arg_ptr).is_addressable = true;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionId(arg_ptr: *mut PJRT_DeviceDescription_Id_Args) -> *mut PJRT_Error {
    println!("DeviceDescriptionId was called...");

    (*arg_ptr).id = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptiopnProcessIndex(arg_ptr: *mut PJRT_DeviceDescription_ProcessIndex_Args) -> *mut PJRT_Error {
    println!("DeviceDescriptiopnProcessIndex was called...");

    (*arg_ptr).process_index = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionAttributes(arg_ptr: *mut PJRT_DeviceDescription_Attributes_Args) -> *mut PJRT_Error {
    println!("DeviceDescriptionAttributes was called...");

    (*arg_ptr).num_attributes = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionKind(arg_ptr: *mut PJRT_DeviceDescription_Kind_Args) -> *mut PJRT_Error {
    println!("DeviceDescriptionKind was called...");

    (*arg_ptr).device_kind = COREML_DEVICES[0].description.as_ptr();
    (*arg_ptr).device_kind_size = COREML_DEVICES[0].description.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionDebugString(arg_ptr: *mut PJRT_DeviceDescription_DebugString_Args) -> *mut PJRT_Error {
    println!("DeviceDescriptionDebugString was called...");

    (*arg_ptr).debug_string = COREML_DEVICES[0].description.as_ptr();
    (*arg_ptr).debug_string_size = COREML_DEVICES[0].description.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionToString(arg_ptr: *mut PJRT_DeviceDescription_ToString_Args) -> *mut PJRT_Error {
    println!("DeviceDescriptionToString was called...");

    (*arg_ptr).to_string = COREML_DEVICES[0].description.as_ptr();
    (*arg_ptr).to_string_size = COREML_DEVICES[0].description.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientAddressableDevices(arg_ptr: *mut PJRT_Client_AddressableDevices_Args) -> *mut PJRT_Error {
    println!("ClientAddressableDevices was called...");

    (*arg_ptr).num_addressable_devices = COREML_DEVICES.len();
    (*arg_ptr).addressable_devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientAddressableMemories(arg_ptr: *mut PJRT_Client_AddressableMemories_Args) -> *mut PJRT_Error {
    println!("ClientAddressableMemories was called...");

    (*arg_ptr).num_addressable_memories = COREML_MEMORYS.len();
    (*arg_ptr).addressable_memories = COREML_MEMORYS_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientCompile(arg_ptr: *mut PJRT_Client_Compile_Args) -> *mut PJRT_Error {
    println!("ClientCompile was called...");

    let executable = Box::new(PJRT_LoadedExecutable {
        _unused: [0; 0],
    });

    (*arg_ptr).executable = Box::into_raw(executable);

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableGetExecutable(arg_ptr: *mut PJRT_LoadedExecutable_GetExecutable_Args) -> *mut PJRT_Error {
    println!("LoadedExecutableGetExecutable was called...");

    let executable = Box::new(PJRT_Executable {
        _unused: [0; 0],
    });

    (*arg_ptr).executable = Box::into_raw(executable);

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceAddressableMemories(arg_ptr: *mut PJRT_Device_AddressableMemories_Args) -> *mut PJRT_Error {
    println!("DeviceAddressableMemories was called...");

    (*arg_ptr).num_memories = COREML_MEMORYS.len();
    (*arg_ptr).memories = COREML_MEMORYS_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn PJRT_Client_TopologyDescription(arg_ptr: *mut PJRT_Client_TopologyDescription_Args) -> *mut PJRT_Error {
    println!("PJRT_Client_TopologyDescription was called...");

    let topology = Box::new(PJRT_TopologyDescription {
        _unused: [0; 0],
    });

    (*arg_ptr).topology = Box::into_raw(topology) as *mut PJRT_TopologyDescription;
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyDescriptionCreate(arg_ptr: *mut PJRT_TopologyDescription_Create_Args) -> *mut PJRT_Error {
    println!("TopologyDescriptionCreate was called...");

    let topology = Box::new(PJRT_TopologyDescription {
        _unused: [0; 0],
    });

    (*arg_ptr).topology = Box::into_raw(topology) as *mut PJRT_TopologyDescription;
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyDescriptionDestroy(arg_ptr: *mut PJRT_TopologyDescription_Destroy_Args) -> *mut PJRT_Error {
    println!("TopologyDescriptionDestroy was called...");

    if !(*arg_ptr).topology.is_null() {
        // Free the client memory
        unsafe { drop(Box::from_raw((*arg_ptr).topology)) };
    }
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyPlatformName(arg_ptr: *mut PJRT_TopologyDescription_PlatformName_Args) -> *mut PJRT_Error {
    println!("TopologyPlatformName was called...");

    (*arg_ptr).platform_name_size = PLATFORM_NAME.count_bytes();
    (*arg_ptr).platform_name = PLATFORM_NAME.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyPlatformVersion(arg_ptr: *mut PJRT_TopologyDescription_PlatformVersion_Args) -> *mut PJRT_Error {
    println!("TopologyPlatformVersion was called...");

    (*arg_ptr).platform_version_size = VERSION.count_bytes();
    (*arg_ptr).platform_version = VERSION.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyDeviceDescriptions(arg_ptr: *mut PJRT_TopologyDescription_GetDeviceDescriptions_Args) -> *mut PJRT_Error {
    println!("TopologyDeviceDescriptions was called...");

    (*arg_ptr).num_descriptions = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologySerialize(arg_ptr: *mut PJRT_TopologyDescription_Serialize_Args) -> *mut PJRT_Error {
    println!("TopologySerialize was called...");

    todo!("TopologySerialize is not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn TopologyAttributes(arg_ptr: *mut PJRT_TopologyDescription_Attributes_Args) -> *mut PJRT_Error {
    println!("TopologyAttributes was called...");

    (*arg_ptr).num_attributes = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn MemoryAddressableByDevices(arg_ptr: *mut PJRT_Memory_AddressableByDevices_Args) -> *mut PJRT_Error {
    println!("MemoryAddressableByDevices was called...");

    (*arg_ptr).num_devices = COREML_DEVICES.len();
    (*arg_ptr).devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableDestroy(arg_ptr: *mut PJRT_Executable_Destroy_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableDestroy")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableName(arg_ptr: *mut PJRT_Executable_Name_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableName")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableNumReplicas(arg_ptr: *mut PJRT_Executable_NumReplicas_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableNumReplicas")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableNumPartitions(arg_ptr: *mut PJRT_Executable_NumPartitions_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableNumPartitions")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableNumOutputs(arg_ptr: *mut PJRT_Executable_NumOutputs_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableNumOutputs")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableSizeOfGeneratedCodeInBytes(arg_ptr: *mut PJRT_Executable_SizeOfGeneratedCodeInBytes_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableSizeOfGeneratedCodeInBytes")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableGetCostAnalysis(arg_ptr: *mut PJRT_Executable_GetCostAnalysis_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableGetCostAnalysis")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableOutputMemoryKinds(arg_ptr: *mut PJRT_Executable_OutputMemoryKinds_Args) -> *mut PJRT_Error {
    println!("ExecutableOutputMemoryKinds was called");

    for i in 0..(*arg_ptr).num_outputs {
        let sizes_ptr = (*arg_ptr).memory_kind_sizes.add(i) as *mut usize;
        let memory_kind_ptr = (*arg_ptr).memory_kinds.add(i) as *mut *const i8;

        println!("Sizes ptr: {:?} - {:?}", sizes_ptr, (*arg_ptr).memory_kind_sizes);
        println!("Memory kind ptr: {:?} - {:?}", memory_kind_ptr, (*arg_ptr).memory_kinds);

        *sizes_ptr = PLATFORM_NAME.count_bytes();
        *memory_kind_ptr = PLATFORM_NAME.as_ptr();
    }

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableOptimizedProgram(arg_ptr: *mut PJRT_Executable_OptimizedProgram_Args) -> *mut PJRT_Error {
    println!("ExecutableOptimizedProgram was called");

    (*(*arg_ptr).program).format = PLATFORM_NAME.as_ptr();
    (*(*arg_ptr).program).format_size = PLATFORM_NAME.count_bytes();

    if (*(*arg_ptr).program).code.is_null() {
        // Calculate the size of the code
        (*(*arg_ptr).program).code_size = 16;
    } else {
        let num_bytes = (*(*arg_ptr).program).code_size;
        let bytes = vec![0; num_bytes];
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), (*(*arg_ptr).program).code, num_bytes);
    }

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableSerialize(arg_ptr: *mut PJRT_Executable_Serialize_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableSerialize")
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableDestroy(arg_ptr: *mut PJRT_LoadedExecutable_Destroy_Args) -> *mut PJRT_Error {
    todo!("Implement LoadedExecutableDestroy")
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableAddressableDevices(arg_ptr: *mut PJRT_LoadedExecutable_AddressableDevices_Args) -> *mut PJRT_Error {
    println!("LoadedExecutableAddressableDevices was called...");

    (*arg_ptr).num_addressable_devices = 1;
    (*arg_ptr).addressable_devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableDelete(arg_ptr: *mut PJRT_LoadedExecutable_Delete_Args) -> *mut PJRT_Error {
    todo!("Implement LoadedExecutableDelete")
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableIsDeleted(arg_ptr: *mut PJRT_LoadedExecutable_IsDeleted_Args) -> *mut PJRT_Error {
    todo!("Implement LoadedExecutableIsDeleted")
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableExecute(arg_ptr: *mut PJRT_LoadedExecutable_Execute_Args) -> *mut PJRT_Error {
    todo!("Implement LoadedExecutableExecute")
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableDeserializeAndLoad(arg_ptr: *mut PJRT_Executable_DeserializeAndLoad_Args) -> *mut PJRT_Error {
    todo!("Implement ExecutableDeserializeAndLoad")
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableFingerprint(arg_ptr: *mut PJRT_LoadedExecutable_Fingerprint_Args) -> *mut PJRT_Error {
    todo!("Implement LoadedExecutableFingerprint")
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDefaultMemory(arg_ptr: *mut PJRT_Device_DefaultMemory_Args) -> *mut PJRT_Error {
    println!("DeviceDefaultMemory was called...");

    (*arg_ptr).memory = COREML_MEMORYS_POINTERS[0];

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceMemoryStats(arg_ptr: *mut PJRT_Device_MemoryStats_Args) -> *mut PJRT_Error {
    println!("DeviceMemoryStats was called...");

    let mut sys = System::new_all();
    sys.refresh_all();

    (*arg_ptr).bytes_in_use = sys.used_memory() as i64;

    // TODO(knielsen): Fill in real values for these
    (*arg_ptr).peak_bytes_in_use_is_set = false;
    (*arg_ptr).num_allocs_is_set = false;
    (*arg_ptr).largest_alloc_size_is_set = false;
    (*arg_ptr).bytes_limit_is_set = false;

    (*arg_ptr).bytes_reserved_is_set = false;
    (*arg_ptr).peak_bytes_reserved_is_set = false;
    (*arg_ptr).bytes_reservable_limit_is_set = false;

    (*arg_ptr).largest_free_block_bytes_is_set = false;

    (*arg_ptr).pool_bytes_is_set = false;
    (*arg_ptr).peak_pool_bytes_is_set = false;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn MemoryId(arg_ptr: *mut PJRT_Memory_Id_Args) -> *mut PJRT_Error {
    println!("MemoryId was called...");

    todo!("MemoryId not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn MemoryKind(arg_ptr: *mut PJRT_Memory_Kind_Args) -> *mut PJRT_Error {
    println!("MemoryKind was called...");

    (*arg_ptr).kind = COREML_MEMORYS[0].kind.as_ptr();
    (*arg_ptr).kind_size = COREML_MEMORYS[0].kind.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn MemoryDebugString(arg_ptr: *mut PJRT_Memory_DebugString_Args) -> *mut PJRT_Error {
    println!("MemoryDebugString was called...");

    todo!("MemoryDebugString not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn MemoryToString(arg_ptr: *mut PJRT_Memory_ToString_Args) -> *mut PJRT_Error {
    println!("MemoryToString was called...");

    todo!("MemoryToString not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn PJRT_Compile(arg_ptr: *mut PJRT_Compile_Args) -> *mut PJRT_Error {
    println!("PJRT_Compile was called...");

    todo!("PJRT_Compile not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn PJRT_MemoryKindId(arg_ptr: *mut PJRT_Memory_Kind_Id_Args) -> *mut PJRT_Error {
    println!("PJRT_MemoryKindId was called...");

    todo!("PJRT_MemoryKindId not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn GetPjrtApi() -> *mut PJRT_Api {
    println!("Attempted to get the CoreML PjrtApi");

    let pjrt_version = PJRT_Api_Version {
        struct_size: 0,
        extension_start: ptr::null_mut(),
        major_version: PJRT_API_MAJOR as i32,
        minor_version: PJRT_API_MINOR as i32,
    };

    let pjrt_api = Box::new(PJRT_Api {
        struct_size: 0,
        extension_start: ptr::null_mut(),
        pjrt_api_version: pjrt_version,
        PJRT_Error_Destroy: None,
        PJRT_Error_Message: None,
        PJRT_Error_GetCode: None,
        PJRT_Plugin_Initialize: Some(PluginInitialize),
        PJRT_Plugin_Attributes: Some(PluginAttributes),
        PJRT_Event_Destroy: None,
        PJRT_Event_IsReady: None,
        PJRT_Event_Error: None,
        PJRT_Event_Await: None,
        PJRT_Event_OnReady: None,
        PJRT_Client_Create: Some(ClientCreate),
        PJRT_Client_Destroy: Some(ClientDestroy),
        PJRT_Client_PlatformName: Some(ClientPlatformName),
        PJRT_Client_ProcessIndex: Some(ClientProcessIndex),
        PJRT_Client_PlatformVersion: Some(ClientPlatformVersion),
        PJRT_Client_Devices: Some(ClientDevices),
        PJRT_Client_AddressableDevices: Some(ClientAddressableDevices),
        PJRT_Client_LookupDevice: None,
        PJRT_Client_LookupAddressableDevice: None,
        PJRT_Client_AddressableMemories: Some(ClientAddressableMemories),
        PJRT_Client_Compile: Some(ClientCompile),
        PJRT_Client_DefaultDeviceAssignment: None,
        PJRT_Client_BufferFromHostBuffer: None,
        PJRT_DeviceDescription_Id: Some(DeviceDescriptionId),
        PJRT_DeviceDescription_ProcessIndex: Some(DeviceDescriptiopnProcessIndex),
        PJRT_DeviceDescription_Attributes: Some(DeviceDescriptionAttributes),
        PJRT_DeviceDescription_Kind: Some(DeviceDescriptionKind),
        PJRT_DeviceDescription_DebugString: Some(DeviceDescriptionDebugString),
        PJRT_DeviceDescription_ToString: Some(DeviceDescriptionToString),
        PJRT_Device_GetDescription: Some(DeviceGetDescription),
        PJRT_Device_IsAddressable: Some(DeviceIsAddressable),
        PJRT_Device_LocalHardwareId: None,
        PJRT_Device_AddressableMemories: Some(DeviceAddressableMemories),
        PJRT_Device_DefaultMemory: Some(DeviceDefaultMemory),
        PJRT_Device_MemoryStats: Some(DeviceMemoryStats),
        PJRT_Memory_Id: Some(MemoryId),
        PJRT_Memory_Kind: Some(MemoryKind),
        PJRT_Memory_DebugString: Some(MemoryDebugString),
        PJRT_Memory_ToString: Some(MemoryToString),
        PJRT_Memory_AddressableByDevices: Some(MemoryAddressableByDevices),
        PJRT_Executable_Destroy: Some(ExecutableDestroy),
        PJRT_Executable_Name: Some(ExecutableName),
        PJRT_Executable_NumReplicas: Some(ExecutableNumReplicas),
        PJRT_Executable_NumPartitions: Some(ExecutableNumPartitions),
        PJRT_Executable_NumOutputs: Some(ExecutableNumOutputs),
        PJRT_Executable_SizeOfGeneratedCodeInBytes: Some(ExecutableSizeOfGeneratedCodeInBytes),
        PJRT_Executable_GetCostAnalysis: Some(ExecutableGetCostAnalysis),
        PJRT_Executable_OutputMemoryKinds: Some(ExecutableOutputMemoryKinds),
        PJRT_Executable_OptimizedProgram: Some(ExecutableOptimizedProgram),
        PJRT_Executable_Serialize: Some(ExecutableSerialize),
        PJRT_LoadedExecutable_Destroy: Some(LoadedExecutableDestroy),
        PJRT_LoadedExecutable_GetExecutable: Some(LoadedExecutableGetExecutable),
        PJRT_LoadedExecutable_AddressableDevices: Some(LoadedExecutableAddressableDevices),
        PJRT_LoadedExecutable_Delete: Some(LoadedExecutableDelete),
        PJRT_LoadedExecutable_IsDeleted: Some(LoadedExecutableIsDeleted),
        PJRT_LoadedExecutable_Execute: Some(LoadedExecutableExecute),
        PJRT_Executable_DeserializeAndLoad: Some(ExecutableDeserializeAndLoad),
        PJRT_LoadedExecutable_Fingerprint: Some(LoadedExecutableFingerprint),
        PJRT_Buffer_Destroy: None,
        PJRT_Buffer_ElementType: None,
        PJRT_Buffer_Dimensions: None,
        PJRT_Buffer_UnpaddedDimensions: None,
        PJRT_Buffer_DynamicDimensionIndices: None,
        PJRT_Buffer_GetMemoryLayout: None,
        PJRT_Buffer_OnDeviceSizeInBytes: None,
        PJRT_Buffer_Device: None,
        PJRT_Buffer_Memory: None,
        PJRT_Buffer_Delete: None,
        PJRT_Buffer_IsDeleted: None,
        PJRT_Buffer_CopyToDevice: None,
        PJRT_Buffer_ToHostBuffer: None,
        PJRT_Buffer_IsOnCpu: None,
        PJRT_Buffer_ReadyEvent: None,
        PJRT_Buffer_UnsafePointer: None,
        PJRT_Buffer_IncreaseExternalReferenceCount: None,
        PJRT_Buffer_DecreaseExternalReferenceCount: None,
        PJRT_Buffer_OpaqueDeviceMemoryDataPointer: None,
        PJRT_CopyToDeviceStream_Destroy: None,
        PJRT_CopyToDeviceStream_AddChunk: None,
        PJRT_CopyToDeviceStream_TotalBytes: None,
        PJRT_CopyToDeviceStream_GranuleSize: None,
        PJRT_CopyToDeviceStream_CurrentBytes: None,
        PJRT_TopologyDescription_Create: Some(TopologyDescriptionCreate),
        PJRT_TopologyDescription_Destroy: Some(TopologyDescriptionDestroy),
        PJRT_TopologyDescription_PlatformName: Some(TopologyPlatformName),
        PJRT_TopologyDescription_PlatformVersion: Some(TopologyPlatformVersion),
        PJRT_TopologyDescription_GetDeviceDescriptions: Some(TopologyDeviceDescriptions),
        PJRT_TopologyDescription_Serialize: Some(TopologySerialize),
        PJRT_TopologyDescription_Attributes: Some(TopologyAttributes),
        PJRT_Compile: Some(PJRT_Compile),
        PJRT_Executable_OutputElementTypes: None,
        PJRT_Executable_OutputDimensions: None,
        PJRT_Buffer_CopyToMemory: None,
        PJRT_Client_CreateViewOfDeviceBuffer: None,
        PJRT_Executable_Fingerprint: None,
        PJRT_Client_TopologyDescription: Some(PJRT_Client_TopologyDescription),
        PJRT_Executable_GetCompiledMemoryStats: None,
        PJRT_Memory_Kind_Id: Some(PJRT_MemoryKindId),
        PJRT_ExecuteContext_Create: None,
        PJRT_ExecuteContext_Destroy: None,
    });
    Box::into_raw(pjrt_api) as *mut PJRT_Api
}
