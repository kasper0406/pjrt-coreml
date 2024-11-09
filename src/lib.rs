#![feature(vec_into_raw_parts, box_as_ptr, async_closure)]

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));


#[macro_use]
extern crate lazy_static;

mod coreml;

use core::slice;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ffi::CString;
use std::ptr;
use std::sync::RwLock;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::alloc::{alloc, dealloc, Layout};
use uuid::Uuid;
use std::future::Future;
use async_std::prelude::*;
use async_std::task;
use async_std::task::JoinHandle;
use std::sync::{Arc, Mutex};

use sysinfo::System;

use log::{debug, info, warn};

use coreml::CoreMLBuffer;


static mut COREML_DEVICES_POINTERS: Vec<*mut PJRT_Device> = vec![];
static mut COREML_MEMORYS_POINTERS: Vec<*mut PJRT_Memory> = vec![];

struct Error {
    pjrt_error: PJRT_Error,

    message: CString,
    error_code: PJRT_Error_Code,
}

impl Error {
    fn new_alloc(message: String) -> *mut PJRT_Error {
        Box::into_raw(Box::new(Error {
            pjrt_error: PJRT_Error { _unused: [0; 0] },
            message: CString::new(message).unwrap(),
            // TODO(knielsen): Support additional error codes...
            error_code: PJRT_Error_Code_PJRT_Error_Code_INTERNAL,
        })) as *mut PJRT_Error
    }
}

struct Device {
    device: Box<PJRT_Device>,
    description: CString,
}

struct Memory {
    memory: Box<PJRT_Memory>,
    kind: CString,
}

#[derive(Debug)]
struct Executable {
    pjrt_executable: PJRT_Executable,

    memory_kinds: Vec<CString>,
    fingerprint: CString,

    // C pointers
    memory_kinds_ptr: Vec<*const i8>,
    memory_kinds_sizes: Vec<usize>,
}

impl Executable {
    fn new() -> Self {
        // TODO(knielsen): Fix these memory_kinds to line up with COREML_MEMORYS
        let memory_kinds = vec![CString::new("CoreML Unified Memory").unwrap()];
        let memory_kinds_ptr: Vec<*const i8> = memory_kinds.iter()
            .map(|memory_kind| memory_kind.as_ptr())
            .collect();
        let memory_kinds_sizes = memory_kinds.iter()
            .map(|memory_kind| memory_kind.count_bytes())
            .collect();

        Executable {
            pjrt_executable: PJRT_Executable{ _unused: [0; 0] },
            // TODO(knielsen): Implement proper fingerprinting!
            fingerprint: CString::new(Uuid::new_v4().to_string()).unwrap(),
            memory_kinds,
            memory_kinds_ptr,
            memory_kinds_sizes,
        }
    }
}

struct LoadedExecutable {
    pjrt_loaded_executable: PJRT_LoadedExecutable,

    executable: Executable,
}

impl LoadedExecutable {
    fn new(executable: Executable) -> Self {
        LoadedExecutable {
            pjrt_loaded_executable: PJRT_LoadedExecutable { _unused: [0; 0] },
            executable,
        }
    }
}

struct Buffer {
    pjrt_buffer: PJRT_Buffer,
    ref_count: usize,

    buffer: Option<CoreMLBuffer>,

    // Buffered fields to provide to the C API
    dims: Vec<i64>,
}

impl Buffer {
    fn new(mut maybe_buffer: Option<CoreMLBuffer>) -> Self {
        let dims = match maybe_buffer {
            Some(ref mut tensor) => tensor.shape(),
            None => vec![],
        };
        debug!["Created buffer with shape: {:?}", dims];

        Buffer {
            pjrt_buffer: PJRT_Buffer { _unused: [0; 0] },
            ref_count: 0,
            buffer: maybe_buffer,
            dims,
        }
    }

    pub fn raw_data_pointer(&mut self) -> Option<*mut c_void> {
        self.buffer.as_mut().map(|mut buffer| buffer.raw_data_pointer())
    }
}

#[derive(Debug)]
struct EventCallback {
    callback: PJRT_Event_OnReadyCallback,
    user_arg: *mut c_void,
}
unsafe impl Send for EventCallback {}

struct CallbackInfo {
    task_completed: bool,
    callbacks: Vec<EventCallback>,
}

impl CallbackInfo {
    fn new() -> Self {
        CallbackInfo {
            task_completed: false,
            callbacks: vec![],
        }
    }
}

struct Event {
    event: PJRT_Event,

    future: Option<JoinHandle<()>>,
    callback_info: Arc<Mutex<CallbackInfo>>,
}

impl Event {
    fn new(task: JoinHandle<()>) -> Self {
        let mut event = Event {
            event: PJRT_Event { _unused: [0; 0] },
            future: None,
            callback_info: Arc::new(Mutex::new(CallbackInfo::new())),
        };

        let callback_info_ref = event.callback_info.clone();
        let run_task_and_perform_callbacks = task::spawn(async move {
            task.await;

            let mut callback_info = callback_info_ref.lock().unwrap();
            callback_info.task_completed = true;
            for callback in callback_info.callbacks.iter() {
                debug!("Background task completed, calling callback: {:?}", callback);
                unsafe { callback.callback.unwrap()(ptr::null_mut(), callback.user_arg) };
            }
            callback_info.callbacks.clear();
        });
        event.future = Some(run_task_and_perform_callbacks);

        event
    }

    fn add_callback(&mut self, callback: EventCallback) {
        let mut callback_info = self.callback_info.lock().unwrap();

        if callback_info.task_completed {
            debug!("Task was already completed, calling callback directly");
            // The task already completed, just call the callback directly
            unsafe { callback.callback.unwrap()(ptr::null_mut(), callback.user_arg) };
        } else {
            callback_info.callbacks.push(callback);
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
}

#[no_mangle]
pub unsafe extern "C" fn ErrorDestroy(arg_ptr: *mut PJRT_Error_Destroy_Args) -> () {
    info!("ErrorDestroy was called");

    let error_ptr = (*arg_ptr).error;
    if error_ptr.is_null() {
        debug!("ErrorDestroy called on a nullptr");
    } else {
        drop(Box::from_raw(error_ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ErrorMessage(arg_ptr: *mut PJRT_Error_Message_Args) -> () {
    info!("ErrorMessage was called");

    let error_ptr = (*arg_ptr).error as *mut Error;
    (*arg_ptr).message = (*error_ptr).message.as_ptr();
    (*arg_ptr).message_size = (*error_ptr).message.count_bytes();
}

#[no_mangle]
pub unsafe extern "C" fn ErrorGetCode(arg_ptr: *mut PJRT_Error_GetCode_Args) -> *mut PJRT_Error {
    info!("ErrorGetCode was called");

    let error_ptr = (*arg_ptr).error as *mut Error;
    (*arg_ptr).code = (*error_ptr).error_code;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn PluginInitialize(arg_ptr: *mut PJRT_Plugin_Initialize_Args) -> *mut PJRT_Error {
    info!("PluginInitialize was called...");
    env_logger::init();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn PluginAttributes(arg_ptr: *mut PJRT_Plugin_Attributes_Args) -> *mut PJRT_Error {
    info!("PluginAttributes was called...");

    (*arg_ptr).num_attributes = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn EventDestroy(arg_ptr: *mut PJRT_Event_Destroy_Args) -> *mut PJRT_Error {
    info!("EventDestroy was called...");

    let event_ptr = (*arg_ptr).event as *mut Event;
    if !event_ptr.is_null() {
        drop(Box::from_raw(event_ptr));
    }

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn EventIsReady(arg_ptr: *mut PJRT_Event_IsReady_Args) -> *mut PJRT_Error {
    todo!("EventIsReady was called...")
}

#[no_mangle]
pub unsafe extern "C" fn EventError(arg_ptr: *mut PJRT_Event_Error_Args) -> *mut PJRT_Error {
    todo!("EventError was called...")
}

#[no_mangle]
pub unsafe extern "C" fn EventAwait(arg_ptr: *mut PJRT_Event_Await_Args) -> *mut PJRT_Error {
    todo!("EventAwait was called...")
}

#[no_mangle]
pub unsafe extern "C" fn EventOnReady(arg_ptr: *mut PJRT_Event_OnReady_Args) -> *mut PJRT_Error {
    info!("EventOnReady was called...");

    let mut event_ptr = (*arg_ptr).event as *mut Event;
    (*event_ptr).add_callback(EventCallback {
        callback: (*arg_ptr).callback,
        user_arg: (*arg_ptr).user_arg,
    });

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientCreate(arg_ptr: *mut PJRT_Client_Create_Args) -> *mut PJRT_Error {
    info!("ClientCreate was called...");

    let client = Box::new(PJRT_Client {
        _unused: [0; 0],
    });

    (*arg_ptr).client = Box::into_raw(client) as *mut PJRT_Client;
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientDestroy(arg_ptr: *mut PJRT_Client_Destroy_Args) -> *mut PJRT_Error {
    info!("ClientDestroy was called...");

    if !(*arg_ptr).client.is_null() {
        // Free the client memory
        unsafe { drop(Box::from_raw((*arg_ptr).client)) };
    }
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientPlatformName(arg_ptr: *mut PJRT_Client_PlatformName_Args) -> *mut PJRT_Error {
    info!("ClientPlatformName was called...");

    (*arg_ptr).platform_name_size = PLATFORM_NAME.count_bytes();
    (*arg_ptr).platform_name = PLATFORM_NAME.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientProcessIndex(arg_ptr: *mut PJRT_Client_ProcessIndex_Args) -> *mut PJRT_Error {
    info!("ClientProcessIndex was called...");

    (*arg_ptr).process_index = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientPlatformVersion(arg_ptr: *mut PJRT_Client_PlatformVersion_Args) -> *mut PJRT_Error {
    info!("ClientPlatformVersion was called...");

    (*arg_ptr).platform_version_size = VERSION.count_bytes();
    (*arg_ptr).platform_version = VERSION.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientDevices(arg_ptr: *mut PJRT_Client_Devices_Args) -> *mut PJRT_Error {
    info!("ClientDevices was called...");

    (*arg_ptr).num_devices = COREML_DEVICES.len();
    (*arg_ptr).devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceGetDescription(arg_ptr: *mut PJRT_Device_GetDescription_Args) -> *mut PJRT_Error {
    info!("DeviceGetDescription was called...");

    let description = Box::new(PJRT_DeviceDescription {
        _unused: [0; 0],
    });
    (*arg_ptr).device_description = Box::into_raw(description) as *mut PJRT_DeviceDescription;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceIsAddressable(arg_ptr: *mut PJRT_Device_IsAddressable_Args) -> *mut PJRT_Error {
    info!("DeviceIsAddressable was called...");

    (*arg_ptr).is_addressable = true;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionId(arg_ptr: *mut PJRT_DeviceDescription_Id_Args) -> *mut PJRT_Error {
    info!("DeviceDescriptionId was called...");

    (*arg_ptr).id = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptiopnProcessIndex(arg_ptr: *mut PJRT_DeviceDescription_ProcessIndex_Args) -> *mut PJRT_Error {
    info!("DeviceDescriptiopnProcessIndex was called...");

    (*arg_ptr).process_index = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionAttributes(arg_ptr: *mut PJRT_DeviceDescription_Attributes_Args) -> *mut PJRT_Error {
    info!("DeviceDescriptionAttributes was called...");

    (*arg_ptr).num_attributes = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionKind(arg_ptr: *mut PJRT_DeviceDescription_Kind_Args) -> *mut PJRT_Error {
    info!("DeviceDescriptionKind was called...");

    (*arg_ptr).device_kind = COREML_DEVICES[0].description.as_ptr();
    (*arg_ptr).device_kind_size = COREML_DEVICES[0].description.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionDebugString(arg_ptr: *mut PJRT_DeviceDescription_DebugString_Args) -> *mut PJRT_Error {
    info!("DeviceDescriptionDebugString was called...");

    (*arg_ptr).debug_string = COREML_DEVICES[0].description.as_ptr();
    (*arg_ptr).debug_string_size = COREML_DEVICES[0].description.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDescriptionToString(arg_ptr: *mut PJRT_DeviceDescription_ToString_Args) -> *mut PJRT_Error {
    info!("DeviceDescriptionToString was called...");

    (*arg_ptr).to_string = COREML_DEVICES[0].description.as_ptr();
    (*arg_ptr).to_string_size = COREML_DEVICES[0].description.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientAddressableDevices(arg_ptr: *mut PJRT_Client_AddressableDevices_Args) -> *mut PJRT_Error {
    info!("ClientAddressableDevices was called...");

    (*arg_ptr).num_addressable_devices = COREML_DEVICES.len();
    (*arg_ptr).addressable_devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientAddressableMemories(arg_ptr: *mut PJRT_Client_AddressableMemories_Args) -> *mut PJRT_Error {
    info!("ClientAddressableMemories was called...");

    (*arg_ptr).num_addressable_memories = COREML_MEMORYS.len();
    (*arg_ptr).addressable_memories = COREML_MEMORYS_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientCompile(arg_ptr: *mut PJRT_Client_Compile_Args) -> *mut PJRT_Error {
    info!("ClientCompile was called...");

    let executable = Box::new(LoadedExecutable::new(Executable::new()));
    (*arg_ptr).executable = Box::into_raw(executable) as *mut PJRT_LoadedExecutable;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableGetExecutable(arg_ptr: *mut PJRT_LoadedExecutable_GetExecutable_Args) -> *mut PJRT_Error {
    info!("LoadedExecutableGetExecutable was called...");

    let loaded_executable_ptr = (*arg_ptr).loaded_executable as *mut LoadedExecutable;
    (*arg_ptr).executable = ptr::addr_of_mut!((*loaded_executable_ptr).executable) as *mut PJRT_Executable;
    info!("Sending back executable: {:?}", *arg_ptr);

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceAddressableMemories(arg_ptr: *mut PJRT_Device_AddressableMemories_Args) -> *mut PJRT_Error {
    info!("DeviceAddressableMemories was called...");

    (*arg_ptr).num_memories = COREML_MEMORYS.len();
    (*arg_ptr).memories = COREML_MEMORYS_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn Executable_Fingerprint(arg_ptr: *mut PJRT_Executable_Fingerprint_Args) -> *mut PJRT_Error {
    info!("PJRT_Executable_Fingerprint was called...");

    let executable_ptr = (*arg_ptr).executable as *mut Executable;

    (*arg_ptr).executable_fingerprint_size = (*executable_ptr).fingerprint.count_bytes();
    (*arg_ptr).executable_fingerprint = (*executable_ptr).fingerprint.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn Client_TopologyDescription(arg_ptr: *mut PJRT_Client_TopologyDescription_Args) -> *mut PJRT_Error {
    info!("PJRT_Client_TopologyDescription was called...");

    let topology = Box::new(PJRT_TopologyDescription {
        _unused: [0; 0],
    });

    (*arg_ptr).topology = Box::into_raw(topology) as *mut PJRT_TopologyDescription;
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn CopyToDeviceStreamDestroy(arg_ptr: *mut PJRT_CopyToDeviceStream_Destroy_Args) -> *mut PJRT_Error {
    todo!("Implement CopyToDeviceStreamDestroy")
}

#[no_mangle]
pub unsafe extern "C" fn CopyToDeviceStreamAddChunk(arg_ptr: *mut PJRT_CopyToDeviceStream_AddChunk_Args) -> *mut PJRT_Error {
    todo!("Implement CopyToDeviceStreamAddChunk")
}

#[no_mangle]
pub unsafe extern "C" fn CopyToDeviceStreamTotalBytes(arg_ptr: *mut PJRT_CopyToDeviceStream_TotalBytes_Args) -> *mut PJRT_Error {
    todo!("Implement CopyToDeviceStreamTotalBytes")
}

#[no_mangle]
pub unsafe extern "C" fn CopyToDeviceStreamGranuleSize(arg_ptr: *mut PJRT_CopyToDeviceStream_GranuleSize_Args) -> *mut PJRT_Error {
    todo!("Implement CopyToDeviceStreamGranuleSize")
}

#[no_mangle]
pub unsafe extern "C" fn CopyToDeviceStreamCurrentBytes(arg_ptr: *mut PJRT_CopyToDeviceStream_CurrentBytes_Args) -> *mut PJRT_Error {
    todo!("Implement CopyToDeviceStreamCurrentBytes")
}

#[no_mangle]
pub unsafe extern "C" fn TopologyDescriptionCreate(arg_ptr: *mut PJRT_TopologyDescription_Create_Args) -> *mut PJRT_Error {
    info!("TopologyDescriptionCreate was called...");

    let topology = Box::new(PJRT_TopologyDescription {
        _unused: [0; 0],
    });

    (*arg_ptr).topology = Box::into_raw(topology) as *mut PJRT_TopologyDescription;
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyDescriptionDestroy(arg_ptr: *mut PJRT_TopologyDescription_Destroy_Args) -> *mut PJRT_Error {
    info!("TopologyDescriptionDestroy was called...");

    if !(*arg_ptr).topology.is_null() {
        // Free the client memory
        unsafe { drop(Box::from_raw((*arg_ptr).topology)) };
    }
    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyPlatformName(arg_ptr: *mut PJRT_TopologyDescription_PlatformName_Args) -> *mut PJRT_Error {
    info!("TopologyPlatformName was called...");

    (*arg_ptr).platform_name_size = PLATFORM_NAME.count_bytes();
    (*arg_ptr).platform_name = PLATFORM_NAME.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyPlatformVersion(arg_ptr: *mut PJRT_TopologyDescription_PlatformVersion_Args) -> *mut PJRT_Error {
    info!("TopologyPlatformVersion was called...");

    (*arg_ptr).platform_version_size = VERSION.count_bytes();
    (*arg_ptr).platform_version = VERSION.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologyDeviceDescriptions(arg_ptr: *mut PJRT_TopologyDescription_GetDeviceDescriptions_Args) -> *mut PJRT_Error {
    info!("TopologyDeviceDescriptions was called...");

    (*arg_ptr).num_descriptions = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn TopologySerialize(arg_ptr: *mut PJRT_TopologyDescription_Serialize_Args) -> *mut PJRT_Error {
    info!("TopologySerialize was called...");

    todo!("TopologySerialize is not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn TopologyAttributes(arg_ptr: *mut PJRT_TopologyDescription_Attributes_Args) -> *mut PJRT_Error {
    info!("TopologyAttributes was called...");

    (*arg_ptr).num_attributes = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn MemoryAddressableByDevices(arg_ptr: *mut PJRT_Memory_AddressableByDevices_Args) -> *mut PJRT_Error {
    info!("MemoryAddressableByDevices was called...");

    (*arg_ptr).num_devices = COREML_DEVICES.len();
    (*arg_ptr).devices = COREML_DEVICES_POINTERS.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableDestroy(arg_ptr: *mut PJRT_Executable_Destroy_Args) -> *mut PJRT_Error {
    info!("ExecutableDestroy was called...");

    // TODO(knielsen): Consider if this is ok?
    // We keep the executable around for as long as the LoadedExecutable is alive

    // let executable_ptr = (*arg_ptr).executable as *mut Executable;
    // let executable = Box::from_raw(executable_ptr);
    // drop(executable);

    ptr::null_mut()
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
    info!("ExecutableNumOutputs was called...");

    let executable_ptr = (*arg_ptr).executable as *mut Executable;
    (*arg_ptr).num_outputs = (*executable_ptr).memory_kinds.len();

    ptr::null_mut()
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
    info!("ExecutableOutputMemoryKinds was called");

    let executable_ptr = (*arg_ptr).executable as *mut Executable;

    (*arg_ptr).num_outputs = (*executable_ptr).memory_kinds.len();
    (*arg_ptr).memory_kind_sizes = (*executable_ptr).memory_kinds_sizes.as_ptr();
    (*arg_ptr).memory_kinds = (*executable_ptr).memory_kinds_ptr.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ExecutableOptimizedProgram(arg_ptr: *mut PJRT_Executable_OptimizedProgram_Args) -> *mut PJRT_Error {
    info!("ExecutableOptimizedProgram was called");

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
    info!("LoadedExecutableDestroy was called...");

    let loaded_executable_ptr = (*arg_ptr).executable as *mut LoadedExecutable;
    let loaded_executable = Box::from_raw(loaded_executable_ptr);
    drop(loaded_executable);

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn LoadedExecutableAddressableDevices(arg_ptr: *mut PJRT_LoadedExecutable_AddressableDevices_Args) -> *mut PJRT_Error {
    info!("LoadedExecutableAddressableDevices was called...");

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
    info!("LoadedExecutableExecute was called...");

    // TODO(knielsen): Implement actual execution...

    let loaded_executable_ptr = (*arg_ptr).executable as *mut LoadedExecutable;
    let num_outputs = (*loaded_executable_ptr).executable.memory_kinds.len();

    if (*arg_ptr).num_devices != 1 {
        panic!("There should be exactly 1 CoreML device!");
    }

    for output_idx in 0..num_outputs {
        // TODO(knielsen): Get this information from the program. For now just make up something...
        let elementType = coreml::ElementType::F16;
        let shape = vec![3, 2];
        let strides = vec![2, 1];

        let coreml_buffer = coreml::CoreMLBuffer::allocate_shape(elementType, shape, strides);
        let buffer = Box::new(Buffer::new(Some(coreml_buffer)));

        let device_outputs = *(*arg_ptr).output_lists;
        *device_outputs.add(output_idx) = Box::into_raw(buffer) as *mut PJRT_Buffer;
    }

    if !(*arg_ptr).device_complete_events.is_null() {
        let future = task::spawn(async {
            info!("Pretending to run computation in the background...");
            task::sleep(std::time::Duration::from_secs(10)).await;
            info!("Finished pretended computation...");
        });
        let event = Box::new(Event::new(future));
        let events_ptr = (*arg_ptr).device_complete_events;
        *events_ptr = Box::into_raw(event) as *mut PJRT_Event;
    }

    ptr::null_mut()
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
pub unsafe extern "C" fn BufferDestroy(arg_ptr: *mut PJRT_Buffer_Destroy_Args) -> *mut PJRT_Error {
    info!("BufferDestroy was called...");

    let buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    let buffer = Box::from_raw(buffer_ptr);
    drop(buffer);

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferElementType(arg_ptr: *mut PJRT_Buffer_ElementType_Args) -> *mut PJRT_Error {
    info!("BufferElementType was called...");

    // TODO(knielsen): Make a proper implementation of this
    (*arg_ptr).type_ = PJRT_Buffer_Type_PJRT_Buffer_Type_F32;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferDimensions(arg_ptr: *mut PJRT_Buffer_Dimensions_Args) -> *mut PJRT_Error {
    info!("BufferDimensions was called...");

    let buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    (*arg_ptr).num_dims = (*buffer_ptr).dims.len();
    (*arg_ptr).dims = (*buffer_ptr).dims.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferUnpaddedDimensions(arg_ptr: *mut PJRT_Buffer_UnpaddedDimensions_Args) -> *mut PJRT_Error {
    info!("BufferUnpaddedDimensions was called...");

    let buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    (*arg_ptr).num_dims = (*buffer_ptr).dims.len();
    (*arg_ptr).unpadded_dims = (*buffer_ptr).dims.as_ptr();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferDynamicDimensionIndices(arg_ptr: *mut PJRT_Buffer_DynamicDimensionIndices_Args) -> *mut PJRT_Error {
    info!("BufferDynamicDimensionIndices was called...");

    // let buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    (*arg_ptr).num_dynamic_dims = 0;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferMemoryLayout(arg_ptr: *mut PJRT_Buffer_GetMemoryLayout_Args) -> *mut PJRT_Error {
    todo!("Implement BufferMemoryLayout")
}

#[no_mangle]
pub unsafe extern "C" fn BufferDeviceSizeInBytes(arg_ptr: *mut PJRT_Buffer_OnDeviceSizeInBytes_Args) -> *mut PJRT_Error {
    todo!("Implement BufferDeviceSizeInBytes")
}

#[no_mangle]
pub unsafe extern "C" fn BufferDevice(arg_ptr: *mut PJRT_Buffer_Device_Args) -> *mut PJRT_Error {
    info!("BufferDevice was called...");

    (*arg_ptr).device = COREML_DEVICES_POINTERS[0];

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferMemory(arg_ptr: *mut PJRT_Buffer_Memory_Args) -> *mut PJRT_Error {
    info!("BufferMemory was called...");

    (*arg_ptr).memory = COREML_MEMORYS_POINTERS[0];

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferDelete(arg_ptr: *mut PJRT_Buffer_Delete_Args) -> *mut PJRT_Error {
    todo!("Implement BufferDelete")
}

#[no_mangle]
pub unsafe extern "C" fn BufferIsDeleted(arg_ptr: *mut PJRT_Buffer_IsDeleted_Args) -> *mut PJRT_Error {
    info!("BufferIsDeleted was called...");

    // TODO(knielsen): Support deletion...
    (*arg_ptr).is_deleted = false;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferCopyToDevice(arg_ptr: *mut PJRT_Buffer_CopyToDevice_Args) -> *mut PJRT_Error {
    todo!("Implement BufferCopyToDevice")
}

#[no_mangle]
pub unsafe extern "C" fn BufferToHostBuffer(arg_ptr: *mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PJRT_Error {
    todo!("Implement BufferToHostBuffer")
}

#[no_mangle]
pub unsafe extern "C" fn BufferIsOnCpu(arg_ptr: *mut PJRT_Buffer_IsOnCpu_Args) -> *mut PJRT_Error {
    info!("BufferIsOnCpu was called...");

    // The CoreML buffers are always on the CPU
    (*arg_ptr).is_on_cpu = true;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferReadyEvent(arg_ptr: *mut PJRT_Buffer_ReadyEvent_Args) -> *mut PJRT_Error {
    info!("BufferReadyEvent was called...");

    let buffer_immediately_ready = task::spawn(async {});
    let event = Box::new(Event::new(buffer_immediately_ready));
    (*arg_ptr).event = Box::into_raw(event) as *mut PJRT_Event;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferUnsafePointer(arg_ptr: *mut PJRT_Buffer_UnsafePointer_Args) -> *mut PJRT_Error {
    todo!("Implement BufferUnsafePointer")
}

#[no_mangle]
pub unsafe extern "C" fn BufferIncreaseRefCount(arg_ptr: *mut PJRT_Buffer_IncreaseExternalReferenceCount_Args) -> *mut PJRT_Error {
    info!("BufferIncreaseRefCount was called...");

    let buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    (*buffer_ptr).ref_count += 1;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferDecreaseRefCount(arg_ptr: *mut PJRT_Buffer_DecreaseExternalReferenceCount_Args) -> *mut PJRT_Error {
    info!("BufferDecreaseRefCount was called...");

    let buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    if (*buffer_ptr).ref_count == 0 {
        // TODO(knielsen): Make this return a proper error...
        panic!("Ref count reached <0!");
    }
    (*buffer_ptr).ref_count -= 1;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferOpaqueDeviceMemoryDataPointer(arg_ptr: *mut PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args) -> *mut PJRT_Error {
    info!("BufferDecreaseRefCount was called...");

    let mut buffer_ptr = (*arg_ptr).buffer as *mut Buffer;
    (*arg_ptr).device_memory_ptr = (*buffer_ptr).raw_data_pointer().unwrap();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn BufferCopyToMemory(arg_ptr: *mut PJRT_Buffer_CopyToMemory_Args) -> *mut PJRT_Error {
    todo!("BufferCopyToMemory missing")
}

#[no_mangle]
pub unsafe extern "C" fn ClientBufferFromHostBuffer(arg_ptr: *mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PJRT_Error {
    info!("ClientBufferFromHostBuffer was called");
    
    // Compute shape
    let shape = unsafe { slice::from_raw_parts((*arg_ptr).dims, (*arg_ptr).num_dims) };
    info!["Shape: {:?}", shape];

    // Compute strides
    let mut strides = vec![];
    if (*arg_ptr).device_layout.is_null() {
        // Major-to-minor order
        strides.push(1);
        for (axis, dimension) in shape.iter().skip(1).rev().enumerate() {
            strides.push(dimension * strides[axis]);
        }
        info!["Calculated strides: {:?}", strides];
    } else {
        let device_layout = *(*arg_ptr).device_layout;
        match device_layout.type_ {
            PJRT_Buffer_MemoryLayout_Type_PJRT_Buffer_MemoryLayout_Type_Tiled => {
                todo!("Implement tiled buffers")
            },
            PJRT_Buffer_MemoryLayout_Type_PJRT_Buffer_MemoryLayout_Type_Strides => {
                let num_strides = device_layout.__bindgen_anon_1.strides.num_byte_strides;
                todo!("Strided!")
            },
            _ => todo!("Unexpected memory layout!"),
        }
    }

    // Compute element type
    let element_type = match (*arg_ptr).type_ {
        PJRT_Buffer_Type_PJRT_Buffer_Type_F16 => coreml::ElementType::F16,
        PJRT_Buffer_Type_PJRT_Buffer_Type_F32 => coreml::ElementType::F32,
        PJRT_Buffer_Type_PJRT_Buffer_Type_F64 => coreml::ElementType::F64,
        PJRT_Buffer_Type_PJRT_Buffer_Type_S32 => coreml::ElementType::I32,
        unsupported_type => todo!("Type not yet supported: {:?}", unsupported_type)
    };

    let numBytes = (shape.iter().product::<i64>() as usize) * element_type.width();
    let data = unsafe { slice::from_raw_parts((*arg_ptr).data as *const u8, numBytes) };

    let coreml_buffer = CoreMLBuffer::allocate_from_data(data, element_type, shape, &strides);
    let buffer = Box::new(Buffer::new(Some(coreml_buffer)));

    (*arg_ptr).buffer = Box::into_raw(buffer) as *mut PJRT_Buffer;

    let data_ptr_can_be_freed_immediately = task::spawn(async {});
    let event = Box::new(Event::new(data_ptr_can_be_freed_immediately));
    (*arg_ptr).done_with_host_buffer = Box::into_raw(event) as *mut PJRT_Event;

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn ClientCreateViewOfDeviceBuffer(arg_ptr: *mut PJRT_Client_CreateViewOfDeviceBuffer_Args) -> *mut PJRT_Error {
    todo!("ClientCreateViewOfDeviceBuffer missing")
}

#[no_mangle]
pub unsafe extern "C" fn DeviceDefaultMemory(arg_ptr: *mut PJRT_Device_DefaultMemory_Args) -> *mut PJRT_Error {
    info!("DeviceDefaultMemory was called...");

    (*arg_ptr).memory = COREML_MEMORYS_POINTERS[0];

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn DeviceMemoryStats(arg_ptr: *mut PJRT_Device_MemoryStats_Args) -> *mut PJRT_Error {
    info!("DeviceMemoryStats was called...");

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
    info!("MemoryId was called...");

    todo!("MemoryId not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn MemoryKind(arg_ptr: *mut PJRT_Memory_Kind_Args) -> *mut PJRT_Error {
    info!("MemoryKind was called...");

    (*arg_ptr).kind = COREML_MEMORYS[0].kind.as_ptr();
    (*arg_ptr).kind_size = COREML_MEMORYS[0].kind.count_bytes();

    ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn MemoryDebugString(arg_ptr: *mut PJRT_Memory_DebugString_Args) -> *mut PJRT_Error {
    info!("MemoryDebugString was called...");

    todo!("MemoryDebugString not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn MemoryToString(arg_ptr: *mut PJRT_Memory_ToString_Args) -> *mut PJRT_Error {
    info!("MemoryToString was called...");

    todo!("MemoryToString not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn PJRT_Compile(arg_ptr: *mut PJRT_Compile_Args) -> *mut PJRT_Error {
    info!("PJRT_Compile was called...");

    todo!("PJRT_Compile not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn MemoryKindId(arg_ptr: *mut PJRT_Memory_Kind_Id_Args) -> *mut PJRT_Error {
    info!("PJRT_MemoryKindId was called...");

    todo!("PJRT_MemoryKindId not yet implemented");
}

#[no_mangle]
pub unsafe extern "C" fn GetPjrtApi() -> *mut PJRT_Api {
    info!("Attempted to get the CoreML PjrtApi");

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
        PJRT_Error_Destroy: Some(ErrorDestroy),
        PJRT_Error_Message: Some(ErrorMessage),
        PJRT_Error_GetCode: Some(ErrorGetCode),
        PJRT_Plugin_Initialize: Some(PluginInitialize),
        PJRT_Plugin_Attributes: Some(PluginAttributes),
        PJRT_Event_Destroy: Some(EventDestroy),
        PJRT_Event_IsReady: Some(EventIsReady),
        PJRT_Event_Error: Some(EventError),
        PJRT_Event_Await: Some(EventAwait),
        PJRT_Event_OnReady: Some(EventOnReady),
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
        PJRT_Client_BufferFromHostBuffer: Some(ClientBufferFromHostBuffer),
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
        PJRT_Buffer_Destroy: Some(BufferDestroy),
        PJRT_Buffer_ElementType: Some(BufferElementType),
        PJRT_Buffer_Dimensions: Some(BufferDimensions),
        PJRT_Buffer_UnpaddedDimensions: Some(BufferUnpaddedDimensions),
        PJRT_Buffer_DynamicDimensionIndices: Some(BufferDynamicDimensionIndices),
        PJRT_Buffer_GetMemoryLayout: Some(BufferMemoryLayout),
        PJRT_Buffer_OnDeviceSizeInBytes: Some(BufferDeviceSizeInBytes),
        PJRT_Buffer_Device: Some(BufferDevice),
        PJRT_Buffer_Memory: Some(BufferMemory),
        PJRT_Buffer_Delete: Some(BufferDelete),
        PJRT_Buffer_IsDeleted: Some(BufferIsDeleted),
        PJRT_Buffer_CopyToDevice: Some(BufferCopyToDevice),
        PJRT_Buffer_ToHostBuffer: Some(BufferToHostBuffer),
        PJRT_Buffer_IsOnCpu: Some(BufferIsOnCpu),
        PJRT_Buffer_ReadyEvent: Some(BufferReadyEvent),
        PJRT_Buffer_UnsafePointer: Some(BufferUnsafePointer),
        PJRT_Buffer_IncreaseExternalReferenceCount: Some(BufferIncreaseRefCount),
        PJRT_Buffer_DecreaseExternalReferenceCount: Some(BufferDecreaseRefCount),
        PJRT_Buffer_OpaqueDeviceMemoryDataPointer: Some(BufferOpaqueDeviceMemoryDataPointer),
        PJRT_CopyToDeviceStream_Destroy: Some(CopyToDeviceStreamDestroy),
        PJRT_CopyToDeviceStream_AddChunk: Some(CopyToDeviceStreamAddChunk),
        PJRT_CopyToDeviceStream_TotalBytes: Some(CopyToDeviceStreamTotalBytes),
        PJRT_CopyToDeviceStream_GranuleSize: Some(CopyToDeviceStreamGranuleSize),
        PJRT_CopyToDeviceStream_CurrentBytes: Some(CopyToDeviceStreamCurrentBytes),
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
        PJRT_Buffer_CopyToMemory: Some(BufferCopyToMemory),
        PJRT_Client_CreateViewOfDeviceBuffer: Some(ClientCreateViewOfDeviceBuffer),
        PJRT_Executable_Fingerprint: Some(Executable_Fingerprint),
        PJRT_Client_TopologyDescription: Some(Client_TopologyDescription),
        PJRT_Executable_GetCompiledMemoryStats: None,
        PJRT_Memory_Kind_Id: Some(MemoryKindId),
        PJRT_ExecuteContext_Create: None,
        PJRT_ExecuteContext_Destroy: None,
    });
    Box::into_raw(pjrt_api) as *mut PJRT_Api
}
