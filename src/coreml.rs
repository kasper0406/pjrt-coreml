use autocxx::prelude::*;
use std::ffi::c_void;

use log::{debug};

include_cpp! {
    #include "swift/CleanInterface.hpp"
    safety!(unsafe_ffi)
    generate_ns!("coreml")
}

pub struct CoreMLBuffer {
    buffer: UniquePtr<ffi::coreml::WrappedBuffer>,
}

// impl Drop for CoreMLBuffer {
//     fn drop(&mut self) {
//         debug!("Freeing CoreML tensor");
//         ffi::swift_coreml::mltensor_destroy(&self.buffer);
//     }
// }

impl CoreMLBuffer {
    pub fn new() -> Self {
        debug!("Allocating MLTensor in Swift");
        CoreMLBuffer {
            buffer: ffi::coreml::WrappedBuffer::new().within_unique_ptr(),
        }
    }

    pub fn shape(&mut self) -> Vec<i64> {
        let foo = self.buffer.pin_mut().getShape();
        return vec![foo];
    }

    pub fn raw_data_pointer(&mut self) -> *mut c_void {
        self.buffer.pin_mut().getRawDataPointer() as *mut std::ffi::c_void
    }
}
