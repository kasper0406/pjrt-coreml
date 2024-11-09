use autocxx::prelude::*;
use std::ffi::c_void;

use log::{debug, info};

include_cpp! {
    #include "swift/CleanInterface.hpp"
    safety!(unsafe_ffi)
    generate_ns!("coreml")
}

#[derive(Debug, Copy, Clone)]
pub enum ElementType {
    F16,
    F32,
    F64,
    I32,
}

impl ElementType {
    fn to_cxx(self) -> ffi::coreml::ElementType {
        match self {
            ElementType::F16 => ffi::coreml::ElementType::F16,
            ElementType::F32 => ffi::coreml::ElementType::F32,
            ElementType::F64 => ffi::coreml::ElementType::F64,
            ElementType::I32 => ffi::coreml::ElementType::F32,
        }
    }

    pub fn width(self) -> usize {
        match self {
            ElementType::F16 => 2,
            ElementType::F32 => 4,
            ElementType::F64 => 8,
            ElementType::I32 => 4,
        }
    }
}

pub struct CoreMLBuffer {
    buffer: UniquePtr<ffi::coreml::WrappedBuffer>,
}

impl CoreMLBuffer {
    /**
     * Allocates a new buffer with the given output size
     */
    pub fn allocate_shape(element_type: ElementType, shape: Vec<i64>, strides: Vec<i64>) -> Self {
        assert!(shape.len() == strides.len(), "shape and strides must have the same length");

        let shape: Vec<_> = shape.into_iter().map(|a| a as usize).collect();
        let strides: Vec<_> = strides.into_iter().map(|a| a as usize).collect();

        let rank = shape.len();
        let buffer = unsafe { ffi::coreml::WrappedBuffer::new(
            element_type.to_cxx(),
            rank,
            shape.as_ptr(),
            strides.as_ptr()
        ) }.within_unique_ptr();

        CoreMLBuffer { buffer }
    }

    /**
     * Copies the data from the `data` slice, and creates a new CoreMLBuffer with a copy
     */
    pub fn allocate_from_data(data: &[u8], element_type: ElementType, shape: &[i64], strides: &[i64]) -> Self {
        assert!(shape.len() == strides.len(), "shape and strides must have the same length");

        let shape: Vec<_> = shape.iter().map(|a| *a as usize).collect();
        let strides: Vec<_> = strides.iter().map(|a| *a as usize).collect();

        let rank = shape.len();
        let buffer = unsafe { ffi::coreml::WrappedBuffer::new1(
            data.as_ptr() as *const autocxx::c_void,
            element_type.to_cxx(),
            rank,
            shape.as_ptr(),
            strides.as_ptr()
        ) }.within_unique_ptr();

        CoreMLBuffer { buffer }
    }

    pub fn shape(&mut self) -> Vec<i64> {
        let rank = self.buffer.pin_mut().getRank();
        let mut shape = vec![0; rank];
        unsafe { self.buffer.pin_mut().getShape(shape.as_mut_ptr()) };
        return shape.into_iter().map(|a| a as i64).collect();
    }

    pub fn strides(&mut self) -> Vec<i64> {
        let rank = self.buffer.pin_mut().getRank();
        let mut strides= vec![0; rank];
        unsafe { self.buffer.pin_mut().getStrides(strides.as_mut_ptr()) };
        return strides.into_iter().map(|a| a as i64).collect();
    }

    pub fn raw_data_pointer(&mut self) -> *mut c_void {
        self.buffer.pin_mut().getRawDataPointer() as *mut std::ffi::c_void
    }
}
