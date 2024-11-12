use numpy::ndarray::ArcArray;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyBytes;
use pyo3::types::PyList;

use numpy::PyArray;
use numpy::PyArrayDyn;
use ndarray::ShapeBuilder;

use tempfile::TempDir;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use log::{debug, info, warn};

#[derive(Debug)]
pub struct BufferSpec {
}

#[derive(Debug)]
pub struct Model {
    model: Py<PyAny>,
}

impl Model {
    pub fn from_mlir(mlir_module: &[u8]) -> Result<Model, ()> {
        debug!["Attempting to call Python!"];

        let model_result: Result<Model, pyo3::PyErr> = Python::with_gil(|py| {
            // Construct the MLIR module
            let jax_mlir = py.import_bound("jax._src.interpreters.mlir")?;
            let mlir = py.import_bound("jax._src.lib.mlir")?;
            let ir = mlir.getattr("ir")?;

            let jax_lib = py.import_bound("jax._src.lib")?;
            let xla_extension = jax_lib.getattr("xla_extension")?.getattr("mlir")?;
            
            let make_context: Py<PyAny> = jax_mlir.getattr("make_ir_context")?.into();
            let mlir_context = make_context.call0(py)?;

            let mhlo_to_stablehlo = xla_extension.getattr("mhlo_to_stablehlo")?;
            let mlir_module_bytes = PyBytes::new_bound(py, mlir_module);
            debug!["Got mlir bytes: {:?}", mlir_module_bytes];
            let stablehlo_bytes = mhlo_to_stablehlo.call((mlir_module_bytes, ), None)?;
            debug!["Got stablehlo bytes: {:?}", stablehlo_bytes];

            let module_parse = ir.getattr("Module")?.getattr("parse")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("context", mlir_context)?;

            let hlo_module = module_parse.call((stablehlo_bytes, ), Some(&kwargs))?;
            debug!["Contstructed module {:?}", hlo_module];

            // Convert to MIL
            let stablehlo_coreml = py.import_bound("stablehlo_coreml")?;
            let ct = py.import_bound("coremltools")?;
            let mil_converter = stablehlo_coreml.getattr("convert")?;
            let target_macos15 = ct.getattr("target")?.getattr("macOS15")?;
            let mil_program = mil_converter.call((hlo_module, &target_macos15), None)?;
            debug!["Constructed MIL program: {:?}", mil_program];

            // Convert to CoreML
            let empty_pipeline = ct.getattr("PassPipeline")?.getattr("EMPTY")?;
            let coreml_converter = ct.getattr("convert")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("source", "milinternal")?;
            kwargs.set_item("minimum_deployment_target", &target_macos15)?;
            kwargs.set_item("pass_pipeline", empty_pipeline)?;
            let coreml_model = coreml_converter.call((mil_program, ), Some(&kwargs))?;
            debug!["Constructed CoreML model: {:?}", coreml_model];

            Ok(Model { model: coreml_model.into() })
        });
        info!("Result of constructing CoreML model: {:?}", &model_result);

        // TODO(knielsen): Add a proper error return!
        model_result.map_err(|err| ())
    }

    pub fn outputs(&self) -> Vec<BufferSpec> {
        Python::with_gil(|py| {
            let model = self.model.bind(py);
            let mil_program = model.getattr("_mil_program").unwrap();

            let functions = mil_program.getattr("functions").unwrap();
            let default_func = mil_program.getattr("default_function_name").unwrap();
            let main_func = functions.get_item(default_func).unwrap();

            let mut outputs: Vec<BufferSpec> = vec![];
            for output in main_func.getattr("outputs").unwrap().downcast::<PyList>().unwrap() {
                outputs.push(BufferSpec {})
            }
            outputs
        })
    }

    pub fn predict(&self, inputs: &[&Buffer]) -> Vec<Buffer> {
        info!["Calling CoreML model with {} inputs", inputs.len()];
        warn!("Currently returning some bogus data...");
        vec![
            Buffer::Float32(InternalBuffer::<f32>::new(&vec![3], &vec![1], &vec![10.1, 20.2, 1.0]))
        ]
    }
}

pub enum Buffer {
    // Float16(coreml::Buffer<f16>),
    Float32(InternalBuffer<f32>),
    Float64(InternalBuffer<f64>),
    Int32(InternalBuffer<i32>),
    None,
}

impl Buffer {
    pub fn shape(&self) -> Vec<i64> {
        match self {
            // Self::Float16(buf) => buf.shape(),
            Self::Float32(buf) => buf.shape(),
            Self::Float64(buf) => buf.shape(),
            Self::Int32(buf) => buf.shape(),
            Self::None => vec![],
        }
    }
    
    pub unsafe fn raw_data_pointer(&mut self) -> Option<*mut std::ffi::c_void> {
        match self {
            // Self::Float16(buf) => Some(buf.raw_data_pointer()),
            Self::Float32(buf) => Some(buf.raw_data_pointer()),
            Self::Float64(buf) => Some(buf.raw_data_pointer()),
            Self::Int32(buf) => Some(buf.raw_data_pointer()),
            Self::None => None,
        }
    }
}

pub struct InternalBuffer<T> {
    buffer: Py<PyArrayDyn<T>>,
}

impl<T: Clone + numpy::Element> InternalBuffer<T> {
    pub fn new(raw_shape: &[i64], raw_strides: &[i64], data: &[T]) -> InternalBuffer<T> {
        let raw_shape: Vec<_> = raw_shape.iter().map(|s| *s as usize).collect();
        let raw_strides: Vec<_> = raw_strides.iter().map(|s| *s as usize).collect();

        let shape = raw_shape.strides(raw_strides);
        let array = ndarray::ArrayView::from_shape(shape, data).unwrap();
        let owned_array = array.into_owned(); // Notice, this will trigger a data copy

        let buffer = Python::with_gil(|py| {
            numpy::PyArrayDyn::from_owned_array_bound(py, owned_array).unbind()
        });

        InternalBuffer {
            buffer
        }
    }

    pub fn shape(&self) -> Vec<i64> {
        Python::with_gil(|py| {
            self.buffer.bind(py).shape().into_iter()
                .map(|s| *s as i64)
                .collect()
        })
    }

    pub fn strides(&self) -> Vec<i64> {
        Python::with_gil(|py| {
            self.buffer.bind(py).strides().into_iter()
                .map(|s| *s as i64)
                .collect()
        })
    }

    pub unsafe fn raw_data_pointer(&self) -> *mut std::ffi::c_void {
        // TODO(knielsen): Consider if this is actually ok...
        Python::with_gil(|py| {
            self.buffer.bind_borrowed(py).data() as *mut std::ffi::c_void
        })
    }
}
