use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyBytes;
use pyo3::types::PyList;
use pyo3::types::IntoPyDict;

use numpy::PyArrayDyn;
use ndarray::ShapeBuilder;

use std::sync::Arc;
use std::sync::Mutex;

use log::{debug, info, warn};

#[derive(Debug)]
pub struct BufferSpec {
}

#[derive(Debug)]
pub struct Model {
    model: Py<PyAny>,
}


#[pyclass]
#[derive(Clone)]
struct PyOutputCapture {
    buffer: Arc<Mutex<Option<Vec<String>>>>,
}

#[pymethods]
impl PyOutputCapture {
    fn write(&self, data: &str) {
        let mut maybe_buffer = self.buffer.lock().unwrap();
        // TODO(knielsen): Is there a cleaner way of doing this?
        if maybe_buffer.is_some() {
            let buffer = maybe_buffer.as_mut().unwrap();
            buffer.push(String::from(data));
        } else {
            print!("{}", data);
        }
    }

    fn flush(&self) {}
}

impl PyOutputCapture {
    fn start_capture(&mut self) {
        let mut maybe_buffer = self.buffer.lock().unwrap();
        *maybe_buffer = Some(vec![]);
    }

    fn end_capture(&mut self) -> Vec<String> {
        let mut maybe_buffer = self.buffer.lock().unwrap();
        if maybe_buffer.is_some() {
            let buffer = maybe_buffer.as_mut().unwrap();
            let result = buffer.clone();
            *maybe_buffer = None;

            result
        } else {
            vec![]
        }
    }
}

lazy_static! {
    static ref PY_OUTPUT_BUFFER: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
    static ref PY_OUTPUT_CAPTURE: PyOutputCapture = PyOutputCapture { buffer: PY_OUTPUT_BUFFER.clone() };
}

pub fn capture_python_stdout() {
    Python::with_gil(|py| {
        let sys = py.import_bound("sys").unwrap();
        sys.setattr("stdout", PY_OUTPUT_CAPTURE.clone().into_py(py)).unwrap();
        sys.setattr("stderr", PY_OUTPUT_CAPTURE.clone().into_py(py)).unwrap();
    })
}

impl Model {
    pub fn from_mlir(mlir_module: &[u8]) -> Result<Model, ()> {
        debug!["Attempting to call Python!"];

        let mut output_capture = PY_OUTPUT_CAPTURE.clone();
        output_capture.start_capture();

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

            // Convert to CoreML
            // let empty_pipeline = ct.getattr("PassPipeline")?.getattr("EMPTY")?;
            let default_hlo_pipeline = stablehlo_coreml.getattr("DEFAULT_HLO_PIPELINE")?;
            let coreml_converter = ct.getattr("convert")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("source", "milinternal")?;
            kwargs.set_item("minimum_deployment_target", &target_macos15)?;
            kwargs.set_item("pass_pipeline", default_hlo_pipeline)?;
            let coreml_model = coreml_converter.call((mil_program, ), Some(&kwargs))?;
            debug!["CoreML program: {:?}", coreml_model.getattr("_mil_program")?];

            Ok(Model { model: coreml_model.into() })
        });
        let python_output = output_capture.end_capture();
        debug!["Python output converting model {:?}: {:?}", &model_result, python_output];

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

    pub fn predict(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>, PyErr> {
        info!["Calling CoreML model with {} inputs", inputs.len()];

        let mut output_capture = PY_OUTPUT_CAPTURE.clone();
        output_capture.start_capture();

        let prediction_result = Python::with_gil(|py| {
            let model = self.model.bind(py);
            let predict_func = model.getattr("predict")?;

            let list_func = py.import_bound("builtins")?.getattr("list")?;
            let input_description_obj = model.getattr("input_description")?;
            let inputs_list = list_func.call1((input_description_obj, ))?;
            let input_names = inputs_list.downcast::<PyList>()?;

            let mut model_inputs = vec![];
            for (input_name, input_value) in input_names.iter().zip(inputs.iter()) {
                model_inputs.push((input_name, input_value.py_buffer()));
            }

            let model_input_dict = model_inputs.into_py_dict_bound(py);
            debug!["Calling the model with inputs {:?}", model_input_dict];
            let model_result = predict_func.call((model_input_dict, ), None)?;
            debug!["Evaluated model prediction and got: {:?}", model_result];

            // TODO(knielsen): Refactor this to a helper function
            let output_description_obj = model.getattr("output_description")?;
            let outputs_list = list_func.call1((output_description_obj, ))?;
            let output_names = outputs_list.downcast::<PyList>()?;

            let result_buffers = output_names.iter()
                .map(|output_name| model_result.get_item(output_name).unwrap())
                .map(|py_output| Buffer::from_py(py_output))
                .collect();

            Ok(result_buffers)
        });
        
        let python_output = output_capture.end_capture();
        debug!["Python output for model prediction: {:?}", python_output];

        return prediction_result;
    }
}

pub enum Buffer {
    Float16(InternalBuffer<WrappedF16>),
    Float32(InternalBuffer<f32>),
    Float64(InternalBuffer<f64>),
    Int32(InternalBuffer<i32>),
    UInt32(InternalBuffer<i32>), // TODO(knielsen): Fix this :'(
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct WrappedF16(f16);
unsafe impl numpy::Element for WrappedF16 {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, numpy::PyArrayDescr> {
        u16::get_dtype_bound(py)
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }
}

impl Buffer {
    pub fn shape(&self) -> Vec<i64> {
        match self {
            Self::Float16(buf) => buf.shape(),
            Self::Float32(buf) => buf.shape(),
            Self::Float64(buf) => buf.shape(),
            Self::Int32(buf) => buf.shape(),
            Self::UInt32(buf) => buf.shape(),
            Self::None => vec![],
        }
    }
    
    pub unsafe fn raw_data_pointer(&mut self) -> Option<*mut std::ffi::c_void> {
        match self {
            Self::Float16(buf) => Some(buf.raw_data_pointer()),
            Self::Float32(buf) => Some(buf.raw_data_pointer()),
            Self::Float64(buf) => Some(buf.raw_data_pointer()),
            Self::Int32(buf) => Some(buf.raw_data_pointer()),
            Self::UInt32(buf) => Some(buf.raw_data_pointer()),
            Self::None => None,
        }
    }

    pub fn py_buffer(&self) -> &Py<PyAny> {
        match self {
            Self::Float16(buf) => buf.buffer.as_any(),
            Self::Float32(buf) => buf.buffer.as_any(),
            Self::Float64(buf) => buf.buffer.as_any(),
            Self::Int32(buf) => buf.buffer.as_any(),
            Self::UInt32(buf) => buf.buffer.as_any(),
            Self::None => todo!("The Python buffer should not be None at this point!"),
        }
    }
    
    pub fn from_py<'py>(py_obj: Bound<'py, PyAny>) -> Buffer {
        // Find a better way of doing this...
        let dtype = py_obj.getattr("dtype").unwrap().to_string();
        debug!["Discovered dtype: {:?}", dtype];

        match dtype.as_str() {
            "float16" => {
                let internal_buffer = py_obj.downcast_into::<PyArrayDyn<WrappedF16>>().unwrap().unbind();
                Buffer::Float16(InternalBuffer { buffer: internal_buffer })
            },
            "float32" => {
                let internal_buffer = py_obj.downcast_into::<PyArrayDyn<f32>>().unwrap().unbind();
                Buffer::Float32(InternalBuffer { buffer: internal_buffer })
            },
            "float64" => {
                let internal_buffer = py_obj.downcast_into::<PyArrayDyn<f64>>().unwrap().unbind();
                Buffer::Float64(InternalBuffer { buffer: internal_buffer })
            },
            "int32" => {
                let internal_buffer = py_obj.downcast_into::<PyArrayDyn<i32>>().unwrap().unbind();
                Buffer::Int32(InternalBuffer { buffer: internal_buffer })
            },
            "uint32" => {
                let internal_buffer = py_obj.downcast_into::<PyArrayDyn<i32>>().unwrap().unbind();
                Buffer::UInt32(InternalBuffer { buffer: internal_buffer })
            },
            _ => todo!("Unsupported numpy dtype: {:?}", dtype)
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
        debug!("Constructing array with shape: {:?}", shape);
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
