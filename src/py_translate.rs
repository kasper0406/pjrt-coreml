use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyBytes;

use tempfile::TempDir;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use log::{debug, info, warn};

#[derive(Debug)]
pub struct CoreMLModel {
    directory: TempDir,
    filename: String,
}

pub fn stablehlo_to_coreml(mlir_module: &[u8]) -> Result<CoreMLModel, ()> {
    debug!["Attempting to call Python!"];

    let model_result: Result<CoreMLModel, pyo3::PyErr> = Python::with_gil(|py| {
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

        // Save the model to a directory so it  can be loaded later in Swift for prediction
        // TODO(knielsen): Consider getting rid of this step and just doing things through Python?
        let coreml_model_dir = TempDir::new()?;
        let model_filename = "cml.mlpackage";
        let coreml_model_file = coreml_model_dir.path().join(model_filename);
        coreml_model.getattr("save")?.call((coreml_model_file, ), None)?;

        Ok(CoreMLModel { directory: coreml_model_dir, filename: String::from(model_filename) })
    });
    info!("Result of constructing CoreML model: {:?}", &model_result);

    // TODO(knielsen): Add a proper error return!
    model_result.map_err(|err| ())
}
