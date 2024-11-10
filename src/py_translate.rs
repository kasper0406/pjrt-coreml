use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyBytes;

use log::{debug, info, warn};

pub struct CoreMLModel {
    bytes: Vec<u8>,
}

pub fn stablehlo_to_coreml(mlir_module: &[u8]) -> CoreMLModel {
    debug!["Attempting to call Python!"];

    let foo: Result<(), pyo3::PyErr> = Python::with_gil(|py| {
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

        // let locals = [("mlir_module", &mlir_module_bytes)].into_py_dict_bound(py);
        // let code = "print(f\"Processing mlir_module with type {type(mlir_module)}: {mlir_module}\")";
        // let coreml_bytes: Vec<u8> = py.eval_bound(code, None, Some(&locals))?.extract()?;
        // let coreml_bytes: Vec<u8> = py.eval_bound(code, None, None)?.extract()?;
        // info!["Got CoreML binary data: {:?}", coreml_bytes];

        let module = module_parse.call((stablehlo_bytes, ), Some(&kwargs))?;

        info!["Contstructed module {:?}", module];

        Ok(())
    });
    info!("Result of constructing CoreML model: {:?}", foo);
 
    CoreMLModel { bytes: vec![] }
}
