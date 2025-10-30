use std::convert::TryFrom;
use std::ffi::CString;

use pyo3::{
    prelude::*,
    types::{PyAny, PyDict, PyList, PyModule},
};
use voxelsim::{Action, ActionIntent, Agent, MoveDir, VoxelGrid};

use crate::backend::ControlBackend;

pub struct PythonBackend {
    module: Py<PyModule>,
    update_action_fn: Py<PyAny>,
}

impl PythonBackend {
    pub fn from_script(script: &str) -> PyResult<Self> {
        // Ensure the Python interpreter is initialized for use on any thread
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Register voxelsim_core module so scripts can `import voxelsim_core` and return Action
            let core = PyModule::new(py, "voxelsim_core")?;
            voxelsim::py::voxelsim_core(&core)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("voxelsim_core", core)?;

            // Compile the provided script as a module
            let code = CString::new(script).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("script contains NUL")
            })?;
            let file = CString::new("<embedded>").unwrap();
            let name = CString::new("voxelsim_backend").unwrap();
            let module = PyModule::from_code(py, &code, &file, &name)?;
            let func = module.getattr("update_action")?;
            Ok(Self {
                module: module.unbind(),
                update_action_fn: func.unbind(),
            })
        })
    }
}

impl ControlBackend for PythonBackend {
    fn update_action(&mut self, agent: &Agent, _fw: &VoxelGrid) -> Option<Action> {
        Python::with_gil(|py| {
            let func = self.update_action_fn.bind(py);
            let ret = func.call0().ok()?; // Bound<'py, PyAny>

            // Expect a voxelsim_core.Action instance; extract and clone to native Rust Action
            if let Ok(a) = ret.extract::<pyo3::PyRef<voxelsim::Action>>() {
                return Some(a.clone());
            }

            None
        })
    }
}
