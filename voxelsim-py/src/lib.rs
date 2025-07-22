use ::voxelsim as voxelsim_core;
use pyo3::prelude::*;

#[pymodule]
fn voxelsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    voxelsim_core::py::voxelsim(m)?;
    voxelsim_compute::py::voxelsim_compute(m)?;
    Ok(())
}
