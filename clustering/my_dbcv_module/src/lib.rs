use pyo3::prelude::*;

#[pyfunction]
fn double_input(x: usize) -> usize {
    2 * x
}

#[pymodule]
fn my_dbcv_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double_input, m)?)?;
    Ok(())
}
