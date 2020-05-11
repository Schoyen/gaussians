use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn ret_str() -> PyResult<String> {
    Ok("Hello, rusty world!".to_string())
}

#[pymodule(gaussian_lib)]
fn init_mod(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(ret_str))?;

    Ok(())
}
