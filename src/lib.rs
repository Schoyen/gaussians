use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn ret_str() -> PyResult<String> {
    Ok("Hello, rusty world!".to_string())
}

#[pyfunction]
fn mul_arr(_py: Python, x: &PyArrayDyn<f64>) -> PyResult<()> {
    let mut x = unsafe { x.as_array_mut() };
    x.map_inplace(|y| *y *= 2.0);

    Ok(())
}

#[pymodule]
fn gaussian_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(ret_str))?;
    m.add_wrapped(wrap_pyfunction!(mul_arr))?;

    Ok(())
}
