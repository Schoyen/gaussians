use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use numpy::{PyArray2, ToPyArray};

use gs_lib::one_dim::G1D;

fn set_up_g1d_vec(g1d_params: &PyList) -> Vec<G1D> {
    let g1d_param_vec =
        g1d_params.extract::<Vec<(u32, f64, f64, char)>>().unwrap();
    let mut g1d_vec = Vec::new();

    for (i, a, center, symbol) in g1d_param_vec.into_iter() {
        g1d_vec.push(G1D::new(i, a, center, symbol));
    }

    g1d_vec
}

#[pyfunction]
pub fn construct_overlap_matrix_elements<'a>(
    py: Python<'a>,
    g1d_params: &'a PyList,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g1d_vec(g1d_params);
    let s = gs_lib::one_dim::construct_overlap_matrix_elements(&gaussians);

    s.to_pyarray(py)
}

#[pyfunction]
pub fn construct_multipole_moment_matrix_elements<'a>(
    py: Python<'a>,
    e: u32,
    center: f64,
    g1d_params: &'a PyList,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g1d_vec(g1d_params);
    let s_e = gs_lib::one_dim::construct_multipole_moment_matrix_elements(
        e, center, &gaussians,
    );

    s_e.to_pyarray(py)
}

#[pymodule]
fn one_dim_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(construct_overlap_matrix_elements))?;
    m.add_wrapped(wrap_pyfunction!(
        construct_multipole_moment_matrix_elements
    ))?;

    Ok(())
}
