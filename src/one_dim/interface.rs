use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::wrap_pyfunction;

use numpy::{PyArray2, ToPyArray};

use gs_lib::one_dim::G1D;

fn set_up_g1d_vec(g1d_params: &PySequence) -> Vec<G1D> {
    let g1d_param_vec =
        g1d_params.extract::<Vec<(u32, f64, f64, char)>>().unwrap();

    g1d_param_vec
        .into_iter()
        .map(|(i, a, center, symbol)| G1D::new(i, a, center, symbol))
        .collect()
}

#[pyfunction]
pub fn construct_overlap_matrix_elements<'a>(
    py: Python<'a>,
    g1d_params: &'a PySequence,
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
    g1d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g1d_vec(g1d_params);
    let s_e = gs_lib::one_dim::construct_multipole_moment_matrix_elements(
        e, center, &gaussians,
    );

    s_e.to_pyarray(py)
}

#[pyfunction]
pub fn construct_kinetic_operator_matrix_elements<'a>(
    py: Python<'a>,
    g1d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g1d_vec(g1d_params);
    let t =
        gs_lib::one_dim::construct_kinetic_operator_matrix_elements(&gaussians);

    t.to_pyarray(py)
}

#[pyfunction]
pub fn construct_differential_operator_matrix_elements<'a>(
    py: Python<'a>,
    e: u32,
    g1d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g1d_vec(g1d_params);
    let d_e = gs_lib::one_dim::construct_differential_operator_matrix_elements(
        e, &gaussians,
    );

    d_e.to_pyarray(py)
}

#[pymodule]
fn one_dim_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(construct_overlap_matrix_elements))?;
    m.add_wrapped(wrap_pyfunction!(
        construct_multipole_moment_matrix_elements
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        construct_kinetic_operator_matrix_elements
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        construct_differential_operator_matrix_elements
    ))?;

    Ok(())
}