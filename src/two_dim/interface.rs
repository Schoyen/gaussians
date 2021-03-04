use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::wrap_pyfunction;

use numpy::{PyArray2, PyArray4, ToPyArray};

use gs_lib::two_dim::G2D;

fn set_up_g2d_vec(g2d_params: &PySequence) -> Vec<G2D> {
    let g2d_param_vec = g2d_params
        .extract::<Vec<((u32, u32), f64, (f64, f64))>>()
        .unwrap();

    g2d_param_vec
        .into_iter()
        .map(|(alpha, a, centers)| G2D::new(alpha, a, centers))
        .collect()
}

#[pyfunction]
pub fn construct_overlap_matrix_elements<'a>(
    py: Python<'a>,
    g2d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let s = gs_lib::two_dim::construct_overlap_matrix_elements(&gaussians);

    s.to_pyarray(py)
}

#[pyfunction]
pub fn construct_multipole_moment_matrix_elements<'a>(
    py: Python<'a>,
    e: &'a PySequence,
    centers: &'a PySequence,
    g2d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let s_e = gs_lib::two_dim::construct_multipole_moment_matrix_elements(
        e.extract::<(u32, u32)>().unwrap(),
        centers.extract::<(f64, f64)>().unwrap(),
        &gaussians,
    );

    s_e.to_pyarray(py)
}

#[pyfunction]
pub fn construct_kinetic_operator_matrix_elements<'a>(
    py: Python<'a>,
    g2d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let t =
        gs_lib::two_dim::construct_kinetic_operator_matrix_elements(&gaussians);

    t.to_pyarray(py)
}

#[pyfunction]
pub fn construct_differential_operator_matrix_elements<'a>(
    py: Python<'a>,
    e: &'a PySequence,
    g2d_params: &'a PySequence,
) -> &'a PyArray2<f64> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let d_e = gs_lib::two_dim::construct_differential_operator_matrix_elements(
        e.extract::<(u32, u32)>().unwrap(),
        &gaussians,
    );

    d_e.to_pyarray(py)
}

#[pyfunction]
pub fn construct_coulomb_operator_matrix_elements<'a>(
    py: Python<'a>,
    g2d_params: &'a PySequence,
) -> &'a PyArray4<f64> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let u =
        gs_lib::two_dim::construct_coulomb_operator_matrix_elements(&gaussians);

    u.to_pyarray(py)
}

#[pymodule]
fn two_dim_lib(_py: Python, m: &PyModule) -> PyResult<()> {
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
    m.add_wrapped(wrap_pyfunction!(
        construct_coulomb_operator_matrix_elements
    ))?;

    Ok(())
}