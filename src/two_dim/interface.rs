use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::wrap_pyfunction;

use numpy::{IntoPyArray, PyArray2, PyArray4};

use gs_lib::two_dim::G2D;

fn set_up_g2d_vec<'py>(g2d_params: &Bound<'py, PySequence>) -> Vec<G2D> {
    let g2d_param_vec = g2d_params
        .extract::<Vec<((u32, u32), f64, (f64, f64))>>()
        .unwrap();

    g2d_param_vec
        .into_iter()
        .map(|(alpha, a, centers)| G2D::new(alpha, a, centers))
        .collect()
}

#[pyfunction]
pub fn construct_overlap_matrix_elements<'py>(
    py: Python<'py>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let s = gs_lib::two_dim::construct_overlap_matrix_elements(&gaussians);

    s.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_multipole_moment_matrix_elements<'py>(
    py: Python<'py>,
    e: &Bound<'py, PySequence>,
    centers: &Bound<'py, PySequence>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let s_e = gs_lib::two_dim::construct_multipole_moment_matrix_elements(
        e.extract::<(u32, u32)>().unwrap(),
        centers.extract::<(f64, f64)>().unwrap(),
        &gaussians,
    );

    s_e.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_kinetic_operator_matrix_elements<'py>(
    py: Python<'py>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let t =
        gs_lib::two_dim::construct_kinetic_operator_matrix_elements(&gaussians);

    t.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_differential_operator_matrix_elements<'py>(
    py: Python<'py>,
    f: &Bound<'py, PySequence>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let d_e = gs_lib::two_dim::construct_differential_operator_matrix_elements(
        f.extract::<(u32, u32)>().unwrap(),
        &gaussians,
    );

    d_e.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_angular_moment_z_matrix_elements<'py>(
    py: Python<'py>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let l_z =
        gs_lib::two_dim::construct_angular_moment_z_matrix_elements(&gaussians);

    l_z.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_diff_mm_matrix_elements<'py>(
    py: Python<'py>,
    e: &Bound<'py, PySequence>,
    f: &Bound<'py, PySequence>,
    centers: &Bound<'py, PySequence>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let l_ef = gs_lib::two_dim::construct_diff_mm_matrix_elements(
        e.extract::<(u32, u32)>().unwrap(),
        f.extract::<(u32, u32)>().unwrap(),
        centers.extract::<(f64, f64)>().unwrap(),
        &gaussians,
    );

    l_ef.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_gaussian_operator_matrix_elements<'py>(
    py: Python<'py>,
    op_params: &Bound<'py, PySequence>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    // We only look at a single operator at a time
    let op = &set_up_g2d_vec(op_params)[0];
    let gaussians = set_up_g2d_vec(g2d_params);
    let gop_k = gs_lib::two_dim::construct_gaussian_operator_matrix_elements(
        op, &gaussians,
    );

    gop_k.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_coulomb_attraction_operator_matrix_elements<'py>(
    py: Python<'py>,
    center: &Bound<'py, PySequence>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray2<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let v_ab =
        gs_lib::two_dim::construct_coulomb_attraction_operator_matrix_elements(
            center.extract::<(f64, f64)>().unwrap(),
            &gaussians,
        );

    v_ab.into_pyarray_bound(py)
}

#[pyfunction]
pub fn construct_coulomb_interaction_operator_matrix_elements<'py>(
    py: Python<'py>,
    g2d_params: &Bound<'py, PySequence>,
) -> Bound<'py, PyArray4<f64>> {
    let gaussians = set_up_g2d_vec(g2d_params);
    let u =
        gs_lib::two_dim::construct_coulomb_interaction_operator_matrix_elements(
            &gaussians,
        );

    u.into_pyarray_bound(py)
}

#[pymodule]
fn two_dim_lib<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(construct_overlap_matrix_elements, m)?)?;
    m.add_function(wrap_pyfunction!(
        construct_multipole_moment_matrix_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        construct_kinetic_operator_matrix_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        construct_differential_operator_matrix_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        construct_angular_moment_z_matrix_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(construct_diff_mm_matrix_elements, m)?)?;
    m.add_function(wrap_pyfunction!(
        construct_gaussian_operator_matrix_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        construct_coulomb_attraction_operator_matrix_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        construct_coulomb_interaction_operator_matrix_elements,
        m
    )?)?;

    Ok(())
}
