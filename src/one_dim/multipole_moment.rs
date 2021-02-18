use super::G1D;
use super::OD1D;

fn construct_overlap_matrix_elements(gaussians: &Vec<G1D>) -> Vec<Vec<f64>> {
    construct_multipole_moment_matrix_elements(0, 0.0, gaussians)
}

fn construct_multipole_moment_matrix_elements(
    e: u32,
    center: f64,
    gaussians: &Vec<G1D>,
) -> Vec<Vec<f64>> {
    let l = gaussians.len();
    let mut s_e = vec![vec![0.0; l]; l];

    for i in 0..l {
        let g_i = &gaussians[i];

        s_e[i][i] = g_i.norm.powi(2) * s(e, center, OD1D::new(&g_i, &g_i));

        for j in (i + 1)..l {
            let g_j = &gaussians[j];

            let val = g_i.norm * g_j.norm * s(e, center, OD1D::new(&g_i, &g_j));

            s_e[i][j] = val;
            s_e[j][i] = val;
        }
    }

    s_e
}

fn s(e: u32, center: f64, od: OD1D) -> f64 {
    let mut val = 0.0;

    for t in 0..(std::cmp::min(od.i + od.j, e) + 1) {
        val += od.expansion_coefficients(t as i32)
            * m(e as i32, t as i32, od.tot_exp, od.com, center);
    }

    val
}

fn m(e: i32, t: i32, p: f64, od_center: f64, center: f64) -> f64 {
    if t > e {
        return 0.0;
    }

    if t < 0 || e < 0 {
        return 0.0;
    }

    if e == 0 {
        return if t == 0 {
            0.0
        } else {
            (std::f64::consts::PI / p).sqrt()
        };
    }

    (t as f64) * m(e - 1, t - 1, p, od_center, center)
        + (od_center - center) * m(e - 1, t, p, od_center, center)
        + 1.0 / (2.0 * p) * m(e - 1, t + 1, p, od_center, center)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap() {
        let g_list =
            vec![G1D::new(0, 2.0, 2.0, 'x'), G1D::new(0, 2.0, -2.0, 'x')];

        let s = construct_overlap_matrix_elements(&g_list);

        for i in 0..s.len() {
            for j in 0..s[i].len() {
                assert!(s[i][j].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_multipole_moment_matrix_elements() {
        let g_list = vec![
            G1D::new(0, 2.0, -4.0, 'x'),
            G1D::new(0, 2.0, 4.0, 'x'),
            G1D::new(1, 1.0, 0.0, 'x'),
            G1D::new(2, 1.0, 0.0, 'x'),
        ];

        let d = construct_multipole_moment_matrix_elements(1, 1.0, &g_list);

        for i in 0..d.len() {
            for j in 0..d[i].len() {
                assert!((d[i][j] - d[j][i]).abs() < 1e-12);
            }
        }
    }
}
