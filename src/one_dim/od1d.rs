use super::G1D;

pub struct OD1D {
    g_i: G1D,
    g_j: G1D,

    tot_exp: f64,
    red_exp: f64,

    center_diff: f64,
    com: f64,
    i_com: f64,
    j_com: f64,

    exp_weight: f64,
    norm: f64,
}

impl OD1D {
    pub fn new(g_i: G1D, g_j: G1D) -> Self {
        let tot_exp = g_i.a + g_j.a; // p
        let red_exp = g_i.a * g_j.a / tot_exp; // mu
        let center_diff = g_i.center - g_j.center; // X_AB
        let com = (g_i.a * g_i.center + g_j.a * g_j.center) / tot_exp; // P
        let i_com = com - g_i.center; // X_PA
        let j_com = com - g_j.center; // X_PB
        let exp_weight = (-red_exp * center_diff.powi(2)).exp(); // K_AB
        let norm = g_i.norm * g_j.norm;

        OD1D {
            g_i,
            g_j,
            tot_exp,
            red_exp,
            center_diff,
            com,
            i_com,
            j_com,
            exp_weight,
            norm,
        }
    }

    pub fn evaluate_point(&self, x: f64, with_norm: bool) -> f64 {
        self.g_i.evaluate_point(x, with_norm)
            * self.g_j.evaluate_point(x, with_norm)
    }

    pub fn evaluate(&self, x: &Vec<f64>, with_norm: bool) -> Vec<f64> {
        let mut res = vec![0.0; x.len()];

        for i in 0..x.len() {
            res[i] = self.evaluate_point(x[i], with_norm);
        }

        res
    }

    pub fn expansion_coefficients(&self, t: i32) -> f64 {
        self._expansion_coefficients(self.g_i.i as i32, self.g_j.i as i32, t)
    }

    pub fn _expansion_coefficients(&self, i: i32, j: i32, t: i32) -> f64 {
        if i == 0 && j == 0 && t == 0 {
            return self.exp_weight;
        }

        if t < 0 || t > (i + j) || i < 0 || j < 0 {
            return 0.0;
        }

        if i == 0 {
            return 1.0 / (2.0 * self.tot_exp)
                * self._expansion_coefficients(i, j - 1, t - 1)
                + self.j_com * self._expansion_coefficients(i, j - 1, t)
                + ((t + 1) as f64)
                    * self._expansion_coefficients(i, j - 1, t + 1);
        }

        1.0 / (2.0 * self.tot_exp)
            * self._expansion_coefficients(i - 1, j, t - 1)
            + self.i_com * self._expansion_coefficients(i - 1, j, t)
            + ((t + 1) as f64) * self._expansion_coefficients(i - 1, j, t + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let od_01 =
            OD1D::new(G1D::new(0, 1.0, 0.0, 'x'), G1D::new(1, 1.0, 0.5, 'x'));

        assert_eq!(
            od_01.red_exp,
            od_01.g_i.a * od_01.g_j.a / (od_01.g_i.a + od_01.g_j.a)
        );
        assert_eq!(od_01.center_diff, od_01.g_i.center - od_01.g_j.center);
    }

    #[test]
    fn test_expansion_coefficients() {
        let od_01 =
            OD1D::new(G1D::new(0, 1.0, -0.5, 'x'), G1D::new(1, 0.7, 0.5, 'x'));
        let od_02 =
            OD1D::new(G1D::new(0, 1.0, -0.5, 'x'), G1D::new(2, 1.2, 0.3, 'x'));
        let od_21 =
            OD1D::new(G1D::new(2, 1.2, 0.3, 'x'), G1D::new(1, 0.7, 0.5, 'x'));

        assert!((od_01.expansion_coefficients(0) - (-0.38969419)).abs() < 1e-7);
        assert!((od_01.expansion_coefficients(1) - 0.19484709).abs() < 1e-7);
        assert!((od_01.expansion_coefficients(2)).abs() < 1e-10);
        assert!((od_02.expansion_coefficients(0) - 0.25356869).abs() < 1e-7);
        assert!((od_02.expansion_coefficients(1) - (-0.11658330)).abs() < 1e-7);
        assert!((od_21.expansion_coefficients(0) - 0.00476926).abs() < 1e-7);
        assert!((od_21.expansion_coefficients(2) - 0.00143238).abs() < 1e-7);
    }
}
