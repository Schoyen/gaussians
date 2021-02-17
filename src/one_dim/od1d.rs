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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let od_01 =
            OD1D::new(G1D::new(0, 1.0, 0.0, 'x'), G1D::new(1, 1.0, 0.5, 'x'));
    }
}
