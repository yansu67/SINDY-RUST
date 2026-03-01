use ndarray::{Array1, Array2};
use super::{Differentiation, TimeStep};
use crate::error::{Result, SINDyError};
#[derive(Debug, Clone)]
pub struct FiniteDifference {
    pub order: usize,
    pub d: usize,
    pub drop_endpoints: bool,
}
impl Default for FiniteDifference {
    fn default() -> Self {
        Self {
            order: 2,
            d: 1,
            drop_endpoints: false,
        }
    }
}
impl FiniteDifference {
    pub fn new(order: usize, d: usize, drop_endpoints: bool) -> Result<Self> {
        if order == 0 {
            return Err(SINDyError::InvalidParameter(
                "order must be a positive integer".into(),
            ));
        }
        if d == 0 {
            return Err(SINDyError::InvalidParameter(
                "differentiation order d must be a positive integer".into(),
            ));
        }
        Ok(Self {
            order,
            d,
            drop_endpoints,
        })
    }
    fn constant_coefficients(&self, dt: f64) -> Array1<f64> {
        let n_stencil = self.n_stencil();
        let half = (n_stencil - 1) / 2;
        let mut v = Array2::<f64>::zeros((n_stencil, n_stencil));
        for i in 0..n_stencil {
            for j in 0..n_stencil {
                let offset = (j as f64) - (half as f64);
                v[[i, j]] = (dt * offset).powi(i as i32);
            }
        }
        let mut b = Array1::<f64>::zeros(n_stencil);
        b[self.d] = factorial(self.d);
        solve_linear_system(&v, &b)
    }
    fn forward_coefficients(&self, dt: f64) -> Array1<f64> {
        let n_stencil_forward = self.d + self.order;
        let mut v = Array2::<f64>::zeros((n_stencil_forward, n_stencil_forward));
        for i in 0..n_stencil_forward {
            for j in 0..n_stencil_forward {
                v[[i, j]] = (dt * (j as f64)).powi(i as i32);
            }
        }
        let mut b = Array1::<f64>::zeros(n_stencil_forward);
        b[self.d] = factorial(self.d);
        solve_linear_system(&v, &b)
    }
    fn backward_coefficients(&self, dt: f64) -> Array1<f64> {
        let n_stencil_forward = self.d + self.order;
        let mut v = Array2::<f64>::zeros((n_stencil_forward, n_stencil_forward));
        for i in 0..n_stencil_forward {
            for j in 0..n_stencil_forward {
                let offset = -((n_stencil_forward - 1 - j) as f64);
                v[[i, j]] = (dt * offset).powi(i as i32);
            }
        }
        let mut b = Array1::<f64>::zeros(n_stencil_forward);
        b[self.d] = factorial(self.d);
        solve_linear_system(&v, &b)
    }
    fn n_stencil(&self) -> usize {
        2 * self.d.div_ceil(2) - 1 + self.order
    }
}
impl Differentiation for FiniteDifference {
    fn differentiate(&self, x: &Array2<f64>, t: &TimeStep) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples < 2 {
            return Err(SINDyError::InvalidShape(
                "Need at least 2 samples to differentiate".into(),
            ));
        }
        let _dt = match t {
            TimeStep::Uniform(dt) => *dt,
            TimeStep::Array(arr) => {
                if arr.len() != n_samples {
                    return Err(SINDyError::InvalidShape(format!(
                        "Time array length {} != n_samples {}",
                        arr.len(),
                        n_samples
                    )));
                }
                arr[1] - arr[0]
            }
        };
        let n_stencil = self.n_stencil();
        let half = (n_stencil - 1) / 2;
        let mut x_dot = Array2::<f64>::from_elem((n_samples, n_features), f64::NAN);
        match t {
            TimeStep::Uniform(dt) => {
                let coeffs = self.constant_coefficients(*dt);
                for row in half..(n_samples - half) {
                    for col in 0..n_features {
                        let mut val = 0.0;
                        for k in 0..n_stencil {
                            let idx = row + k - half;
                            val += coeffs[k] * x[[idx, col]];
                        }
                        x_dot[[row, col]] = val;
                    }
                }
                if !self.drop_endpoints {
                    let fwd_coeffs = self.forward_coefficients(*dt);
                    let bwd_coeffs = self.backward_coefficients(*dt);
                    let n_fwd = self.d + self.order;
                    for row in 0..half {
                        for col in 0..n_features {
                            let mut val = 0.0;
                            for k in 0..n_fwd {
                                val += fwd_coeffs[k] * x[[k, col]];
                            }
                            x_dot[[row, col]] = val;
                        }
                    }
                    for row in (n_samples - half)..n_samples {
                        for col in 0..n_features {
                            let mut val = 0.0;
                            for k in 0..n_fwd {
                                val += bwd_coeffs[k] * x[[n_samples - n_fwd + k, col]];
                            }
                            x_dot[[row, col]] = val;
                        }
                    }
                }
            }
            TimeStep::Array(t_arr) => {
                for row in half..(n_samples - half) {
                    let coeffs =
                        compute_local_coefficients(t_arr, row, half, n_stencil, self.d);
                    for col in 0..n_features {
                        let mut val = 0.0;
                        for k in 0..n_stencil {
                            let idx = row + k - half;
                            val += coeffs[k] * x[[idx, col]];
                        }
                        x_dot[[row, col]] = val;
                    }
                }
                if !self.drop_endpoints {
                    let n_fwd = self.d + self.order;
                    for row in 0..half {
                        let coeffs = compute_forward_coefficients(t_arr, row, n_fwd, self.d);
                        for col in 0..n_features {
                            let mut val = 0.0;
                            for k in 0..n_fwd {
                                val += coeffs[k] * x[[k, col]];
                            }
                            x_dot[[row, col]] = val;
                        }
                    }
                    for row in (n_samples - half)..n_samples {
                        let coeffs = compute_backward_coefficients(
                            t_arr, row, n_fwd, n_samples, self.d,
                        );
                        for col in 0..n_features {
                            let mut val = 0.0;
                            for k in 0..n_fwd {
                                val += coeffs[k] * x[[n_samples - n_fwd + k, col]];
                            }
                            x_dot[[row, col]] = val;
                        }
                    }
                }
            }
        }
        Ok(x_dot)
    }
}
fn compute_local_coefficients(
    t: &[f64],
    center: usize,
    half: usize,
    n_stencil: usize,
    d: usize,
) -> Array1<f64> {
    let t_center = t[center];
    let mut v = Array2::<f64>::zeros((n_stencil, n_stencil));
    for i in 0..n_stencil {
        for j in 0..n_stencil {
            let idx = center + j - half;
            let dt = t[idx] - t_center;
            v[[i, j]] = dt.powi(i as i32);
        }
    }
    let mut b = Array1::<f64>::zeros(n_stencil);
    b[d] = factorial(d);
    solve_linear_system(&v, &b)
}
fn compute_forward_coefficients(
    t: &[f64],
    eval_point: usize,
    n_stencil: usize,
    d: usize,
) -> Array1<f64> {
    let t_eval = t[eval_point];
    let mut v = Array2::<f64>::zeros((n_stencil, n_stencil));
    for i in 0..n_stencil {
        for j in 0..n_stencil {
            let dt = t[j] - t_eval;
            v[[i, j]] = dt.powi(i as i32);
        }
    }
    let mut b = Array1::<f64>::zeros(n_stencil);
    b[d] = factorial(d);
    solve_linear_system(&v, &b)
}
fn compute_backward_coefficients(
    t: &[f64],
    eval_point: usize,
    n_stencil: usize,
    n_total: usize,
    d: usize,
) -> Array1<f64> {
    let t_eval = t[eval_point];
    let mut v = Array2::<f64>::zeros((n_stencil, n_stencil));
    for i in 0..n_stencil {
        for j in 0..n_stencil {
            let idx = n_total - n_stencil + j;
            let dt = t[idx] - t_eval;
            v[[i, j]] = dt.powi(i as i32);
        }
    }
    let mut b = Array1::<f64>::zeros(n_stencil);
    b[d] = factorial(d);
    solve_linear_system(&v, &b)
}
fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, i| acc * i as f64)
}
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-15 {
            return Array1::zeros(n);
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }
    x
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_constant_derivative() {
        let fd = FiniteDifference::default();
        let x = array![[5.0, 3.0], [5.0, 3.0], [5.0, 3.0], [5.0, 3.0], [5.0, 3.0]];
        let t = TimeStep::Uniform(0.1);
        let x_dot = fd.differentiate(&x, &t).unwrap();
        for row in 0..x_dot.nrows() {
            for col in 0..x_dot.ncols() {
                if !x_dot[[row, col]].is_nan() {
                    assert!(
                        x_dot[[row, col]].abs() < 1e-10,
                        "Expected ~0 for constant, got {}",
                        x_dot[[row, col]]
                    );
                }
            }
        }
    }
    #[test]
    fn test_linear_derivative() {
        let fd = FiniteDifference::default();
        let n = 10;
        let dt = 0.1;
        let mut x = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x[[i, 0]] = i as f64 * dt;
        }
        let t = TimeStep::Uniform(dt);
        let x_dot = fd.differentiate(&x, &t).unwrap();
        for row in 0..n {
            if !x_dot[[row, 0]].is_nan() {
                assert!(
                    (x_dot[[row, 0]] - 1.0).abs() < 1e-10,
                    "Expected 1.0, got {} at row {}",
                    x_dot[[row, 0]],
                    row
                );
            }
        }
    }
    #[test]
    fn test_sin_derivative() {
        let fd = FiniteDifference::default();
        let n = 100;
        let dt = 0.01;
        let mut x = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x[[i, 0]] = (i as f64 * dt).sin();
        }
        let t = TimeStep::Uniform(dt);
        let x_dot = fd.differentiate(&x, &t).unwrap();
        for i in 5..95 {
            let expected = (i as f64 * dt).cos();
            let got = x_dot[[i, 0]];
            if !got.is_nan() {
                assert!(
                    (got - expected).abs() < 1e-3,
                    "At t={}, expected {}, got {}",
                    i as f64 * dt,
                    expected,
                    got
                );
            }
        }
    }
    #[test]
    fn test_nonuniform_grid() {
        let fd = FiniteDifference::default();
        let t_arr: Vec<f64> = (0..20).map(|i| (i as f64 * 0.05).powi(2) + i as f64 * 0.05).collect();
        let n = t_arr.len();
        let mut x = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x[[i, 0]] = t_arr[i] * t_arr[i];
        }
        let t = TimeStep::Array(t_arr.clone());
        let x_dot = fd.differentiate(&x, &t).unwrap();
        for i in 3..17 {
            let expected = 2.0 * t_arr[i];
            let got = x_dot[[i, 0]];
            if !got.is_nan() {
                assert!(
                    (got - expected).abs() < 0.1,
                    "At t={}, expected {}, got {}",
                    t_arr[i], expected, got
                );
            }
        }
    }
}
