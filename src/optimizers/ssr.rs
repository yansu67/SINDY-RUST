use ndarray::{Array2, Axis};
use linfa_linalg::qr::QR;
use std::f64;
use super::Optimizer;
use crate::error::Result;
use crate::utils::validation::drop_nan_rows;
pub struct SSR {
    pub threshold: f64,
    pub max_iter: usize,
    pub ridge_kw: f64,
    coef: Option<Array2<f64>>,
}
impl Default for SSR {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            max_iter: 30,
            ridge_kw: 1e-5,
            coef: None,
        }
    }
}
impl SSR {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}
impl Optimizer for SSR {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let (x_clean, y_clean) = drop_nan_rows(x, y);
        let n_samples = x_clean.nrows();
        let n_features = x_clean.ncols();
        let n_targets = y_clean.ncols();
        let mut all_coefs = Array2::<f64>::zeros((n_targets, n_features));
        for j in 0..n_targets {
            let mut active_features: Vec<usize> = (0..n_features).collect();
            let y_col = y_clean.column(j).to_owned().into_shape_with_order((n_samples, 1)).unwrap();
            let mut current_coefs = vec![0.0; n_features];
            for _iter in 0..self.max_iter {
                if active_features.is_empty() {
                    break;
                }
                let x_subset = x_clean.select(Axis(1), &active_features);
                let mut h = x_subset.t().dot(&x_subset);
                for i in 0..active_features.len() {
                    h[[i, i]] += self.ridge_kw;
                }
                let rhs = x_subset.t().dot(&y_col);
                let mut iteration_done = false;
                if let Ok(qr) = h.qr() {
                    if let Ok(w) = qr.solve_into(rhs) {
                        let mut min_abs_val = f64::MAX;
                        let mut min_idx = 0;
                        for (idx, &feat) in active_features.iter().enumerate() {
                            let val = w[[idx, 0]];
                            current_coefs[feat] = val;
                            if val.abs() < min_abs_val {
                                min_abs_val = val.abs();
                                min_idx = idx;
                            }
                        }
                        if min_abs_val >= self.threshold {
                            iteration_done = true;
                        } else {
                            current_coefs[active_features[min_idx]] = 0.0;
                            active_features.remove(min_idx);
                        }
                    } else {
                        iteration_done = true;
                    }
                } else {
                    iteration_done = true;
                }
                if iteration_done || active_features.is_empty() {
                    break;
                }
            }
            for (feat, &val) in current_coefs.iter().enumerate() {
                all_coefs[[j, feat]] = val;
            }
        }
        self.coef = Some(all_coefs);
        Ok(())
    }
    fn coef(&self) -> &Array2<f64> {
        self.coef.as_ref().expect("SSR Optimizer is not fitted")
    }
    fn complexity(&self) -> usize {
        if let Some(c) = &self.coef {
            c.iter().filter(|&&x| x.abs() > 1e-10).count()
        } else {
            0
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_ssr() {
        let x = array![
            [1.0, 0.1, 0.5],
            [2.0, 0.2, 1.0],
            [3.0, 0.3, 1.5],
            [4.0, 0.4, 2.0]
        ];
        let y = array![
            [2.0],
            [4.0],
            [6.0],
            [8.0]
        ];
        let mut optimizer = SSR::new(1.0);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        assert_eq!(coef.dim(), (1, 3));
        let error = (&x.dot(&coef.t()) - &y).mapv(|a| a.powi(2)).sum();
        assert!(error < 1e-5);
    }
}
