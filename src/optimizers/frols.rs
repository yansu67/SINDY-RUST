use ndarray::{Array2, Axis};
use linfa_linalg::qr::QR;
use std::f64;
use super::Optimizer;
use crate::error::Result;
use crate::utils::validation::drop_nan_rows;
pub struct FROLS {
    pub max_iter: usize,
    pub alpha: f64,
    pub ridge_kw: f64,
    coef: Option<Array2<f64>>,
}
impl Default for FROLS {
    fn default() -> Self {
        Self {
            max_iter: 10,
            alpha: 1e-4,
            ridge_kw: 1e-5,
            coef: None,
        }
    }
}
impl FROLS {
    pub fn new(max_iter: usize, alpha: f64) -> Self {
        Self {
            max_iter,
            alpha,
            ..Default::default()
        }
    }
}
impl Optimizer for FROLS {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let (x_clean, y_clean) = drop_nan_rows(x, y);
        let n_samples = x_clean.nrows();
        let n_features = x_clean.ncols();
        let n_targets = y_clean.ncols();
        let mut all_coefs = Array2::<f64>::zeros((n_targets, n_features));
        for j in 0..n_targets {
            let mut selected_features = Vec::new();
            let mut remaining_features: Vec<usize> = (0..n_features).collect();
            let y_col = y_clean.column(j).to_owned().into_shape_with_order((n_samples, 1)).unwrap();
            let mut current_coefs = vec![0.0; n_features];
            let mut best_error = f64::MAX;
            for _iter in 0..self.max_iter {
                if remaining_features.is_empty() {
                    break;
                }
                let mut best_feature_to_add = None;
                let mut best_coef_for_step = Vec::new();
                let mut min_step_error = best_error;
                for &candidate in &remaining_features {
                    let mut test_features = selected_features.clone();
                    test_features.push(candidate);
                    let x_subset = x_clean.select(Axis(1), &test_features);
                    let mut h = x_subset.t().dot(&x_subset);
                    for i in 0..test_features.len() {
                        h[[i, i]] += self.ridge_kw;
                    }
                    let rhs = x_subset.t().dot(&y_col);
                    if let Ok(qr) = h.qr() {
                        if let Ok(w) = qr.solve_into(rhs) {
                            let pred = x_subset.dot(&w);
                            let mut step_error = 0.0;
                            for i in 0..n_samples {
                                let diff = pred[[i, 0]] - y_col[[i, 0]];
                                step_error += diff * diff;
                            }
                            if step_error < min_step_error {
                                min_step_error = step_error;
                                best_feature_to_add = Some(candidate);
                                let mut temp_coefs = vec![0.0; test_features.len()];
                                for (idx, _) in test_features.iter().enumerate() {
                                    temp_coefs[idx] = w[[idx, 0]];
                                }
                                best_coef_for_step = temp_coefs;
                            }
                        }
                    }
                }
                if let Some(feat) = best_feature_to_add {
                    if best_error - min_step_error >= self.alpha {
                        selected_features.push(feat);
                        remaining_features.retain(|&f| f != feat);
                        best_error = min_step_error;
                        current_coefs = vec![0.0; n_features];
                        for (idx, &f) in selected_features.iter().enumerate() {
                            current_coefs[f] = best_coef_for_step[idx];
                        }
                    } else {
                        break;
                    }
                } else {
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
        self.coef.as_ref().expect("FROLS Optimizer is not fitted")
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
    fn test_frols() {
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
        let mut optimizer = FROLS::new(1, 1e-4);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        assert_eq!(coef.dim(), (1, 3));
        assert_eq!(optimizer.complexity(), 1);
        let error = (&x.dot(&coef.t()) - &y).mapv(|a| a.powi(2)).sum();
        assert!(error < 1e-5);
    }
}
