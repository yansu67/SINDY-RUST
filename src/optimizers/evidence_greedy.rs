use ndarray::{Array2, Axis};
use linfa_linalg::qr::QR;
use std::f64;
use super::Optimizer;
use crate::error::Result;
use crate::utils::validation::drop_nan_rows;
pub struct EvidenceGreedy {
    pub threshold: f64,
    pub max_iter: usize,
    pub ridge_kw: f64,
    coef: Option<Array2<f64>>,
}
impl Default for EvidenceGreedy {
    fn default() -> Self {
        Self {
            threshold: 1e-4,
            max_iter: 30,
            ridge_kw: 1e-5,
            coef: None,
        }
    }
}
impl EvidenceGreedy {
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
impl Optimizer for EvidenceGreedy {
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
                        let pred = x_subset.dot(&w);
                        let mut base_error = 0.0;
                        for i in 0..n_samples {
                            let diff = pred[[i, 0]] - y_col[[i, 0]];
                            base_error += diff * diff;
                        }
                        let mut min_error_increase = f64::MAX;
                        let mut min_idx_to_drop = 0;
                        if active_features.len() == 1 {
                            let val = w[[0, 0]];
                            current_coefs[active_features[0]] = val;
                            if val.abs() < self.threshold {
                                active_features.clear();
                            }
                            break;
                        }
                        for (idx, _) in active_features.iter().enumerate() {
                            let mut test_features = active_features.clone();
                            test_features.remove(idx);
                            let x_sub_test = x_clean.select(Axis(1), &test_features);
                            let mut h_sub = x_sub_test.t().dot(&x_sub_test);
                            for i in 0..test_features.len() {
                                h_sub[[i, i]] += self.ridge_kw;
                            }
                            let rhs_sub = x_sub_test.t().dot(&y_col);
                            if let Ok(qr_sub) = h_sub.qr() {
                                if let Ok(w_sub) = qr_sub.solve_into(rhs_sub) {
                                    let pred_sub = x_sub_test.dot(&w_sub);
                                    let mut drop_error = 0.0;
                                    for i in 0..n_samples {
                                        let diff = pred_sub[[i, 0]] - y_col[[i, 0]];
                                        drop_error += diff * diff;
                                    }
                                    let error_increase = drop_error - base_error;
                                    if error_increase < min_error_increase {
                                        min_error_increase = error_increase;
                                        min_idx_to_drop = idx;
                                    }
                                }
                            }
                        }
                        for (idx, &feat) in active_features.iter().enumerate() {
                            current_coefs[feat] = w[[idx, 0]];
                        }
                        if min_error_increase > self.threshold {
                            iteration_done = true;
                        } else {
                            current_coefs[active_features[min_idx_to_drop]] = 0.0;
                            active_features.remove(min_idx_to_drop);
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
        self.coef.as_ref().expect("EvidenceGreedy Optimizer is not fitted")
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
    fn test_evidence_greedy() {
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
        let mut optimizer = EvidenceGreedy::new(1e-2);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        assert_eq!(coef.dim(), (1, 3));
        let error = (&x.dot(&coef.t()) - &y).mapv(|a| a.powi(2)).sum();
        assert!(error < 1e-5);
    }
}
