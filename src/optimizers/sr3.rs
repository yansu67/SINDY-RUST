use ndarray::{Array2, Axis};
use linfa_linalg::qr::QR;
use std::f64;
use super::Optimizer;
use crate::error::Result;
use crate::utils::validation::drop_nan_rows;
use crate::utils::regularization::{prox_l0, prox_l1};
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrimType {
    L0,
    L1,
}
pub struct SR3 {
    pub threshold: f64,
    pub nu: f64,
    pub trim_type: TrimType,
    pub max_iter: usize,
    pub tol: f64,
    pub ridge_kw: f64,
    pub thresholder: TrimType,
    pub fit_intercept: bool,
    coef: Option<Array2<f64>>,
    pub unbias: bool,
}
impl Default for SR3 {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            nu: 1.0,
            trim_type: TrimType::L0,
            max_iter: 30,
            tol: 1e-5,
            ridge_kw: 1e-5,
            thresholder: TrimType::L0,
            fit_intercept: false,
            coef: None,
            unbias: true,
        }
    }
}
impl SR3 {
    pub fn new(threshold: f64, nu: f64) -> Self {
        Self {
            threshold,
            nu,
            ..Default::default()
        }
    }
    pub fn with_trim_type(mut self, trim_type: TrimType) -> Self {
        self.trim_type = trim_type;
        self.thresholder = trim_type;
        self
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}
impl Optimizer for SR3 {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let (x_clean, y_clean) = drop_nan_rows(x, y);
        let n_samples = x_clean.nrows();
        let n_features = x_clean.ncols();
        let n_targets = y_clean.ncols();
        let mut h = x_clean.t().dot(&x_clean);
        for i in 0..n_features {
            h[[i, i]] += self.nu + self.ridge_kw;
        }
        let x_ty = x_clean.t().dot(&y_clean);
        let qr = h.qr().unwrap();
        let mut xi = qr.solve_into(x_ty.clone()).unwrap();
        let mut w = xi.clone();
        for _iter in 0..self.max_iter {
            let w_old = w.clone();
            let proxy_threshold = self.threshold / self.nu;
            w = match self.trim_type {
                TrimType::L0 => prox_l0(&xi, proxy_threshold),
                TrimType::L1 => prox_l1(&xi, proxy_threshold),
            };
            let mut diff_max = 0.0;
            for i in 0..n_features {
                for j in 0..n_targets {
                    let d = (w[[i, j]] - w_old[[i, j]]).abs();
                    if d > diff_max {
                        diff_max = d;
                    }
                }
            }
            if diff_max < self.tol {
                break;
            }
            let mut rhs = x_ty.clone();
            for i in 0..n_features {
                for j in 0..n_targets {
                    rhs[[i, j]] += self.nu * w[[i, j]];
                }
            }
            xi = qr.solve(&rhs).unwrap();
        }
        if self.unbias {
            for j in 0..n_targets {
                let mask: Vec<usize> = (0..n_features).filter(|&i| w[[i, j]].abs() > 1e-10).collect();
                if mask.is_empty() {
                    continue;
                }
                let x_subset = x_clean.select(Axis(1), &mask);
                let y_col = y_clean.column(j).to_owned().into_shape_with_order((n_samples, 1)).unwrap();
                let mut h_sub = x_subset.t().dot(&x_subset);
                for i in 0..mask.len() {
                    h_sub[[i, i]] += self.ridge_kw;
                }
                let rhs_sub = x_subset.t().dot(&y_col);
                if let Ok(qr_sub) = h_sub.qr() {
                    if let Ok(coef_sub) = qr_sub.solve_into(rhs_sub) {
                        for (idx, &feature_idx) in mask.iter().enumerate() {
                            w[[feature_idx, j]] = coef_sub[[idx, 0]];
                        }
                    }
                }
            }
        }
        self.coef = Some(w.t().to_owned());
        Ok(())
    }
    fn coef(&self) -> &Array2<f64> {
        self.coef.as_ref().expect("SR3 Optimizer is not fitted")
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
    fn test_sr3_l0() {
        let x = array![
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0]
        ];
        let y = array![
            [2.0],
            [4.0],
            [6.0],
            [8.0]
        ];
        let mut optimizer = SR3::new(0.5, 1.0).with_trim_type(TrimType::L0);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        assert_eq!(coef.dim(), (1, 2));
        let error = (&x.dot(&coef.t()) - &y).mapv(|a| a.powi(2)).sum();
        assert!(error < 1e-5);
    }
    #[test]
    fn test_sr3_l1() {
        let x = array![
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0]
        ];
        let y = array![
            [2.0],
            [4.0],
            [6.0],
            [8.0]
        ];
        let mut optimizer = SR3::new(0.01, 1.0).with_trim_type(TrimType::L1);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        let error = (&x.dot(&coef.t()) - &y).mapv(|a| a.powi(2)).sum();
        assert!(error < 1e-4);
    }
}
