use ndarray::{Array1, Array2};
use super::Optimizer;
use crate::error::{Result, SINDyError};
#[derive(Debug, Clone)]
pub struct STLSQ {
    pub threshold: f64,
    pub alpha: f64,
    pub max_iter: usize,
    pub unbias: bool,
    pub verbose: bool,
    coef_: Option<Array2<f64>>,
    ind_: Option<Array2<bool>>,
}
impl Default for STLSQ {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            alpha: 0.05,
            max_iter: 20,
            unbias: true,
            verbose: false,
            coef_: None,
            ind_: None,
        }
    }
}
impl STLSQ {
    pub fn new(threshold: f64, alpha: f64) -> Result<Self> {
        if threshold < 0.0 {
            return Err(SINDyError::InvalidParameter(
                "threshold cannot be negative".into(),
            ));
        }
        if alpha < 0.0 {
            return Err(SINDyError::InvalidParameter(
                "alpha cannot be negative".into(),
            ));
        }
        Ok(Self {
            threshold,
            alpha,
            ..Default::default()
        })
    }
    fn ridge_regression(x: &Array2<f64>, y: &Array1<f64>, alpha: f64) -> Array1<f64> {
        let n_features = x.ncols();
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);
        let mut a = xtx;
        for i in 0..n_features {
            a[[i, i]] += alpha;
        }
        solve_system(&a, &xty)
    }
    fn least_squares(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
        Self::ridge_regression(x, y, 0.0)
    }
    fn select_columns(x: &Array2<f64>, active: &[bool]) -> Array2<f64> {
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .map(|(i, _)| i)
            .collect();
        if active_indices.is_empty() {
            return Array2::zeros((x.nrows(), 0));
        }
        let n_rows = x.nrows();
        let n_active = active_indices.len();
        let mut result = Array2::<f64>::zeros((n_rows, n_active));
        for (j, &col_idx) in active_indices.iter().enumerate() {
            for i in 0..n_rows {
                result[[i, j]] = x[[i, col_idx]];
            }
        }
        result
    }
}
impl Optimizer for STLSQ {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_targets = y.ncols();
        if n_samples != y.nrows() {
            return Err(SINDyError::InvalidShape(format!(
                "x has {} rows but y has {} rows",
                n_samples,
                y.nrows()
            )));
        }
        let mut ind = Array2::<bool>::from_elem((n_targets, n_features), true);
        let mut coef = Array2::<f64>::zeros((n_targets, n_features));
        let mut history: Vec<Array2<f64>> = Vec::new();
        for _iter in 0..self.max_iter {
            let mut new_coef = Array2::<f64>::zeros((n_targets, n_features));
            for target in 0..n_targets {
                let active: Vec<bool> = ind.row(target).iter().copied().collect();
                let n_active: usize = active.iter().filter(|&&a| a).count();
                if n_active == 0 {
                    eprintln!(
                        "Warning: threshold {} eliminated all coefficients for target {}",
                        self.threshold, target
                    );
                    continue;
                }
                let x_active = Self::select_columns(x, &active);
                let y_target = y.column(target).to_owned();
                let coef_active = Self::ridge_regression(&x_active, &y_target, self.alpha);
                let mut full_coef = Array1::<f64>::zeros(n_features);
                let mut j = 0;
                for i in 0..n_features {
                    if active[i] {
                        full_coef[i] = coef_active[j];
                        j += 1;
                    }
                }
                for i in 0..n_features {
                    if full_coef[i].abs() < self.threshold {
                        full_coef[i] = 0.0;
                        ind[[target, i]] = false;
                    } else {
                        ind[[target, i]] = true;
                    }
                }
                new_coef.row_mut(target).assign(&full_coef);
            }
            let old_support: Vec<bool> = if let Some(prev) = history.last() {
                prev.iter().map(|&v| v.abs() > 0.0).collect()
            } else {
                vec![false; n_targets * n_features]
            };
            let new_support: Vec<bool> = new_coef.iter().map(|&v| v.abs() > 0.0).collect();
            history.push(new_coef.clone());
            if self.verbose {
                let residual: f64 = {
                    let pred = x.dot(&new_coef.t());
                    (y - &pred).mapv(|v| v * v).sum()
                };
                let l2_penalty = self.alpha * new_coef.mapv(|v| v * v).sum();
                let l0 = new_coef.iter().filter(|&&v| v.abs() > 0.0).count();
                eprintln!(
                    "Iter {:3}: |y-Xw|²={:.4e}  α|w|²={:.4e}  |w|₀={}  Total={:.4e}",
                    _iter, residual, l2_penalty, l0, residual + l2_penalty
                );
            }
            coef = new_coef;
            if old_support == new_support {
                break;
            }
        }
        if self.unbias {
            for target in 0..n_targets {
                let active: Vec<bool> = ind.row(target).iter().copied().collect();
                let n_active: usize = active.iter().filter(|&&a| a).count();
                if n_active == 0 {
                    continue;
                }
                let x_active = Self::select_columns(x, &active);
                let y_target = y.column(target).to_owned();
                let coef_unbiased = Self::least_squares(&x_active, &y_target);
                let mut j = 0;
                for i in 0..n_features {
                    if active[i] {
                        coef[[target, i]] = coef_unbiased[j];
                        j += 1;
                    }
                }
            }
        }
        self.coef_ = Some(coef);
        self.ind_ = Some(ind);
        Ok(())
    }
    fn coef(&self) -> &Array2<f64> {
        self.coef_.as_ref().expect("STLSQ not fitted")
    }
    fn complexity(&self) -> usize {
        match &self.coef_ {
            Some(c) => c.iter().filter(|&&v| v.abs() > f64::EPSILON).count(),
            None => 0,
        }
    }
}
fn solve_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    if n == 0 {
        return Array1::zeros(0);
    }
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
            continue;
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
        let diag = aug[[i, i]];
        if diag.abs() < 1e-15 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / diag;
    }
    x
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_simple_linear_regression() {
        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 0.5],
            [0.5, 3.0],
        ];
        let y = x.dot(&array![[2.0], [3.0]]);
        let mut opt = STLSQ {
            threshold: 0.01,
            alpha: 0.0,
            unbias: true,
            ..Default::default()
        };
        opt.fit(&x, &y).unwrap();
        let coef = opt.coef();
        assert!(
            (coef[[0, 0]] - 2.0).abs() < 1e-6,
            "Expected coef ~2.0, got {}",
            coef[[0, 0]]
        );
        assert!(
            (coef[[0, 1]] - 3.0).abs() < 1e-6,
            "Expected coef ~3.0, got {}",
            coef[[0, 1]]
        );
    }
    #[test]
    fn test_sparse_regression() {
        let n = 50;
        let mut x = Array2::<f64>::zeros((n, 5));
        let mut y = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let t = i as f64 * 0.1;
            x[[i, 0]] = t;
            x[[i, 1]] = 0.01 * (t * 1.1).sin();
            x[[i, 2]] = 0.01 * (t * 2.3).cos();
            x[[i, 3]] = 0.01 * t * t;
            x[[i, 4]] = 0.005;
            y[[i, 0]] = 2.0 * t;
        }
        let mut opt = STLSQ {
            threshold: 0.1,
            alpha: 0.01,
            unbias: true,
            ..Default::default()
        };
        opt.fit(&x, &y).unwrap();
        let coef = opt.coef();
        assert!(
            (coef[[0, 0]] - 2.0).abs() < 0.2,
            "Expected coef[0] ~2.0, got {}",
            coef[[0, 0]]
        );
        assert!(
            coef[[0, 1]].abs() < 0.5,
            "Expected coef[1] ~0, got {}",
            coef[[0, 1]]
        );
    }
    #[test]
    fn test_multi_target() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]];
        let y = x.clone();
        let mut opt = STLSQ {
            threshold: 0.01,
            alpha: 0.0,
            unbias: true,
            ..Default::default()
        };
        opt.fit(&x, &y).unwrap();
        let coef = opt.coef();
        assert!((coef[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(coef[[0, 1]].abs() < 1e-6);
        assert!(coef[[1, 0]].abs() < 1e-6);
        assert!((coef[[1, 1]] - 1.0).abs() < 1e-6);
    }
}
