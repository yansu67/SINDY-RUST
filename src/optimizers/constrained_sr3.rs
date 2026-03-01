use ndarray::{Array1, Array2};
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSolver, DefaultSettings, NonnegativeConeT, ZeroConeT, IPSolver};
use std::f64;
use super::{Optimizer, TrimType};
use crate::error::Result;
use crate::utils::validation::drop_nan_rows;
use crate::utils::regularization::{prox_l0, prox_l1};
pub struct ConstrainedSR3 {
    pub threshold: f64,
    pub nu: f64,
    pub trim_type: TrimType,
    pub max_iter: usize,
    pub tol: f64,
    pub ridge_kw: f64,
    pub constraint_lhs: Option<Array2<f64>>,
    pub constraint_rhs: Option<Array1<f64>>,
    pub inequality_lhs: Option<Array2<f64>>,
    pub inequality_rhs: Option<Array1<f64>>,
    coef: Option<Array2<f64>>,
}
impl Default for ConstrainedSR3 {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            nu: 1.0,
            trim_type: TrimType::L0,
            max_iter: 30,
            tol: 1e-5,
            ridge_kw: 1e-5,
            constraint_lhs: None,
            constraint_rhs: None,
            inequality_lhs: None,
            inequality_rhs: None,
            coef: None,
        }
    }
}
impl ConstrainedSR3 {
    pub fn new(threshold: f64, nu: f64) -> Self {
        Self {
            threshold,
            nu,
            ..Default::default()
        }
    }
    pub fn with_trim_type(mut self, trim_type: TrimType) -> Self {
        self.trim_type = trim_type;
        self
    }
    pub fn with_equality_constraints(mut self, lhs: Array2<f64>, rhs: Array1<f64>) -> Self {
        self.constraint_lhs = Some(lhs);
        self.constraint_rhs = Some(rhs);
        self
    }
    pub fn with_inequality_constraints(mut self, lhs: Array2<f64>, rhs: Array1<f64>) -> Self {
        self.inequality_lhs = Some(lhs);
        self.inequality_rhs = Some(rhs);
        self
    }
    fn to_csc_upper(mat: &Array2<f64>) -> CscMatrix {
        let n = mat.nrows();
        let mut col_ptr = vec![0; n + 1];
        let mut row_ind = Vec::new();
        let mut data = Vec::new();
        let mut nnz = 0;
        for j in 0..n {
            col_ptr[j] = nnz;
            for i in 0..=j {
                let val = mat[[i, j]];
                if val.abs() > 1e-14 {
                    row_ind.push(i);
                    data.push(val);
                    nnz += 1;
                }
            }
        }
        col_ptr[n] = nnz;
        CscMatrix::new(n, n, col_ptr, row_ind, data)
    }
    fn to_csc_full(mat: &Array2<f64>) -> CscMatrix {
        let m = mat.nrows();
        let n = mat.ncols();
        let mut col_ptr = vec![0; n + 1];
        let mut row_ind = Vec::new();
        let mut data = Vec::new();
        let mut nnz = 0;
        for j in 0..n {
            col_ptr[j] = nnz;
            for i in 0..m {
                let val = mat[[i, j]];
                if val.abs() > 1e-14 {
                    row_ind.push(i);
                    data.push(val);
                    nnz += 1;
                }
            }
        }
        col_ptr[n] = nnz;
        CscMatrix::new(m, n, col_ptr, row_ind, data)
    }
}
impl Optimizer for ConstrainedSR3 {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let (x_clean, y_clean) = drop_nan_rows(x, y);
        let _n_samples = x_clean.nrows();
        let n_features = x_clean.ncols();
        let n_targets = y_clean.ncols();
        let n_vars = n_features * n_targets;
        let mut h_sub = x_clean.t().dot(&x_clean);
        for i in 0..n_features {
            h_sub[[i, i]] += self.nu + self.ridge_kw;
        }
        let mut p_dense = Array2::<f64>::zeros((n_vars, n_vars));
        for j in 0..n_targets {
            let offset = j * n_features;
            for r in 0..n_features {
                for c in 0..n_features {
                    p_dense[[offset + r, offset + c]] = h_sub[[r, c]];
                }
            }
        }
        let p_csc = Self::to_csc_upper(&p_dense);
        let x_ty = x_clean.t().dot(&y_clean);
        let num_eq = self.constraint_lhs.as_ref().map(|mat| mat.nrows()).unwrap_or(0);
        let num_ineq = self.inequality_lhs.as_ref().map(|mat| mat.nrows()).unwrap_or(0);
        let num_total_cons = num_eq + num_ineq;
        let mut a_dense = Array2::<f64>::zeros((num_total_cons, n_vars));
        let mut b_vec = vec![0.0; num_total_cons];
        if let Some(c_lhs) = &self.constraint_lhs {
            let c_rhs = self.constraint_rhs.as_ref().unwrap();
            for r in 0..num_eq {
                b_vec[r] = c_rhs[r];
                for c in 0..n_vars {
                    a_dense[[r, c]] = c_lhs[[r, c]];
                }
            }
        }
        if let Some(i_lhs) = &self.inequality_lhs {
            let i_rhs = self.inequality_rhs.as_ref().unwrap();
            for r in 0..num_ineq {
                b_vec[num_eq + r] = i_rhs[r];
                for c in 0..n_vars {
                    a_dense[[num_eq + r, c]] = i_lhs[[r, c]];
                }
            }
        }
        let a_csc = Self::to_csc_full(&a_dense);
        let cones = if num_total_cons > 0 {
            let mut c = Vec::new();
            if num_eq > 0 {
                c.push(ZeroConeT(num_eq));
            }
            if num_ineq > 0 {
                c.push(NonnegativeConeT(num_ineq));
            }
            c
        } else {
            vec![ZeroConeT(0)]
        };
        let mut xi = Array2::<f64>::zeros((n_features, n_targets));
        let mut w = Array2::<f64>::zeros((n_features, n_targets));
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
                    if d > diff_max { diff_max = d; }
                }
            }
            if diff_max < self.tol && _iter > 0 {
                break;
            }
            let mut q_vec = vec![0.0; n_vars];
            for j in 0..n_targets {
                let offset = j * n_features;
                for i in 0..n_features {
                    q_vec[offset + i] = -(x_ty[[i, j]] + self.nu * w[[i, j]]);
                }
            }
            let settings = DefaultSettings {
                verbose: false,
                ..Default::default()
            };
            let mut solver = DefaultSolver::new(&p_csc, &q_vec, &a_csc, &b_vec, &cones, settings).unwrap();
            solver.solve();
            let solution = solver.solution;
            for j in 0..n_targets {
                let offset = j * n_features;
                for i in 0..n_features {
                    xi[[i, j]] = solution.x[offset + i];
                }
            }
        }
        self.coef = Some(w.t().to_owned());
        Ok(())
    }
    fn coef(&self) -> &Array2<f64> {
        self.coef.as_ref().expect("ConstrainedSR3 Optimizer is not fitted")
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
    fn test_constrained_sr3_equality() {
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
        let c_lhs = array![[1.0, 1.0]];
        let c_rhs = array![3.0];
        let mut optimizer = ConstrainedSR3::new(0.01, 1.0)
            .with_trim_type(TrimType::L1)
            .with_equality_constraints(c_lhs, c_rhs);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        let sum = coef[[0, 0]] + coef[[0, 1]];
        assert!((sum - 3.0).abs() < 0.2);
    }
    #[test]
    fn test_constrained_sr3_inequality() {
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
        let i_lhs = array![[1.0, 0.0]];
        let i_rhs = array![1.0];
        let mut optimizer = ConstrainedSR3::new(0.01, 1.0)
            .with_trim_type(TrimType::L1)
            .with_inequality_constraints(i_lhs, i_rhs);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        println!("INEQUALITY COEF = {:?}", coef);
        assert!(coef[[0, 0]] <= 1.0 + 1e-2);
    }
}
