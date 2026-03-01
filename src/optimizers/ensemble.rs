use ndarray::{Array2, Axis};
use rand::Rng;
use std::f64;
use super::Optimizer;
use crate::error::Result;
use crate::utils::validation::drop_nan_rows;
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnsembleType {
    Bagging,
    SubSample,
    Library,
}
pub struct EnsembleOptimizer {
    pub base_optimizer: Box<dyn Optimizer>,
    pub n_models: usize,
    pub ensemble_type: EnsembleType,
    pub sample_frac: f64,
    coef: Option<Array2<f64>>,
    pub coef_list: Vec<Array2<f64>>,
}
impl EnsembleOptimizer {
    pub fn new(base_optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            base_optimizer,
            n_models: 20,
            ensemble_type: EnsembleType::Bagging,
            sample_frac: 0.8,
            coef: None,
            coef_list: Vec::new(),
        }
    }
    pub fn with_n_models(mut self, n_models: usize) -> Self {
        self.n_models = n_models;
        self
    }
    pub fn with_ensemble_type(mut self, ensemble_type: EnsembleType) -> Self {
        self.ensemble_type = ensemble_type;
        self
    }
    pub fn with_sample_frac(mut self, sample_frac: f64) -> Self {
        self.sample_frac = sample_frac;
        self
    }
}
impl Optimizer for EnsembleOptimizer {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let (x_clean, y_clean) = drop_nan_rows(x, y);
        let n_samples = x_clean.nrows();
        let n_features = x_clean.ncols();
        let n_targets = y_clean.ncols();
        self.coef_list.clear();
        let mut rng = rand::thread_rng();
        for _ in 0..self.n_models {
            let mut x_subset: Array2<f64>;
            let mut y_subset: Array2<f64>;
            let mut active_features: Vec<usize> = (0..n_features).collect();
            match self.ensemble_type {
                EnsembleType::Bagging => {
                    let n_sub = (n_samples as f64 * self.sample_frac).round() as usize;
                    let n_sub = n_sub.max(1);
                    x_subset = Array2::zeros((n_sub, n_features));
                    y_subset = Array2::zeros((n_sub, n_targets));
                    for i in 0..n_sub {
                        let idx = rng.gen_range(0..n_samples);
                        x_subset.row_mut(i).assign(&x_clean.row(idx));
                        y_subset.row_mut(i).assign(&y_clean.row(idx));
                    }
                }
                EnsembleType::SubSample => {
                    let n_sub = (n_samples as f64 * self.sample_frac).round() as usize;
                    let n_sub = n_sub.max(1).min(n_samples);
                    x_subset = Array2::zeros((n_sub, n_features));
                    y_subset = Array2::zeros((n_sub, n_targets));
                    let mut indices: Vec<usize> = (0..n_samples).collect();
                    for i in 0..n_sub {
                        let swap_idx = rng.gen_range(i..n_samples);
                        indices.swap(i, swap_idx);
                        x_subset.row_mut(i).assign(&x_clean.row(indices[i]));
                        y_subset.row_mut(i).assign(&y_clean.row(indices[i]));
                    }
                }
                EnsembleType::Library => {
                    let n_sub_feat = (n_features as f64 * self.sample_frac).round() as usize;
                    let n_sub_feat = n_sub_feat.max(1).min(n_features);
                    let mut feat_indices: Vec<usize> = (0..n_features).collect();
                    for i in 0..n_sub_feat {
                        let swap_idx = rng.gen_range(i..n_features);
                        feat_indices.swap(i, swap_idx);
                    }
                    feat_indices.truncate(n_sub_feat);
                    feat_indices.sort_unstable();
                    active_features = feat_indices.clone();
                    x_subset = x_clean.select(Axis(1), &feat_indices);
                    y_subset = y_clean.clone();
                }
            }
            self.base_optimizer.fit(&x_subset, &y_subset)?;
            let base_coef = self.base_optimizer.coef();
            let mut iter_coef = Array2::zeros((n_targets, n_features));
            for j in 0..n_targets {
                for (idx, &feat) in active_features.iter().enumerate() {
                    iter_coef[[j, feat]] = base_coef[[j, idx]];
                }
            }
            self.coef_list.push(iter_coef);
        }
        let mut final_coef = Array2::zeros((n_targets, n_features));
        for model_coef in &self.coef_list {
            final_coef += model_coef;
        }
        final_coef.mapv_inplace(|v| v / self.n_models as f64);
        self.coef = Some(final_coef);
        Ok(())
    }
    fn coef(&self) -> &Array2<f64> {
        self.coef.as_ref().expect("EnsembleOptimizer is not fitted")
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
    use crate::optimizers::STLSQ;
    #[test]
    fn test_ensemble_bagging() {
        let x = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4]
        ];
        let y = array![
            [2.0],
            [4.0],
            [6.0],
            [8.0]
        ];
        let base_opt = Box::new(STLSQ::default());
        let mut optimizer = EnsembleOptimizer::new(base_opt)
            .with_n_models(5)
            .with_ensemble_type(EnsembleType::Bagging);
        optimizer.fit(&x, &y).unwrap();
        let coef = optimizer.coef();
        assert_eq!(coef.dim(), (1, 2));
        assert_eq!(optimizer.coef_list.len(), 5);
        let error = (&x.dot(&coef.t()) - &y).mapv(|a| a.powi(2)).sum();
        assert!(error < 1e-4);
    }
}
