use ndarray::{Array1, Array2, Axis};
use crate::error::{Result, SINDyError};
use crate::feature_library::{polynomial::PolynomialLibrary, FeatureLibrary};
use crate::optimizers::{stlsq::STLSQ, Optimizer};
pub struct DiscreteSINDy {
    pub feature_library: Box<dyn FeatureLibrary>,
    pub optimizer: Box<dyn Optimizer>,
    n_input_features: Option<usize>,
    n_control_features: Option<usize>,
    n_output_features: Option<usize>,
    feature_names: Vec<String>,
}
impl Default for DiscreteSINDy {
    fn default() -> Self {
        Self {
            feature_library: Box::new(PolynomialLibrary::default()),
            optimizer: Box::new(STLSQ::default()),
            n_input_features: None,
            n_control_features: None,
            n_output_features: None,
            feature_names: Vec::new(),
        }
    }
}
impl DiscreteSINDy {
    pub fn new(
        feature_library: Box<dyn FeatureLibrary>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        Self {
            feature_library,
            optimizer,
            n_input_features: None,
            n_control_features: None,
            n_output_features: None,
            feature_names: Vec::new(),
        }
    }
    pub fn fit(
        &mut self,
        x: &[Array2<f64>],
        u: Option<&[Array2<f64>]>,
        feature_names: Option<&[&str]>,
    ) -> Result<()> {
        if x.is_empty() {
            return Err(SINDyError::InvalidShape("No trajectories provided".into()));
        }
        let n_state_features = x[0].ncols();
        let n_control_features = u.map(|us| us[0].ncols()).unwrap_or(0);
        self.n_input_features = Some(n_state_features);
        self.n_control_features = Some(n_control_features);
        self.feature_names = match feature_names {
            Some(names) => names.iter().map(|s| s.to_string()).collect(),
            None => {
                let mut names = (0..n_state_features).map(|i| format!("x{}", i)).collect::<Vec<_>>();
                names.extend((0..n_control_features).map(|i| format!("u{}", i)));
                names
            }
        };
        let mut x_concatenated = Vec::new();
        let mut x_next_concatenated = Vec::new();
        for i in 0..x.len() {
            let xi = &x[i];
            if xi.nrows() < 2 {
                return Err(SINDyError::InvalidShape(format!(
                    "Trajectory {} has < 2 samples, insufficient for discrete map",
                    i
                )));
            }
            let xi_t = xi.slice(ndarray::s![0..xi.nrows()-1, ..]).to_owned();
            let xi_next = xi.slice(ndarray::s![1..xi.nrows(), ..]).to_owned();
            let xi_full = match u {
                Some(us) => {
                    let ui = &us[i];
                    if xi.nrows() != ui.nrows() {
                        return Err(SINDyError::InvalidShape(format!(
                            "Trajectory {}: x rows ({}) != u rows ({})",
                            i, xi.nrows(), ui.nrows()
                        )));
                    }
                    let ui_t = ui.slice(ndarray::s![0..ui.nrows()-1, ..]);
                    ndarray::concatenate(ndarray::Axis(1), &[xi_t.view(), ui_t.view()]).map_err(|e| {
                        SINDyError::InvalidShape(format!("Failed to concatenate x and u: {}", e))
                    })?
                }
                None => xi_t,
            };
            let (xi_clean, xi_next_clean) = remove_nan_rows(&xi_full, &xi_next);
            if xi_clean.nrows() > 0 {
                x_concatenated.push(xi_clean);
                x_next_concatenated.push(xi_next_clean);
            }
        }
        if x_concatenated.is_empty() {
             return Err(SINDyError::InvalidShape(
                "Not enough non-NaN sample pairs across trajectories".into(),
            ));
        }
        let full_x = concat_arrays(&x_concatenated)?;
        let full_x_next = concat_arrays(&x_next_concatenated)?;
        self.feature_library.fit(&full_x)?;
        let theta = self.feature_library.transform(&full_x)?;
        self.n_output_features = Some(self.feature_library.n_output_features());
        self.optimizer.fit(&theta, &full_x_next)?;
        Ok(())
    }
    pub fn predict(&self, x: &Array2<f64>, u: Option<&Array2<f64>>) -> Result<Array2<f64>> {
        self.check_fitted()?;
        let n_control = self.n_control_features.unwrap_or(0);
        if n_control > 0 && u.is_none() {
            return Err(SINDyError::InvalidShape("Model was fit with control variables, but none were provided for predict".into()));
        }
        let x_full = match u {
            Some(ui) if n_control > 0 => {
                ndarray::concatenate(ndarray::Axis(1), &[x.view(), ui.view()]).map_err(|e| {
                    SINDyError::InvalidShape(format!("Failed to concatenate x and u: {}", e))
                })?
            },
            _ => x.clone()
        };
        let theta = self.feature_library.transform(&x_full)?;
        let coef = self.optimizer.coef();
        Ok(theta.dot(&coef.t()))
    }
    pub fn coefficients(&self) -> Result<&Array2<f64>> {
        self.check_fitted()?;
        Ok(self.optimizer.coef())
    }
    pub fn equations(&self, precision: usize) -> Result<Vec<String>> {
        self.check_fitted()?;
        let coef = self.optimizer.coef();
        let lib_names = self
            .feature_library
            .get_feature_names(Some(&self.feature_names));
        let mut equations = Vec::new();
        for target in 0..coef.nrows() {
            let mut terms: Vec<String> = Vec::new();
            for (j, name) in lib_names.iter().enumerate() {
                let c = coef[[target, j]];
                if c.abs() > f64::EPSILON {
                    terms.push(format!("{:.prec$} {}", c, name, prec = precision));
                }
            }
            if terms.is_empty() {
                equations.push("0".to_string());
            } else {
                equations.push(terms.join(" + "));
            }
        }
        Ok(equations)
    }
    pub fn print_model(&self, precision: usize) -> Result<()> {
        let eqs = self.equations(precision)?;
        for (i, eq) in eqs.iter().enumerate() {
            let lhs = if i < self.feature_names.len() {
                format!("{}(k+1)", self.feature_names[i])
            } else {
                format!("x{}(k+1)", i)
            };
            println!("{} = {}", lhs, eq);
        }
        Ok(())
    }
    pub fn score(
        &self,
        x: &[Array2<f64>],
        u: Option<&[Array2<f64>]>,
    ) -> Result<f64> {
        self.check_fitted()?;
        let mut x_next_clean_all = Vec::new();
        let mut x_pred_all = Vec::new();
        for i in 0..x.len() {
            let xi = &x[i];
            if xi.nrows() < 2 {
                continue;
            }
            let xi_t = xi.slice(ndarray::s![0..xi.nrows()-1, ..]).to_owned();
            let xi_next = xi.slice(ndarray::s![1..xi.nrows(), ..]).to_owned();
            let mut x_full_view = vec![xi_t.view()];
            if let Some(us) = u {
                let ui_t = us[i].slice(ndarray::s![0..us[i].nrows()-1, ..]);
                x_full_view.push(ui_t);
            }
            let xi_full = ndarray::concatenate(ndarray::Axis(1), &x_full_view).unwrap_or(xi_t.clone());
            let (xi_clean, xi_next_clean) = remove_nan_rows(&xi_full, &xi_next);
            if xi_clean.nrows() > 0 {
                let xi_state_clean = xi_clean.slice(ndarray::s![.., 0..self.n_input_features.unwrap_or(xi_clean.ncols())]).to_owned();
                let ui_clean = if u.is_some() {
                    Some(xi_clean.slice(ndarray::s![.., self.n_input_features.unwrap()..]).to_owned())
                } else {
                    None
                };
                let xi_pred = self.predict(&xi_state_clean, ui_clean.as_ref())?;
                x_next_clean_all.push(xi_next_clean);
                x_pred_all.push(xi_pred);
            }
        }
        if x_next_clean_all.is_empty() {
             return Err(SINDyError::InvalidShape(
                "Not enough non-NaN sample pairs to score".into(),
            ));
        }
        let full_x_next = concat_arrays(&x_next_clean_all)?;
        let full_tmp_pred = concat_arrays(&x_pred_all)?;
        let ss_res: f64 = (&full_x_next - &full_tmp_pred)
            .mapv(|v| v * v)
            .sum();
        let mean = full_x_next.mean_axis(Axis(0)).unwrap();
        let ss_tot: f64 = full_x_next
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .zip(mean.iter())
                    .map(|(&v, &m)| (v - m) * (v - m))
                    .sum::<f64>()
            })
            .sum();
        if ss_tot < f64::EPSILON {
            Ok(1.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
    pub fn simulate(&self, x0: &Array1<f64>, n_steps: usize, u: Option<&Array2<f64>>) -> Result<Array2<f64>> {
        self.check_fitted()?;
        let n_features = x0.len();
        if let Some(control) = u {
            if control.nrows() != n_steps {
                return Err(SINDyError::InvalidShape(format!("Control array 'u' must have {} rows to match 'n_steps'", n_steps)));
            }
        }
        let mut trajectory = Array2::<f64>::zeros((n_steps + 1, n_features));
        trajectory.row_mut(0).assign(x0);
        for i in 0..n_steps {
            let x_current = trajectory.row(i).to_owned();
            let x_2d = x_current.view().insert_axis(Axis(0)).to_owned();
            let u_2d = u.map(|us| us.row(i).to_owned().view().insert_axis(Axis(0)).to_owned());
            let x_next = self.predict(&x_2d, u_2d.as_ref())?;
            for j in 0..n_features {
                trajectory[[i + 1, j]] = x_next[[0, j]];
            }
        }
        Ok(trajectory)
    }
    pub fn get_feature_names(&self) -> Vec<String> {
        self.feature_library
            .get_feature_names(Some(&self.feature_names))
    }
    pub fn complexity(&self) -> usize {
        self.optimizer.complexity()
    }
    fn check_fitted(&self) -> Result<()> {
        if self.n_input_features.is_none() {
            return Err(SINDyError::NotFitted("DiscreteSINDy model".into()));
        }
        Ok(())
    }
}
fn remove_nan_rows(x: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut keep = Vec::new();
    for i in 0..x.nrows().min(y.nrows()) {
        let x_ok = x.row(i).iter().all(|v| !v.is_nan());
        let y_ok = y.row(i).iter().all(|v| !v.is_nan());
        if x_ok && y_ok {
            keep.push(i);
        }
    }
    let x_out = x.select(Axis(0), &keep);
    let y_out = y.select(Axis(0), &keep);
    (x_out, y_out)
}
fn concat_arrays(arrays: &[Array2<f64>]) -> Result<Array2<f64>> {
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::concatenate(Axis(0), &views).map_err(|e| {
        SINDyError::InvalidShape(format!("Failed to concatenate arrays vertically: {}", e))
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_discrete_sindy_logistic_map() {
        let n = 100;
        let r = 3.8;
        let mut data = Array2::<f64>::zeros((n, 1));
        data[[0, 0]] = 0.5;
        for i in 0..n-1 {
            let x_t = data[[i, 0]];
            data[[i+1, 0]] = r * x_t * (1.0 - x_t);
        }
        let mut model = DiscreteSINDy::default();
        model
            .fit(&[data.clone()], None, Some(&["x"]))
            .unwrap();
        let coefs = model.coefficients().unwrap();
        let names = model.get_feature_names();
        let x_idx = names.iter().position(|n| n == "x").unwrap();
        let x2_idx = names.iter().position(|n| n == "x^2").unwrap();
        assert!((coefs[[0, x_idx]] - r).abs() < 1e-4, "Expected x coef to be {}, got {}", r, coefs[[0, x_idx]]);
        assert!((coefs[[0, x2_idx]] - (-r)).abs() < 1e-4, "Expected x^2 coef to be {}, got {}", -r, coefs[[0, x2_idx]]);
        let score = model.score(&[data], None).unwrap();
        assert!(score > 0.99, "Expected R² > 0.99 for noise-free analytic map");
    }
    #[test]
    fn test_discrete_sindy_simulate() {
        let n = 20;
        let mut data = Array2::<f64>::zeros((n, 1));
        data[[0, 0]] = 1.0;
        for i in 0..n-1 {
            data[[i+1, 0]] = 2.0 * data[[i, 0]];
        }
        let mut model = DiscreteSINDy::new(
            Box::new(PolynomialLibrary::new(1).with_bias(false)),
            Box::new(STLSQ::default())
        );
        model.fit(&[data], None, Some(&["x"])).unwrap();
        let x0 = array![1.0];
        let sim = model.simulate(&x0, 10, None).unwrap();
        assert_eq!(sim.nrows(), 11);
        assert!((sim[[10, 0]] - 1024.0).abs() < 1e-4);
    }
}
