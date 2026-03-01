use ndarray::{Array1, Array2, Axis};
use crate::differentiation::{finite_difference::FiniteDifference, Differentiation, TimeStep};
use crate::error::{Result, SINDyError};
use crate::feature_library::{polynomial::PolynomialLibrary, FeatureLibrary};
use crate::optimizers::{stlsq::STLSQ, Optimizer};
pub struct SINDy {
    pub feature_library: Box<dyn FeatureLibrary>,
    pub optimizer: Box<dyn Optimizer>,
    pub differentiation_method: Box<dyn Differentiation>,
    n_input_features: Option<usize>,
    n_control_features: Option<usize>,
    n_output_features: Option<usize>,
    feature_names: Vec<String>,
}
impl Default for SINDy {
    fn default() -> Self {
        Self {
            feature_library: Box::new(PolynomialLibrary::default()),
            optimizer: Box::new(STLSQ::default()),
            differentiation_method: Box::new(FiniteDifference::default()),
            n_input_features: None,
            n_control_features: None,
            n_output_features: None,
            feature_names: Vec::new(),
        }
    }
}
impl SINDy {
    pub fn new(
        feature_library: Box<dyn FeatureLibrary>,
        optimizer: Box<dyn Optimizer>,
        differentiation_method: Box<dyn Differentiation>,
    ) -> Self {
        Self {
            feature_library,
            optimizer,
            differentiation_method,
            n_input_features: None,
            n_control_features: None,
            n_output_features: None,
            feature_names: Vec::new(),
        }
    }
    pub fn fit(
        &mut self,
        x: &[Array2<f64>],
        t: &[TimeStep],
        x_dot: Option<&[Array2<f64>]>,
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
        let mut x_dot_concatenated = Vec::new();
        for i in 0..x.len() {
            let xi = &x[i];
            let ti = &t[i];
            let xd_computed;
            let xi_dot = match x_dot {
                Some(xds) => &xds[i],
                None => {
                    xd_computed = self.differentiation_method.differentiate(xi, ti)?;
                    &xd_computed
                }
            };
            let xi_full = match u {
                Some(us) => {
                    let ui = &us[i];
                    if xi.nrows() != ui.nrows() {
                        return Err(SINDyError::InvalidShape(format!(
                            "Trajectory {}: x rows ({}) != u rows ({})",
                            i, xi.nrows(), ui.nrows()
                        )));
                    }
                    ndarray::concatenate(ndarray::Axis(1), &[xi.view(), ui.view()]).map_err(|e| {
                        SINDyError::InvalidShape(format!("Failed to concatenate x and u: {}", e))
                    })?
                }
                None => xi.clone(),
            };
            let (xi_clean, xi_dot_clean) = remove_nan_rows(&xi_full, xi_dot);
            if xi_clean.nrows() > 0 {
                x_concatenated.push(xi_clean);
                x_dot_concatenated.push(xi_dot_clean);
            }
        }
        if x_concatenated.is_empty() {
             return Err(SINDyError::InvalidShape(
                "Not enough non-NaN samples across trajectories".into(),
            ));
        }
        let full_x = concat_arrays(&x_concatenated)?;
        let full_x_dot = concat_arrays(&x_dot_concatenated)?;
        self.feature_library.fit(&full_x)?;
        let theta = self.feature_library.transform(&full_x)?;
        self.n_output_features = Some(self.feature_library.n_output_features());
        self.optimizer.fit(&theta, &full_x_dot)?;
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
                format!("{}'", self.feature_names[i])
            } else {
                format!("x{}'", i)
            };
            println!("{} = {}", lhs, eq);
        }
        Ok(())
    }
    pub fn score(
        &self,
        x: &[Array2<f64>],
        t: &[TimeStep],
        x_dot: Option<&[Array2<f64>]>,
        u: Option<&[Array2<f64>]>,
    ) -> Result<f64> {
        self.check_fitted()?;
        let mut x_dot_clean_all = Vec::new();
        let mut x_dot_pred_all = Vec::new();
        for i in 0..x.len() {
            let xi = &x[i];
            let ti = &t[i];
            let xd_computed;
            let xi_dot = match x_dot {
                Some(xds) => &xds[i],
                None => {
                    xd_computed = self.differentiation_method.differentiate(xi, ti)?;
                    &xd_computed
                }
            };
            let mut x_full_view = vec![xi.view()];
            if let Some(us) = u {
                x_full_view.push(us[i].view());
            }
            let xi_full = ndarray::concatenate(ndarray::Axis(1), &x_full_view).unwrap_or(xi.clone());
            let (xi_clean, xi_dot_clean) = remove_nan_rows(&xi_full, xi_dot);
            if xi_clean.nrows() > 0 {
                let xi_state_clean = xi_clean.slice(ndarray::s![.., 0..self.n_input_features.unwrap_or(xi_clean.ncols())]).to_owned();
                let ui_clean = if u.is_some() {
                    Some(xi_clean.slice(ndarray::s![.., self.n_input_features.unwrap()..]).to_owned())
                } else {
                    None
                };
                let xi_dot_pred = self.predict(&xi_state_clean, ui_clean.as_ref())?;
                x_dot_clean_all.push(xi_dot_clean);
                x_dot_pred_all.push(xi_dot_pred);
            }
        }
        if x_dot_clean_all.is_empty() {
             return Err(SINDyError::InvalidShape(
                "Not enough non-NaN samples to score".into(),
            ));
        }
        let full_xd_clean = concat_arrays(&x_dot_clean_all)?;
        let full_xd_pred = concat_arrays(&x_dot_pred_all)?;
        let ss_res: f64 = (&full_xd_clean - &full_xd_pred)
            .mapv(|v| v * v)
            .sum();
        let mean = full_xd_clean.mean_axis(Axis(0)).unwrap();
        let ss_tot: f64 = full_xd_clean
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
    pub fn simulate(&self, x0: &Array1<f64>, t: &[f64], u: Option<&Array2<f64>>) -> Result<Array2<f64>> {
        self.check_fitted()?;
        let n_features = x0.len();
        let n_steps = t.len();
        if let Some(control) = u {
            if control.nrows() != n_steps {
                return Err(SINDyError::InvalidShape(format!("Control array 'u' must have {} rows to match 't'", n_steps)));
            }
        }
        let mut trajectory = Array2::<f64>::zeros((n_steps, n_features));
        trajectory.row_mut(0).assign(x0);
        for i in 1..n_steps {
            let dt = t[i] - t[i - 1];
            let x_current = trajectory.row(i - 1).to_owned();
            let x_2d = x_current.view().insert_axis(Axis(0)).to_owned();
            let u_2d = u.map(|us| us.row(i - 1).to_owned().view().insert_axis(Axis(0)).to_owned());
            let x_dot = self.predict(&x_2d, u_2d.as_ref())?;
            for j in 0..n_features {
                trajectory[[i, j]] = trajectory[[i - 1, j]] + dt * x_dot[[0, j]];
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
            return Err(SINDyError::NotFitted("SINDy model".into()));
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
    fn test_sindy_linear_system() {
        let n = 100;
        let dt = 0.01;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let x0 = 3.0;
        let y0 = 0.5;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = x0 * (-2.0 * t[i]).exp();
            data[[i, 1]] = y0 * t[i].exp();
        }
        let mut model = SINDy::default();
        model
            .fit(
                &[data.clone()],
                &[TimeStep::Uniform(dt)],
                None,
                None,
                Some(&["x", "y"]),
            )
            .unwrap();
        let coef = model.coefficients().unwrap();
        let names = model.get_feature_names();
        let x_idx = names.iter().position(|n| n == "x").unwrap();
        let y_idx = names.iter().position(|n| n == "y").unwrap();
        assert!(
            (coef[[0, x_idx]] - (-2.0)).abs() < 0.1,
            "Expected x' coef of x ~-2.0, got {}",
            coef[[0, x_idx]]
        );
        assert!(
            (coef[[1, y_idx]] - 1.0).abs() < 0.1,
            "Expected y' coef of y ~1.0, got {}",
            coef[[1, y_idx]]
        );
        let score = model
            .score(&[data], &[TimeStep::Uniform(dt)], None, None)
            .unwrap();
        assert!(
            score > 0.95,
            "Expected R² > 0.95, got {}",
            score
        );
        model.print_model(3).unwrap();
    }
    #[test]
    fn test_sindy_simulate() {
        let n = 200;
        let dt = 0.01;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = (-t[i]).exp();
        }
        let mut model = SINDy::default();
        model
            .fit(&[data], &[TimeStep::Uniform(dt)], None, None, Some(&["x"]))
            .unwrap();
        let x0 = array![1.0];
        let sim = model.simulate(&x0, &t, None).unwrap();
        let t_final = t[n - 1];
        let expected = (-t_final).exp();
        let got = sim[[n - 1, 0]];
        assert!(
            (got - expected).abs() < 0.1,
            "Simulation at t={}: expected {}, got {}",
            t_final, expected, got
        );
    }
    #[test]
    fn test_sindy_control_inputs() {
        let n = 100;
        let dt = 0.01;
        let t_step = TimeStep::Uniform(dt);
        let mut data_x = Array2::<f64>::zeros((n, 1));
        let mut data_u = Array2::<f64>::zeros((n, 1));
        let mut data_xd = Array2::<f64>::zeros((n, 1));
        let mut x_val = 1.0;
        for i in 0..n {
            let u_val = (i as f64 * dt).sin();
            let xd_val = -2.0 * x_val + u_val;
            data_x[[i, 0]] = x_val;
            data_u[[i, 0]] = u_val;
            data_xd[[i, 0]] = xd_val;
            x_val += xd_val * dt;
        }
        let mut model = SINDy::new(
            Box::new(PolynomialLibrary::new(1).with_bias(false).with_interaction(false)),
            Box::new(STLSQ::new(1e-3, 0.05).unwrap()),
            Box::new(FiniteDifference::default()),
        );
        model.fit(&[data_x.clone()], std::slice::from_ref(&t_step), Some(&[data_xd.clone()]), Some(&[data_u.clone()]), Some(&["x", "u"])).unwrap();
        let coefs = model.coefficients().unwrap();
        let names = model.get_feature_names();
        let x_idx = names.iter().position(|n| n == "x").unwrap();
        let u_idx = names.iter().position(|n| n == "u").unwrap();
        assert!((coefs[[0, x_idx]] - (-2.0)).abs() < 0.1, "Expected x coef ~-2.0");
        assert!((coefs[[0, u_idx]] - 1.0).abs() < 0.1, "Expected u coef ~1.0");
    }
}
