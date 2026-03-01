use ndarray::Array2;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
use crate::differentiation::{Differentiation, TimeStep};
use crate::differentiation::finite_difference::FiniteDifference;
pub struct WeakPDELibrary {
    pub derivative_order: usize,
    pub include_interactions: bool,
    pub num_subdomains: usize,
    pub spatial_grid_shape: Vec<usize>,
    pub spatial_grid_spacing: Vec<f64>,
    n_features_in: Option<usize>,
    n_output_features: Option<usize>,
}
impl Default for WeakPDELibrary {
    fn default() -> Self {
        Self {
            derivative_order: 1,
            include_interactions: true,
            num_subdomains: 10,
            spatial_grid_shape: vec![],
            spatial_grid_spacing: vec![],
            n_features_in: None,
            n_output_features: None,
        }
    }
}
impl WeakPDELibrary {
    pub fn new(
        derivative_order: usize,
        num_subdomains: usize,
        spatial_grid_shape: Vec<usize>,
        spatial_grid_spacing: Vec<f64>,
    ) -> Result<Self> {
        if spatial_grid_shape.len() != spatial_grid_spacing.len() {
            return Err(SINDyError::InvalidParameter(
                "spatial_grid_shape and spatial_grid_spacing must have same length".into()
            ));
        }
        Ok(Self {
            derivative_order,
            num_subdomains,
            spatial_grid_shape,
            spatial_grid_spacing,
            ..Default::default()
        })
    }
    pub fn with_interactions(mut self, include_interactions: bool) -> Self {
        self.include_interactions = include_interactions;
        self
    }
}
impl FeatureLibrary for WeakPDELibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        let n_samples = x.nrows();
        let spatial_size: usize = self.spatial_grid_shape.iter().product();
        if spatial_size == 0 {
            return Err(SINDyError::InvalidParameter("spatial_grid_shape must not contain zeros".into()));
        }
        if !n_samples.is_multiple_of(spatial_size) {
            return Err(SINDyError::InvalidShape(format!(
                "Number of samples ({}) must be a multiple of the spatial grid size ({})",
                n_samples, spatial_size
            )));
        }
        self.n_features_in = Some(n_features);
        let n_spatial_dims = self.spatial_grid_shape.len();
        let n_basic_derivatives = n_features * n_spatial_dims * self.derivative_order;
        let mut n_out = n_features + n_basic_derivatives;
        if self.include_interactions {
            n_out += n_features * n_basic_derivatives;
        }
        self.n_output_features = Some(n_out);
        Ok(())
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = self.n_features_in.ok_or_else(|| {
            SINDyError::NotFitted("transform".into())
        })?;
        let n_out = self.n_output_features.unwrap();
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, n_out));
        let mut col_ptr = 0;
        for j in 0..n_features {
            let smoothed_col = x.column(j).to_owned();
            result.slice_mut(ndarray::s![.., col_ptr]).assign(&smoothed_col);
            col_ptr += 1;
        }
        let n_spatial_dims = self.spatial_grid_shape.len();
        let mut active_derivatives = Vec::new();
        for _dim in 0..n_spatial_dims {
            for feature in 0..n_features {
                let feature_data = x.column(feature).to_owned();
                let mut current_deriv = feature_data.clone();
                for _order in 1..=self.derivative_order {
                    let mut next_deriv = ndarray::Array1::<f64>::zeros(n_samples);
                    let diff = FiniteDifference::default();
                    let line = current_deriv.clone().into_shape_with_order((n_samples, 1)).unwrap();
                    let d_line = diff.differentiate(&line, &TimeStep::Uniform(1.0)).unwrap();
                    next_deriv.assign(&d_line.column(0));
                    result.slice_mut(ndarray::s![.., col_ptr]).assign(&next_deriv);
                    active_derivatives.push(next_deriv.clone());
                    col_ptr += 1;
                    current_deriv = next_deriv;
                }
            }
        }
        if self.include_interactions {
            for j in 0..n_features {
                let base_feat = x.column(j);
                for deriv in &active_derivatives {
                    for i in 0..n_samples {
                        result[[i, col_ptr]] = base_feat[i] * deriv[i];
                    }
                    col_ptr += 1;
                }
            }
        }
        Ok(result)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n_features = self.n_features_in.unwrap_or(0);
        let default_names: Vec<String> = (0..n_features).map(|i| format!("x{}", i)).collect();
        let names = input_features.unwrap_or(&default_names);
        let mut output_names = Vec::new();
        for name in names {
            output_names.push(name.clone());
        }
        let n_spatial_dims = self.spatial_grid_shape.len();
        let mut spatial_axes: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
        if n_spatial_dims > 3 {
            for i in 3..n_spatial_dims {
                spatial_axes.push(format!("x{}", i));
            }
        }
        let mut deriv_names = Vec::new();
        for dim in 0..n_spatial_dims {
            let axis_name = &spatial_axes[dim];
            for feature in 0..n_features {
                let feat_name = &names[feature];
                for order in 1..=self.derivative_order {
                    let mut deriv_suffix = String::new();
                    for _ in 0..order {
                        deriv_suffix.push_str(axis_name);
                    }
                    let name = format!("{}_{}", feat_name, deriv_suffix);
                    output_names.push(name.clone());
                    deriv_names.push(name);
                }
            }
        }
        if self.include_interactions {
            for (feature, _) in names.iter().enumerate() {
                let feat_name = &names[feature];
                for d_name in &deriv_names {
                    output_names.push(format!("{} {}", feat_name, d_name));
                }
            }
        }
        output_names
    }
    fn n_output_features(&self) -> usize {
        self.n_output_features.unwrap_or(0)
    }
}
