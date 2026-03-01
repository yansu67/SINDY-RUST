use ndarray::Array2;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
use crate::differentiation::{Differentiation, TimeStep};
use crate::differentiation::finite_difference::FiniteDifference;
pub struct PDELibrary {
    pub derivative_order: usize,
    pub include_interactions: bool,
    pub spatial_grid_shape: Vec<usize>,
    pub spatial_grid_spacing: Vec<f64>,
    pub differentiator: Box<dyn Differentiation>,
    n_features_in: Option<usize>,
    n_output_features: Option<usize>,
}
impl Default for PDELibrary {
    fn default() -> Self {
        Self {
            derivative_order: 1,
            include_interactions: true,
            spatial_grid_shape: vec![],
            spatial_grid_spacing: vec![],
            differentiator: Box::new(FiniteDifference::default()),
            n_features_in: None,
            n_output_features: None,
        }
    }
}
impl PDELibrary {
    pub fn new(
        derivative_order: usize,
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
            spatial_grid_shape,
            spatial_grid_spacing,
            ..Default::default()
        })
    }
    pub fn with_interactions(mut self, include_interactions: bool) -> Self {
        self.include_interactions = include_interactions;
        self
    }
    pub fn with_differentiator(mut self, differentiator: Box<dyn Differentiation>) -> Self {
        self.differentiator = differentiator;
        self
    }
}
impl FeatureLibrary for PDELibrary {
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
        let spatial_size: usize = self.spatial_grid_shape.iter().product();
        let _n_time_steps = n_samples / spatial_size;
        let n_spatial_dims = self.spatial_grid_shape.len();
        let mut result = Array2::<f64>::zeros((n_samples, n_out));
        let mut col_ptr = 0;
        for j in 0..n_features {
            result.slice_mut(ndarray::s![.., col_ptr]).assign(&x.column(j));
            col_ptr += 1;
        }
        let mut active_derivatives = Vec::new();
        for dim in 0..n_spatial_dims {
            let dx = self.spatial_grid_spacing[dim];
            let n_points_in_dim = self.spatial_grid_shape[dim];
            let mut stride = 1;
            for next_dim in (dim + 1)..n_spatial_dims {
                stride *= self.spatial_grid_shape[next_dim];
            }
            let chunk_size = stride * n_points_in_dim;
            for feature in 0..n_features {
                let feature_data = x.column(feature).to_owned();
                let mut current_deriv = feature_data.clone();
                for _order in 1..=self.derivative_order {
                    let mut next_deriv = ndarray::Array1::<f64>::zeros(n_samples);
                    for start_idx in (0..n_samples).step_by(chunk_size) {
                        for offset in 0..stride {
                            let mut line = ndarray::Array2::<f64>::zeros((n_points_in_dim, 1));
                            for p in 0..n_points_in_dim {
                                let idx = start_idx + p * stride + offset;
                                line[[p, 0]] = current_deriv[idx];
                            }
                            let d_line = self.differentiator.differentiate(&line, &TimeStep::Uniform(dx))?;
                            for p in 0..n_points_in_dim {
                                let idx = start_idx + p * stride + offset;
                                next_deriv[idx] = d_line[[p, 0]];
                            }
                        }
                    }
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
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_pde_library_1d() {
        let x = array![
            [0.0],
            [1.0],
            [4.0],
            [9.0],
            [16.0]
        ];
        let mut pde_lib = PDELibrary::new(
            2,
            vec![5],
            vec![1.0]
        ).unwrap().with_interactions(true);
        pde_lib.fit(&x).unwrap();
        assert_eq!(pde_lib.n_output_features(), 5);
        let out = pde_lib.transform(&x).unwrap();
        assert_eq!(out.ncols(), 5);
        assert_eq!(out[[2, 0]], 4.0);
        assert!((out[[2, 1]] - 4.0).abs() < 1e-10);
        assert!((out[[2, 2]] - 2.0).abs() < 1e-10);
        assert!((out[[2, 3]] - 16.0).abs() < 1e-10);
        assert!((out[[2, 4]] - 8.0).abs() < 1e-10);
        let names = pde_lib.get_feature_names(Some(&["u".into()]));
        assert_eq!(names, vec![
            "u", "u_x", "u_xx", "u u_x", "u u_xx"
        ]);
    }
}
