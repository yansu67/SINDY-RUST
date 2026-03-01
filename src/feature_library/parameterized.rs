use ndarray::Array2;
use super::{FeatureLibrary, GeneralizedLibrary};
use crate::error::{Result, SINDyError};
pub struct ParameterizedLibrary {
    internal: GeneralizedLibrary,
}
impl ParameterizedLibrary {
    pub fn new(
        state_library: Box<dyn FeatureLibrary>,
        parameter_library: Box<dyn FeatureLibrary>,
        n_state_features: usize,
        n_input_features: usize,
    ) -> Result<Self> {
        if n_state_features > n_input_features {
            return Err(SINDyError::InvalidParameter(format!(
                "n_state_features ({}) cannot be greater than total n_input_features ({})",
                n_state_features, n_input_features
            )));
        }
        let state_indices: Vec<usize> = (0..n_state_features).collect();
        let param_indices: Vec<usize> = (n_state_features..n_input_features).collect();
        let generalized = GeneralizedLibrary::new(vec![state_library, parameter_library])
            .with_tensor_array(vec![true, true])
            .with_inputs(vec![Some(state_indices), Some(param_indices)]);
        Ok(Self {
            internal: generalized,
        })
    }
}
impl FeatureLibrary for ParameterizedLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        self.internal.fit(x)
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.internal.transform(x)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        self.internal.get_feature_names(input_features)
    }
    fn n_output_features(&self) -> usize {
        self.internal.n_output_features()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::feature_library::polynomial::PolynomialLibrary;
    #[test]
    fn test_parameterized_library() {
        let x = array![
            [1.0, 2.0, 3.0]
        ];
        let state_lib = Box::new(PolynomialLibrary::new(1).with_bias(false));
        let param_lib = Box::new(PolynomialLibrary::new(1).with_bias(false));
        let mut param_gen_lib = ParameterizedLibrary::new(
            state_lib,
            param_lib,
            2,
            3,
        ).unwrap();
        param_gen_lib.fit(&x).unwrap();
        assert_eq!(param_gen_lib.n_output_features(), 2);
        let out = param_gen_lib.transform(&x).unwrap();
        assert_eq!(out[[0, 0]], 3.0);
        assert_eq!(out[[0, 1]], 6.0);
        let names = param_gen_lib.get_feature_names(Some(&[
            "x0".to_string(),
            "x1".to_string(),
            "u0".to_string()
        ]));
        assert_eq!(names, vec!["x0 u0", "x1 u0"]);
    }
}
