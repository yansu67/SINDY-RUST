use ndarray::Array2;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
pub struct SINDyPILibrary {
    pub base_library: Box<dyn FeatureLibrary>,
    pub n_features_in: Option<usize>,
    pub n_output_features: Option<usize>,
    pub feature_names: Vec<String>,
}
impl SINDyPILibrary {
    pub fn new(base_library: Box<dyn FeatureLibrary>) -> Self {
        Self {
            base_library,
            n_features_in: None,
            n_output_features: None,
            feature_names: Vec::new(),
        }
    }
}
impl FeatureLibrary for SINDyPILibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        self.n_features_in = Some(x.ncols());
        self.base_library.fit(x)?;
        self.n_output_features = Some(self.base_library.n_output_features());
        Ok(())
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.n_features_in.is_none() {
            return Err(SINDyError::NotFitted("transform".into()));
        }
        self.base_library.transform(x)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        self.base_library.get_feature_names(input_features)
    }
    fn n_output_features(&self) -> usize {
        self.n_output_features.unwrap_or(0)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::feature_library::polynomial::PolynomialLibrary;
    #[test]
    fn test_sindy_pi_library() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let base_lib = PolynomialLibrary::new(2);
        let mut pi_lib = SINDyPILibrary::new(Box::new(base_lib));
        pi_lib.fit(&x).unwrap();
        assert_eq!(pi_lib.n_output_features(), 6);
        let out = pi_lib.transform(&x).unwrap();
        assert_eq!(out.ncols(), 6);
        let names = pi_lib.get_feature_names(Some(&["x".into(), "y".into()]));
        assert_eq!(names, vec!["1", "x", "y", "x^2", "x y", "y^2"]);
    }
}
