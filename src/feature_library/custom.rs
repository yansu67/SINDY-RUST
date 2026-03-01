use ndarray::Array2;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
#[derive(Clone)]
pub struct CustomLibrary {
    functions: Vec<(String, fn(f64) -> f64)>,
    n_features_in: Option<usize>,
    n_output_features: Option<usize>,
}
impl std::fmt::Debug for CustomLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.functions.iter().map(|(n, _)| n.as_str()).collect();
        f.debug_struct("CustomLibrary")
            .field("function_names", &names)
            .finish()
    }
}
impl CustomLibrary {
    pub fn new(functions: Vec<(String, fn(f64) -> f64)>) -> Self {
        Self {
            functions,
            n_features_in: None,
            n_output_features: None,
        }
    }
}
impl FeatureLibrary for CustomLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        self.n_features_in = Some(n_features);
        self.n_output_features = Some(n_features * self.functions.len());
        Ok(())
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_out = self.n_output_features.ok_or_else(|| {
            SINDyError::NotFitted("transform".into())
        })?;
        let n_features = self.n_features_in.unwrap();
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, n_out));
        let mut col = 0;
        for (_, func) in &self.functions {
            for feat in 0..n_features {
                for row in 0..n_samples {
                    result[[row, col]] = func(x[[row, feat]]);
                }
                col += 1;
            }
        }
        Ok(result)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n_features = self.n_features_in.unwrap_or(0);
        let default_names: Vec<String> = (0..n_features).map(|i| format!("x{}", i)).collect();
        let names = input_features.unwrap_or(&default_names);
        let mut result = Vec::new();
        for (func_name, _) in &self.functions {
            for feat in 0..n_features {
                let name = if feat < names.len() { &names[feat] } else { "?" };
                result.push(format!("{}({})", func_name, name));
            }
        }
        result
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
    fn test_custom_library() {
        let mut lib = CustomLibrary::new(vec![
            ("sq".into(), |x: f64| x * x),
            ("cube".into(), |x: f64| x * x * x),
        ]);
        let x = array![[2.0, 3.0], [4.0, 5.0]];
        lib.fit(&x).unwrap();
        assert_eq!(lib.n_output_features(), 4);
        let out = lib.transform(&x).unwrap();
        assert!((out[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((out[[0, 1]] - 9.0).abs() < 1e-10);
        assert!((out[[0, 2]] - 8.0).abs() < 1e-10);
        assert!((out[[0, 3]] - 27.0).abs() < 1e-10);
    }
}
