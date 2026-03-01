use ndarray::Array2;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
#[derive(Debug, Clone)]
pub struct FourierLibrary {
    pub n_frequencies: usize,
    pub include_sin: bool,
    pub include_cos: bool,
    n_features_in: Option<usize>,
    n_output_features: Option<usize>,
}
impl Default for FourierLibrary {
    fn default() -> Self {
        Self {
            n_frequencies: 1,
            include_sin: true,
            include_cos: true,
            n_features_in: None,
            n_output_features: None,
        }
    }
}
impl FourierLibrary {
    pub fn new(n_frequencies: usize) -> Self {
        Self {
            n_frequencies,
            ..Default::default()
        }
    }
}
impl FeatureLibrary for FourierLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        self.n_features_in = Some(n_features);
        let terms_per_feature = if self.include_sin && self.include_cos {
            2 * self.n_frequencies
        } else {
            self.n_frequencies
        };
        self.n_output_features = Some(n_features * terms_per_feature);
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
        for feat in 0..n_features {
            for freq in 1..=self.n_frequencies {
                if self.include_sin {
                    for row in 0..n_samples {
                        result[[row, col]] = (freq as f64 * x[[row, feat]]).sin();
                    }
                    col += 1;
                }
                if self.include_cos {
                    for row in 0..n_samples {
                        result[[row, col]] = (freq as f64 * x[[row, feat]]).cos();
                    }
                    col += 1;
                }
            }
        }
        Ok(result)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n_features = self.n_features_in.unwrap_or(0);
        let default_names: Vec<String> = (0..n_features).map(|i| format!("x{}", i)).collect();
        let names = input_features.unwrap_or(&default_names);
        let mut result = Vec::new();
        for feat in 0..n_features {
            let name = if feat < names.len() { &names[feat] } else { "?" };
            for freq in 1..=self.n_frequencies {
                if self.include_sin {
                    if freq == 1 {
                        result.push(format!("sin({})", name));
                    } else {
                        result.push(format!("sin({}{})", freq, name));
                    }
                }
                if self.include_cos {
                    if freq == 1 {
                        result.push(format!("cos({})", name));
                    } else {
                        result.push(format!("cos({}{})", freq, name));
                    }
                }
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
    use std::f64::consts::PI;
    #[test]
    fn test_fourier_basic() {
        let mut lib = FourierLibrary::new(2);
        let x = array![[0.0], [PI / 2.0], [PI]];
        lib.fit(&x).unwrap();
        assert_eq!(lib.n_output_features(), 4);
        let out = lib.transform(&x).unwrap();
        assert!((out[[0, 0]]).abs() < 1e-10);
        assert!((out[[0, 1]] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_fourier_names() {
        let mut lib = FourierLibrary::new(2);
        let x = array![[0.0]];
        lib.fit(&x).unwrap();
        let names = lib.get_feature_names(Some(&["t".into()]));
        assert_eq!(names, vec!["sin(t)", "cos(t)", "sin(2t)", "cos(2t)"]);
    }
}
