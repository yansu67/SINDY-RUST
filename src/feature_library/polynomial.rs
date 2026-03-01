use ndarray::Array2;
use std::collections::HashMap;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
#[derive(Debug, Clone)]
pub struct PolynomialLibrary {
    pub degree: usize,
    pub include_interaction: bool,
    pub interaction_only: bool,
    pub include_bias: bool,
    n_features_in: Option<usize>,
    n_output_features: Option<usize>,
    combinations: Vec<Vec<(usize, usize)>>,
}
impl Default for PolynomialLibrary {
    fn default() -> Self {
        Self {
            degree: 2,
            include_interaction: true,
            interaction_only: false,
            include_bias: true,
            n_features_in: None,
            n_output_features: None,
            combinations: Vec::new(),
        }
    }
}
impl PolynomialLibrary {
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            ..Default::default()
        }
    }
    pub fn with_interaction(mut self, include_interaction: bool) -> Self {
        self.include_interaction = include_interaction;
        self
    }
    pub fn with_interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }
    pub fn with_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }
    fn generate_combinations(
        n_features: usize,
        degree: usize,
        include_interaction: bool,
        interaction_only: bool,
        include_bias: bool,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut combos = Vec::new();
        if include_bias {
            combos.push(Vec::new());
        }
        for deg in 1..=degree {
            let mut current_combo: Vec<usize> = Vec::new();
            Self::generate_combos_recursive(
                n_features,
                deg,
                0,
                &mut current_combo,
                &mut combos,
                include_interaction,
                interaction_only,
            );
        }
        combos
    }
    fn generate_combos_recursive(
        n_features: usize,
        remaining_degree: usize,
        min_feature: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<(usize, usize)>>,
        include_interaction: bool,
        interaction_only: bool,
    ) {
        if remaining_degree == 0 {
            let mut powers: HashMap<usize, usize> = HashMap::new();
            for &feat in current.iter() {
                *powers.entry(feat).or_default() += 1;
            }
            let term: Vec<(usize, usize)> = {
                let mut v: Vec<_> = powers.into_iter().collect();
                v.sort_by_key(|&(f, _)| f);
                v
            };
            let n_distinct = term.len();
            if interaction_only && n_distinct < 2 && current.len() > 1 {
                return;
            }
            if !include_interaction && n_distinct > 1 {
                return;
            }
            result.push(term);
            return;
        }
        for feat in min_feature..n_features {
            current.push(feat);
            Self::generate_combos_recursive(
                n_features,
                remaining_degree - 1,
                feat,
                current,
                result,
                include_interaction,
                interaction_only,
            );
            current.pop();
        }
    }
}
impl FeatureLibrary for PolynomialLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        self.n_features_in = Some(n_features);
        self.combinations = Self::generate_combinations(
            n_features,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        );
        self.n_output_features = Some(self.combinations.len());
        Ok(())
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_out = self.n_output_features.ok_or_else(|| {
            SINDyError::NotFitted("transform".into())
        })?;
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, n_out));
        for (col_idx, combo) in self.combinations.iter().enumerate() {
            for row in 0..n_samples {
                let mut val = 1.0;
                for &(feat, power) in combo.iter() {
                    val *= x[[row, feat]].powi(power as i32);
                }
                result[[row, col_idx]] = val;
            }
        }
        Ok(result)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n_features = self.n_features_in.unwrap_or(0);
        let default_names: Vec<String> = (0..n_features).map(|i| format!("x{}", i)).collect();
        let names = input_features.unwrap_or(&default_names);
        let mut result = Vec::new();
        for combo in &self.combinations {
            if combo.is_empty() {
                result.push("1".to_string());
            } else {
                let mut parts = Vec::new();
                for &(feat, power) in combo {
                    let name = if feat < names.len() {
                        names[feat].as_str()
                    } else {
                        "?"
                    };
                    if power == 1 {
                        parts.push(name.to_string());
                    } else {
                        parts.push(format!("{}^{}", name, power));
                    }
                }
                result.push(parts.join(" "));
            }
        }
        result
    }
    fn n_output_features(&self) -> usize {
        self.n_output_features.unwrap_or(0)
    }
}
pub fn identity_library() -> PolynomialLibrary {
    PolynomialLibrary {
        degree: 1,
        include_bias: false,
        ..Default::default()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_polynomial_degree2_with_bias() {
        let mut lib = PolynomialLibrary::default();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        lib.fit(&x).unwrap();
        assert_eq!(lib.n_output_features(), 6);
        let transformed = lib.transform(&x).unwrap();
        assert_eq!(transformed.ncols(), 6);
        assert_eq!(transformed.nrows(), 2);
        assert!((transformed[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((transformed[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((transformed[[0, 2]] - 2.0).abs() < 1e-10);
        assert!((transformed[[0, 3]] - 1.0).abs() < 1e-10);
        assert!((transformed[[0, 4]] - 2.0).abs() < 1e-10);
        assert!((transformed[[0, 5]] - 4.0).abs() < 1e-10);
    }
    #[test]
    fn test_feature_names() {
        let mut lib = PolynomialLibrary::default();
        let x = array![[1.0, 2.0]];
        lib.fit(&x).unwrap();
        let names = lib.get_feature_names(Some(&["x".into(), "y".into()]));
        assert_eq!(names, vec!["1", "x", "y", "x^2", "x y", "y^2"]);
    }
    #[test]
    fn test_no_bias() {
        let mut lib = PolynomialLibrary::new(1).with_bias(false);
        let x = array![[1.0, 2.0, 3.0]];
        lib.fit(&x).unwrap();
        assert_eq!(lib.n_output_features(), 3);
    }
    #[test]
    fn test_identity_library() {
        let mut lib = identity_library();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        lib.fit(&x).unwrap();
        let out = lib.transform(&x).unwrap();
        assert_eq!(out, x);
    }
    #[test]
    fn test_no_interaction() {
        let mut lib = PolynomialLibrary::new(2)
            .with_interaction(false)
            .with_bias(false);
        let x = array![[2.0, 3.0]];
        lib.fit(&x).unwrap();
        assert_eq!(lib.n_output_features(), 4);
        let out = lib.transform(&x).unwrap();
        assert!((out[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((out[[0, 1]] - 3.0).abs() < 1e-10);
        assert!((out[[0, 2]] - 4.0).abs() < 1e-10);
        assert!((out[[0, 3]] - 9.0).abs() < 1e-10);
    }
}
