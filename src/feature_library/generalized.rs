use ndarray::{Array2, Axis};
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
pub struct GeneralizedLibrary {
    pub libraries: Vec<Box<dyn FeatureLibrary>>,
    pub tensor_array: Vec<bool>,
    pub inputs_per_library: Vec<Option<Vec<usize>>>,
    pub exclude_tensors: bool,
    n_features_in: Option<usize>,
    n_output_features: Option<usize>,
}
impl GeneralizedLibrary {
    pub fn new(
        libraries: Vec<Box<dyn FeatureLibrary>>,
    ) -> Self {
        let n = libraries.len();
        Self {
            libraries,
            tensor_array: vec![false; n],
            inputs_per_library: vec![None; n],
            exclude_tensors: false,
            n_features_in: None,
            n_output_features: None,
        }
    }
    pub fn with_tensor_array(mut self, tensor_array: Vec<bool>) -> Self {
        self.tensor_array = tensor_array;
        self
    }
    pub fn with_inputs(mut self, inputs_per_library: Vec<Option<Vec<usize>>>) -> Self {
        self.inputs_per_library = inputs_per_library;
        self
    }
}
impl FeatureLibrary for GeneralizedLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        self.n_features_in = Some(n_features);
        let n_libs = self.libraries.len();
        if self.tensor_array.len() != n_libs {
            return Err(SINDyError::InvalidParameter("tensor_array length must match libraries".into()));
        }
        if self.inputs_per_library.len() != n_libs {
            return Err(SINDyError::InvalidParameter("inputs_per_library length must match libraries".into()));
        }
        let mut n_out = 0;
        let mut tensor_group_out = 0;
        for i in 0..n_libs {
            let x_sub = match &self.inputs_per_library[i] {
                Some(indices) => x.select(Axis(1), indices),
                None => x.clone(),
            };
            self.libraries[i].fit(&x_sub)?;
            let lib_out = self.libraries[i].n_output_features();
            if self.tensor_array[i] {
                if tensor_group_out == 0 {
                    tensor_group_out = lib_out;
                } else {
                    tensor_group_out *= lib_out;
                }
            } else {
                n_out += lib_out;
            }
        }
        if tensor_group_out > 0 {
            n_out += tensor_group_out;
        }
        self.n_output_features = Some(n_out);
        Ok(())
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_out = self.n_output_features.ok_or_else(|| {
            SINDyError::NotFitted("transform".into())
        })?;
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, n_out));
        let mut current_col = 0;
        let mut tensor_blocks: Vec<Array2<f64>> = Vec::new();
        for i in 0..self.libraries.len() {
            let x_sub = match &self.inputs_per_library[i] {
                Some(indices) => x.select(Axis(1), indices),
                None => x.clone(),
            };
            let lib_transformed = self.libraries[i].transform(&x_sub)?;
            if self.tensor_array[i] {
                tensor_blocks.push(lib_transformed);
            } else {
                let cols = lib_transformed.ncols();
                result.slice_mut(ndarray::s![.., current_col..current_col+cols])
                      .assign(&lib_transformed);
                current_col += cols;
            }
        }
        if !tensor_blocks.is_empty() {
            let tensor_result = generate_tensor_product(&tensor_blocks, n_samples)?;
            let cols = tensor_result.ncols();
            result.slice_mut(ndarray::s![.., current_col..current_col+cols])
                  .assign(&tensor_result);
        }
        Ok(result)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n_features = self.n_features_in.unwrap_or(0);
        let default_names: Vec<String> = (0..n_features).map(|i| format!("x{}", i)).collect();
        let names = input_features.unwrap_or(&default_names);
        let mut all_names = Vec::new();
        let mut tensor_names: Vec<Vec<String>> = Vec::new();
        for i in 0..self.libraries.len() {
            let lib_names = match &self.inputs_per_library[i] {
                Some(indices) => {
                    let sub_names: Vec<String> = indices.iter().map(|&idx| names[idx].clone()).collect();
                    self.libraries[i].get_feature_names(Some(&sub_names))
                }
                None => self.libraries[i].get_feature_names(Some(names)),
            };
            if self.tensor_array[i] {
                tensor_names.push(lib_names);
            } else {
                all_names.extend(lib_names);
            }
        }
        if !tensor_names.is_empty() {
            let mut current_tensors = tensor_names[0].clone();
            for next_block in tensor_names.iter().skip(1) {
                let mut new_tensors = Vec::new();
                for t1 in &current_tensors {
                    for t2 in next_block {
                        if t1 == "1" {
                            new_tensors.push(t2.clone());
                        } else if t2 == "1" {
                            new_tensors.push(t1.clone());
                        } else {
                            new_tensors.push(format!("{} {}", t1, t2));
                        }
                    }
                }
                current_tensors = new_tensors;
            }
            all_names.extend(current_tensors);
        }
        all_names
    }
    fn n_output_features(&self) -> usize {
        self.n_output_features.unwrap_or(0)
    }
}
fn generate_tensor_product(blocks: &[Array2<f64>], n_samples: usize) -> Result<Array2<f64>> {
    if blocks.is_empty() {
        return Ok(Array2::zeros((n_samples, 0)));
    }
    let mut current = blocks[0].clone();
    for next_block in blocks.iter().skip(1) {
        let n_cols1 = current.ncols();
        let n_cols2 = next_block.ncols();
        let mut new_result = Array2::<f64>::zeros((n_samples, n_cols1 * n_cols2));
        for c1 in 0..n_cols1 {
            for c2 in 0..n_cols2 {
                let col_idx = c1 * n_cols2 + c2;
                for r in 0..n_samples {
                    new_result[[r, col_idx]] = current[[r, c1]] * next_block[[r, c2]];
                }
            }
        }
        current = new_result;
    }
    Ok(current)
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::feature_library::polynomial::PolynomialLibrary;
    #[test]
    fn test_tensor_product() {
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let lib1 = PolynomialLibrary::new(1);
        let lib2 = PolynomialLibrary::new(1);
        let mut gen_lib = GeneralizedLibrary::new(vec![
            Box::new(lib1),
            Box::new(lib2)
        ])
        .with_tensor_array(vec![true, true])
        .with_inputs(vec![Some(vec![0]), Some(vec![1])]);
        gen_lib.fit(&x).unwrap();
        assert_eq!(gen_lib.n_output_features(), 4);
        let out = gen_lib.transform(&x).unwrap();
        assert_eq!(out.ncols(), 4);
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
        assert_eq!(out[[0, 2]], 1.0);
        assert_eq!(out[[0, 3]], 2.0);
        let names = gen_lib.get_feature_names(Some(&["x0".into(), "x1".into()]));
        assert_eq!(names, vec!["1", "x1", "x0", "x0 x1"]);
    }
    #[test]
    fn test_mixed_concat_and_tensor() {
        let x = array![[1.0, 2.0, 3.0]];
        let lib1 = PolynomialLibrary::new(1).with_bias(false);
        let lib2 = PolynomialLibrary::new(1).with_bias(false);
        let lib3 = PolynomialLibrary::new(1).with_bias(false);
        let mut gen_lib = GeneralizedLibrary::new(vec![
            Box::new(lib1),
            Box::new(lib2),
            Box::new(lib3)
        ])
        .with_tensor_array(vec![true, true, false])
        .with_inputs(vec![Some(vec![0, 1]), Some(vec![2]), None]);
        gen_lib.fit(&x).unwrap();
        assert_eq!(gen_lib.n_output_features(), 5);
        let out = gen_lib.transform(&x).unwrap();
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
        assert_eq!(out[[0, 2]], 3.0);
        assert_eq!(out[[0, 3]], 3.0);
        assert_eq!(out[[0, 4]], 6.0);
    }
}
