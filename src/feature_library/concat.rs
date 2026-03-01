use ndarray::Array2;
use super::FeatureLibrary;
use crate::error::{Result, SINDyError};
pub struct ConcatLibrary {
    libraries: Vec<Box<dyn FeatureLibrary>>,
    n_output_features: Option<usize>,
}
impl ConcatLibrary {
    pub fn new(libraries: Vec<Box<dyn FeatureLibrary>>) -> Self {
        Self {
            libraries,
            n_output_features: None,
        }
    }
}
impl FeatureLibrary for ConcatLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let mut total = 0;
        for lib in self.libraries.iter_mut() {
            lib.fit(x)?;
            total += lib.n_output_features();
        }
        self.n_output_features = Some(total);
        Ok(())
    }
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_out = self.n_output_features.ok_or_else(|| {
            SINDyError::NotFitted("transform".into())
        })?;
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, n_out));
        let mut col_offset = 0;
        for lib in &self.libraries {
            let part = lib.transform(x)?;
            let n_cols = part.ncols();
            result
                .slice_mut(ndarray::s![.., col_offset..col_offset + n_cols])
                .assign(&part);
            col_offset += n_cols;
        }
        Ok(result)
    }
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String> {
        let mut names = Vec::new();
        for lib in &self.libraries {
            names.extend(lib.get_feature_names(input_features));
        }
        names
    }
    fn n_output_features(&self) -> usize {
        self.n_output_features.unwrap_or(0)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_library::polynomial::PolynomialLibrary;
    use crate::feature_library::fourier::FourierLibrary;
    use ndarray::array;
    #[test]
    fn test_concat_library() {
        let poly = PolynomialLibrary::new(1).with_bias(false);
        let fourier = FourierLibrary::new(1);
        let mut concat = ConcatLibrary::new(vec![
            Box::new(poly),
            Box::new(fourier),
        ]);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        concat.fit(&x).unwrap();
        assert_eq!(concat.n_output_features(), 6);
        let out = concat.transform(&x).unwrap();
        assert_eq!(out.ncols(), 6);
        assert!((out[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((out[[0, 1]] - 2.0).abs() < 1e-10);
    }
}
