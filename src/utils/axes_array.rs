use ndarray::{Array, Array2, IxDyn};
use std::collections::HashMap;
use crate::error::{Result, SINDyError};
#[derive(Debug, Clone)]
pub struct AxesArray<A> {
    pub data: Array<A, IxDyn>,
    pub axes: HashMap<usize, String>,
}
impl<A> AxesArray<A>
where
    A: Clone,
{
    pub fn new<D: ndarray::Dimension>(data: Array<A, D>, axes: HashMap<usize, String>) -> Result<Self> {
        let ndim = data.ndim();
        for &axis in axes.keys() {
            if axis >= ndim {
                return Err(SINDyError::InvalidShape(format!(
                    "Axis {} out of bounds for array with {} dimensions",
                    axis, ndim
                )));
            }
        }
        Ok(Self { data: data.into_dyn(), axes })
    }
    pub fn axis_name(&self, axis: usize) -> Option<&String> {
        self.axes.get(&axis)
    }
    pub fn set_axis_name(&mut self, axis: usize, name: String) -> Result<()> {
        if axis >= self.data.ndim() {
            return Err(SINDyError::InvalidShape(format!(
                "Axis {} out of bounds for array with {} dimensions",
                axis,
                self.data.ndim()
            )));
        }
        self.axes.insert(axis, name);
        Ok(())
    }
    pub fn find_axis(&self, name: &str) -> Option<usize> {
        self.axes.iter().find(|(_, v)| *v == name).map(|(k, _)| *k)
    }
    pub fn flatten_to_2d(&self, feature_axis: usize) -> Result<Array2<A>> {
        let ndim = self.data.ndim();
        if feature_axis >= ndim {
            return Err(SINDyError::InvalidShape(format!(
                "Feature axis {} out of bounds for array with {} dimensions",
                feature_axis, ndim
            )));
        }
        let mut perm: Vec<usize> = (0..ndim).filter(|&i| i != feature_axis).collect();
        perm.push(feature_axis);
        let permuted = self.data.view().permuted_axes(perm).to_owned();
        let shape = permuted.shape();
        let n_features = shape[ndim - 1];
        let n_samples: usize = shape[..ndim - 1].iter().product();
        let mut vec = Vec::with_capacity(n_samples * n_features);
        for item in permuted.iter() {
            vec.push(item.clone());
        }
        let result = Array2::from_shape_vec((n_samples, n_features), vec).map_err(|e| {
            SINDyError::InvalidShape(format!("Failed to reshape to 2D: {}", e))
        })?;
        Ok(result)
    }
}
impl AxesArray<f64> {
    pub fn flatten_f64(&self, feature_axis: usize) -> Result<Array2<f64>> {
        let ndim = self.data.ndim();
        if feature_axis >= ndim {
            return Err(SINDyError::InvalidShape(format!(
                "Feature axis {} out of bounds for array with {} dimensions",
                feature_axis, ndim
            )));
        }
        let mut perm: Vec<usize> = (0..ndim).filter(|&i| i != feature_axis).collect();
        perm.push(feature_axis);
        let permuted = self.data.view().permuted_axes(perm).to_owned();
        let shape = permuted.shape();
        let n_features = shape[ndim - 1];
        let n_samples: usize = shape[..ndim - 1].iter().product();
        let mut flattened = Array2::<f64>::zeros((n_samples, n_features));
        for (i, item) in permuted.iter().enumerate() {
            let row = i / n_features;
            let col = i % n_features;
            flattened[[row, col]] = *item;
        }
        Ok(flattened)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};
    #[test]
    fn test_axes_array_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut axes = HashMap::new();
        axes.insert(0, "time".to_string());
        axes.insert(1, "space".to_string());
        let arr = AxesArray::new(data, axes.clone()).unwrap();
        assert_eq!(arr.axis_name(0), Some(&"time".to_string()));
        assert_eq!(arr.find_axis("space"), Some(1));
    }
    #[test]
    fn test_axes_array_flatten() {
        let mut data = Array3::<f64>::zeros((2, 2, 3));
        for t in 0..2 {
            for s in 0..2 {
                for f in 0..3 {
                    data[[t, s, f]] = (t * 100 + s * 10 + f) as f64;
                }
            }
        }
        let mut axes = HashMap::new();
        axes.insert(0, "time".to_string());
        axes.insert(1, "space".to_string());
        axes.insert(2, "ax_feature".to_string());
        let arr = AxesArray::new(data.into_dyn(), axes).unwrap();
        let feat_ax = arr.find_axis("ax_feature").unwrap();
        let flat = arr.flatten_f64(feat_ax).unwrap();
        assert_eq!(flat.shape(), &[4, 3]);
        assert_eq!(flat[[0, 0]], 0.0);
        assert_eq!(flat[[0, 1]], 1.0);
        assert_eq!(flat[[0, 2]], 2.0);
        assert_eq!(flat[[2, 0]], 100.0);
        assert_eq!(flat[[2, 1]], 101.0);
        assert_eq!(flat[[2, 2]], 102.0);
    }
}
