use ndarray::{Array2, Axis};
use crate::error::{Result, SINDyError};
pub fn validate_input(x: &Array2<f64>) -> Result<()> {
    if x.nrows() == 0 {
        return Err(SINDyError::InvalidShape(
            "Input array must have at least one sample (row)".into(),
        ));
    }
    if x.ncols() == 0 {
        return Err(SINDyError::InvalidShape(
            "Input array must have at least one feature (column)".into(),
        ));
    }
    Ok(())
}
pub fn validate_time(t: &[f64], n_samples: usize) -> Result<()> {
    if t.len() != n_samples {
        return Err(SINDyError::InvalidShape(format!(
            "Time array length {} does not match number of samples {}",
            t.len(),
            n_samples,
        )));
    }
    for i in 1..t.len() {
        if t[i] <= t[i - 1] {
            return Err(SINDyError::InvalidParameter(
                "Time array must be strictly increasing".into(),
            ));
        }
    }
    Ok(())
}
pub fn drop_nan_rows(x: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    assert_eq!(x.nrows(), y.nrows(), "x and y must have same number of rows");
    let mut keep = Vec::new();
    for i in 0..x.nrows() {
        let x_has_nan = x.row(i).iter().any(|v| v.is_nan());
        let y_has_nan = y.row(i).iter().any(|v| v.is_nan());
        if !x_has_nan && !y_has_nan {
            keep.push(i);
        }
    }
    let x_out = x.select(Axis(0), &keep);
    let y_out = y.select(Axis(0), &keep);
    (x_out, y_out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_validate_input_ok() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(validate_input(&x).is_ok());
    }
    #[test]
    fn test_validate_input_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        assert!(validate_input(&x).is_err());
    }
    #[test]
    fn test_validate_time_ok() {
        let t = vec![0.0, 0.1, 0.2, 0.3];
        assert!(validate_time(&t, 4).is_ok());
    }
    #[test]
    fn test_validate_time_wrong_length() {
        let t = vec![0.0, 0.1];
        assert!(validate_time(&t, 4).is_err());
    }
    #[test]
    fn test_drop_nan_rows() {
        let x = array![[1.0, 2.0], [f64::NAN, 4.0], [5.0, 6.0]];
        let y = array![[10.0], [20.0], [30.0]];
        let (xo, yo) = drop_nan_rows(&x, &y);
        assert_eq!(xo.nrows(), 2);
        assert_eq!(yo.nrows(), 2);
    }
}
