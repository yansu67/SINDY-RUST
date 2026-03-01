use ndarray::{Array2, Zip};
pub fn prox_l0(x: &Array2<f64>, weight: f64) -> Array2<f64> {
    let threshold = (2.0 * weight).sqrt();
    x.mapv(|v| if v.abs() < threshold { 0.0 } else { v })
}
pub fn prox_l1(x: &Array2<f64>, weight: f64) -> Array2<f64> {
    x.mapv(|v| {
        let sign = v.signum();
        let shrunk = v.abs() - weight;
        if shrunk > 0.0 {
            sign * shrunk
        } else {
            0.0
        }
    })
}
pub fn prox_l2(x: &Array2<f64>, weight: f64) -> Array2<f64> {
    x.mapv(|v| v / (1.0 + 2.0 * weight))
}
pub fn prox_weighted_l0(x: &Array2<f64>, weights: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    Zip::from(&mut result)
        .and(x)
        .and(weights)
        .for_each(|r, &xi, &w| {
            let threshold = (2.0 * w).sqrt();
            *r = if xi.abs() < threshold { 0.0 } else { xi };
        });
    result
}
pub fn prox_weighted_l1(x: &Array2<f64>, weights: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    Zip::from(&mut result)
        .and(x)
        .and(weights)
        .for_each(|r, &xi, &w| {
            let sign = xi.signum();
            let shrunk = xi.abs() - w;
            *r = if shrunk > 0.0 { sign * shrunk } else { 0.0 };
        });
    result
}
pub fn regularization_l0(x: &Array2<f64>, weight: f64) -> f64 {
    weight * x.iter().filter(|&&v| v.abs() > f64::EPSILON).count() as f64
}
pub fn regularization_l1(x: &Array2<f64>, weight: f64) -> f64 {
    weight * x.iter().map(|v| v.abs()).sum::<f64>()
}
pub fn regularization_l2(x: &Array2<f64>, weight: f64) -> f64 {
    weight * x.iter().map(|v| v * v).sum::<f64>()
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_prox_l0() {
        let x = array![[0.5, 0.01], [-3.0, 0.005]];
        let result = prox_l0(&x, 0.05);
        assert_eq!(result[[0, 0]], 0.5);
        assert_eq!(result[[0, 1]], 0.0);
        assert_eq!(result[[1, 0]], -3.0);
        assert_eq!(result[[1, 1]], 0.0);
    }
    #[test]
    fn test_prox_l1() {
        let x = array![[1.0, 0.05], [-2.0, 0.0]];
        let result = prox_l1(&x, 0.1);
        assert!((result[[0, 0]] - 0.9).abs() < 1e-10);
        assert_eq!(result[[0, 1]], 0.0);
        assert!((result[[1, 0]] - (-1.9)).abs() < 1e-10);
    }
    #[test]
    fn test_regularization_l1() {
        let x = array![[1.0, -2.0], [3.0, 0.0]];
        let val = regularization_l1(&x, 0.5);
        assert!((val - 3.0).abs() < 1e-10);
    }
}
