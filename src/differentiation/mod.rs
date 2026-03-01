pub mod finite_difference;
use ndarray::Array2;
use crate::error::Result;
#[derive(Debug, Clone)]
pub enum TimeStep {
    Uniform(f64),
    Array(Vec<f64>),
}
pub trait Differentiation {
    fn differentiate(&self, x: &Array2<f64>, t: &TimeStep) -> Result<Array2<f64>>;
}
