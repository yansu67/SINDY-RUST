pub mod polynomial;
pub mod fourier;
pub mod custom;
pub mod concat;
pub mod generalized;
pub mod parameterized;
pub mod pde;
pub mod weak_pde;
pub mod sindy_pi;
pub use concat::ConcatLibrary;
pub use custom::CustomLibrary;
pub use fourier::FourierLibrary;
pub use polynomial::PolynomialLibrary;
pub use generalized::GeneralizedLibrary;
pub use parameterized::ParameterizedLibrary;
pub use pde::PDELibrary;
pub use weak_pde::WeakPDELibrary;
pub use sindy_pi::SINDyPILibrary;
use ndarray::Array2;
use crate::error::Result;
pub trait FeatureLibrary {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn get_feature_names(&self, input_features: Option<&[String]>) -> Vec<String>;
    fn n_output_features(&self) -> usize;
}
