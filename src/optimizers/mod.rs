pub mod stlsq;
pub mod sr3;
pub mod ssr;
pub mod frols;
pub mod evidence_greedy;
pub mod ensemble;
pub mod constrained_sr3;
pub use stlsq::STLSQ;
pub use sr3::{SR3, TrimType};
pub use ssr::SSR;
pub use frols::FROLS;
pub use evidence_greedy::EvidenceGreedy;
pub use ensemble::{EnsembleOptimizer, EnsembleType};
pub use constrained_sr3::ConstrainedSR3;
use ndarray::Array2;
use crate::error::Result;
pub trait Optimizer {
    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()>;
    fn coef(&self) -> &Array2<f64>;
    fn complexity(&self) -> usize;
}
