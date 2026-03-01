use thiserror::Error;
#[derive(Error, Debug)]
pub enum SINDyError {
    #[error("Invalid input shape: {0}")]
    InvalidShape(String),
    #[error("Not yet fitted: call fit() before {0}")]
    NotFitted(String),
    #[error("Linear algebra error: {0}")]
    LinAlg(String),
    #[error("Convergence error: {0}")]
    Convergence(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}
pub type Result<T> = std::result::Result<T, SINDyError>;
