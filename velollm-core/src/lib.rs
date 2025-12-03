// VeloLLM Core Library
//
// Core functionality for hardware detection, optimization, and performance monitoring

pub mod error;
pub mod hardware;
pub mod optimizer;
pub mod paged_attention;
pub mod parser;
pub mod scheduler;

// Re-export common error types for convenience
pub use error::HardwareError;

#[cfg(test)]
mod hardware_tests;
