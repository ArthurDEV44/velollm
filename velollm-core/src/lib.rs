// VeloLLM Core Library
//
// Core functionality for hardware detection, optimization, and performance monitoring

pub mod hardware;
pub mod optimizer;
pub mod paged_attention;
pub mod scheduler;

#[cfg(test)]
mod hardware_tests;
