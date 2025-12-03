//! Optimization modules for VeloLLM proxy.
//!
//! This module contains optimizers that enhance the reliability and performance
//! of LLM inference, particularly for tool calling with Mistral and Llama models.

pub mod tools;

pub use tools::ToolOptimizer;
