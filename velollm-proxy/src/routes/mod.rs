//! HTTP route handlers for VeloLLM proxy.
//!
//! This module organizes all route handlers:
//! - `health`: Health check and metrics endpoints
//! - `ollama`: Ollama-native API routes
//! - `openai`: OpenAI-compatible API routes

pub mod health;
pub mod ollama;
pub mod openai;

// Re-export handlers for convenience
pub use health::{health, live, metrics, ready};
pub use ollama::{chat, generate, tags};
pub use openai::{chat_completions, get_model, list_models};
