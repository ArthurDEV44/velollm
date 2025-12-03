//! Request batching and queuing for VeloLLM proxy.
//!
//! This module provides intelligent request queuing and concurrency control
//! to maximize throughput while working with Ollama's native batching.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Incoming Requests                     │
//! └───────────────────────────┬─────────────────────────────┘
//!                             │
//!                    ┌────────▼────────┐
//!                    │  RequestQueue   │ ← Per-model queues
//!                    │  ┌───────────┐  │
//!                    │  │ llama3.2  │  │
//!                    │  │ mistral   │  │
//!                    │  │ ...       │  │
//!                    │  └───────────┘  │
//!                    └────────┬────────┘
//!                             │
//!                    ┌────────▼────────┐
//!                    │   Semaphore     │ ← Concurrency limit
//!                    │ (max_parallel)  │
//!                    └────────┬────────┘
//!                             │
//!                    ┌────────▼────────┐
//!                    │     Ollama      │ ← Native batching
//!                    └─────────────────┘
//! ```
//!
//! # Features
//!
//! - **Concurrency Control**: Limits concurrent requests to Ollama
//! - **Model Grouping**: Groups requests by model for better cache efficiency
//! - **Fair Queuing**: FIFO with configurable priority
//! - **Timeout Handling**: Configurable queue timeouts
//! - **Metrics**: Track queue depth, wait times, throughput
//!
//! # References
//!
//! - [How Ollama Handles Parallel Requests](https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/)
//! - [Ollama FAQ](https://docs.ollama.com/faq)

mod queue;
mod config;
mod metrics;

pub use config::BatcherConfig;
pub use metrics::{BatcherMetrics, MetricsSnapshot, RequestTimer};
pub use queue::{QueueError, QueuedRequest, RequestQueue};
