//! Metrics module for VeloLLM proxy
//!
//! Provides Prometheus metrics for monitoring and observability.

pub mod prometheus;

// Re-export commonly used items
pub use prometheus::{
    encode_metrics, register_metrics, set_backend_healthy, set_cache_size, set_queue_size,
    RequestTimer, ACTIVE_REQUESTS, CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL, MAX_CONCURRENT_REQUESTS,
    REQUESTS_REJECTED_TOTAL, REQUESTS_TIMEOUT_TOTAL, TOKENS_PER_SECOND,
};
