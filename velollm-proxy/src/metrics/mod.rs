//! Metrics module for VeloLLM proxy
//!
//! Provides Prometheus metrics for monitoring and observability.

pub mod prometheus;

// Re-export commonly used items
pub use prometheus::{
    encode_metrics, record_compression, record_compression_skipped, record_messages_summarized,
    record_prefetch_cache_hit, record_prefetch_dropped, record_prefetch_executed,
    record_prefetch_permit_acquired, record_prefetch_permit_unavailable,
    record_prefetch_predictions, record_prefetch_queued, record_system_prompt_cache_hit,
    register_metrics, set_backend_healthy, set_cache_size, set_queue_size, RequestTimer,
    ACTIVE_REQUESTS, CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL, MAX_CONCURRENT_REQUESTS,
    REQUESTS_REJECTED_TOTAL, REQUESTS_TIMEOUT_TOTAL, TOKENS_PER_SECOND,
};
