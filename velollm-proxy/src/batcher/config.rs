//! Configuration for the request batcher.

use std::time::Duration;

/// Configuration for the request batcher
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of concurrent requests to Ollama
    /// This should match or be less than OLLAMA_NUM_PARALLEL
    pub max_concurrent: usize,

    /// Maximum number of requests waiting in queue per model
    pub max_queue_per_model: usize,

    /// Maximum total requests in all queues
    pub max_queue_total: usize,

    /// Maximum time a request can wait in queue before timeout
    pub queue_timeout: Duration,

    /// Enable model-aware queuing (prioritize same-model batches)
    pub model_aware_queuing: bool,

    /// Minimum batch size before processing (for throughput optimization)
    /// Set to 1 to disable waiting for batches
    pub min_batch_size: usize,

    /// Maximum time to wait for min_batch_size to be reached
    pub batch_wait_timeout: Duration,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            // Match Ollama's default OLLAMA_NUM_PARALLEL
            max_concurrent: 4,
            max_queue_per_model: 100,
            max_queue_total: 500,
            queue_timeout: Duration::from_secs(300), // 5 minutes
            model_aware_queuing: true,
            min_batch_size: 1, // Process immediately by default
            batch_wait_timeout: Duration::from_millis(50),
        }
    }
}

impl BatcherConfig {
    /// Create config optimized for low latency (single user)
    pub fn low_latency() -> Self {
        Self {
            max_concurrent: 1,
            max_queue_per_model: 10,
            max_queue_total: 20,
            queue_timeout: Duration::from_secs(60),
            model_aware_queuing: false,
            min_batch_size: 1,
            batch_wait_timeout: Duration::from_millis(0),
        }
    }

    /// Create config optimized for high throughput (many concurrent users)
    pub fn high_throughput() -> Self {
        Self {
            max_concurrent: 8,
            max_queue_per_model: 200,
            max_queue_total: 1000,
            queue_timeout: Duration::from_secs(600), // 10 minutes
            model_aware_queuing: true,
            min_batch_size: 4,
            batch_wait_timeout: Duration::from_millis(100),
        }
    }

    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("VELOLLM_MAX_CONCURRENT") {
            if let Ok(n) = val.parse() {
                config.max_concurrent = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_MAX_QUEUE") {
            if let Ok(n) = val.parse() {
                config.max_queue_total = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_QUEUE_TIMEOUT_SECS") {
            if let Ok(n) = val.parse() {
                config.queue_timeout = Duration::from_secs(n);
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_MIN_BATCH_SIZE") {
            if let Ok(n) = val.parse() {
                config.min_batch_size = n;
            }
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BatcherConfig::default();
        assert_eq!(config.max_concurrent, 4);
        assert_eq!(config.min_batch_size, 1);
    }

    #[test]
    fn test_low_latency_config() {
        let config = BatcherConfig::low_latency();
        assert_eq!(config.max_concurrent, 1);
        assert!(!config.model_aware_queuing);
    }

    #[test]
    fn test_high_throughput_config() {
        let config = BatcherConfig::high_throughput();
        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.min_batch_size, 4);
    }
}
