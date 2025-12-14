//! Configuration for speculative prefetch.

use std::time::Duration;

/// Configuration for speculative prefetch system
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Enable prefetch (default: false)
    pub enabled: bool,

    /// Maximum predictions per request (default: 2)
    pub max_predictions: usize,

    /// TTL for prefetched responses in seconds (default: 300)
    pub cache_ttl_secs: u64,

    /// Minimum confidence threshold to trigger prefetch (default: 0.7)
    pub min_confidence: f32,

    /// Maximum pending prefetch tasks in queue (default: 10)
    pub max_queue_size: usize,

    /// Interval between worker poll cycles in ms (default: 100)
    pub worker_poll_interval_ms: u64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_predictions: 2,
            cache_ttl_secs: 300,
            min_confidence: 0.7,
            max_queue_size: 10,
            worker_poll_interval_ms: 100,
        }
    }
}

impl PrefetchConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("VELOLLM_PREFETCH_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            max_predictions: std::env::var("VELOLLM_PREFETCH_MAX_PREDICTIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2),
            cache_ttl_secs: std::env::var("VELOLLM_PREFETCH_CACHE_TTL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
            min_confidence: std::env::var("VELOLLM_PREFETCH_MIN_CONFIDENCE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            max_queue_size: std::env::var("VELOLLM_PREFETCH_MAX_QUEUE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            worker_poll_interval_ms: std::env::var("VELOLLM_PREFETCH_POLL_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
        }
    }

    /// Get cache TTL as Duration
    pub fn cache_ttl(&self) -> Duration {
        Duration::from_secs(self.cache_ttl_secs)
    }

    /// Get worker poll interval as Duration
    pub fn worker_poll_interval(&self) -> Duration {
        Duration::from_millis(self.worker_poll_interval_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PrefetchConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_predictions, 2);
        assert_eq!(config.cache_ttl_secs, 300);
        assert!((config.min_confidence - 0.7).abs() < 0.001);
        assert_eq!(config.max_queue_size, 10);
    }

    #[test]
    fn test_cache_ttl_duration() {
        let config = PrefetchConfig::default();
        assert_eq!(config.cache_ttl(), Duration::from_secs(300));
    }
}
