//! Application state for VeloLLM proxy.

use std::sync::Arc;

use crate::batcher::{BatcherConfig, BatcherMetrics, RequestQueue};
use crate::cache::{CacheConfig, ResponseCache};
use crate::compressor::{CompressionConfig, PromptCompressor};
use crate::optimizer::ToolOptimizer;
use crate::prefetch::{PrefetchConfig, PrefetchService};
use crate::proxy::OllamaProxy;
use crate::types::openai::ChatCompletionRequest;
use tokio::sync::Mutex;

/// Application state shared across all handlers
pub struct AppState {
    /// Ollama proxy client
    pub proxy: OllamaProxy,

    /// Tool optimizer for enhanced tool calling
    pub tool_optimizer: Mutex<ToolOptimizer>,

    /// Request queue for batching and concurrency control
    pub request_queue: RequestQueue<ChatCompletionRequest>,

    /// Batcher metrics for monitoring
    pub batcher_metrics: Arc<BatcherMetrics>,

    /// Batcher configuration
    pub batcher_config: BatcherConfig,

    /// Response cache (exact + semantic)
    pub response_cache: ResponseCache,

    /// Cache configuration
    pub cache_config: CacheConfig,

    /// Prompt compressor
    pub prompt_compressor: PromptCompressor,

    /// Compression configuration
    pub compressor_config: CompressionConfig,

    /// Speculative prefetch service
    pub prefetch_service: PrefetchService,

    /// Prefetch configuration
    pub prefetch_config: PrefetchConfig,

    /// Runtime statistics
    pub stats: Mutex<ProxyStats>,

    /// Configuration
    pub config: ProxyConfig,
}

impl AppState {
    /// Create new application state with default configurations
    pub fn new(config: ProxyConfig) -> Self {
        Self::with_configs(
            config,
            BatcherConfig::from_env(),
            CacheConfig::from_env(),
            CompressionConfig::from_env(),
            PrefetchConfig::from_env(),
        )
    }

    /// Create new application state with custom configurations
    pub fn with_configs(
        config: ProxyConfig,
        batcher_config: BatcherConfig,
        cache_config: CacheConfig,
        compressor_config: CompressionConfig,
        prefetch_config: PrefetchConfig,
    ) -> Self {
        let batcher_metrics = Arc::new(BatcherMetrics::new());
        let request_queue = RequestQueue::new(batcher_config.clone(), batcher_metrics.clone());
        let response_cache = ResponseCache::new(cache_config.clone());
        let prompt_compressor = PromptCompressor::new(compressor_config.clone());
        let prefetch_service = PrefetchService::new(prefetch_config.clone());

        Self {
            proxy: OllamaProxy::new(&config.ollama_url),
            tool_optimizer: Mutex::new(ToolOptimizer::new()),
            request_queue,
            batcher_metrics,
            batcher_config,
            response_cache,
            cache_config,
            prompt_compressor,
            compressor_config,
            prefetch_service,
            prefetch_config,
            stats: Mutex::new(ProxyStats::default()),
            config,
        }
    }
}

/// Proxy configuration
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Port to listen on
    pub port: u16,

    /// Ollama backend URL
    pub ollama_url: String,

    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self { port: 8000, ollama_url: "http://localhost:11434".to_string(), verbose: false }
    }
}

impl ProxyConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("VELOLLM_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8000),
            ollama_url: std::env::var("OLLAMA_HOST")
                .or_else(|_| std::env::var("OLLAMA_URL"))
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            verbose: std::env::var("VELOLLM_VERBOSE")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
        }
    }
}

/// Runtime statistics
#[derive(Debug, Default)]
pub struct ProxyStats {
    /// Total requests received
    pub requests_total: u64,

    /// Successful requests
    pub requests_success: u64,

    /// Failed requests
    pub requests_failed: u64,

    /// Total tokens generated
    pub tokens_generated: u64,

    /// Total generation time in milliseconds
    pub generation_time_ms: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,
}

impl ProxyStats {
    /// Calculate average tokens per second
    pub fn avg_tokens_per_second(&self) -> f64 {
        if self.generation_time_ms == 0 {
            0.0
        } else {
            (self.tokens_generated as f64 / self.generation_time_ms as f64) * 1000.0
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.requests_total == 0 {
            1.0
        } else {
            self.requests_success as f64 / self.requests_total as f64
        }
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProxyConfig::default();
        assert_eq!(config.port, 8000);
        assert_eq!(config.ollama_url, "http://localhost:11434");
    }

    #[test]
    fn test_stats_calculations() {
        let stats = ProxyStats {
            requests_total: 100,
            requests_success: 95,
            requests_failed: 5,
            tokens_generated: 10000,
            generation_time_ms: 5000,
            cache_hits: 30,
            cache_misses: 70,
        };

        assert!((stats.avg_tokens_per_second() - 2000.0).abs() < 0.001);
        assert!((stats.success_rate() - 0.95).abs() < 0.001);
        assert!((stats.cache_hit_rate() - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_stats_edge_cases() {
        let stats = ProxyStats::default();

        assert_eq!(stats.avg_tokens_per_second(), 0.0);
        assert_eq!(stats.success_rate(), 1.0); // No requests = 100% success
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }
}
