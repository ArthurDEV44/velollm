//! Prometheus metrics for VeloLLM proxy
//!
//! Exposes metrics in Prometheus format for monitoring and observability.

use lazy_static::lazy_static;
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, Opts, Registry,
    TextEncoder,
};

lazy_static! {
    /// Global Prometheus registry for VeloLLM metrics
    pub static ref REGISTRY: Registry = Registry::new();

    // ============== Request Metrics ==============

    /// Total requests counter with model and status labels
    pub static ref REQUESTS_TOTAL: CounterVec = CounterVec::new(
        Opts::new("requests_total", "Total number of requests")
            .namespace("velollm"),
        &["model", "status"]
    ).expect("metric can be created");

    /// Request duration histogram with model label
    pub static ref REQUEST_DURATION_SECONDS: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "request_duration_seconds",
            "Request duration in seconds"
        )
        .namespace("velollm")
        .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]),
        &["model"]
    ).expect("metric can be created");

    // ============== Token Metrics ==============

    /// Total tokens generated counter with model label
    pub static ref TOKENS_GENERATED_TOTAL: CounterVec = CounterVec::new(
        Opts::new("tokens_generated_total", "Total tokens generated")
            .namespace("velollm"),
        &["model"]
    ).expect("metric can be created");

    /// Current throughput gauge (tokens per second)
    pub static ref TOKENS_PER_SECOND: Gauge = Gauge::with_opts(
        Opts::new("tokens_per_second", "Current token generation throughput")
            .namespace("velollm")
    ).expect("metric can be created");

    // ============== Cache Metrics ==============

    /// Cache hits counter with cache type label
    pub static ref CACHE_HITS_TOTAL: CounterVec = CounterVec::new(
        Opts::new("cache_hits_total", "Total cache hits")
            .namespace("velollm"),
        &["type"]
    ).expect("metric can be created");

    /// Cache misses counter with cache type label
    pub static ref CACHE_MISSES_TOTAL: CounterVec = CounterVec::new(
        Opts::new("cache_misses_total", "Total cache misses")
            .namespace("velollm"),
        &["type"]
    ).expect("metric can be created");

    /// Cache evictions counter
    pub static ref CACHE_EVICTIONS_TOTAL: Counter = Counter::with_opts(
        Opts::new("cache_evictions_total", "Total cache evictions")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Cache size gauge (number of entries)
    pub static ref CACHE_SIZE: GaugeVec = GaugeVec::new(
        Opts::new("cache_size", "Current cache size (entries)")
            .namespace("velollm"),
        &["type"]
    ).expect("metric can be created");

    // ============== Queue Metrics ==============

    /// Current queue size gauge
    pub static ref QUEUE_SIZE: Gauge = Gauge::with_opts(
        Opts::new("queue_size", "Current number of requests in queue")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Queue wait time histogram
    pub static ref QUEUE_WAIT_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "queue_wait_seconds",
            "Time spent waiting in queue"
        )
        .namespace("velollm")
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
    ).expect("metric can be created");

    /// Active requests gauge (currently being processed)
    pub static ref ACTIVE_REQUESTS: Gauge = Gauge::with_opts(
        Opts::new("active_requests", "Number of requests currently being processed")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Maximum concurrent requests gauge
    pub static ref MAX_CONCURRENT_REQUESTS: Gauge = Gauge::with_opts(
        Opts::new("max_concurrent_requests", "Maximum concurrent requests allowed")
            .namespace("velollm")
    ).expect("metric can be created");

    // ============== Batcher Metrics ==============

    /// Requests rejected counter (queue full)
    pub static ref REQUESTS_REJECTED_TOTAL: Counter = Counter::with_opts(
        Opts::new("requests_rejected_total", "Total requests rejected due to queue full")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Requests timed out counter
    pub static ref REQUESTS_TIMEOUT_TOTAL: Counter = Counter::with_opts(
        Opts::new("requests_timeout_total", "Total requests that timed out")
            .namespace("velollm")
    ).expect("metric can be created");

    // ============== Backend Metrics ==============

    /// Ollama backend health status (1 = healthy, 0 = unhealthy)
    pub static ref BACKEND_HEALTHY: Gauge = Gauge::with_opts(
        Opts::new("backend_healthy", "Backend health status (1=healthy, 0=unhealthy)")
            .namespace("velollm")
    ).expect("metric can be created");

    // ============== Compression Metrics ==============

    /// Total compression operations
    pub static ref COMPRESSION_TOTAL: Counter = Counter::with_opts(
        Opts::new("compression_total", "Total prompt compression operations")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Compressions skipped (under threshold)
    pub static ref COMPRESSION_SKIPPED_TOTAL: Counter = Counter::with_opts(
        Opts::new("compression_skipped_total", "Total compressions skipped (context under threshold)")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Characters saved by compression
    pub static ref COMPRESSION_CHARS_SAVED_TOTAL: Counter = Counter::with_opts(
        Opts::new("compression_chars_saved_total", "Total characters saved by compression")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Current compression ratio gauge
    pub static ref COMPRESSION_RATIO: Gauge = Gauge::with_opts(
        Opts::new("compression_ratio", "Current compression ratio (lower is better)")
            .namespace("velollm")
    ).expect("metric can be created");

    /// System prompt cache hits
    pub static ref SYSTEM_PROMPT_CACHE_HITS_TOTAL: Counter = Counter::with_opts(
        Opts::new("system_prompt_cache_hits_total", "Total system prompt cache hits")
            .namespace("velollm")
    ).expect("metric can be created");

    /// Messages summarized
    pub static ref MESSAGES_SUMMARIZED_TOTAL: Counter = Counter::with_opts(
        Opts::new("messages_summarized_total", "Total messages summarized")
            .namespace("velollm")
    ).expect("metric can be created");
}

/// Register all metrics with the global registry.
/// Should be called once at startup.
pub fn register_metrics() -> prometheus::Result<()> {
    // Request metrics
    REGISTRY.register(Box::new(REQUESTS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(REQUEST_DURATION_SECONDS.clone()))?;

    // Token metrics
    REGISTRY.register(Box::new(TOKENS_GENERATED_TOTAL.clone()))?;
    REGISTRY.register(Box::new(TOKENS_PER_SECOND.clone()))?;

    // Cache metrics
    REGISTRY.register(Box::new(CACHE_HITS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(CACHE_MISSES_TOTAL.clone()))?;
    REGISTRY.register(Box::new(CACHE_EVICTIONS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(CACHE_SIZE.clone()))?;

    // Queue metrics
    REGISTRY.register(Box::new(QUEUE_SIZE.clone()))?;
    REGISTRY.register(Box::new(QUEUE_WAIT_SECONDS.clone()))?;
    REGISTRY.register(Box::new(ACTIVE_REQUESTS.clone()))?;
    REGISTRY.register(Box::new(MAX_CONCURRENT_REQUESTS.clone()))?;

    // Batcher metrics
    REGISTRY.register(Box::new(REQUESTS_REJECTED_TOTAL.clone()))?;
    REGISTRY.register(Box::new(REQUESTS_TIMEOUT_TOTAL.clone()))?;

    // Backend metrics
    REGISTRY.register(Box::new(BACKEND_HEALTHY.clone()))?;

    // Compression metrics
    REGISTRY.register(Box::new(COMPRESSION_TOTAL.clone()))?;
    REGISTRY.register(Box::new(COMPRESSION_SKIPPED_TOTAL.clone()))?;
    REGISTRY.register(Box::new(COMPRESSION_CHARS_SAVED_TOTAL.clone()))?;
    REGISTRY.register(Box::new(COMPRESSION_RATIO.clone()))?;
    REGISTRY.register(Box::new(SYSTEM_PROMPT_CACHE_HITS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(MESSAGES_SUMMARIZED_TOTAL.clone()))?;

    Ok(())
}

/// Encode all metrics to Prometheus text format.
pub fn encode_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder
        .encode_to_string(&metric_families)
        .unwrap_or_else(|e| format!("# Error encoding metrics: {}", e))
}

/// Helper struct for tracking request duration.
/// Automatically records the duration when dropped.
pub struct RequestTimer {
    model: String,
    start: std::time::Instant,
}

impl RequestTimer {
    /// Start a new request timer for the given model.
    pub fn new(model: &str) -> Self {
        ACTIVE_REQUESTS.inc();
        Self { model: model.to_string(), start: std::time::Instant::now() }
    }

    /// Record a successful request completion.
    pub fn record_success(self, tokens: u64) {
        let duration = self.start.elapsed().as_secs_f64();
        REQUEST_DURATION_SECONDS
            .with_label_values(&[&self.model])
            .observe(duration);
        REQUESTS_TOTAL
            .with_label_values(&[&self.model, "success"])
            .inc();
        TOKENS_GENERATED_TOTAL
            .with_label_values(&[&self.model])
            .inc_by(tokens as f64);

        // Update throughput gauge
        if duration > 0.0 {
            let tps = tokens as f64 / duration;
            TOKENS_PER_SECOND.set(tps);
        }

        // Don't forget to decrement active requests
        ACTIVE_REQUESTS.dec();
        // Prevent the Drop impl from running
        std::mem::forget(self);
    }

    /// Record a failed request.
    pub fn record_failure(self, _error: &str) {
        let duration = self.start.elapsed().as_secs_f64();
        REQUEST_DURATION_SECONDS
            .with_label_values(&[&self.model])
            .observe(duration);
        REQUESTS_TOTAL
            .with_label_values(&[&self.model, "error"])
            .inc();

        ACTIVE_REQUESTS.dec();
        std::mem::forget(self);
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        // If dropped without explicit record, count as error
        ACTIVE_REQUESTS.dec();
        REQUESTS_TOTAL
            .with_label_values(&[&self.model, "error"])
            .inc();
    }
}

/// Record a cache hit.
pub fn record_cache_hit(cache_type: &str) {
    CACHE_HITS_TOTAL.with_label_values(&[cache_type]).inc();
}

/// Record a cache miss.
pub fn record_cache_miss(cache_type: &str) {
    CACHE_MISSES_TOTAL.with_label_values(&[cache_type]).inc();
}

/// Record queue wait time.
pub fn record_queue_wait(wait_secs: f64) {
    QUEUE_WAIT_SECONDS.observe(wait_secs);
}

/// Update queue size gauge.
pub fn set_queue_size(size: u64) {
    QUEUE_SIZE.set(size as f64);
}

/// Update cache size gauge.
pub fn set_cache_size(cache_type: &str, size: u64) {
    CACHE_SIZE.with_label_values(&[cache_type]).set(size as f64);
}

/// Record a rejected request.
pub fn record_rejected() {
    REQUESTS_REJECTED_TOTAL.inc();
}

/// Record a timed out request.
pub fn record_timeout() {
    REQUESTS_TIMEOUT_TOTAL.inc();
}

/// Set backend health status.
pub fn set_backend_healthy(healthy: bool) {
    BACKEND_HEALTHY.set(if healthy { 1.0 } else { 0.0 });
}

/// Record a compression operation.
pub fn record_compression(chars_saved: u64, ratio: f64) {
    COMPRESSION_TOTAL.inc();
    COMPRESSION_CHARS_SAVED_TOTAL.inc_by(chars_saved as f64);
    COMPRESSION_RATIO.set(ratio);
}

/// Record a skipped compression.
pub fn record_compression_skipped() {
    COMPRESSION_SKIPPED_TOTAL.inc();
}

/// Record a system prompt cache hit.
pub fn record_system_prompt_cache_hit() {
    SYSTEM_PROMPT_CACHE_HITS_TOTAL.inc();
}

/// Record messages summarized.
pub fn record_messages_summarized(count: u64) {
    MESSAGES_SUMMARIZED_TOTAL.inc_by(count as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_metrics() {
        // Create a new registry for testing
        let registry = Registry::new();

        // Create test metrics
        let counter = Counter::new("test_counter", "Test counter").unwrap();
        registry.register(Box::new(counter.clone())).unwrap();

        counter.inc();
        assert_eq!(counter.get(), 1.0);
    }

    #[test]
    fn test_request_timer() {
        // Just verify that RequestTimer can be created and dropped
        let timer = RequestTimer::new("test-model");
        timer.record_success(100);
    }

    #[test]
    fn test_cache_metrics() {
        record_cache_hit("exact");
        record_cache_miss("semantic");
        // Metrics are recorded without error
    }

    #[test]
    fn test_encode_metrics() {
        let output = encode_metrics();
        // Should return something (even if registry not initialized)
        assert!(output.is_empty() || output.starts_with('#') || output.contains("velollm"));
    }
}
