//! Metrics for prompt compression.

use std::sync::atomic::{AtomicU64, Ordering};

/// Metrics for monitoring compression performance
#[derive(Debug, Default)]
pub struct CompressionMetrics {
    /// Total compression operations performed
    pub compressions_total: AtomicU64,

    /// Compressions skipped (context under threshold)
    pub compressions_skipped: AtomicU64,

    /// Total characters before compression
    pub chars_before_total: AtomicU64,

    /// Total characters after compression
    pub chars_after_total: AtomicU64,

    /// Deduplication operations performed
    pub dedup_operations: AtomicU64,

    /// Patterns deduplicated
    pub patterns_deduplicated: AtomicU64,

    /// System prompt cache hits
    pub system_cache_hits: AtomicU64,

    /// System prompt cache misses
    pub system_cache_misses: AtomicU64,

    /// Messages summarized
    pub messages_summarized: AtomicU64,

    /// Summarization operations performed
    pub summarization_operations: AtomicU64,
}

impl CompressionMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a compression operation
    pub fn record_compression(&self, chars_before: usize, chars_after: usize) {
        self.compressions_total.fetch_add(1, Ordering::Relaxed);
        self.chars_before_total
            .fetch_add(chars_before as u64, Ordering::Relaxed);
        self.chars_after_total
            .fetch_add(chars_after as u64, Ordering::Relaxed);
    }

    /// Record a skipped compression
    pub fn record_skip(&self) {
        self.compressions_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record deduplication
    pub fn record_dedup(&self, patterns_found: usize) {
        self.dedup_operations.fetch_add(1, Ordering::Relaxed);
        self.patterns_deduplicated
            .fetch_add(patterns_found as u64, Ordering::Relaxed);
    }

    /// Record a system prompt cache hit
    pub fn record_system_cache_hit(&self) {
        self.system_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a system prompt cache miss
    pub fn record_system_cache_miss(&self) {
        self.system_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record summarization
    pub fn record_summarization(&self, messages_count: usize) {
        self.summarization_operations
            .fetch_add(1, Ordering::Relaxed);
        self.messages_summarized
            .fetch_add(messages_count as u64, Ordering::Relaxed);
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> CompressionStats {
        let compressions_total = self.compressions_total.load(Ordering::Relaxed);
        let chars_before = self.chars_before_total.load(Ordering::Relaxed);
        let chars_after = self.chars_after_total.load(Ordering::Relaxed);
        let system_hits = self.system_cache_hits.load(Ordering::Relaxed);
        let system_misses = self.system_cache_misses.load(Ordering::Relaxed);

        let compression_ratio = if chars_before > 0 {
            chars_after as f64 / chars_before as f64
        } else {
            1.0
        };

        let system_cache_total = system_hits + system_misses;
        let system_cache_hit_rate = if system_cache_total > 0 {
            system_hits as f64 / system_cache_total as f64
        } else {
            0.0
        };

        CompressionStats {
            compressions_total,
            compressions_skipped: self.compressions_skipped.load(Ordering::Relaxed),
            chars_before_total: chars_before,
            chars_after_total: chars_after,
            chars_saved: chars_before.saturating_sub(chars_after),
            compression_ratio,
            dedup_operations: self.dedup_operations.load(Ordering::Relaxed),
            patterns_deduplicated: self.patterns_deduplicated.load(Ordering::Relaxed),
            system_cache_hits: system_hits,
            system_cache_misses: system_misses,
            system_cache_hit_rate,
            messages_summarized: self.messages_summarized.load(Ordering::Relaxed),
            summarization_operations: self.summarization_operations.load(Ordering::Relaxed),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.compressions_total.store(0, Ordering::Relaxed);
        self.compressions_skipped.store(0, Ordering::Relaxed);
        self.chars_before_total.store(0, Ordering::Relaxed);
        self.chars_after_total.store(0, Ordering::Relaxed);
        self.dedup_operations.store(0, Ordering::Relaxed);
        self.patterns_deduplicated.store(0, Ordering::Relaxed);
        self.system_cache_hits.store(0, Ordering::Relaxed);
        self.system_cache_misses.store(0, Ordering::Relaxed);
        self.messages_summarized.store(0, Ordering::Relaxed);
        self.summarization_operations.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of compression statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompressionStats {
    /// Total compression operations
    pub compressions_total: u64,

    /// Compressions skipped
    pub compressions_skipped: u64,

    /// Total characters before compression
    pub chars_before_total: u64,

    /// Total characters after compression
    pub chars_after_total: u64,

    /// Total characters saved
    pub chars_saved: u64,

    /// Overall compression ratio (lower is better)
    pub compression_ratio: f64,

    /// Deduplication operations performed
    pub dedup_operations: u64,

    /// Patterns deduplicated
    pub patterns_deduplicated: u64,

    /// System prompt cache hits
    pub system_cache_hits: u64,

    /// System prompt cache misses
    pub system_cache_misses: u64,

    /// System prompt cache hit rate
    pub system_cache_hit_rate: f64,

    /// Messages summarized
    pub messages_summarized: u64,

    /// Summarization operations performed
    pub summarization_operations: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = CompressionMetrics::new();

        metrics.record_compression(1000, 600);
        metrics.record_compression(500, 300);
        metrics.record_dedup(3);
        metrics.record_system_cache_hit();
        metrics.record_system_cache_miss();

        let stats = metrics.snapshot();
        assert_eq!(stats.compressions_total, 2);
        assert_eq!(stats.chars_before_total, 1500);
        assert_eq!(stats.chars_after_total, 900);
        assert_eq!(stats.chars_saved, 600);
        assert_eq!(stats.patterns_deduplicated, 3);
    }

    #[test]
    fn test_compression_ratio() {
        let metrics = CompressionMetrics::new();

        metrics.record_compression(1000, 500);

        let stats = metrics.snapshot();
        assert!((stats.compression_ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cache_hit_rate() {
        let metrics = CompressionMetrics::new();

        metrics.record_system_cache_hit();
        metrics.record_system_cache_hit();
        metrics.record_system_cache_hit();
        metrics.record_system_cache_miss();

        let stats = metrics.snapshot();
        assert!((stats.system_cache_hit_rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let metrics = CompressionMetrics::new();
        metrics.record_compression(1000, 500);
        metrics.record_dedup(5);

        metrics.reset();

        let stats = metrics.snapshot();
        assert_eq!(stats.compressions_total, 0);
        assert_eq!(stats.patterns_deduplicated, 0);
    }
}
