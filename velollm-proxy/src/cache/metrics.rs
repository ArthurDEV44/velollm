//! Metrics for the response cache.

use std::sync::atomic::{AtomicU64, Ordering};

/// Metrics for monitoring cache performance
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Exact cache hits
    pub exact_hits: AtomicU64,
    /// Exact cache misses
    pub exact_misses: AtomicU64,
    /// Semantic cache hits
    pub semantic_hits: AtomicU64,
    /// Semantic cache misses
    pub semantic_misses: AtomicU64,
    /// Total cache puts
    pub puts: AtomicU64,
    /// Evictions from exact cache
    pub exact_evictions: AtomicU64,
    /// Expired entries removed
    pub expirations: AtomicU64,
}

impl CacheMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an exact cache hit
    pub fn record_exact_hit(&self) {
        self.exact_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an exact cache miss
    pub fn record_exact_miss(&self) {
        self.exact_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a semantic cache hit
    pub fn record_semantic_hit(&self) {
        self.semantic_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a semantic cache miss
    pub fn record_semantic_miss(&self) {
        self.semantic_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache put
    pub fn record_put(&self) {
        self.puts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an eviction
    pub fn record_eviction(&self) {
        self.exact_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an expiration
    pub fn record_expiration(&self) {
        self.expirations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> CacheStats {
        let exact_hits = self.exact_hits.load(Ordering::Relaxed);
        let exact_misses = self.exact_misses.load(Ordering::Relaxed);
        let semantic_hits = self.semantic_hits.load(Ordering::Relaxed);
        let semantic_misses = self.semantic_misses.load(Ordering::Relaxed);

        let total_requests = exact_hits + exact_misses;
        let total_hits = exact_hits + semantic_hits;

        CacheStats {
            exact_hits,
            exact_misses,
            semantic_hits,
            semantic_misses,
            puts: self.puts.load(Ordering::Relaxed),
            evictions: self.exact_evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
            hit_rate: if total_requests > 0 {
                total_hits as f64 / total_requests as f64
            } else {
                0.0
            },
            exact_hit_rate: if total_requests > 0 {
                exact_hits as f64 / total_requests as f64
            } else {
                0.0
            },
            semantic_hit_rate: if exact_misses > 0 {
                semantic_hits as f64 / exact_misses as f64
            } else {
                0.0
            },
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.exact_hits.store(0, Ordering::Relaxed);
        self.exact_misses.store(0, Ordering::Relaxed);
        self.semantic_hits.store(0, Ordering::Relaxed);
        self.semantic_misses.store(0, Ordering::Relaxed);
        self.puts.store(0, Ordering::Relaxed);
        self.exact_evictions.store(0, Ordering::Relaxed);
        self.expirations.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheStats {
    /// Exact cache hits
    pub exact_hits: u64,
    /// Exact cache misses
    pub exact_misses: u64,
    /// Semantic cache hits
    pub semantic_hits: u64,
    /// Semantic cache misses
    pub semantic_misses: u64,
    /// Total cache puts
    pub puts: u64,
    /// Evictions from cache
    pub evictions: u64,
    /// Expired entries removed
    pub expirations: u64,
    /// Overall cache hit rate (exact + semantic)
    pub hit_rate: f64,
    /// Exact cache hit rate
    pub exact_hit_rate: f64,
    /// Semantic cache hit rate (among exact misses)
    pub semantic_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = CacheMetrics::new();

        metrics.record_exact_hit();
        metrics.record_exact_hit();
        metrics.record_exact_miss();
        metrics.record_semantic_hit();

        let stats = metrics.snapshot();
        assert_eq!(stats.exact_hits, 2);
        assert_eq!(stats.exact_misses, 1);
        assert_eq!(stats.semantic_hits, 1);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let metrics = CacheMetrics::new();

        // 3 exact hits, 2 exact misses, 1 semantic hit
        for _ in 0..3 {
            metrics.record_exact_hit();
        }
        for _ in 0..2 {
            metrics.record_exact_miss();
        }
        metrics.record_semantic_hit();

        let stats = metrics.snapshot();

        // Total requests = 3 + 2 = 5
        // Total hits = 3 + 1 = 4
        // Overall hit rate = 4/5 = 0.8
        assert!((stats.hit_rate - 0.8).abs() < 0.001);

        // Exact hit rate = 3/5 = 0.6
        assert!((stats.exact_hit_rate - 0.6).abs() < 0.001);

        // Semantic hit rate (among exact misses) = 1/2 = 0.5
        assert!((stats.semantic_hit_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let metrics = CacheMetrics::new();
        metrics.record_exact_hit();
        metrics.record_put();

        metrics.reset();

        let stats = metrics.snapshot();
        assert_eq!(stats.exact_hits, 0);
        assert_eq!(stats.puts, 0);
    }
}
