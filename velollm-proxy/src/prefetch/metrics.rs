//! Metrics for speculative prefetch.

use std::sync::atomic::{AtomicU64, Ordering};

/// Metrics for monitoring prefetch performance
#[derive(Debug, Default)]
pub struct PrefetchMetrics {
    /// Total predictions made
    pub predictions_total: AtomicU64,

    /// Tasks queued for prefetch
    pub tasks_queued: AtomicU64,

    /// Tasks successfully executed
    pub tasks_executed: AtomicU64,

    /// Tasks dropped (queue full or expired)
    pub tasks_dropped: AtomicU64,

    /// Cache hits from prefetched responses
    pub cache_hits: AtomicU64,

    /// Spare permits successfully acquired
    pub permits_acquired: AtomicU64,

    /// Times no permit was available
    pub permits_unavailable: AtomicU64,
}

impl PrefetchMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record predictions made
    pub fn record_predictions(&self, count: usize) {
        self.predictions_total
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Record a task queued
    pub fn record_queued(&self) {
        self.tasks_queued.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a task executed
    pub fn record_executed(&self) {
        self.tasks_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a task dropped
    pub fn record_dropped(&self) {
        self.tasks_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record successful permit acquisition
    pub fn record_permit_acquired(&self) {
        self.permits_acquired.fetch_add(1, Ordering::Relaxed);
    }

    /// Record failed permit acquisition
    pub fn record_permit_unavailable(&self) {
        self.permits_unavailable.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> PrefetchStats {
        let tasks_queued = self.tasks_queued.load(Ordering::Relaxed);
        let tasks_executed = self.tasks_executed.load(Ordering::Relaxed);
        let permits_acquired = self.permits_acquired.load(Ordering::Relaxed);
        let permits_unavailable = self.permits_unavailable.load(Ordering::Relaxed);

        let execution_rate = if tasks_queued > 0 {
            tasks_executed as f64 / tasks_queued as f64
        } else {
            0.0
        };

        let permit_success_rate = {
            let total = permits_acquired + permits_unavailable;
            if total > 0 {
                permits_acquired as f64 / total as f64
            } else {
                0.0
            }
        };

        PrefetchStats {
            predictions_total: self.predictions_total.load(Ordering::Relaxed),
            tasks_queued,
            tasks_executed,
            tasks_dropped: self.tasks_dropped.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            permits_acquired,
            permits_unavailable,
            execution_rate,
            permit_success_rate,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.predictions_total.store(0, Ordering::Relaxed);
        self.tasks_queued.store(0, Ordering::Relaxed);
        self.tasks_executed.store(0, Ordering::Relaxed);
        self.tasks_dropped.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.permits_acquired.store(0, Ordering::Relaxed);
        self.permits_unavailable.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of prefetch statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefetchStats {
    /// Total predictions made
    pub predictions_total: u64,

    /// Tasks queued for prefetch
    pub tasks_queued: u64,

    /// Tasks successfully executed
    pub tasks_executed: u64,

    /// Tasks dropped
    pub tasks_dropped: u64,

    /// Cache hits from prefetched responses
    pub cache_hits: u64,

    /// Spare permits acquired
    pub permits_acquired: u64,

    /// Times no permit available
    pub permits_unavailable: u64,

    /// Task execution rate (executed / queued)
    pub execution_rate: f64,

    /// Permit acquisition success rate
    pub permit_success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = PrefetchMetrics::new();

        metrics.record_predictions(3);
        metrics.record_queued();
        metrics.record_queued();
        metrics.record_executed();
        metrics.record_dropped();
        metrics.record_cache_hit();
        metrics.record_permit_acquired();
        metrics.record_permit_unavailable();

        let stats = metrics.snapshot();
        assert_eq!(stats.predictions_total, 3);
        assert_eq!(stats.tasks_queued, 2);
        assert_eq!(stats.tasks_executed, 1);
        assert_eq!(stats.tasks_dropped, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.permits_acquired, 1);
        assert_eq!(stats.permits_unavailable, 1);
    }

    #[test]
    fn test_execution_rate() {
        let metrics = PrefetchMetrics::new();

        metrics.record_queued();
        metrics.record_queued();
        metrics.record_queued();
        metrics.record_queued();
        metrics.record_executed();
        metrics.record_executed();

        let stats = metrics.snapshot();
        assert!((stats.execution_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let metrics = PrefetchMetrics::new();
        metrics.record_predictions(5);
        metrics.record_queued();

        metrics.reset();

        let stats = metrics.snapshot();
        assert_eq!(stats.predictions_total, 0);
        assert_eq!(stats.tasks_queued, 0);
    }
}
