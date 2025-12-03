//! Metrics for the request batcher.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Metrics for monitoring batcher performance
#[derive(Debug, Default)]
pub struct BatcherMetrics {
    /// Total requests received
    pub requests_received: AtomicU64,

    /// Requests currently in queue
    pub requests_queued: AtomicU64,

    /// Requests currently being processed
    pub requests_processing: AtomicU64,

    /// Total requests completed successfully
    pub requests_completed: AtomicU64,

    /// Requests that timed out in queue
    pub requests_timed_out: AtomicU64,

    /// Requests rejected due to full queue
    pub requests_rejected: AtomicU64,

    /// Total queue wait time in milliseconds (for averaging)
    pub total_queue_wait_ms: AtomicU64,

    /// Total processing time in milliseconds (for averaging)
    pub total_processing_ms: AtomicU64,

    /// Maximum queue depth observed
    pub max_queue_depth: AtomicU64,

    /// Number of batches processed
    pub batches_processed: AtomicU64,

    /// Total requests in batches (for batch size averaging)
    pub total_batch_requests: AtomicU64,
}

impl BatcherMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new request received
    pub fn record_received(&self) {
        self.requests_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a request entering the queue
    pub fn record_queued(&self) {
        let queued = self.requests_queued.fetch_add(1, Ordering::Relaxed) + 1;
        // Update max queue depth if needed
        let mut current_max = self.max_queue_depth.load(Ordering::Relaxed);
        while queued > current_max {
            match self.max_queue_depth.compare_exchange_weak(
                current_max,
                queued,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    /// Record a request leaving the queue (starting processing)
    pub fn record_dequeued(&self, wait_time: Duration) {
        self.requests_queued.fetch_sub(1, Ordering::Relaxed);
        self.requests_processing.fetch_add(1, Ordering::Relaxed);
        self.total_queue_wait_ms
            .fetch_add(wait_time.as_millis() as u64, Ordering::Relaxed);
    }

    /// Record a request completed successfully
    pub fn record_completed(&self, processing_time: Duration) {
        self.requests_processing.fetch_sub(1, Ordering::Relaxed);
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_ms
            .fetch_add(processing_time.as_millis() as u64, Ordering::Relaxed);
    }

    /// Record a request that timed out
    pub fn record_timeout(&self) {
        self.requests_queued.fetch_sub(1, Ordering::Relaxed);
        self.requests_timed_out.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a request rejected due to full queue
    pub fn record_rejected(&self) {
        self.requests_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a batch being processed
    pub fn record_batch(&self, batch_size: usize) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.total_batch_requests
            .fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    /// Get current queue depth
    pub fn queue_depth(&self) -> u64 {
        self.requests_queued.load(Ordering::Relaxed)
    }

    /// Get current processing count
    pub fn processing_count(&self) -> u64 {
        self.requests_processing.load(Ordering::Relaxed)
    }

    /// Calculate average queue wait time in milliseconds
    pub fn avg_queue_wait_ms(&self) -> f64 {
        let completed = self.requests_completed.load(Ordering::Relaxed);
        if completed == 0 {
            return 0.0;
        }
        let total_wait = self.total_queue_wait_ms.load(Ordering::Relaxed);
        total_wait as f64 / completed as f64
    }

    /// Calculate average processing time in milliseconds
    pub fn avg_processing_ms(&self) -> f64 {
        let completed = self.requests_completed.load(Ordering::Relaxed);
        if completed == 0 {
            return 0.0;
        }
        let total_processing = self.total_processing_ms.load(Ordering::Relaxed);
        total_processing as f64 / completed as f64
    }

    /// Calculate average batch size
    pub fn avg_batch_size(&self) -> f64 {
        let batches = self.batches_processed.load(Ordering::Relaxed);
        if batches == 0 {
            return 0.0;
        }
        let total_requests = self.total_batch_requests.load(Ordering::Relaxed);
        total_requests as f64 / batches as f64
    }

    /// Calculate throughput (requests per second) over a time window
    pub fn throughput(&self, window: Duration) -> f64 {
        let completed = self.requests_completed.load(Ordering::Relaxed);
        if window.as_secs_f64() == 0.0 {
            return 0.0;
        }
        completed as f64 / window.as_secs_f64()
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_received: self.requests_received.load(Ordering::Relaxed),
            requests_queued: self.requests_queued.load(Ordering::Relaxed),
            requests_processing: self.requests_processing.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_timed_out: self.requests_timed_out.load(Ordering::Relaxed),
            requests_rejected: self.requests_rejected.load(Ordering::Relaxed),
            max_queue_depth: self.max_queue_depth.load(Ordering::Relaxed),
            avg_queue_wait_ms: self.avg_queue_wait_ms(),
            avg_processing_ms: self.avg_processing_ms(),
            avg_batch_size: self.avg_batch_size(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.requests_received.store(0, Ordering::Relaxed);
        self.requests_queued.store(0, Ordering::Relaxed);
        self.requests_processing.store(0, Ordering::Relaxed);
        self.requests_completed.store(0, Ordering::Relaxed);
        self.requests_timed_out.store(0, Ordering::Relaxed);
        self.requests_rejected.store(0, Ordering::Relaxed);
        self.total_queue_wait_ms.store(0, Ordering::Relaxed);
        self.total_processing_ms.store(0, Ordering::Relaxed);
        self.max_queue_depth.store(0, Ordering::Relaxed);
        self.batches_processed.store(0, Ordering::Relaxed);
        self.total_batch_requests.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub requests_received: u64,
    pub requests_queued: u64,
    pub requests_processing: u64,
    pub requests_completed: u64,
    pub requests_timed_out: u64,
    pub requests_rejected: u64,
    pub max_queue_depth: u64,
    pub avg_queue_wait_ms: f64,
    pub avg_processing_ms: f64,
    pub avg_batch_size: f64,
}

/// Timer for tracking request lifecycle
pub struct RequestTimer {
    pub queued_at: Instant,
    pub started_at: Option<Instant>,
}

impl RequestTimer {
    pub fn new() -> Self {
        Self { queued_at: Instant::now(), started_at: None }
    }

    pub fn start_processing(&mut self) {
        self.started_at = Some(Instant::now());
    }

    pub fn queue_wait_time(&self) -> Duration {
        self.started_at
            .map(|s| s.duration_since(self.queued_at))
            .unwrap_or_else(|| self.queued_at.elapsed())
    }

    pub fn processing_time(&self) -> Duration {
        self.started_at
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO)
    }
}

impl Default for RequestTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = BatcherMetrics::new();

        metrics.record_received();
        metrics.record_queued();
        assert_eq!(metrics.queue_depth(), 1);

        metrics.record_dequeued(Duration::from_millis(100));
        assert_eq!(metrics.queue_depth(), 0);
        assert_eq!(metrics.processing_count(), 1);

        metrics.record_completed(Duration::from_millis(500));
        assert_eq!(metrics.processing_count(), 0);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.requests_received, 1);
        assert_eq!(snapshot.requests_completed, 1);
        assert_eq!(snapshot.avg_queue_wait_ms, 100.0);
        assert_eq!(snapshot.avg_processing_ms, 500.0);
    }

    #[test]
    fn test_max_queue_depth() {
        let metrics = BatcherMetrics::new();

        // Add 5 requests
        for _ in 0..5 {
            metrics.record_queued();
        }
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 5);

        // Remove 3
        for _ in 0..3 {
            metrics.record_dequeued(Duration::ZERO);
            metrics.record_completed(Duration::ZERO);
        }

        // Max should still be 5
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.queue_depth(), 2);
    }

    #[test]
    fn test_batch_metrics() {
        let metrics = BatcherMetrics::new();

        metrics.record_batch(4);
        metrics.record_batch(6);

        assert_eq!(metrics.avg_batch_size(), 5.0);
    }
}
