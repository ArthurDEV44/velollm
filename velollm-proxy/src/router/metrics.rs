//! Metrics for multi-model load balancing.

use super::complexity::ModelTier;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Metrics for model routing.
#[derive(Debug, Default)]
pub struct RouterMetrics {
    /// Total requests routed.
    pub requests_routed: AtomicU64,

    /// Requests per tier.
    pub requests_by_tier: TierCounters,

    /// Requests where routing was skipped (explicit model specified).
    pub requests_skipped: AtomicU64,

    /// Average complexity score (stored as fixed-point * 1000).
    complexity_sum: AtomicU64,
    complexity_count: AtomicU64,
}

/// Counters per model tier.
#[derive(Debug, Default)]
pub struct TierCounters {
    pub small: AtomicU64,
    pub medium: AtomicU64,
    pub large: AtomicU64,
}

impl RouterMetrics {
    /// Create new metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a routing decision.
    pub fn record_routing(&self, tier: ModelTier, complexity: f32) {
        self.requests_routed.fetch_add(1, Ordering::Relaxed);

        match tier {
            ModelTier::Small => self.requests_by_tier.small.fetch_add(1, Ordering::Relaxed),
            ModelTier::Medium => self.requests_by_tier.medium.fetch_add(1, Ordering::Relaxed),
            ModelTier::Large => self.requests_by_tier.large.fetch_add(1, Ordering::Relaxed),
        };

        // Store complexity as fixed-point (multiply by 1000)
        let complexity_fp = (complexity * 1000.0) as u64;
        self.complexity_sum
            .fetch_add(complexity_fp, Ordering::Relaxed);
        self.complexity_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a skipped routing (explicit model).
    pub fn record_skipped(&self) {
        self.requests_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the average complexity score.
    pub fn avg_complexity(&self) -> f32 {
        let count = self.complexity_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let sum = self.complexity_sum.load(Ordering::Relaxed);
        (sum as f32 / count as f32) / 1000.0
    }

    /// Get total routed requests.
    pub fn total_routed(&self) -> u64 {
        self.requests_routed.load(Ordering::Relaxed)
    }

    /// Get requests by tier.
    pub fn by_tier(&self) -> HashMap<ModelTier, u64> {
        let mut map = HashMap::new();
        map.insert(ModelTier::Small, self.requests_by_tier.small.load(Ordering::Relaxed));
        map.insert(ModelTier::Medium, self.requests_by_tier.medium.load(Ordering::Relaxed));
        map.insert(ModelTier::Large, self.requests_by_tier.large.load(Ordering::Relaxed));
        map
    }

    /// Get tier distribution as percentages.
    pub fn tier_distribution(&self) -> HashMap<ModelTier, f32> {
        let total = self.total_routed();
        if total == 0 {
            let mut map = HashMap::new();
            map.insert(ModelTier::Small, 0.0);
            map.insert(ModelTier::Medium, 0.0);
            map.insert(ModelTier::Large, 0.0);
            return map;
        }

        let by_tier = self.by_tier();
        let mut map = HashMap::new();
        for (tier, count) in by_tier {
            map.insert(tier, count as f32 / total as f32 * 100.0);
        }
        map
    }

    /// Get a summary of metrics.
    pub fn summary(&self) -> RouterMetricsSummary {
        let total = self.total_routed();
        let by_tier = self.by_tier();

        RouterMetricsSummary {
            total_routed: total,
            total_skipped: self.requests_skipped.load(Ordering::Relaxed),
            small_count: by_tier.get(&ModelTier::Small).copied().unwrap_or(0),
            medium_count: by_tier.get(&ModelTier::Medium).copied().unwrap_or(0),
            large_count: by_tier.get(&ModelTier::Large).copied().unwrap_or(0),
            avg_complexity: self.avg_complexity(),
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.requests_routed.store(0, Ordering::Relaxed);
        self.requests_skipped.store(0, Ordering::Relaxed);
        self.requests_by_tier.small.store(0, Ordering::Relaxed);
        self.requests_by_tier.medium.store(0, Ordering::Relaxed);
        self.requests_by_tier.large.store(0, Ordering::Relaxed);
        self.complexity_sum.store(0, Ordering::Relaxed);
        self.complexity_count.store(0, Ordering::Relaxed);
    }
}

/// Summary of router metrics.
#[derive(Debug, Clone)]
pub struct RouterMetricsSummary {
    pub total_routed: u64,
    pub total_skipped: u64,
    pub small_count: u64,
    pub medium_count: u64,
    pub large_count: u64,
    pub avg_complexity: f32,
}

impl RouterMetricsSummary {
    /// Get small tier percentage.
    pub fn small_pct(&self) -> f32 {
        if self.total_routed == 0 {
            0.0
        } else {
            self.small_count as f32 / self.total_routed as f32 * 100.0
        }
    }

    /// Get medium tier percentage.
    pub fn medium_pct(&self) -> f32 {
        if self.total_routed == 0 {
            0.0
        } else {
            self.medium_count as f32 / self.total_routed as f32 * 100.0
        }
    }

    /// Get large tier percentage.
    pub fn large_pct(&self) -> f32 {
        if self.total_routed == 0 {
            0.0
        } else {
            self.large_count as f32 / self.total_routed as f32 * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = RouterMetrics::new();

        metrics.record_routing(ModelTier::Small, 0.2);
        metrics.record_routing(ModelTier::Medium, 0.5);
        metrics.record_routing(ModelTier::Large, 0.8);
        metrics.record_skipped();

        assert_eq!(metrics.total_routed(), 3);
        assert_eq!(metrics.requests_skipped.load(Ordering::Relaxed), 1);

        let by_tier = metrics.by_tier();
        assert_eq!(by_tier.get(&ModelTier::Small), Some(&1));
        assert_eq!(by_tier.get(&ModelTier::Medium), Some(&1));
        assert_eq!(by_tier.get(&ModelTier::Large), Some(&1));
    }

    #[test]
    fn test_avg_complexity() {
        let metrics = RouterMetrics::new();

        metrics.record_routing(ModelTier::Small, 0.2);
        metrics.record_routing(ModelTier::Medium, 0.4);
        metrics.record_routing(ModelTier::Large, 0.6);

        let avg = metrics.avg_complexity();
        assert!((avg - 0.4).abs() < 0.01, "Expected ~0.4, got {}", avg);
    }

    #[test]
    fn test_tier_distribution() {
        let metrics = RouterMetrics::new();

        // 2 small, 2 medium, 1 large = 5 total
        metrics.record_routing(ModelTier::Small, 0.1);
        metrics.record_routing(ModelTier::Small, 0.2);
        metrics.record_routing(ModelTier::Medium, 0.5);
        metrics.record_routing(ModelTier::Medium, 0.5);
        metrics.record_routing(ModelTier::Large, 0.9);

        let dist = metrics.tier_distribution();
        assert!((dist.get(&ModelTier::Small).unwrap() - 40.0).abs() < 0.1);
        assert!((dist.get(&ModelTier::Medium).unwrap() - 40.0).abs() < 0.1);
        assert!((dist.get(&ModelTier::Large).unwrap() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_reset() {
        let metrics = RouterMetrics::new();

        metrics.record_routing(ModelTier::Small, 0.2);
        metrics.record_routing(ModelTier::Large, 0.8);

        assert_eq!(metrics.total_routed(), 2);

        metrics.reset();

        assert_eq!(metrics.total_routed(), 0);
        assert_eq!(metrics.avg_complexity(), 0.0);
    }

    #[test]
    fn test_summary() {
        let metrics = RouterMetrics::new();

        metrics.record_routing(ModelTier::Small, 0.2);
        metrics.record_routing(ModelTier::Medium, 0.5);
        metrics.record_skipped();

        let summary = metrics.summary();
        assert_eq!(summary.total_routed, 2);
        assert_eq!(summary.total_skipped, 1);
        assert_eq!(summary.small_count, 1);
        assert_eq!(summary.medium_count, 1);
        assert_eq!(summary.large_count, 0);
    }
}
