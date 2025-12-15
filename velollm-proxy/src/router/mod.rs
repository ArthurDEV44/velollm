//! Multi-model load balancing for VeloLLM proxy.
//!
//! This module routes requests to the optimal model based on complexity analysis.
//! Simple queries go to small/fast models, complex queries go to larger models.
//!
//! # Example
//!
//! ```ignore
//! use velollm_proxy::router::{ModelRouter, RouterConfig};
//!
//! let config = RouterConfig::from_env();
//! let router = ModelRouter::new(config);
//!
//! // Route a request
//! let routed_model = router.route(&request);
//! ```

pub mod complexity;
pub mod config;
pub mod metrics;

pub use complexity::{ComplexityAnalyzer, ComplexityScore, ModelTier};
pub use config::RouterConfig;
pub use metrics::{RouterMetrics, RouterMetricsSummary};

use crate::types::openai::ChatCompletionRequest;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Special model name that triggers automatic routing.
pub const AUTO_MODEL: &str = "auto";

/// Model router for multi-model load balancing.
#[derive(Debug)]
pub struct ModelRouter {
    /// Configuration.
    config: RouterConfig,

    /// Complexity analyzer.
    analyzer: ComplexityAnalyzer,

    /// Metrics collector.
    metrics: Arc<RouterMetrics>,

    /// Available models (populated from Ollama).
    available_models: std::sync::RwLock<HashSet<String>>,
}

impl ModelRouter {
    /// Create a new model router with the given configuration.
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            analyzer: ComplexityAnalyzer::default(),
            metrics: Arc::new(RouterMetrics::new()),
            available_models: std::sync::RwLock::new(HashSet::new()),
        }
    }

    /// Check if routing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get metrics reference.
    pub fn metrics(&self) -> Arc<RouterMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Get configuration reference.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Update available models from Ollama.
    pub fn update_available_models(&self, models: Vec<String>) {
        let mut available = self.available_models.write().unwrap();
        available.clear();
        for model in models {
            available.insert(model);
        }
        info!(count = available.len(), "Updated available models for routing");
    }

    /// Check if a model is available.
    pub fn is_model_available(&self, model: &str) -> bool {
        let available = self.available_models.read().unwrap();
        if available.is_empty() {
            // No models detected yet, assume available
            true
        } else {
            available.contains(model)
        }
    }

    /// Route a request to the optimal model.
    ///
    /// Returns the model name to use. If routing is disabled or an explicit
    /// model is specified (not "auto"), returns the original model.
    pub fn route(&self, request: &ChatCompletionRequest) -> RoutingDecision {
        // Check if routing is enabled
        if !self.config.enabled {
            return RoutingDecision::passthrough(&request.model, "routing disabled");
        }

        // Check if explicit model specified (not "auto")
        let model = &request.model;
        if !model.eq_ignore_ascii_case(AUTO_MODEL) && !model.is_empty() {
            self.metrics.record_skipped();
            return RoutingDecision::passthrough(model, "explicit model specified");
        }

        // Analyze complexity
        let score = self.analyzer.analyze(request);
        let tier = ModelTier::from_score(
            score.total,
            self.config.small_threshold,
            self.config.large_threshold,
        );

        // Get model for tier
        let selected_model = self.model_for_tier(tier);

        // Check availability and fallback if needed
        let (final_model, fallback) = self.select_with_fallback(&selected_model, tier);

        // Record metrics
        self.metrics.record_routing(tier, score.total);

        debug!(
            complexity = score.total,
            tier = %tier,
            selected = %selected_model,
            final_model = %final_model,
            fallback = fallback,
            "Routed request"
        );

        RoutingDecision {
            original_model: model.clone(),
            selected_model: final_model,
            tier,
            complexity: score,
            reason: if fallback {
                format!("{} (fallback, {} unavailable)", tier, selected_model)
            } else {
                format!("{} tier", tier)
            },
            was_routed: true,
        }
    }

    /// Get the model name for a given tier.
    fn model_for_tier(&self, tier: ModelTier) -> String {
        match tier {
            ModelTier::Small => self.config.small_model.clone(),
            ModelTier::Medium => self.config.medium_model.clone(),
            ModelTier::Large => self.config.large_model.clone(),
        }
    }

    /// Select a model with fallback if the preferred one is unavailable.
    fn select_with_fallback(&self, preferred: &str, tier: ModelTier) -> (String, bool) {
        if self.is_model_available(preferred) {
            return (preferred.to_string(), false);
        }

        warn!(
            model = preferred,
            tier = %tier,
            "Preferred model unavailable, trying fallback"
        );

        // Fallback order: try adjacent tiers
        let fallbacks = match tier {
            ModelTier::Small => vec![&self.config.medium_model, &self.config.large_model],
            ModelTier::Medium => vec![&self.config.small_model, &self.config.large_model],
            ModelTier::Large => vec![&self.config.medium_model, &self.config.small_model],
        };

        for fallback in fallbacks {
            if self.is_model_available(fallback) {
                return (fallback.clone(), true);
            }
        }

        // Last resort: use the preferred model anyway (might fail at Ollama)
        warn!(model = preferred, "No fallback models available, using preferred model");
        (preferred.to_string(), false)
    }

    /// Analyze complexity without routing (for debugging).
    pub fn analyze_only(&self, request: &ChatCompletionRequest) -> ComplexityScore {
        self.analyzer.analyze(request)
    }
}

/// Result of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Original model from request.
    pub original_model: String,

    /// Selected model after routing.
    pub selected_model: String,

    /// Model tier selected.
    pub tier: ModelTier,

    /// Complexity analysis.
    pub complexity: ComplexityScore,

    /// Human-readable reason for the decision.
    pub reason: String,

    /// Whether routing was actually performed.
    pub was_routed: bool,
}

impl RoutingDecision {
    /// Create a passthrough decision (no routing performed).
    fn passthrough(model: &str, reason: &str) -> Self {
        Self {
            original_model: model.to_string(),
            selected_model: model.to_string(),
            tier: ModelTier::Medium, // Default tier for passthrough
            complexity: ComplexityScore::default(),
            reason: reason.to_string(),
            was_routed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ChatMessage, Role};

    fn make_request(model: &str, content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(content.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        }
    }

    #[test]
    fn test_routing_disabled() {
        let config = RouterConfig { enabled: false, ..Default::default() };
        let router = ModelRouter::new(config);

        let request = make_request("llama3.2:3b", "Hello");
        let decision = router.route(&request);

        assert!(!decision.was_routed);
        assert_eq!(decision.selected_model, "llama3.2:3b");
    }

    #[test]
    fn test_explicit_model_not_routed() {
        let config = RouterConfig { enabled: true, ..Default::default() };
        let router = ModelRouter::new(config);

        let request = make_request("llama3.2:3b", "Hello");
        let decision = router.route(&request);

        assert!(!decision.was_routed);
        assert_eq!(decision.selected_model, "llama3.2:3b");
    }

    #[test]
    fn test_auto_model_routes_simple_query() {
        let config = RouterConfig {
            enabled: true,
            small_model: "small-model".to_string(),
            medium_model: "medium-model".to_string(),
            large_model: "large-model".to_string(),
            small_threshold: 0.3,
            large_threshold: 0.7,
            ..Default::default()
        };
        let router = ModelRouter::new(config);

        let request = make_request("auto", "Hello!");
        let decision = router.route(&request);

        assert!(decision.was_routed);
        assert_eq!(decision.tier, ModelTier::Small);
        assert_eq!(decision.selected_model, "small-model");
    }

    #[test]
    fn test_auto_model_routes_complex_query() {
        let config = RouterConfig {
            enabled: true,
            small_model: "small-model".to_string(),
            medium_model: "medium-model".to_string(),
            large_model: "large-model".to_string(),
            small_threshold: 0.3,
            large_threshold: 0.7,
            ..Default::default()
        };
        let router = ModelRouter::new(config);

        let request = make_request(
            "auto",
            "Write code to implement a distributed consensus algorithm with Raft protocol",
        );
        let decision = router.route(&request);

        assert!(decision.was_routed);
        // Complex coding task should route to large model
        assert!(
            decision.tier == ModelTier::Large || decision.tier == ModelTier::Medium,
            "Expected Large or Medium tier, got {:?}",
            decision.tier
        );
    }

    #[test]
    fn test_fallback_when_model_unavailable() {
        let config = RouterConfig {
            enabled: true,
            small_model: "unavailable-small".to_string(),
            medium_model: "available-medium".to_string(),
            large_model: "unavailable-large".to_string(),
            ..Default::default()
        };
        let router = ModelRouter::new(config);

        // Set available models
        router.update_available_models(vec!["available-medium".to_string()]);

        let request = make_request("auto", "Hello!");
        let decision = router.route(&request);

        assert!(decision.was_routed);
        // Should fallback to medium since small is unavailable
        assert_eq!(decision.selected_model, "available-medium");
    }

    #[test]
    fn test_metrics_recorded() {
        let config = RouterConfig { enabled: true, ..Default::default() };
        let router = ModelRouter::new(config);

        let request = make_request("auto", "Hello!");
        router.route(&request);

        let metrics = router.metrics();
        assert_eq!(metrics.total_routed(), 1);
    }

    #[test]
    fn test_analyze_only() {
        let config = RouterConfig::default();
        let router = ModelRouter::new(config);

        let request = make_request("any-model", "Write code to implement quicksort");
        let score = router.analyze_only(&request);

        assert!(score.total > 0.0);
        assert!(score.factors.task_type > 0.0);
    }
}
