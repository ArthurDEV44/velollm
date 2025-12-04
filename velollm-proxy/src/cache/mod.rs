//! Intelligent caching for VeloLLM proxy.
//!
//! This module provides two levels of caching for LLM responses:
//!
//! 1. **Exact Cache**: Fast LRU cache with hash-based lookup for identical queries
//! 2. **Semantic Cache**: Embedding-based cache for semantically similar queries
//!
//! # Architecture
//!
//! ```text
//! Incoming Request
//!        │
//!        ▼
//! ┌──────────────┐
//! │ Exact Cache  │ ─── Hash lookup (< 1ms)
//! │   (LRU)      │
//! └──────┬───────┘
//!        │ Miss
//!        ▼
//! ┌──────────────┐
//! │Semantic Cache│ ─── Embedding + similarity (10-50ms)
//! │ (Optional)   │
//! └──────┬───────┘
//!        │ Miss
//!        ▼
//!    Forward to LLM
//! ```
//!
//! # Features
//!
//! - **Exact Cache**: Always enabled, zero external dependencies
//! - **Semantic Cache**: Enabled with `semantic-cache` feature, uses local embeddings
//!
//! # References
//!
//! - [GPTCache](https://github.com/zilliztech/GPTCache)
//! - [Semantic Caching with Qdrant & Rust](https://www.shuttle.dev/blog/2024/05/30/semantic-caching-qdrant-rust)

mod config;
mod exact;
mod metrics;

#[cfg(feature = "semantic-cache")]
mod semantic;

pub use config::CacheConfig;
pub use exact::ExactCache;
pub use metrics::{CacheMetrics, CacheStats};

#[cfg(feature = "semantic-cache")]
pub use semantic::SemanticCache;

use crate::types::openai::{ChatCompletionRequest, ChatCompletionResponse};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Unified cache that combines exact and semantic caching
pub struct ResponseCache {
    /// Exact match cache (always enabled)
    exact: RwLock<ExactCache>,
    /// Semantic similarity cache (optional)
    #[cfg(feature = "semantic-cache")]
    semantic: Option<RwLock<SemanticCache>>,
    /// Cache metrics
    metrics: Arc<CacheMetrics>,
    /// Configuration
    config: CacheConfig,
}

impl ResponseCache {
    /// Create a new response cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        let metrics = Arc::new(CacheMetrics::new());
        let exact = RwLock::new(ExactCache::new(config.exact_cache_size, config.exact_cache_ttl));

        #[cfg(feature = "semantic-cache")]
        let semantic = if config.semantic_cache_enabled {
            match SemanticCache::new(
                config.semantic_cache_size,
                config.similarity_threshold,
                metrics.clone(),
            ) {
                Ok(cache) => {
                    tracing::info!("Semantic cache initialized successfully");
                    Some(RwLock::new(cache))
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize semantic cache: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            exact,
            #[cfg(feature = "semantic-cache")]
            semantic,
            metrics,
            config,
        }
    }

    /// Try to get a cached response for the request
    pub async fn get(&self, request: &ChatCompletionRequest) -> Option<ChatCompletionResponse> {
        // Try exact cache first
        {
            let mut exact = self.exact.write().await;
            if let Some(response) = exact.get(request) {
                self.metrics.record_exact_hit();
                tracing::debug!("Exact cache hit for request");
                return Some(response);
            }
        }
        self.metrics.record_exact_miss();

        // Try semantic cache if available
        #[cfg(feature = "semantic-cache")]
        if let Some(ref semantic) = self.semantic {
            let semantic = semantic.read().await;
            if let Some(response) = semantic.get(request).await {
                self.metrics.record_semantic_hit();
                tracing::debug!("Semantic cache hit for request");
                return Some(response);
            }
            self.metrics.record_semantic_miss();
        }

        None
    }

    /// Store a response in the cache
    pub async fn put(&self, request: &ChatCompletionRequest, response: &ChatCompletionResponse) {
        // Don't cache streaming responses or tool calls
        if request.stream {
            return;
        }
        if response
            .choices
            .iter()
            .any(|c| c.message.tool_calls.is_some())
        {
            return;
        }

        // Store in exact cache
        {
            let mut exact = self.exact.write().await;
            exact.put(request, response.clone());
        }

        // Store in semantic cache if available
        #[cfg(feature = "semantic-cache")]
        if let Some(ref semantic) = self.semantic {
            let mut semantic = semantic.write().await;
            if let Err(e) = semantic.put(request, response.clone()).await {
                tracing::warn!("Failed to store in semantic cache: {}", e);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.metrics.snapshot()
    }

    /// Clear all caches
    pub async fn clear(&self) {
        {
            let mut exact = self.exact.write().await;
            exact.clear();
        }

        #[cfg(feature = "semantic-cache")]
        if let Some(ref semantic) = self.semantic {
            let mut semantic = semantic.write().await;
            semantic.clear();
        }

        self.metrics.reset();
    }

    /// Get the cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ChatMessage, Role, Usage};

    #[tokio::test]
    async fn test_cache_basic() {
        let config = CacheConfig::default();
        let cache = ResponseCache::new(config);

        let request = ChatCompletionRequest {
            model: "llama3.2:3b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            n: None,
            user: None,
            seed: None,
            response_format: None,
        };

        // First request should miss
        assert!(cache.get(&request).await.is_none());

        // Create a response
        let response = ChatCompletionResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "llama3.2:3b".to_string(),
            choices: vec![],
            usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            system_fingerprint: None,
        };

        // Store it
        cache.put(&request, &response).await;

        // Second request should hit
        let cached = cache.get(&request).await;
        assert!(cached.is_some());
    }
}
