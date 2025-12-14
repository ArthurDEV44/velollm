//! Speculative prefetch system for VeloLLM proxy.
//!
//! This module implements pattern-based query prediction and speculative
//! response generation to reduce latency for common follow-up questions.
//!
//! # Architecture
//!
//! ```text
//! User Request → Response
//!                   │
//!                   ▼
//!            PrefetchPredictor
//!                   │
//!                   ▼ (predicted queries)
//!            PrefetchQueue
//!                   │
//!                   ▼ (when spare permits available)
//!            Background Worker → Ollama → ResponseCache
//! ```
//!
//! # Features
//!
//! - **Pattern-based prediction**: Detects query types and predicts likely follow-ups
//! - **Priority queue**: Tasks sorted by confidence score
//! - **Zero user impact**: Only uses spare concurrency permits
//! - **Shared cache**: Prefetched responses stored in existing ResponseCache

mod config;
mod metrics;
mod predictor;
mod queue;

pub use config::PrefetchConfig;
pub use metrics::{PrefetchMetrics, PrefetchStats};
pub use predictor::{PredictedQuery, QueryPredictor, QueryType};
pub use queue::{PrefetchQueue, PrefetchTask};

use crate::cache::ResponseCache;
use crate::proxy::OllamaProxy;
use crate::types::ollama::{ChatRequest, Message};
use crate::types::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Role, Usage,
};
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;
use tracing::{debug, info, warn};

/// Main prefetch service
pub struct PrefetchService {
    /// Configuration
    config: PrefetchConfig,
    /// Query predictor
    predictor: QueryPredictor,
    /// Task queue
    queue: PrefetchQueue,
    /// Metrics
    metrics: Arc<PrefetchMetrics>,
}

impl PrefetchService {
    /// Create a new prefetch service
    pub fn new(config: PrefetchConfig) -> Self {
        let queue = PrefetchQueue::new(config.max_queue_size);
        Self {
            config,
            predictor: QueryPredictor::new(),
            queue,
            metrics: Arc::new(PrefetchMetrics::new()),
        }
    }

    /// Check if prefetch is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Called after a successful response to predict and queue prefetch tasks
    ///
    /// # Arguments
    /// * `messages` - The original conversation messages
    /// * `model` - The model used for generation
    /// * `_response` - The response content (for future context analysis)
    pub async fn on_response(&self, messages: &[Message], model: &str, _response: &str) {
        if !self.config.enabled {
            return;
        }

        // Predict follow-up queries
        let predictions = self
            .predictor
            .predict(messages, self.config.max_predictions);

        if predictions.is_empty() {
            return;
        }

        self.metrics.record_predictions(predictions.len());

        // Queue tasks for predictions above confidence threshold
        for prediction in predictions {
            if prediction.confidence < self.config.min_confidence {
                continue;
            }

            // Build messages for the prefetch task (original + predicted follow-up)
            let mut task_messages = messages.to_vec();
            task_messages.push(Message {
                role: "user".to_string(),
                content: prediction.query.clone(),
                images: None,
                tool_calls: None,
            });

            let task = PrefetchTask::new(task_messages, model.to_string(), prediction);

            if self.queue.push(task).await {
                self.metrics.record_queued();
                debug!("Prefetch task queued");
            } else {
                self.metrics.record_dropped();
                debug!("Prefetch task dropped (queue full)");
            }
        }
    }

    /// Try to get a prefetched response for the given request
    ///
    /// Returns Some(response) if a prefetch cache hit was found
    pub async fn check_cache(
        &self,
        request: &ChatCompletionRequest,
        cache: &ResponseCache,
    ) -> Option<ChatCompletionResponse> {
        if !self.config.enabled {
            return None;
        }

        // Check the shared cache
        if let Some(response) = cache.get(request).await {
            self.metrics.record_cache_hit();
            info!("Prefetch cache hit");
            return Some(response);
        }

        None
    }

    /// Background worker that processes the prefetch queue
    ///
    /// This worker runs continuously and processes tasks when:
    /// 1. There are tasks in the queue
    /// 2. A spare permit is available (non-blocking)
    ///
    /// # Arguments
    /// * `proxy` - The Ollama proxy for making requests
    /// * `cache` - The response cache for storing results
    /// * `try_get_permit` - Function to try to get a spare permit
    pub async fn run_worker<F, Fut>(
        &self,
        proxy: Arc<OllamaProxy>,
        cache: Arc<ResponseCache>,
        try_get_permit: F,
    ) where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Option<OwnedSemaphorePermit>> + Send,
    {
        if !self.config.enabled {
            info!("Prefetch disabled, worker not starting");
            return;
        }

        info!("Prefetch worker started");
        let poll_interval = self.config.worker_poll_interval();
        let cache_ttl = self.config.cache_ttl();

        loop {
            // Clean up expired tasks
            let expired = self.queue.remove_expired(cache_ttl).await;
            if expired > 0 {
                debug!(expired, "Removed expired prefetch tasks");
            }

            // Try to get a task
            let task = match self.queue.pop().await {
                Some(t) => t,
                None => {
                    // No tasks, wait and try again
                    tokio::time::sleep(poll_interval).await;
                    continue;
                }
            };

            // Try to get a spare permit (non-blocking)
            let permit = match try_get_permit().await {
                Some(p) => {
                    self.metrics.record_permit_acquired();
                    p
                }
                None => {
                    // No permit available, put task back and wait
                    self.metrics.record_permit_unavailable();
                    self.queue.push(task).await;
                    tokio::time::sleep(poll_interval).await;
                    continue;
                }
            };

            // Execute the prefetch task
            debug!(
                query_type = ?task.prediction.query_type,
                confidence = task.prediction.confidence,
                "Executing prefetch task"
            );

            let result = self.execute_task(&task, &proxy).await;

            // Release the permit
            drop(permit);

            match result {
                Ok(response) => {
                    // Store in cache
                    let openai_request = self.task_to_openai_request(&task);
                    cache.put(&openai_request, &response).await;
                    self.metrics.record_executed();
                    debug!("Prefetch task completed and cached");
                }
                Err(e) => {
                    warn!(error = %e, "Prefetch task failed");
                }
            }
        }
    }

    /// Execute a prefetch task
    async fn execute_task(
        &self,
        task: &PrefetchTask,
        proxy: &OllamaProxy,
    ) -> Result<ChatCompletionResponse, crate::error::ProxyError> {
        // Build Ollama chat request
        let ollama_request = ChatRequest {
            model: task.model.clone(),
            messages: task.messages.clone(),
            stream: Some(false),
            format: None,
            options: None,
            keep_alive: None,
            tools: None,
        };

        // Execute the request
        let ollama_response = proxy.chat(&ollama_request).await?;

        // Get the response content from the message
        let content = ollama_response.message.as_ref().map(|m| m.content.clone());

        // Convert to OpenAI format
        let response = ChatCompletionResponse {
            id: format!("prefetch-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: task.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: ollama_response.prompt_eval_count.unwrap_or(0) as u32,
                completion_tokens: ollama_response.eval_count.unwrap_or(0) as u32,
                total_tokens: (ollama_response.prompt_eval_count.unwrap_or(0)
                    + ollama_response.eval_count.unwrap_or(0)) as u32,
            },
            system_fingerprint: None,
        };

        Ok(response)
    }

    /// Convert a prefetch task to an OpenAI request for cache lookup
    fn task_to_openai_request(&self, task: &PrefetchTask) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: task.model.clone(),
            messages: task
                .messages
                .iter()
                .map(|m| ChatMessage {
                    role: match m.role.as_str() {
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        "system" => Role::System,
                        _ => Role::User,
                    },
                    content: Some(m.content.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                })
                .collect(),
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

    /// Get prefetch statistics
    pub fn stats(&self) -> PrefetchStats {
        self.metrics.snapshot()
    }

    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        self.queue.len().await
    }

    /// Get the configuration
    pub fn config(&self) -> &PrefetchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    #[tokio::test]
    async fn test_prefetch_disabled() {
        let config = PrefetchConfig { enabled: false, ..Default::default() };
        let service = PrefetchService::new(config);

        assert!(!service.is_enabled());

        // Should not queue anything when disabled
        let messages = vec![make_message("user", "What is Rust?")];
        service
            .on_response(&messages, "test-model", "test response")
            .await;

        assert_eq!(service.queue_size().await, 0);
    }

    #[tokio::test]
    async fn test_prefetch_queuing() {
        let config = PrefetchConfig {
            enabled: true,
            min_confidence: 0.5,
            max_predictions: 2,
            ..Default::default()
        };
        let service = PrefetchService::new(config);

        // A query that should trigger predictions
        let messages = vec![make_message("user", "What is a closure in Rust?")];
        service
            .on_response(&messages, "test-model", "A closure is...")
            .await;

        // Should have queued some tasks
        assert!(service.queue_size().await > 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let config = PrefetchConfig { enabled: true, ..Default::default() };
        let service = PrefetchService::new(config);

        let stats = service.stats();
        assert_eq!(stats.predictions_total, 0);
        assert_eq!(stats.tasks_queued, 0);
    }
}
