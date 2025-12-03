//! Semantic cache using embeddings and cosine similarity.
//!
//! This cache identifies semantically similar queries using local
//! embedding models and returns cached responses for similar questions.

use std::sync::Arc;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::types::openai::{ChatCompletionRequest, ChatCompletionResponse, Role};

use super::metrics::CacheMetrics;

/// Error type for semantic cache operations
#[derive(Debug, thiserror::Error)]
pub enum SemanticCacheError {
    #[error("Embedding model initialization failed: {0}")]
    ModelInitFailed(String),

    #[error("Embedding generation failed: {0}")]
    EmbeddingFailed(String),
}

/// A cached entry with its embedding vector
struct SemanticEntry {
    /// The embedding vector for the query
    embedding: Vec<f32>,
    /// The cached response
    response: ChatCompletionResponse,
    /// Original query text (for debugging)
    query_text: String,
}

/// Semantic cache using embeddings and cosine similarity
///
/// This cache provides O(n) lookup for semantically similar queries.
/// It uses local embedding models via fastembed for privacy and speed.
pub struct SemanticCache {
    /// Embedding model
    model: TextEmbedding,
    /// Cached entries with embeddings
    entries: Vec<SemanticEntry>,
    /// Maximum number of entries
    max_size: usize,
    /// Similarity threshold (0.0 - 1.0)
    threshold: f32,
    /// Metrics reference
    _metrics: Arc<CacheMetrics>,
}

impl SemanticCache {
    /// Create a new semantic cache
    pub fn new(
        max_size: usize,
        threshold: f32,
        metrics: Arc<CacheMetrics>,
    ) -> Result<Self, SemanticCacheError> {
        // Initialize the embedding model
        // Using BGE-small-en-v1.5 by default (384 dimensions, fast)
        let model = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::BGESmallENV15,
            show_download_progress: true,
            ..Default::default()
        })
        .map_err(|e| SemanticCacheError::ModelInitFailed(e.to_string()))?;

        tracing::info!(
            model = "BAAI/bge-small-en-v1.5",
            max_size = max_size,
            threshold = threshold,
            "Semantic cache initialized"
        );

        Ok(Self {
            model,
            entries: Vec::with_capacity(max_size),
            max_size,
            threshold,
            _metrics: metrics,
        })
    }

    /// Extract query text from request for embedding
    fn extract_query_text(request: &ChatCompletionRequest) -> String {
        // Get the last user message as the query
        request
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .filter_map(|m| m.content.as_ref())
            .last()
            .cloned()
            .unwrap_or_default()
    }

    /// Generate embedding for text
    fn embed(&self, text: &str) -> Result<Vec<f32>, SemanticCacheError> {
        let embeddings = self
            .model
            .embed(vec![text], None)
            .map_err(|e| SemanticCacheError::EmbeddingFailed(e.to_string()))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| SemanticCacheError::EmbeddingFailed("No embedding generated".to_string()))
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Try to get a cached response for a semantically similar query
    pub async fn get(&self, request: &ChatCompletionRequest) -> Option<ChatCompletionResponse> {
        let query_text = Self::extract_query_text(request);
        if query_text.is_empty() {
            return None;
        }

        // Generate embedding for the query
        let query_embedding = match self.embed(&query_text) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to generate query embedding: {}", e);
                return None;
            }
        };

        // Find the most similar cached entry
        let mut best_match: Option<(f32, &SemanticEntry)> = None;

        for entry in &self.entries {
            let similarity = Self::cosine_similarity(&query_embedding, &entry.embedding);

            if similarity >= self.threshold {
                if best_match
                    .as_ref()
                    .map(|(s, _)| similarity > *s)
                    .unwrap_or(true)
                {
                    best_match = Some((similarity, entry));
                }
            }
        }

        if let Some((similarity, entry)) = best_match {
            tracing::debug!(
                similarity = similarity,
                query = %query_text,
                cached_query = %entry.query_text,
                "Semantic cache hit"
            );
            return Some(entry.response.clone());
        }

        None
    }

    /// Store a response in the semantic cache
    pub async fn put(
        &mut self,
        request: &ChatCompletionRequest,
        response: ChatCompletionResponse,
    ) -> Result<(), SemanticCacheError> {
        let query_text = Self::extract_query_text(request);
        if query_text.is_empty() {
            return Ok(());
        }

        // Generate embedding
        let embedding = self.embed(&query_text)?;

        // Check if we need to evict (FIFO eviction for simplicity)
        if self.entries.len() >= self.max_size {
            self.entries.remove(0);
        }

        // Add new entry
        self.entries.push(SemanticEntry {
            embedding,
            response,
            query_text,
        });

        Ok(())
    }

    /// Get the current number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ChatMessage, Choice, Usage};

    fn create_request(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "llama3.2:3b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(content.to_string()),
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
        }
    }

    fn create_response(content: &str) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "llama3.2:3b".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: Some(content.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            system_fingerprint: None,
        }
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors should have similarity 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((SemanticCache::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity 0.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(SemanticCache::cosine_similarity(&a, &b).abs() < 0.001);

        // Opposite vectors should have similarity -1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((SemanticCache::cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_query_text() {
        let request = create_request("What is the capital of France?");
        let text = SemanticCache::extract_query_text(&request);
        assert_eq!(text, "What is the capital of France?");
    }

    // Integration tests require the model to be downloaded
    // They are marked as ignored by default
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_semantic_cache_similar_queries() {
        let metrics = Arc::new(CacheMetrics::new());
        let mut cache = SemanticCache::new(100, 0.8, metrics).unwrap();

        // Store a response for a weather query
        let request1 = create_request("What is the weather in Paris?");
        let response1 = create_response("The weather in Paris is sunny.");
        cache.put(&request1, response1).await.unwrap();

        // A similar query should hit the cache
        let request2 = create_request("How is the weather in Paris today?");
        let cached = cache.get(&request2).await;
        assert!(cached.is_some());
    }

    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_semantic_cache_different_queries() {
        let metrics = Arc::new(CacheMetrics::new());
        let mut cache = SemanticCache::new(100, 0.9, metrics).unwrap();

        // Store a response for a weather query
        let request1 = create_request("What is the weather in Paris?");
        let response1 = create_response("The weather in Paris is sunny.");
        cache.put(&request1, response1).await.unwrap();

        // A completely different query should miss
        let request2 = create_request("How do I cook pasta?");
        let cached = cache.get(&request2).await;
        assert!(cached.is_none());
    }
}
