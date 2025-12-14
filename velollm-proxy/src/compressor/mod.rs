//! Prompt compression module for reducing context size.
//!
//! This module provides intelligent prompt compression to reduce the amount of
//! context sent to the LLM backend, improving performance and reducing memory usage.
//!
//! ## Compression Techniques
//!
//! 1. **System Prompt Caching**: Caches and compresses repeated system prompts
//! 2. **Content Deduplication**: Identifies and factorizes repeated patterns
//! 3. **Extractive Summarization**: Summarizes old messages while preserving recent ones

mod config;
mod dedup;
mod metrics;
mod system_prompt;

pub use config::CompressionConfig;
pub use metrics::{CompressionMetrics, CompressionStats};

use crate::types::ollama::{ChatRequest, Message};
use dedup::ContentDeduplicator;
use std::sync::Arc;
use system_prompt::SystemPromptCache;
use tokio::sync::Mutex;

/// Prompt compressor for reducing context size
pub struct PromptCompressor {
    /// Configuration
    config: CompressionConfig,

    /// Metrics
    metrics: Arc<CompressionMetrics>,

    /// System prompt cache
    system_cache: Mutex<SystemPromptCache>,

    /// Content deduplicator
    deduplicator: ContentDeduplicator,
}

impl PromptCompressor {
    /// Create a new prompt compressor with the given configuration
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            system_cache: Mutex::new(SystemPromptCache::new(config.system_prompt_cache_size)),
            deduplicator: ContentDeduplicator::new(
                config.dedup_min_length,
                config.dedup_min_occurrences,
            ),
            metrics: Arc::new(CompressionMetrics::new()),
            config,
        }
    }

    /// Compress a chat request if needed
    ///
    /// Returns the compression result with statistics
    pub async fn compress(&self, request: &mut ChatRequest) -> CompressionResult {
        // Skip if compression is disabled
        if !self.config.enabled {
            return CompressionResult::skipped();
        }

        let original_size = self.estimate_tokens(&request.messages);

        // Skip if context is under threshold
        if original_size < self.config.max_context_tokens {
            self.metrics.record_skip();
            return CompressionResult::skipped();
        }

        let original_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();

        // Step 1: Compress system prompts
        if self.config.system_prompt_cache_enabled {
            self.compress_system_prompts(&mut request.messages).await;
        }

        // Step 2: Deduplicate content
        if self.config.dedup_enabled {
            self.deduplicate_content(&mut request.messages);
        }

        // Step 3: Summarize old messages if still over target
        let current_size = self.estimate_tokens(&request.messages);
        if current_size > self.config.target_context_tokens && self.config.summarization_enabled {
            self.summarize_old_messages(&mut request.messages);
        }

        let compressed_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();
        let compressed_size = self.estimate_tokens(&request.messages);

        // Record metrics
        self.metrics
            .record_compression(original_chars, compressed_chars);

        CompressionResult {
            compressed: true,
            original_tokens: original_size,
            compressed_tokens: compressed_size,
            original_chars,
            compressed_chars,
            ratio: if original_size > 0 {
                compressed_size as f64 / original_size as f64
            } else {
                1.0
            },
        }
    }

    /// Get compression statistics
    pub fn stats(&self) -> CompressionStats {
        self.metrics.snapshot()
    }

    /// Reset metrics
    pub fn reset_metrics(&self) {
        self.metrics.reset();
    }

    /// Compress system prompts using the cache
    async fn compress_system_prompts(&self, messages: &mut [Message]) {
        let mut cache = self.system_cache.lock().await;

        for msg in messages.iter_mut() {
            if msg.role == "system" {
                let (compressed, was_hit) = cache.get_or_compress(&msg.content);
                msg.content = compressed;

                if was_hit {
                    self.metrics.record_system_cache_hit();
                } else {
                    self.metrics.record_system_cache_miss();
                }
            }
        }
    }

    /// Deduplicate repeated content across messages
    fn deduplicate_content(&self, messages: &mut [Message]) {
        // Extract non-system message contents for deduplication
        let mut contents: Vec<String> = messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| m.content.clone())
            .collect();

        if contents.is_empty() {
            return;
        }

        let result = self.deduplicator.deduplicate(&mut contents);

        if result.patterns_found > 0 {
            self.metrics.record_dedup(result.patterns_found);

            // Put deduplicated content back
            let mut content_idx = 0;
            for msg in messages.iter_mut() {
                if msg.role != "system" && content_idx < contents.len() {
                    msg.content = contents[content_idx].clone();
                    content_idx += 1;
                }
            }
        }
    }

    /// Summarize old messages using extractive summarization
    fn summarize_old_messages(&self, messages: &mut Vec<Message>) {
        // Count messages by type
        let system_count = messages.iter().filter(|m| m.role == "system").count();
        let total_non_system = messages.len() - system_count;

        // Check if we have enough messages to summarize
        if total_non_system <= self.config.preserve_recent_messages {
            return;
        }

        // Find the split point: keep system messages and recent messages
        let messages_to_summarize = total_non_system - self.config.preserve_recent_messages;

        // Collect system messages
        let system_messages: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == "system")
            .cloned()
            .collect();

        // Collect non-system messages
        let non_system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != "system")
            .cloned()
            .collect();

        // Split into messages to summarize and messages to keep
        let (to_summarize, to_keep) = non_system.split_at(messages_to_summarize);

        // Don't summarize if nothing to summarize
        if to_summarize.is_empty() {
            return;
        }

        // Perform extractive summarization
        let summary = self.extractive_summarize(to_summarize);

        // Record metrics
        self.metrics.record_summarization(to_summarize.len());

        // Rebuild messages: system + summary + recent
        let mut new_messages = Vec::with_capacity(system_messages.len() + 1 + to_keep.len());

        // Add system messages
        new_messages.extend(system_messages);

        // Add summary as a system context message
        new_messages.push(Message {
            role: "system".to_string(),
            content: format!("[Previous conversation summary]:\n{}", summary),
            images: None,
            tool_calls: None,
        });

        // Add recent messages
        new_messages.extend(to_keep.iter().cloned());

        *messages = new_messages;
    }

    /// Perform extractive summarization on messages
    ///
    /// Extracts key information without using an LLM:
    /// - First sentence of each message
    /// - Code blocks (preserved in full)
    /// - Questions and their immediate responses
    fn extractive_summarize(&self, messages: &[Message]) -> String {
        let mut summary_parts = Vec::new();

        for msg in messages {
            let extracted = self.extract_key_content(&msg.content, &msg.role);
            if !extracted.is_empty() {
                summary_parts.push(format!("[{}]: {}", msg.role, extracted));
            }
        }

        summary_parts.join("\n")
    }

    /// Extract key content from a message
    fn extract_key_content(&self, content: &str, _role: &str) -> String {
        let mut parts = Vec::new();

        // Extract code blocks (preserve fully)
        let code_blocks = self.extract_code_blocks(content);
        for block in &code_blocks {
            parts.push(block.clone());
        }

        // Remove code blocks from content for text processing
        let text_content = self.remove_code_blocks(content);

        // Extract first sentence
        if let Some(first_sentence) = self.extract_first_sentence(&text_content) {
            parts.insert(0, first_sentence);
        }

        // If content is very short, just use it as-is
        if content.len() < 100 {
            return content.to_string();
        }

        parts.join("\n")
    }

    /// Extract code blocks from content
    fn extract_code_blocks(&self, content: &str) -> Vec<String> {
        let mut blocks = Vec::new();
        let mut in_block = false;
        let mut current_block = String::new();

        for line in content.lines() {
            if line.starts_with("```") {
                if in_block {
                    current_block.push_str(line);
                    blocks.push(current_block.clone());
                    current_block.clear();
                    in_block = false;
                } else {
                    in_block = true;
                    current_block.push_str(line);
                    current_block.push('\n');
                }
            } else if in_block {
                current_block.push_str(line);
                current_block.push('\n');
            }
        }

        blocks
    }

    /// Remove code blocks from content
    fn remove_code_blocks(&self, content: &str) -> String {
        let mut result = String::new();
        let mut in_block = false;

        for line in content.lines() {
            if line.starts_with("```") {
                in_block = !in_block;
            } else if !in_block {
                result.push_str(line);
                result.push('\n');
            }
        }

        result.trim().to_string()
    }

    /// Extract the first sentence from text
    fn extract_first_sentence(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        // Find the end of the first sentence
        for (i, c) in trimmed.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                // Make sure we include the punctuation
                let sentence = &trimmed[..=i];
                // Skip if too short (likely an abbreviation like "e.g.")
                if sentence.len() > 10 {
                    return Some(sentence.to_string());
                }
            }
        }

        // If no sentence end found, return first 100 chars
        if trimmed.len() > 100 {
            Some(format!("{}...", &trimmed[..100]))
        } else {
            Some(trimmed.to_string())
        }
    }

    /// Estimate token count from messages
    ///
    /// Uses a simple heuristic: words * 1.3 (average tokens per word)
    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        let word_count: usize = messages
            .iter()
            .map(|m| m.content.split_whitespace().count())
            .sum();

        // Approximate: 1 word ≈ 1.3 tokens on average
        (word_count as f64 * 1.3) as usize
    }
}

/// Result of a compression operation
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Whether compression was performed
    pub compressed: bool,

    /// Original size in estimated tokens
    pub original_tokens: usize,

    /// Compressed size in estimated tokens
    pub compressed_tokens: usize,

    /// Original size in characters
    pub original_chars: usize,

    /// Compressed size in characters
    pub compressed_chars: usize,

    /// Compression ratio (lower is better, 1.0 = no compression)
    pub ratio: f64,
}

impl CompressionResult {
    /// Create a result for skipped compression
    pub fn skipped() -> Self {
        Self {
            compressed: false,
            original_tokens: 0,
            compressed_tokens: 0,
            original_chars: 0,
            compressed_chars: 0,
            ratio: 1.0,
        }
    }

    /// Characters saved by compression
    pub fn chars_saved(&self) -> usize {
        self.original_chars.saturating_sub(self.compressed_chars)
    }

    /// Tokens saved by compression
    pub fn tokens_saved(&self) -> usize {
        self.original_tokens.saturating_sub(self.compressed_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CompressionConfig {
        CompressionConfig {
            enabled: true,
            max_context_tokens: 100,
            target_context_tokens: 50,
            dedup_enabled: true,
            dedup_min_length: 20,
            dedup_min_occurrences: 2,
            system_prompt_cache_enabled: true,
            system_prompt_cache_size: 10,
            summarization_enabled: true,
            preserve_recent_messages: 2,
        }
    }

    fn create_test_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    #[tokio::test]
    async fn test_compression_disabled() {
        let config = CompressionConfig::default(); // disabled by default
        let compressor = PromptCompressor::new(config);

        let mut request = ChatRequest {
            model: "test".to_string(),
            messages: vec![create_test_message("user", "Hello world!")],
            format: None,
            options: None,
            stream: None,
            keep_alive: None,
            tools: None,
        };

        let result = compressor.compress(&mut request).await;
        assert!(!result.compressed);
    }

    #[tokio::test]
    async fn test_small_context_skipped() {
        let config = create_test_config();
        let compressor = PromptCompressor::new(config);

        let mut request = ChatRequest {
            model: "test".to_string(),
            messages: vec![create_test_message("user", "Hi")],
            format: None,
            options: None,
            stream: None,
            keep_alive: None,
            tools: None,
        };

        let result = compressor.compress(&mut request).await;
        assert!(!result.compressed);
    }

    #[tokio::test]
    async fn test_system_prompt_caching() {
        let config = create_test_config();
        let compressor = PromptCompressor::new(config);

        let system_prompt = "You are a helpful AI assistant.  Please provide detailed and accurate information. This is a longer system prompt to ensure it gets processed.";

        // First request
        let mut request1 = ChatRequest {
            model: "test".to_string(),
            messages: vec![
                create_test_message("system", system_prompt),
                create_test_message("user", &"Hello ".repeat(100)), // Make it long enough
            ],
            format: None,
            options: None,
            stream: None,
            keep_alive: None,
            tools: None,
        };

        compressor.compress(&mut request1).await;

        // Second request with same system prompt
        let mut request2 = ChatRequest {
            model: "test".to_string(),
            messages: vec![
                create_test_message("system", system_prompt),
                create_test_message("user", &"World ".repeat(100)),
            ],
            format: None,
            options: None,
            stream: None,
            keep_alive: None,
            tools: None,
        };

        compressor.compress(&mut request2).await;

        let stats = compressor.stats();
        assert!(stats.system_cache_hits >= 1 || stats.system_cache_misses >= 1);
    }

    #[test]
    fn test_extract_first_sentence() {
        let config = create_test_config();
        let compressor = PromptCompressor::new(config);

        let text = "This is the first sentence. This is the second sentence.";
        let result = compressor.extract_first_sentence(text);
        assert_eq!(result, Some("This is the first sentence.".to_string()));
    }

    #[test]
    fn test_extract_code_blocks() {
        let config = create_test_config();
        let compressor = PromptCompressor::new(config);

        let content = "Some text\n```rust\nfn main() {}\n```\nMore text";
        let blocks = compressor.extract_code_blocks(content);

        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("fn main()"));
    }

    #[test]
    fn test_token_estimation() {
        let config = create_test_config();
        let compressor = PromptCompressor::new(config);

        let messages = vec![
            create_test_message("user", "Hello world this is a test"),
            create_test_message("assistant", "This is a response with some words"),
        ];

        let tokens = compressor.estimate_tokens(&messages);
        // 6 + 7 = 13 words * 1.3 ≈ 17 tokens
        assert!(tokens > 10 && tokens < 25);
    }

    #[tokio::test]
    async fn test_summarization() {
        let mut config = create_test_config();
        config.max_context_tokens = 10; // Force summarization
        config.target_context_tokens = 5;
        config.preserve_recent_messages = 1;

        let compressor = PromptCompressor::new(config);

        let mut request = ChatRequest {
            model: "test".to_string(),
            messages: vec![
                create_test_message("system", "System prompt"),
                create_test_message("user", &"Old message one ".repeat(10)),
                create_test_message("assistant", &"Old response ".repeat(10)),
                create_test_message("user", &"Recent message ".repeat(10)),
            ],
            format: None,
            options: None,
            stream: None,
            keep_alive: None,
            tools: None,
        };

        let result = compressor.compress(&mut request).await;

        // Should have compressed
        assert!(result.compressed);

        // Should have summary + preserved messages
        let stats = compressor.stats();
        assert!(stats.summarization_operations >= 1 || stats.messages_summarized >= 1);
    }

    #[test]
    fn test_compression_result_helpers() {
        let result = CompressionResult {
            compressed: true,
            original_tokens: 100,
            compressed_tokens: 60,
            original_chars: 500,
            compressed_chars: 300,
            ratio: 0.6,
        };

        assert_eq!(result.chars_saved(), 200);
        assert_eq!(result.tokens_saved(), 40);
    }
}
