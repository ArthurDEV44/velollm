//! System prompt caching and compression.

use lru::LruCache;
use std::num::NonZeroUsize;
use xxhash_rust::xxh3::xxh3_64;

/// Cache for system prompts to avoid reprocessing identical prompts
pub struct SystemPromptCache {
    /// LRU cache mapping prompt hash to compressed version
    cache: LruCache<u64, CachedPrompt>,
    /// Cache capacity
    capacity: usize,
}

/// A cached system prompt entry
struct CachedPrompt {
    /// Original prompt (for verification)
    original_len: usize,
    /// Compressed version of the prompt
    compressed: String,
    /// Hit count for this entry
    hits: u64,
}

impl SystemPromptCache {
    /// Create a new system prompt cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self { cache: LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap()), capacity }
    }

    /// Get or create a compressed version of a system prompt
    ///
    /// Returns (compressed_prompt, was_cache_hit)
    pub fn get_or_compress(&mut self, prompt: &str) -> (String, bool) {
        let hash = xxh3_64(prompt.as_bytes());

        // Check cache
        if let Some(cached) = self.cache.get_mut(&hash) {
            // Verify it's the same prompt (hash collision check)
            if cached.original_len == prompt.len() {
                cached.hits += 1;
                return (cached.compressed.clone(), true);
            }
        }

        // Compress and cache
        let compressed = self.compress_system_prompt(prompt);

        self.cache.put(
            hash,
            CachedPrompt { original_len: prompt.len(), compressed: compressed.clone(), hits: 1 },
        );

        (compressed, false)
    }

    /// Compress a system prompt by removing redundant content
    fn compress_system_prompt(&self, prompt: &str) -> String {
        let mut result = prompt.to_string();

        // 1. Normalize whitespace (collapse multiple spaces/newlines)
        result = normalize_whitespace(&result);

        // 2. Remove common verbose phrases that don't add information
        result = remove_verbose_phrases(&result);

        // 3. Compress common instruction patterns
        result = compress_common_patterns(&result);

        result
    }

    /// Get cache statistics
    pub fn stats(&self) -> SystemPromptCacheStats {
        let mut total_hits = 0u64;
        let entries = self.cache.len();

        for (_, cached) in self.cache.iter() {
            total_hits += cached.hits;
        }

        SystemPromptCacheStats { entries, capacity: self.capacity, total_hits }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Statistics for the system prompt cache
#[derive(Debug, Clone)]
pub struct SystemPromptCacheStats {
    /// Number of entries in cache
    pub entries: usize,
    /// Cache capacity
    pub capacity: usize,
    /// Total hits across all entries
    pub total_hits: u64,
}

/// Normalize whitespace in text
fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_whitespace = false;
    let mut last_was_newline = false;

    for c in text.chars() {
        if c == '\n' {
            if !last_was_newline {
                result.push('\n');
                last_was_newline = true;
            }
            last_was_whitespace = true;
        } else if c.is_whitespace() {
            if !last_was_whitespace {
                result.push(' ');
            }
            last_was_whitespace = true;
            last_was_newline = false;
        } else {
            result.push(c);
            last_was_whitespace = false;
            last_was_newline = false;
        }
    }

    result.trim().to_string()
}

/// Remove verbose phrases that don't add meaning
fn remove_verbose_phrases(text: &str) -> String {
    let verbose_phrases = [
        "Please note that ",
        "It's important to remember that ",
        "Keep in mind that ",
        "As a reminder, ",
        "For your information, ",
        "I want you to know that ",
        "Be aware that ",
    ];

    let mut result = text.to_string();
    for phrase in &verbose_phrases {
        result = result.replace(phrase, "");
    }

    result
}

/// Compress common instruction patterns
fn compress_common_patterns(text: &str) -> String {
    let patterns = [
        // Format instructions
        (
            "You should always respond in a helpful and professional manner",
            "Be helpful and professional",
        ),
        ("Please provide detailed and accurate information", "Provide accurate details"),
        ("Make sure to follow these instructions carefully", "Follow instructions"),
        // Common AI assistant phrases
        ("You are a helpful AI assistant", "You are a helpful assistant"),
        ("As an AI language model", "As an AI"),
    ];

    let mut result = text.to_string();
    for (long, short) in &patterns {
        result = result.replace(long, short);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let mut cache = SystemPromptCache::new(10);

        let prompt = "You are a helpful assistant.";
        let (compressed1, hit1) = cache.get_or_compress(prompt);
        let (compressed2, hit2) = cache.get_or_compress(prompt);

        assert!(!hit1); // First call is a miss
        assert!(hit2); // Second call is a hit
        assert_eq!(compressed1, compressed2);
    }

    #[test]
    fn test_whitespace_normalization() {
        let input = "Hello    world\n\n\nThis  is   a  test";
        let expected = "Hello world\nThis is a test";
        assert_eq!(normalize_whitespace(input), expected);
    }

    #[test]
    fn test_verbose_phrase_removal() {
        let input = "Please note that you should be helpful. It's important to remember that accuracy matters.";
        let result = remove_verbose_phrases(input);
        assert!(!result.contains("Please note that"));
        assert!(!result.contains("It's important to remember that"));
    }

    #[test]
    fn test_compression() {
        let mut cache = SystemPromptCache::new(10);

        let prompt =
            "You are a helpful AI assistant.  Please provide detailed and accurate information.";
        let (compressed, _) = cache.get_or_compress(prompt);

        // Should be shorter after compression
        assert!(compressed.len() <= prompt.len());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = SystemPromptCache::new(10);

        cache.get_or_compress("Prompt 1");
        cache.get_or_compress("Prompt 1"); // Hit
        cache.get_or_compress("Prompt 2");

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.capacity, 10);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = SystemPromptCache::new(2);

        cache.get_or_compress("Prompt 1");
        cache.get_or_compress("Prompt 2");
        cache.get_or_compress("Prompt 3"); // Should evict Prompt 1

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
    }
}
