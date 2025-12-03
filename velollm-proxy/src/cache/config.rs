//! Configuration for the response cache.

use std::time::Duration;

/// Configuration for the response cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the exact cache
    pub exact_cache_size: usize,

    /// Time-to-live for exact cache entries
    pub exact_cache_ttl: Duration,

    /// Whether semantic caching is enabled
    pub semantic_cache_enabled: bool,

    /// Maximum number of entries in the semantic cache
    pub semantic_cache_size: usize,

    /// Similarity threshold for semantic cache hits (0.0 - 1.0)
    /// Higher values require closer semantic match
    pub similarity_threshold: f32,

    /// Embedding model to use for semantic cache
    /// Options: "bge-small-en-v1.5", "all-MiniLM-L6-v2", etc.
    pub embedding_model: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            exact_cache_size: 1000,
            exact_cache_ttl: Duration::from_secs(3600), // 1 hour
            semantic_cache_enabled: false,              // Disabled by default (requires feature)
            semantic_cache_size: 500,
            similarity_threshold: 0.85, // Require 85% similarity
            embedding_model: "BAAI/bge-small-en-v1.5".to_string(),
        }
    }
}

impl CacheConfig {
    /// Create config optimized for low memory usage
    pub fn low_memory() -> Self {
        Self {
            exact_cache_size: 100,
            exact_cache_ttl: Duration::from_secs(1800), // 30 minutes
            semantic_cache_enabled: false,
            semantic_cache_size: 50,
            similarity_threshold: 0.90,
            embedding_model: "BAAI/bge-small-en-v1.5".to_string(),
        }
    }

    /// Create config optimized for high cache hit rate
    pub fn high_hit_rate() -> Self {
        Self {
            exact_cache_size: 5000,
            exact_cache_ttl: Duration::from_secs(7200), // 2 hours
            semantic_cache_enabled: true,
            semantic_cache_size: 2000,
            similarity_threshold: 0.80, // Accept 80% similarity
            embedding_model: "BAAI/bge-small-en-v1.5".to_string(),
        }
    }

    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("VELOLLM_CACHE_SIZE") {
            if let Ok(n) = val.parse() {
                config.exact_cache_size = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_CACHE_TTL_SECS") {
            if let Ok(n) = val.parse() {
                config.exact_cache_ttl = Duration::from_secs(n);
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_SEMANTIC_CACHE") {
            config.semantic_cache_enabled = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("VELOLLM_SEMANTIC_CACHE_SIZE") {
            if let Ok(n) = val.parse() {
                config.semantic_cache_size = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_SIMILARITY_THRESHOLD") {
            if let Ok(n) = val.parse() {
                config.similarity_threshold = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_EMBEDDING_MODEL") {
            config.embedding_model = val;
        }

        config
    }

    /// Check if semantic cache can be enabled
    #[cfg(feature = "semantic-cache")]
    pub fn can_enable_semantic(&self) -> bool {
        self.semantic_cache_enabled
    }

    #[cfg(not(feature = "semantic-cache"))]
    pub fn can_enable_semantic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default();
        assert_eq!(config.exact_cache_size, 1000);
        assert!(!config.semantic_cache_enabled);
        assert!((config.similarity_threshold - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_low_memory_config() {
        let config = CacheConfig::low_memory();
        assert_eq!(config.exact_cache_size, 100);
    }

    #[test]
    fn test_high_hit_rate_config() {
        let config = CacheConfig::high_hit_rate();
        assert_eq!(config.exact_cache_size, 5000);
        assert!(config.semantic_cache_enabled);
    }
}
