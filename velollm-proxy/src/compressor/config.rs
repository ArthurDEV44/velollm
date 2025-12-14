//! Configuration for prompt compression.

/// Configuration for prompt compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Whether compression is enabled
    pub enabled: bool,

    /// Maximum context size (in estimated tokens) before compression triggers
    pub max_context_tokens: usize,

    /// Target context size after compression
    pub target_context_tokens: usize,

    /// Enable content deduplication
    pub dedup_enabled: bool,

    /// Minimum length for deduplication patterns (in characters)
    pub dedup_min_length: usize,

    /// Minimum occurrences for a pattern to be deduplicated
    pub dedup_min_occurrences: usize,

    /// Enable system prompt caching
    pub system_prompt_cache_enabled: bool,

    /// Maximum entries in system prompt cache
    pub system_prompt_cache_size: usize,

    /// Enable message summarization
    pub summarization_enabled: bool,

    /// Number of recent messages to preserve (not summarize)
    pub preserve_recent_messages: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_context_tokens: 4096,
            target_context_tokens: 2048,
            dedup_enabled: true,
            dedup_min_length: 50,
            dedup_min_occurrences: 2,
            system_prompt_cache_enabled: true,
            system_prompt_cache_size: 100,
            summarization_enabled: true,
            preserve_recent_messages: 4,
        }
    }
}

impl CompressionConfig {
    /// Create config optimized for aggressive compression
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            max_context_tokens: 2048,
            target_context_tokens: 1024,
            dedup_enabled: true,
            dedup_min_length: 30,
            dedup_min_occurrences: 2,
            system_prompt_cache_enabled: true,
            system_prompt_cache_size: 200,
            summarization_enabled: true,
            preserve_recent_messages: 2,
        }
    }

    /// Create config for conservative compression (preserve more content)
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            max_context_tokens: 8192,
            target_context_tokens: 4096,
            dedup_enabled: true,
            dedup_min_length: 100,
            dedup_min_occurrences: 3,
            system_prompt_cache_enabled: true,
            system_prompt_cache_size: 50,
            summarization_enabled: true,
            preserve_recent_messages: 6,
        }
    }

    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_ENABLED") {
            config.enabled = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_MAX_CONTEXT") {
            if let Ok(n) = val.parse() {
                config.max_context_tokens = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_TARGET") {
            if let Ok(n) = val.parse() {
                config.target_context_tokens = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_DEDUP") {
            config.dedup_enabled = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_DEDUP_MIN_LENGTH") {
            if let Ok(n) = val.parse() {
                config.dedup_min_length = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_SYSTEM_CACHE") {
            config.system_prompt_cache_enabled = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_SYSTEM_CACHE_SIZE") {
            if let Ok(n) = val.parse() {
                config.system_prompt_cache_size = n;
            }
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_SUMMARIZE") {
            config.summarization_enabled = val == "1" || val.to_lowercase() == "true";
        }

        if let Ok(val) = std::env::var("VELOLLM_COMPRESSION_PRESERVE_RECENT") {
            if let Ok(n) = val.parse() {
                config.preserve_recent_messages = n;
            }
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompressionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_context_tokens, 4096);
        assert_eq!(config.target_context_tokens, 2048);
        assert!(config.dedup_enabled);
        assert!(config.system_prompt_cache_enabled);
        assert!(config.summarization_enabled);
    }

    #[test]
    fn test_aggressive_config() {
        let config = CompressionConfig::aggressive();
        assert!(config.enabled);
        assert_eq!(config.max_context_tokens, 2048);
        assert_eq!(config.preserve_recent_messages, 2);
    }

    #[test]
    fn test_conservative_config() {
        let config = CompressionConfig::conservative();
        assert!(config.enabled);
        assert_eq!(config.max_context_tokens, 8192);
        assert_eq!(config.preserve_recent_messages, 6);
    }
}
