//! Configuration for multi-model load balancing.

use std::time::Duration;

/// Configuration for the model router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Whether routing is enabled.
    /// When disabled, requests use their original model.
    pub enabled: bool,

    /// Small model for simple queries (fast, low resource).
    /// Example: llama3.2:1b, phi3:mini
    pub small_model: String,

    /// Medium model for moderate complexity (balanced).
    /// Example: llama3.2:3b, mistral:7b
    pub medium_model: String,

    /// Large model for complex queries (slower, higher quality).
    /// Example: llama3.1:8b, llama3.1:70b
    pub large_model: String,

    /// Complexity threshold for small → medium routing.
    /// Requests with complexity below this use the small model.
    pub small_threshold: f32,

    /// Complexity threshold for medium → large routing.
    /// Requests with complexity above this use the large model.
    pub large_threshold: f32,

    /// Whether to auto-detect available models from Ollama.
    pub auto_detect_models: bool,

    /// Timeout for model availability checks.
    pub model_check_timeout: Duration,

    /// Enable routing metrics collection.
    pub metrics_enabled: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            small_model: "llama3.2:1b".to_string(),
            medium_model: "llama3.2:3b".to_string(),
            large_model: "llama3.1:8b".to_string(),
            small_threshold: 0.3,
            large_threshold: 0.7,
            auto_detect_models: true,
            model_check_timeout: Duration::from_secs(5),
            metrics_enabled: true,
        }
    }
}

impl RouterConfig {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `VELOLLM_ROUTER_ENABLED`: Enable routing (default: false)
    /// - `VELOLLM_SMALL_MODEL`: Small model name (default: llama3.2:1b)
    /// - `VELOLLM_MEDIUM_MODEL`: Medium model name (default: llama3.2:3b)
    /// - `VELOLLM_LARGE_MODEL`: Large model name (default: llama3.1:8b)
    /// - `VELOLLM_SMALL_THRESHOLD`: Complexity threshold for small model (default: 0.3)
    /// - `VELOLLM_LARGE_THRESHOLD`: Complexity threshold for large model (default: 0.7)
    /// - `VELOLLM_ROUTER_AUTO_DETECT`: Auto-detect models (default: true)
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("VELOLLM_ROUTER_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            small_model: std::env::var("VELOLLM_SMALL_MODEL")
                .unwrap_or_else(|_| "llama3.2:1b".to_string()),
            medium_model: std::env::var("VELOLLM_MEDIUM_MODEL")
                .unwrap_or_else(|_| "llama3.2:3b".to_string()),
            large_model: std::env::var("VELOLLM_LARGE_MODEL")
                .unwrap_or_else(|_| "llama3.1:8b".to_string()),
            small_threshold: std::env::var("VELOLLM_SMALL_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.3),
            large_threshold: std::env::var("VELOLLM_LARGE_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            auto_detect_models: std::env::var("VELOLLM_ROUTER_AUTO_DETECT")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            model_check_timeout: Duration::from_secs(5),
            metrics_enabled: true,
        }
    }

    /// Check if thresholds are valid (small < large).
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.small_threshold >= self.large_threshold {
            return Err(ConfigError::InvalidThresholds {
                small: self.small_threshold,
                large: self.large_threshold,
            });
        }
        if self.small_threshold < 0.0 || self.small_threshold > 1.0 {
            return Err(ConfigError::ThresholdOutOfRange {
                name: "small_threshold",
                value: self.small_threshold,
            });
        }
        if self.large_threshold < 0.0 || self.large_threshold > 1.0 {
            return Err(ConfigError::ThresholdOutOfRange {
                name: "large_threshold",
                value: self.large_threshold,
            });
        }
        Ok(())
    }
}

/// Configuration error.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid thresholds: small ({small}) must be less than large ({large})")]
    InvalidThresholds { small: f32, large: f32 },

    #[error("Threshold {name} out of range [0, 1]: {value}")]
    ThresholdOutOfRange { name: &'static str, value: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RouterConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.small_model, "llama3.2:1b");
        assert_eq!(config.medium_model, "llama3.2:3b");
        assert_eq!(config.large_model, "llama3.1:8b");
        assert!(config.small_threshold < config.large_threshold);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = RouterConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_thresholds() {
        let config =
            RouterConfig { small_threshold: 0.8, large_threshold: 0.3, ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_threshold_out_of_range() {
        let config = RouterConfig { small_threshold: -0.1, ..Default::default() };
        assert!(config.validate().is_err());

        let config = RouterConfig { large_threshold: 1.5, ..Default::default() };
        assert!(config.validate().is_err());
    }
}
