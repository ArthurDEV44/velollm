use serde::{Deserialize, Serialize};
use std::env;

/// Ollama configuration from environment variables
///
/// Ollama uses environment variables for configuration. This struct provides
/// a type-safe interface to read and write these configurations.
///
/// Reference: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OllamaConfig {
    /// Number of parallel requests to handle simultaneously
    /// Default: 1 (auto-determined based on available memory)
    pub num_parallel: Option<u32>,

    /// Maximum number of models to keep loaded in memory
    /// Default: 1 (system dependent)
    pub max_loaded_models: Option<u32>,

    /// How long to keep models loaded in memory after last use
    /// Format: duration string (e.g., "5m", "1h")
    /// Default: "5m"
    pub keep_alive: Option<String>,

    /// Context window size (number of tokens)
    /// Default: 2048 (model dependent)
    pub num_ctx: Option<u32>,

    /// Batch size for prompt processing
    /// Larger = faster prompt ingestion but more VRAM
    /// Default: 512
    pub num_batch: Option<u32>,

    /// Number of layers to offload to GPU
    /// Set to 999 to offload all layers
    /// Default: -1 (auto-detect)
    pub num_gpu: Option<i32>,

    /// Number of CPU threads to use
    /// Default: auto-detect
    pub num_thread: Option<u32>,

    /// Ollama host address
    /// Default: "127.0.0.1:11434"
    pub ollama_host: Option<String>,

    /// Ollama models directory
    /// Default: system dependent (~/.ollama/models)
    pub ollama_models: Option<String>,

    /// Enable debug logging
    /// Default: false
    pub ollama_debug: Option<bool>,

    /// Disable flash attention
    /// Default: false
    pub ollama_flash_attention: Option<bool>,
}

impl OllamaConfig {
    /// Read current Ollama configuration from environment variables
    ///
    /// # Example
    /// ```
    /// use velollm_adapters_ollama::OllamaConfig;
    ///
    /// let config = OllamaConfig::from_env();
    /// println!("Current config: {:?}", config);
    /// ```
    pub fn from_env() -> Self {
        Self {
            num_parallel: env::var("OLLAMA_NUM_PARALLEL")
                .ok()
                .and_then(|s| s.parse().ok()),
            max_loaded_models: env::var("OLLAMA_MAX_LOADED_MODELS")
                .ok()
                .and_then(|s| s.parse().ok()),
            keep_alive: env::var("OLLAMA_KEEP_ALIVE").ok(),
            num_ctx: env::var("OLLAMA_NUM_CTX").ok().and_then(|s| s.parse().ok()),
            num_batch: env::var("OLLAMA_NUM_BATCH")
                .ok()
                .and_then(|s| s.parse().ok()),
            num_gpu: env::var("OLLAMA_NUM_GPU").ok().and_then(|s| s.parse().ok()),
            num_thread: env::var("OLLAMA_NUM_THREAD")
                .ok()
                .and_then(|s| s.parse().ok()),
            ollama_host: env::var("OLLAMA_HOST").ok(),
            ollama_models: env::var("OLLAMA_MODELS").ok(),
            ollama_debug: env::var("OLLAMA_DEBUG").ok().and_then(|s| s.parse().ok()),
            ollama_flash_attention: env::var("OLLAMA_FLASH_ATTENTION")
                .ok()
                .map(|s| s == "1" || s.to_lowercase() == "true"),
        }
    }

    /// Generate shell export commands for this configuration
    ///
    /// Returns a string containing `export` statements suitable for
    /// sourcing in a shell script or adding to .bashrc/.zshrc
    ///
    /// # Example
    /// ```
    /// use velollm_adapters_ollama::OllamaConfig;
    ///
    /// let mut config = OllamaConfig::default();
    /// config.num_parallel = Some(4);
    /// config.num_gpu = Some(999);
    ///
    /// let exports = config.to_env_exports();
    /// println!("{}", exports);
    /// // Output:
    /// // export OLLAMA_NUM_PARALLEL=4
    /// // export OLLAMA_NUM_GPU=999
    /// ```
    pub fn to_env_exports(&self) -> String {
        let mut exports = Vec::new();

        if let Some(val) = self.num_parallel {
            exports.push(format!("export OLLAMA_NUM_PARALLEL={}", val));
        }
        if let Some(val) = self.max_loaded_models {
            exports.push(format!("export OLLAMA_MAX_LOADED_MODELS={}", val));
        }
        if let Some(ref val) = self.keep_alive {
            exports.push(format!("export OLLAMA_KEEP_ALIVE=\"{}\"", val));
        }
        if let Some(val) = self.num_ctx {
            exports.push(format!("export OLLAMA_NUM_CTX={}", val));
        }
        if let Some(val) = self.num_batch {
            exports.push(format!("export OLLAMA_NUM_BATCH={}", val));
        }
        if let Some(val) = self.num_gpu {
            exports.push(format!("export OLLAMA_NUM_GPU={}", val));
        }
        if let Some(val) = self.num_thread {
            exports.push(format!("export OLLAMA_NUM_THREAD={}", val));
        }
        if let Some(ref val) = self.ollama_host {
            exports.push(format!("export OLLAMA_HOST=\"{}\"", val));
        }
        if let Some(ref val) = self.ollama_models {
            exports.push(format!("export OLLAMA_MODELS=\"{}\"", val));
        }
        if let Some(val) = self.ollama_debug {
            exports.push(format!("export OLLAMA_DEBUG={}", if val { "1" } else { "0" }));
        }
        if let Some(val) = self.ollama_flash_attention {
            exports.push(format!("export OLLAMA_FLASH_ATTENTION={}", if val { "1" } else { "0" }));
        }

        exports.join("\n")
    }

    /// Check if any configuration values are set
    pub fn is_empty(&self) -> bool {
        self.num_parallel.is_none()
            && self.max_loaded_models.is_none()
            && self.keep_alive.is_none()
            && self.num_ctx.is_none()
            && self.num_batch.is_none()
            && self.num_gpu.is_none()
            && self.num_thread.is_none()
            && self.ollama_host.is_none()
            && self.ollama_models.is_none()
            && self.ollama_debug.is_none()
            && self.ollama_flash_attention.is_none()
    }

    /// Merge another config into this one, preferring values from `other`
    pub fn merge(&mut self, other: &OllamaConfig) {
        if other.num_parallel.is_some() {
            self.num_parallel = other.num_parallel;
        }
        if other.max_loaded_models.is_some() {
            self.max_loaded_models = other.max_loaded_models;
        }
        if other.keep_alive.is_some() {
            self.keep_alive = other.keep_alive.clone();
        }
        if other.num_ctx.is_some() {
            self.num_ctx = other.num_ctx;
        }
        if other.num_batch.is_some() {
            self.num_batch = other.num_batch;
        }
        if other.num_gpu.is_some() {
            self.num_gpu = other.num_gpu;
        }
        if other.num_thread.is_some() {
            self.num_thread = other.num_thread;
        }
        if other.ollama_host.is_some() {
            self.ollama_host = other.ollama_host.clone();
        }
        if other.ollama_models.is_some() {
            self.ollama_models = other.ollama_models.clone();
        }
        if other.ollama_debug.is_some() {
            self.ollama_debug = other.ollama_debug;
        }
        if other.ollama_flash_attention.is_some() {
            self.ollama_flash_attention = other.ollama_flash_attention;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OllamaConfig::default();
        assert!(config.is_empty());
        assert_eq!(config.num_parallel, None);
    }

    #[test]
    fn test_to_env_exports() {
        let mut config = OllamaConfig::default();
        config.num_parallel = Some(4);
        config.num_gpu = Some(999);
        config.keep_alive = Some("10m".to_string());

        let exports = config.to_env_exports();

        assert!(exports.contains("export OLLAMA_NUM_PARALLEL=4"));
        assert!(exports.contains("export OLLAMA_NUM_GPU=999"));
        assert!(exports.contains("export OLLAMA_KEEP_ALIVE=\"10m\""));
    }

    #[test]
    fn test_is_empty() {
        let mut config = OllamaConfig::default();
        assert!(config.is_empty());

        config.num_parallel = Some(1);
        assert!(!config.is_empty());
    }

    #[test]
    fn test_merge() {
        let mut config1 = OllamaConfig::default();
        config1.num_parallel = Some(2);
        config1.num_gpu = Some(50);

        let mut config2 = OllamaConfig::default();
        config2.num_parallel = Some(4);
        config2.num_batch = Some(512);

        config1.merge(&config2);

        assert_eq!(config1.num_parallel, Some(4)); // Overwritten
        assert_eq!(config1.num_gpu, Some(50)); // Preserved
        assert_eq!(config1.num_batch, Some(512)); // Added
    }

    #[test]
    fn test_serialization() {
        let mut config = OllamaConfig::default();
        config.num_parallel = Some(4);
        config.keep_alive = Some("5m".to_string());

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OllamaConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_boolean_parsing() {
        let mut config = OllamaConfig::default();
        config.ollama_debug = Some(true);
        config.ollama_flash_attention = Some(false);

        let exports = config.to_env_exports();

        assert!(exports.contains("export OLLAMA_DEBUG=1"));
        assert!(exports.contains("export OLLAMA_FLASH_ATTENTION=0"));
    }
}
