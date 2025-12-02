use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub mod cuda_paged;
pub mod kv_cache;

/// Configuration for speculative decoding with llama.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Path to the main (target) model file
    pub main_model_path: String,
    /// Path to the draft model file (smaller, faster model)
    pub draft_model_path: String,
    /// Number of draft tokens to generate before validation
    pub n_draft: u32,
    /// Number of tokens to predict/generate
    pub n_predict: u32,
    /// Input prompt for generation
    pub prompt: String,
}

/// Runner for executing llama.cpp with speculative decoding
pub struct SpeculativeRunner {
    llama_cpp_path: String,
}

impl SpeculativeRunner {
    /// Create a new SpeculativeRunner
    ///
    /// # Arguments
    /// * `llama_cpp_path` - Path to the llama.cpp build directory
    pub fn new(llama_cpp_path: &str) -> Self {
        Self { llama_cpp_path: llama_cpp_path.to_string() }
    }

    /// Run speculative decoding inference
    ///
    /// # Arguments
    /// * `config` - Configuration for the speculative decoding run
    ///
    /// # Returns
    /// The full output from llama.cpp including timing information
    pub fn run(&self, config: &SpeculativeConfig) -> Result<String> {
        // Determine the correct binary path
        let binary_path = if config.draft_model_path.is_empty() {
            // Vanilla mode - use regular llama-cli
            Path::new(&self.llama_cpp_path)
                .join("build")
                .join("bin")
                .join("llama-cli")
        } else {
            // Speculative mode - use llama-speculative
            Path::new(&self.llama_cpp_path)
                .join("build")
                .join("bin")
                .join("llama-speculative")
        };

        let mut cmd = Command::new(&binary_path);

        // Add common arguments
        cmd.args(["-m", &config.main_model_path]);

        // Add speculative-specific arguments if in speculative mode
        if !config.draft_model_path.is_empty() {
            cmd.args([
                "-md",
                &config.draft_model_path,
                "--draft",
                &config.n_draft.to_string(),
            ]);
        }

        // Add generation parameters
        cmd.args([
            "-n",
            &config.n_predict.to_string(),
            "-p",
            &config.prompt,
            "-ngl",
            "99",                  // Offload all layers to GPU if available
            "--no-display-prompt", // Don't echo the prompt
        ]);

        println!("Executing: {:?}", cmd);

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute llama.cpp binary")?;

        if !output.status.success() {
            anyhow::bail!(
                "llama.cpp execution failed with exit code {:?}:\nStderr: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Parse performance metrics from llama.cpp output
    ///
    /// llama.cpp prints timing information like:
    /// ```text
    /// llama_print_timings:        load time =     123.45 ms
    /// llama_print_timings:      sample time =      10.23 ms /    50 runs   (    0.20 ms per token,  4882.81 tokens per second)
    /// llama_print_timings: prompt eval time =     234.56 ms /    10 tokens (   23.46 ms per token,    42.63 tokens per second)
    /// llama_print_timings:        eval time =    2000.00 ms /    49 runs   (   40.82 ms per token,    24.50 tokens per second)
    /// llama_print_timings:       total time =    2368.24 ms /    59 tokens
    /// ```
    pub fn parse_perf_metrics(&self, output: &str) -> Option<PerfMetrics> {
        let mut metrics = PerfMetrics::default();

        for line in output.lines() {
            if line.contains("llama_print_timings") {
                // Parse prompt eval time (time to first token)
                if line.contains("prompt eval time") {
                    if let Some(time) = extract_time_ms(line) {
                        metrics.time_to_first_token_ms = time;
                    }
                }

                // Parse eval time (generation tokens per second)
                if line.contains("eval time") && !line.contains("prompt eval") {
                    if let Some(tps) = extract_tokens_per_second(line) {
                        metrics.tokens_per_second = tps;
                    }
                }

                // Parse total time
                if line.contains("total time") {
                    if let Some(time) = extract_time_ms(line) {
                        metrics.total_time_ms = time;
                    }
                }
            }
        }

        // Only return metrics if we successfully parsed at least tokens/s
        if metrics.tokens_per_second > 0.0 {
            Some(metrics)
        } else {
            None
        }
    }
}

/// Performance metrics extracted from llama.cpp output
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerfMetrics {
    /// Tokens generated per second (generation speed)
    pub tokens_per_second: f64,
    /// Time to first token in milliseconds (prompt processing time)
    pub time_to_first_token_ms: f64,
    /// Total inference time in milliseconds
    pub total_time_ms: f64,
}

/// Extract time in milliseconds from a llama.cpp timing line
fn extract_time_ms(line: &str) -> Option<f64> {
    // Line format: "llama_print_timings:        eval time =    2000.00 ms / ..."
    line.split('=')
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Extract tokens per second from a llama.cpp timing line
fn extract_tokens_per_second(line: &str) -> Option<f64> {
    // Line format: "... (   40.82 ms per token,    24.50 tokens per second)"
    if let Some(tps_part) = line.split("tokens per second").next() {
        // Get the last number before "tokens per second"
        tps_part.split(',').next_back()?.trim().parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics() {
        let output = r#"
llama_print_timings:        load time =     123.45 ms
llama_print_timings:      sample time =      10.23 ms /    50 runs   (    0.20 ms per token,  4882.81 tokens per second)
llama_print_timings: prompt eval time =     234.56 ms /    10 tokens (   23.46 ms per token,    42.63 tokens per second)
llama_print_timings:        eval time =    2000.00 ms /    49 runs   (   40.82 ms per token,    24.50 tokens per second)
llama_print_timings:       total time =    2368.24 ms /    59 tokens
        "#;

        let runner = SpeculativeRunner::new("/path/to/llama.cpp");
        let metrics = runner.parse_perf_metrics(output).unwrap();

        assert_eq!(metrics.tokens_per_second, 24.50);
        assert_eq!(metrics.time_to_first_token_ms, 234.56);
        assert_eq!(metrics.total_time_ms, 2368.24);
    }

    #[test]
    fn test_parse_metrics_partial() {
        let output = r#"
llama_print_timings:        eval time =    1500.00 ms /    30 runs   (   50.00 ms per token,    20.00 tokens per second)
        "#;

        let runner = SpeculativeRunner::new("/path/to/llama.cpp");
        let metrics = runner.parse_perf_metrics(output).unwrap();

        assert_eq!(metrics.tokens_per_second, 20.0);
        assert_eq!(metrics.time_to_first_token_ms, 0.0); // Not present in output
    }

    #[test]
    fn test_parse_metrics_empty() {
        let output = "No timing information here";

        let runner = SpeculativeRunner::new("/path/to/llama.cpp");
        let result = runner.parse_perf_metrics(output);

        assert!(result.is_none());
    }

    #[test]
    fn test_extract_time_ms() {
        let line = "llama_print_timings:        eval time =    2000.00 ms /    49 runs";
        assert_eq!(extract_time_ms(line), Some(2000.00));

        let line2 = "llama_print_timings: prompt eval time =     234.56 ms /    10 tokens";
        assert_eq!(extract_time_ms(line2), Some(234.56));
    }

    #[test]
    fn test_extract_tokens_per_second() {
        let line = "llama_print_timings:        eval time =    2000.00 ms /    49 runs   (   40.82 ms per token,    24.50 tokens per second)";
        assert_eq!(extract_tokens_per_second(line), Some(24.50));
    }

    #[test]
    fn test_speculative_config_serialization() {
        let config = SpeculativeConfig {
            main_model_path: "/models/llama-3.1-8b.gguf".to_string(),
            draft_model_path: "/models/llama-3.2-1b.gguf".to_string(),
            n_draft: 8,
            n_predict: 100,
            prompt: "Hello world".to_string(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SpeculativeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.main_model_path, deserialized.main_model_path);
        assert_eq!(config.n_draft, deserialized.n_draft);
    }
}
