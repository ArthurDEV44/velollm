// VeloLLM Benchmarking Library
//
// Benchmark suite for measuring LLM inference performance

use anyhow::Context;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkConfig {
    pub name: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub iterations: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub total_tokens: u32,
    pub prompt_eval_count: Option<u32>,
    pub eval_count: Option<u32>,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    #[allow(dead_code)]
    model: String,
    response: String,
    #[allow(dead_code)]
    done: bool,
    #[serde(default)]
    #[allow(dead_code)]
    total_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    load_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    eval_duration: Option<u64>,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    num_predict: u32,
}

pub struct BenchmarkRunner {
    backend: String,
    ollama_url: String,
}

impl BenchmarkRunner {
    pub fn new(backend: &str) -> Self {
        Self { backend: backend.to_string(), ollama_url: "http://localhost:11434".to_string() }
    }

    pub fn with_url(mut self, url: String) -> Self {
        self.ollama_url = url;
        self
    }

    /// Run a single benchmark configuration
    pub async fn run(&self, config: &BenchmarkConfig) -> anyhow::Result<BenchmarkResult> {
        match self.backend.as_str() {
            "ollama" => self.run_ollama(config).await,
            _ => anyhow::bail!("Unsupported backend: {}", self.backend),
        }
    }

    /// Run Ollama benchmark
    async fn run_ollama(&self, config: &BenchmarkConfig) -> anyhow::Result<BenchmarkResult> {
        let client = reqwest::Client::new();
        let mut total_time_ms = 0f64;
        let mut total_tokens = 0u32;
        let mut first_token_times = Vec::new();
        let mut prompt_eval_counts = Vec::new();
        let mut eval_counts = Vec::new();

        println!("Running benchmark: {} ({} iterations)", config.name, config.iterations);

        for i in 0..config.iterations {
            print!("  Iteration {}/{}... ", i + 1, config.iterations);
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let start = Instant::now();

            let request = OllamaGenerateRequest {
                model: config.model.clone(),
                prompt: config.prompt.clone(),
                stream: false,
                options: OllamaOptions { num_predict: config.max_tokens },
            };

            let response = client
                .post(format!("{}/api/generate", self.ollama_url))
                .json(&request)
                .send()
                .await
                .context("Failed to send request to Ollama")?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await?;
                anyhow::bail!("Ollama API error ({}): {}", status, text);
            }

            let ollama_response: OllamaGenerateResponse = response
                .json()
                .await
                .context("Failed to parse Ollama response")?;

            let elapsed = start.elapsed();
            let elapsed_ms = elapsed.as_millis() as f64;
            total_time_ms += elapsed_ms;

            // Count tokens in response
            let tokens = ollama_response.eval_count.unwrap_or_else(|| {
                // Fallback: estimate from response length
                (ollama_response.response.split_whitespace().count() as f64 * 1.3) as u32
            });
            total_tokens += tokens;

            // Calculate TTFT (time to first token)
            // Estimate: prompt_eval_duration + first token generation
            let ttft_ms = if let (Some(prompt_dur), Some(eval_dur), Some(eval_count)) = (
                ollama_response.prompt_eval_duration,
                ollama_response.eval_duration,
                ollama_response.eval_count,
            ) {
                if eval_count > 0 {
                    let prompt_ms = prompt_dur as f64 / 1_000_000.0; // nanoseconds to ms
                    let per_token_ms = (eval_dur as f64 / 1_000_000.0) / eval_count as f64;
                    prompt_ms + per_token_ms
                } else {
                    elapsed_ms / 2.0 // Fallback estimate
                }
            } else {
                elapsed_ms / 10.0 // Rough estimate: 10% of total time
            };

            first_token_times.push(ttft_ms);

            if let Some(count) = ollama_response.prompt_eval_count {
                prompt_eval_counts.push(count);
            }
            if let Some(count) = ollama_response.eval_count {
                eval_counts.push(count);
            }

            let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
                tokens as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };

            println!("{:.1} tok/s ({:.0}ms)", tokens_per_sec, elapsed_ms);
        }

        let avg_tokens_per_sec = total_tokens as f64 / (total_time_ms / 1000.0);
        let avg_ttft_ms = first_token_times.iter().sum::<f64>() / config.iterations as f64;
        let avg_prompt_eval = if !prompt_eval_counts.is_empty() {
            Some(
                (prompt_eval_counts.iter().sum::<u32>() as f64 / prompt_eval_counts.len() as f64)
                    as u32,
            )
        } else {
            None
        };
        let avg_eval = if !eval_counts.is_empty() {
            Some((eval_counts.iter().sum::<u32>() as f64 / eval_counts.len() as f64) as u32)
        } else {
            None
        };

        println!("  Average: {:.1} tok/s, TTFT: {:.1}ms\n", avg_tokens_per_sec, avg_ttft_ms);

        Ok(BenchmarkResult {
            config: config.clone(),
            tokens_per_second: avg_tokens_per_sec,
            time_to_first_token_ms: avg_ttft_ms,
            total_time_ms,
            total_tokens,
            prompt_eval_count: avg_prompt_eval,
            eval_count: avg_eval,
            timestamp: Utc::now().to_rfc3339(),
        })
    }

    /// Check if Ollama is running
    pub async fn check_ollama_available(&self) -> anyhow::Result<bool> {
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/tags", self.ollama_url))
            .send()
            .await;

        Ok(response.is_ok() && response.unwrap().status().is_success())
    }
}

/// Get standard benchmark configurations
pub fn get_standard_benchmarks(model: &str) -> Vec<BenchmarkConfig> {
    vec![
        BenchmarkConfig {
            name: "short_completion".to_string(),
            model: model.to_string(),
            prompt: "Write a hello world program in Python".to_string(),
            max_tokens: 50,
            iterations: 5,
        },
        BenchmarkConfig {
            name: "medium_completion".to_string(),
            model: model.to_string(),
            prompt: "Explain how neural networks learn through backpropagation in detail"
                .to_string(),
            max_tokens: 150,
            iterations: 3,
        },
        BenchmarkConfig {
            name: "code_generation".to_string(),
            model: model.to_string(),
            prompt:
                "Write a Rust function to compute the Fibonacci sequence using dynamic programming"
                    .to_string(),
            max_tokens: 200,
            iterations: 3,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_configs() {
        let benchmarks = get_standard_benchmarks("llama3.2:3b");
        assert_eq!(benchmarks.len(), 3);
        assert_eq!(benchmarks[0].name, "short_completion");
        assert_eq!(benchmarks[0].model, "llama3.2:3b");
    }

    #[test]
    fn test_runner_creation() {
        let runner = BenchmarkRunner::new("ollama");
        assert_eq!(runner.backend, "ollama");
        assert_eq!(runner.ollama_url, "http://localhost:11434");
    }

    #[test]
    fn test_custom_url() {
        let runner = BenchmarkRunner::new("ollama").with_url("http://custom:8080".to_string());
        assert_eq!(runner.ollama_url, "http://custom:8080");
    }
}
