// VeloLLM Benchmarking Library
//
// Placeholder - will be implemented in TASK-004

use serde::{Deserialize, Serialize};

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
    pub memory_usage_mb: Option<u64>,
    pub timestamp: String,
}

pub struct BenchmarkRunner {
    backend: String,
}

impl BenchmarkRunner {
    pub fn new(backend: &str) -> Self {
        Self {
            backend: backend.to_string(),
        }
    }

    pub fn run(&self, _config: &BenchmarkConfig) -> anyhow::Result<BenchmarkResult> {
        // TODO: Implement in TASK-004
        anyhow::bail!("Benchmarking not yet implemented")
    }
}

pub fn get_standard_benchmarks() -> Vec<BenchmarkConfig> {
    vec![
        BenchmarkConfig {
            name: "short_completion".to_string(),
            model: "llama3.2:1b".to_string(),
            prompt: "Write a hello world program in Python".to_string(),
            max_tokens: 50,
            iterations: 10,
        },
        BenchmarkConfig {
            name: "medium_completion".to_string(),
            model: "llama3.1:8b".to_string(),
            prompt: "Explain how speculative decoding works in language models".to_string(),
            max_tokens: 200,
            iterations: 5,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_configs() {
        let benchmarks = get_standard_benchmarks();
        assert_eq!(benchmarks.len(), 2);
        assert_eq!(benchmarks[0].name, "short_completion");
    }
}
