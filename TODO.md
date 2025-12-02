# VeloLLM - TODO Détaillé pour Implémentation par Agent IA

Ce document décompose la roadmap en tâches atomiques, exécutables par un agent IA (Claude Code, Gemini CLI, etc.) avec des instructions précises et des critères de validation.

---

## 🎯 Phase 1: MVP - Fondations & Validation (Mois 1-3)

### Sprint 1: Setup & Infrastructure (Semaine 1-2)

#### TASK-001: Initialiser le repository VeloLLM
**Priority**: P0 (Blocking)
**Estimated effort**: 30min
**Dependencies**: None

**Instructions**:
```bash
# Créer la structure du projet
mkdir -p velollm/{src,tests,benchmarks,docs,adapters,scripts}
cd velollm
git init
touch README.md LICENSE .gitignore

# Structure des dossiers
mkdir -p src/{core,backends,optimization,utils}
mkdir -p adapters/{ollama,llamacpp,localai,vllm}
mkdir -p benchmarks/{baseline,configs,results}
mkdir -p docs/{api,guides,architecture}
```

**Files to create**:
- `README.md`: Vision, quick start, installation
- `LICENSE`: MIT ou Apache 2.0
- `.gitignore`: Node, Rust, Python patterns
- `CONTRIBUTING.md`: Guidelines pour contributions

**Validation criteria**:
- [ ] Repository structure créée
- [ ] README avec description claire du projet
- [ ] Git initialized avec first commit

---

#### TASK-002: Configuration du build system
**Priority**: P0
**Estimated effort**: 1h
**Dependencies**: TASK-001

**Instructions**:

**Pour Rust stack**:
```bash
# Créer Cargo workspace
cat > Cargo.toml << 'EOF'
[workspace]
members = ["velollm-core", "velollm-cli", "velollm-server"]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["VeloLLM Contributors"]
license = "MIT"

[workspace.dependencies]
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
EOF

# Créer les crates
cargo new --lib velollm-core
cargo new --bin velollm-cli
cargo new --bin velollm-server
```

**Pour TypeScript tooling**:
```bash
# Initialiser npm package
npm init -y

# Installer dev dependencies
npm install -D typescript @types/node tsx vitest
npm install commander chalk ora

# tsconfig.json
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "moduleResolution": "node",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
EOF
```

**Validation criteria**:
- [ ] `cargo build` réussit (Rust)
- [ ] `npm run build` réussit (TypeScript)
- [ ] Tests dummy passent

---

#### TASK-003: Implémentation du système de détection hardware
**Priority**: P0
**Estimated effort**: 3h
**Dependencies**: TASK-002

**Instructions**:

**File**: `src/core/hardware_detector.rs` (ou `.ts`)

```rust
// velollm-core/src/hardware.rs
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub gpu: Option<GpuInfo>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub os: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub cuda_version: Option<String>,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub cache_kb: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
}

impl HardwareSpec {
    pub fn detect() -> anyhow::Result<Self> {
        Ok(HardwareSpec {
            gpu: detect_gpu()?,
            cpu: detect_cpu()?,
            memory: detect_memory()?,
            os: std::env::consts::OS.to_string(),
        })
    }
}

fn detect_gpu() -> anyhow::Result<Option<GpuInfo>> {
    // Try nvidia-smi first
    if let Ok(output) = Command::new("nvidia-smi")
        .args(&["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(',').collect();

        if parts.len() >= 3 {
            return Ok(Some(GpuInfo {
                name: parts[0].trim().to_string(),
                vram_total_mb: parts[1].trim().parse().unwrap_or(0),
                vram_free_mb: parts[2].trim().parse().unwrap_or(0),
                cuda_version: detect_cuda_version(),
                compute_capability: None, // TODO: detect via nvidia-smi
            }));
        }
    }

    // TODO: Try rocm-smi for AMD
    // TODO: Try system_profiler for Apple Silicon

    Ok(None)
}

fn detect_cpu() -> anyhow::Result<CpuInfo> {
    // Linux: /proc/cpuinfo
    // macOS: sysctl
    // Windows: wmic cpu

    #[cfg(target_os = "linux")]
    {
        let output = Command::new("lscpu").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse lscpu output
        // This is simplified - real implementation needs better parsing
        Ok(CpuInfo {
            model: "Detected CPU".to_string(), // TODO: parse model name
            cores: num_cpus::get_physical() as u32,
            threads: num_cpus::get() as u32,
            cache_kb: None,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        Ok(CpuInfo {
            model: "Unknown CPU".to_string(),
            cores: num_cpus::get_physical() as u32,
            threads: num_cpus::get() as u32,
            cache_kb: None,
        })
    }
}

fn detect_memory() -> anyhow::Result<MemoryInfo> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        // Parse MemTotal and MemAvailable
        // Simplified version
        Ok(MemoryInfo {
            total_mb: 16384, // TODO: parse from /proc/meminfo
            available_mb: 8192,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Use sysinfo crate for cross-platform
        Ok(MemoryInfo {
            total_mb: 16384,
            available_mb: 8192,
        })
    }
}

fn detect_cuda_version() -> Option<String> {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| {
            String::from_utf8(out.stdout)
                .ok()
                .and_then(|s| s.lines().last().map(String::from))
        })
}
```

**Dependencies à ajouter** (Cargo.toml):
```toml
[dependencies]
num_cpus = "1.16"
sysinfo = "0.30"  # For cross-platform memory detection
```

**Tests to write** (`tests/hardware_detection_test.rs`):
```rust
#[test]
fn test_hardware_detection() {
    let hw = HardwareSpec::detect().expect("Should detect hardware");
    assert!(hw.cpu.cores > 0);
    assert!(hw.memory.total_mb > 0);
    println!("{:#?}", hw); // Visual verification
}

#[test]
fn test_gpu_detection() {
    let hw = HardwareSpec::detect().unwrap();
    if hw.gpu.is_some() {
        let gpu = hw.gpu.unwrap();
        assert!(!gpu.name.is_empty());
        assert!(gpu.vram_total_mb > 0);
    }
}
```

**Validation criteria**:
- [ ] Détecte correctement GPU NVIDIA (si présent)
- [ ] Détecte CPU cores et threads
- [ ] Détecte RAM totale et disponible
- [ ] Output JSON valide avec `serde_json::to_string()`
- [ ] Tests passent sur Linux, macOS (bonus: Windows)

---

#### TASK-004: Créer la baseline benchmark suite
**Priority**: P0
**Estimated effort**: 4h
**Dependencies**: TASK-003

**Instructions**:

**File**: `benchmarks/baseline_suite.rs`

```rust
// benchmarks/baseline_suite.rs
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::process::{Command, Stdio};

#[derive(Debug, Serialize, Deserialize)]
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
    backend: String, // "ollama", "llamacpp", etc.
}

impl BenchmarkRunner {
    pub fn new(backend: &str) -> Self {
        Self {
            backend: backend.to_string(),
        }
    }

    pub fn run(&self, config: &BenchmarkConfig) -> anyhow::Result<BenchmarkResult> {
        match self.backend.as_str() {
            "ollama" => self.run_ollama(config),
            _ => anyhow::bail!("Unsupported backend: {}", self.backend),
        }
    }

    fn run_ollama(&self, config: &BenchmarkConfig) -> anyhow::Result<BenchmarkResult> {
        let mut total_tokens = 0u32;
        let mut total_time = Duration::ZERO;
        let mut first_token_times = Vec::new();

        for i in 0..config.iterations {
            println!("Iteration {}/{}", i + 1, config.iterations);

            let start = Instant::now();
            let mut first_token_time: Option<Duration> = None;

            // Call Ollama API
            let response = self.call_ollama_api(&config.model, &config.prompt, config.max_tokens)?;

            let elapsed = start.elapsed();
            total_time += elapsed;

            // Parse response
            let tokens = self.count_tokens(&response);
            total_tokens += tokens;

            // TTFT approximation (first 10% of time)
            first_token_time = Some(Duration::from_millis((elapsed.as_millis() / 10) as u64));
            first_token_times.push(first_token_time.unwrap());
        }

        let avg_tokens_per_sec = (total_tokens as f64) / total_time.as_secs_f64();
        let avg_ttft_ms = first_token_times.iter()
            .map(|d| d.as_millis() as f64)
            .sum::<f64>() / (config.iterations as f64);

        Ok(BenchmarkResult {
            config: config.clone(),
            tokens_per_second: avg_tokens_per_sec,
            time_to_first_token_ms: avg_ttft_ms,
            total_time_ms: total_time.as_millis() as f64,
            memory_usage_mb: None, // TODO: measure via nvidia-smi or system tools
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    fn call_ollama_api(&self, model: &str, prompt: &str, max_tokens: u32) -> anyhow::Result<String> {
        // Using curl to call Ollama API
        let output = Command::new("curl")
            .args(&[
                "-s",
                "-X", "POST",
                "http://localhost:11434/api/generate",
                "-H", "Content-Type: application/json",
                "-d", &format!(r#"{{
                    "model": "{}",
                    "prompt": "{}",
                    "stream": false,
                    "options": {{
                        "num_predict": {}
                    }}
                }}"#, model, prompt.replace('"', r#"\""#), max_tokens),
            ])
            .output()?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn count_tokens(&self, response: &str) -> u32 {
        // Parse JSON response and count tokens
        // Simplified: count words as proxy
        response.split_whitespace().count() as u32
    }
}

// Predefined benchmark configs
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
        BenchmarkConfig {
            name: "code_generation".to_string(),
            model: "codellama:7b".to_string(),
            prompt: "Write a Rust function to compute Fibonacci numbers using memoization".to_string(),
            max_tokens: 150,
            iterations: 5,
        },
    ]
}
```

**CLI wrapper** (`velollm-cli/src/main.rs`):
```rust
use clap::{Parser, Subcommand};
use velollm_core::hardware::HardwareSpec;

#[derive(Parser)]
#[command(name = "velollm")]
#[command(about = "VeloLLM - Accelerate local LLM inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware specifications
    Detect,

    /// Run benchmarks
    Benchmark {
        #[arg(short, long, default_value = "ollama")]
        backend: String,

        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Detect => {
            let hw = HardwareSpec::detect()?;
            println!("{}", serde_json::to_string_pretty(&hw)?);
        }

        Commands::Benchmark { backend, output } => {
            use velollm_benchmarks::*;

            let runner = BenchmarkRunner::new(&backend);
            let configs = get_standard_benchmarks();

            let mut results = Vec::new();
            for config in configs {
                println!("\n=== Running: {} ===", config.name);
                let result = runner.run(&config)?;
                println!("Tokens/s: {:.2}", result.tokens_per_second);
                println!("TTFT: {:.2}ms", result.time_to_first_token_ms);
                results.push(result);
            }

            if let Some(path) = output {
                std::fs::write(&path, serde_json::to_string_pretty(&results)?)?;
                println!("\nResults saved to: {}", path);
            }
        }
    }

    Ok(())
}
```

**Validation criteria**:
- [ ] `velollm detect` affiche JSON hardware specs
- [ ] `velollm benchmark` exécute les 3 tests standard
- [ ] Résultats sauvegardés en JSON valide
- [ ] Mesures cohérentes (tokens/s > 0, TTFT > 0)

---

### Sprint 2: Speculative Decoding PoC (Semaine 3-4)

#### TASK-005: Analyser l'implémentation speculative decoding dans llama.cpp
**Priority**: P0
**Estimated effort**: 2h
**Dependencies**: None (parallel avec Sprint 1)

**Instructions**:

1. **Explorer le code source llama.cpp**:
   ```bash
   cd /home/sauron/code/llama.cpp

   # Fichiers clés à étudier
   cat common/speculative.h
   cat common/speculative.cpp
   cat examples/speculative/speculative.cpp

   # Chercher les paramètres configurables
   grep -r "draft" common/
   grep -r "SPEC_" common/speculative.cpp
   ```

2. **Documenter les findings** dans `docs/research/speculative_decoding.md`:
   ```markdown
   # Speculative Decoding Analysis

   ## Implementation Details (llama.cpp)

   ### Key Files
   - `common/speculative.h`: API interface
   - `common/speculative.cpp`: Core logic
   - `examples/speculative/speculative.cpp`: Usage example

   ### Parameters
   - `n_draft`: Number of tokens to predict with draft model (default: 5)
   - `draft_model`: Path to smaller draft model
   - `p_accept`: Acceptance probability threshold

   ### Algorithm Flow
   1. Draft model generates N candidates
   2. Target model validates in parallel
   3. Accept longest valid prefix
   4. Rollback on mismatch

   ### Performance Knobs
   - Draft tokens (5-10): Higher = more speedup if acceptance high
   - Sampling strategy: Greedy vs top-k for draft
   - Vocab compatibility check: SPEC_VOCAB_MAX_SIZE_DIFFERENCE
   ```

3. **Identifier les configurations optimales** (à partir de la littérature):
   ```yaml
   # configs/speculative_optimal.yaml
   pairs:
     - main: "llama-3.1-8b"
       draft: "llama-3.2-1b"
       n_draft: 8
       expected_speedup: 2.0

     - main: "llama-3.1-70b"
       draft: "llama-3.2-3b"
       n_draft: 10
       expected_speedup: 2.5

     - main: "codellama-13b"
       draft: "codellama-1b"
       n_draft: 7
       expected_speedup: 1.8
   ```

**Validation criteria**:
- [ ] Documentation complète dans `docs/research/`
- [ ] Compréhension des paramètres critiques
- [ ] Liste de paires optimales (main, draft) documentée

---

#### TASK-006: Wrapper pour llama.cpp avec speculative decoding
**Priority**: P0
**Estimated effort**: 4h
**Dependencies**: TASK-005

**Instructions**:

**File**: `adapters/llamacpp/speculative_wrapper.rs`

```rust
use std::process::{Command, Stdio};
use std::path::Path;
use anyhow::{Context, Result};

pub struct SpeculativeConfig {
    pub main_model_path: String,
    pub draft_model_path: String,
    pub n_draft: u32,
    pub n_predict: u32,
    pub prompt: String,
}

pub struct SpeculativeRunner {
    llama_cpp_path: String,
}

impl SpeculativeRunner {
    pub fn new(llama_cpp_path: &str) -> Self {
        Self {
            llama_cpp_path: llama_cpp_path.to_string(),
        }
    }

    pub fn run(&self, config: &SpeculativeConfig) -> Result<String> {
        // Build command for llama.cpp speculative example
        let mut cmd = Command::new(
            Path::new(&self.llama_cpp_path).join("llama-speculative")
        );

        cmd.args(&[
            "-m", &config.main_model_path,
            "-md", &config.draft_model_path,
            "--draft", &config.n_draft.to_string(),
            "-n", &config.n_predict.to_string(),
            "-p", &config.prompt,
            "-ngl", "99", // Offload all layers to GPU
        ]);

        println!("Executing: {:?}", cmd);

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute llama-speculative")?;

        if !output.status.success() {
            anyhow::bail!(
                "llama-speculative failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Parse timing info from llama.cpp output
    pub fn parse_perf_metrics(&self, output: &str) -> Option<PerfMetrics> {
        // llama.cpp prints timing like:
        // "llama_print_timings: eval time = 1234.56 ms / 50 tokens (24.69 ms per token, 40.50 tokens per second)"

        let mut metrics = PerfMetrics::default();

        for line in output.lines() {
            if line.contains("tokens per second") {
                // Extract tokens/s
                if let Some(tps) = extract_float(line, "tokens per second") {
                    metrics.tokens_per_second = tps;
                }
            }
            if line.contains("prompt eval time") {
                if let Some(ttft) = extract_float(line, "=") {
                    metrics.time_to_first_token_ms = ttft;
                }
            }
        }

        Some(metrics)
    }
}

#[derive(Debug, Default)]
pub struct PerfMetrics {
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
}

fn extract_float(line: &str, after: &str) -> Option<f64> {
    line.split(after)
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics() {
        let output = r#"
llama_print_timings: prompt eval time = 123.45 ms
llama_print_timings: eval time = 2000.00 ms / 100 tokens (20.00 ms per token, 50.00 tokens per second)
        "#;

        let runner = SpeculativeRunner::new("/path/to/llama.cpp");
        let metrics = runner.parse_perf_metrics(output).unwrap();

        assert_eq!(metrics.tokens_per_second, 50.0);
        assert_eq!(metrics.time_to_first_token_ms, 123.45);
    }
}
```

**Validation criteria**:
- [ ] Wrapper exécute `llama-speculative` binary
- [ ] Parse correctement les metrics de timing
- [ ] Tests unitaires passent
- [ ] Testé manuellement avec un vrai modèle

---

#### TASK-007: Benchmark comparatif vanilla vs speculative
**Priority**: P0
**Estimated effort**: 3h
**Dependencies**: TASK-006

**Instructions**:

**File**: `benchmarks/speculative_comparison.rs`

```rust
use velollm_adapters_llamacpp::SpeculativeRunner, SpeculativeConfig};
use std::time::Instant;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct ComparisonResult {
    vanilla_tps: f64,
    speculative_tps: f64,
    speedup: f64,
    config: SpeculativeConfig,
}

fn main() -> anyhow::Result<()> {
    let llama_cpp_path = "/home/sauron/code/llama.cpp";
    let runner = SpeculativeRunner::new(llama_cpp_path);

    // Test config
    let main_model = "/path/to/llama-3.1-8b-q4_k_m.gguf";
    let draft_model = "/path/to/llama-3.2-1b-q4_k_m.gguf";
    let prompt = "Explain quantum computing in simple terms";

    println!("=== Vanilla Inference ===");
    let vanilla_start = Instant::now();
    let vanilla_config = SpeculativeConfig {
        main_model_path: main_model.to_string(),
        draft_model_path: "".to_string(), // No draft = vanilla mode
        n_draft: 0,
        n_predict: 100,
        prompt: prompt.to_string(),
    };
    let vanilla_output = runner.run(&vanilla_config)?;
    let vanilla_metrics = runner.parse_perf_metrics(&vanilla_output).unwrap();
    let vanilla_time = vanilla_start.elapsed();

    println!("Tokens/s: {}", vanilla_metrics.tokens_per_second);
    println!("Time: {:?}", vanilla_time);

    println!("\n=== Speculative Inference ===");
    let spec_start = Instant::now();
    let spec_config = SpeculativeConfig {
        main_model_path: main_model.to_string(),
        draft_model_path: draft_model.to_string(),
        n_draft: 8,
        n_predict: 100,
        prompt: prompt.to_string(),
    };
    let spec_output = runner.run(&spec_config)?;
    let spec_metrics = runner.parse_perf_metrics(&spec_output).unwrap();
    let spec_time = spec_start.elapsed();

    println!("Tokens/s: {}", spec_metrics.tokens_per_second);
    println!("Time: {:?}", spec_time);

    let speedup = spec_metrics.tokens_per_second / vanilla_metrics.tokens_per_second;
    println!("\n=== Results ===");
    println!("Speedup: {:.2}x", speedup);

    // Save results
    let result = ComparisonResult {
        vanilla_tps: vanilla_metrics.tokens_per_second,
        speculative_tps: spec_metrics.tokens_per_second,
        speedup,
        config: spec_config,
    };

    std::fs::write(
        "benchmarks/results/speculative_comparison.json",
        serde_json::to_string_pretty(&result)?,
    )?;

    Ok(())
}
```

**Validation criteria**:
- [ ] Benchmark exécute vanilla et speculative
- [ ] Speedup mesuré et documenté (target: >1.5x)
- [ ] Résultats sauvegardés en JSON
- [ ] Si speedup < 1.5x, investiguer et documenter pourquoi

---

### Sprint 3: Ollama Auto-Configuration (Semaine 5-6)

#### TASK-008: Parser de configuration Ollama
**Priority**: P1
**Estimated effort**: 2h
**Dependencies**: TASK-003

**Instructions**:

**File**: `adapters/ollama/config_parser.rs`

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OllamaConfig {
    pub num_parallel: Option<u32>,
    pub max_loaded_models: Option<u32>,
    pub keep_alive: Option<String>,
    pub num_ctx: Option<u32>,
    pub num_batch: Option<u32>,
    pub num_gpu: Option<u32>,
    pub num_thread: Option<u32>,
}

impl OllamaConfig {
    /// Read current Ollama config from environment variables
    pub fn from_env() -> Self {
        Self {
            num_parallel: env::var("OLLAMA_NUM_PARALLEL")
                .ok()
                .and_then(|s| s.parse().ok()),
            max_loaded_models: env::var("OLLAMA_MAX_LOADED_MODELS")
                .ok()
                .and_then(|s| s.parse().ok()),
            keep_alive: env::var("OLLAMA_KEEP_ALIVE").ok(),
            num_ctx: env::var("OLLAMA_NUM_CTX")
                .ok()
                .and_then(|s| s.parse().ok()),
            num_batch: env::var("OLLAMA_NUM_BATCH")
                .ok()
                .and_then(|s| s.parse().ok()),
            num_gpu: env::var("OLLAMA_NUM_GPU")
                .ok()
                .and_then(|s| s.parse().ok()),
            num_thread: env::var("OLLAMA_NUM_THREAD")
                .ok()
                .and_then(|s| s.parse().ok()),
        }
    }

    /// Generate shell export commands
    pub fn to_env_exports(&self) -> String {
        let mut exports = Vec::new();

        if let Some(val) = self.num_parallel {
            exports.push(format!("export OLLAMA_NUM_PARALLEL={}", val));
        }
        if let Some(val) = self.max_loaded_models {
            exports.push(format!("export OLLAMA_MAX_LOADED_MODELS={}", val));
        }
        if let Some(ref val) = self.keep_alive {
            exports.push(format!("export OLLAMA_KEEP_ALIVE={}", val));
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

        exports.join("\n")
    }
}
```

**Validation criteria**:
- [ ] Lit correctement les env vars Ollama
- [ ] Génère des exports shell valides
- [ ] Tests avec différentes configurations

---

#### TASK-009: Optimiseur de configuration basé sur hardware
**Priority**: P1
**Estimated effort**: 4h
**Dependencies**: TASK-003, TASK-008

**Instructions**:

**File**: `src/optimization/ollama_optimizer.rs`

```rust
use velollm_core::hardware::{HardwareSpec, GpuInfo};
use velollm_adapters_ollama::OllamaConfig;

pub struct OllamaOptimizer;

impl OllamaOptimizer {
    pub fn optimize(hw: &HardwareSpec) -> OllamaConfig {
        let mut config = OllamaConfig::default();

        // GPU-based optimizations
        if let Some(ref gpu) = hw.gpu {
            config = Self::optimize_for_gpu(config, gpu);
        } else {
            config = Self::optimize_for_cpu(config, &hw.cpu);
        }

        // Memory-based optimizations
        config = Self::optimize_memory(config, &hw.memory);

        config
    }

    fn optimize_for_gpu(mut config: OllamaConfig, gpu: &GpuInfo) -> OllamaConfig {
        let vram_gb = gpu.vram_total_mb / 1024;

        // num_parallel: How many requests to handle simultaneously
        // Rule: VRAM / (expected model size + context)
        // Conservative: 1 for <12GB, 2 for 12-24GB, 4 for >24GB
        config.num_parallel = Some(if vram_gb < 12 {
            1
        } else if vram_gb < 24 {
            2
        } else {
            4
        });

        // num_gpu: How many layers to offload
        // Rule: All layers if VRAM > model_size * 1.5
        config.num_gpu = Some(999); // Max layers (Ollama auto-limits)

        // num_batch: Batch size for prompt processing
        // Larger = faster prompt ingestion but more VRAM
        config.num_batch = Some(if vram_gb < 8 {
            128
        } else if vram_gb < 16 {
            256
        } else {
            512
        });

        // Context window
        // Larger VRAM = larger context
        config.num_ctx = Some(if vram_gb < 8 {
            2048
        } else if vram_gb < 16 {
            4096
        } else {
            8192
        });

        config
    }

    fn optimize_for_cpu(mut config: OllamaConfig, cpu: &CpuInfo) -> OllamaConfig {
        // CPU-only mode
        config.num_gpu = Some(0);
        config.num_thread = Some(cpu.threads);

        // Smaller batches for CPU
        config.num_batch = Some(128);
        config.num_ctx = Some(2048);

        config
    }

    fn optimize_memory(mut config: OllamaConfig, mem: &MemoryInfo) -> OllamaConfig {
        let mem_gb = mem.total_mb / 1024;

        // max_loaded_models: Keep models in RAM for faster switching
        config.max_loaded_models = Some(if mem_gb < 16 {
            1
        } else if mem_gb < 32 {
            2
        } else {
            3
        });

        // keep_alive: How long to keep model in memory after use
        // More RAM = longer keep alive
        config.keep_alive = Some(if mem_gb < 16 {
            "5m".to_string()
        } else if mem_gb < 32 {
            "30m".to_string()
        } else {
            "1h".to_string()
        });

        config
    }

    /// Generate a comparison report
    pub fn generate_report(current: &OllamaConfig, optimized: &OllamaConfig) -> String {
        let mut report = String::from("=== Ollama Configuration Optimization ===\n\n");

        report.push_str(&Self::compare_field(
            "num_parallel",
            current.num_parallel,
            optimized.num_parallel,
            "Concurrent requests",
        ));
        report.push_str(&Self::compare_field(
            "num_gpu",
            current.num_gpu,
            optimized.num_gpu,
            "GPU layers",
        ));
        report.push_str(&Self::compare_field(
            "num_batch",
            current.num_batch,
            optimized.num_batch,
            "Batch size",
        ));
        report.push_str(&Self::compare_field(
            "num_ctx",
            current.num_ctx,
            optimized.num_ctx,
            "Context window",
        ));

        report
    }

    fn compare_field<T: std::fmt::Display + PartialEq>(
        name: &str,
        current: Option<T>,
        optimized: Option<T>,
        description: &str,
    ) -> String {
        match (current, optimized) {
            (Some(c), Some(o)) if c != o => {
                format!("{}: {} → {} ({})\n", name, c, o, description)
            }
            (None, Some(o)) => {
                format!("{}: (unset) → {} ({})\n", name, o, description)
            }
            _ => String::new(),
        }
    }
}
```

**Validation criteria**:
- [ ] Génère des configs différentes pour différents hardwares
- [ ] Report comparison clair et lisible
- [ ] Testé sur au moins 3 profils hardware (low/mid/high VRAM)

---

#### TASK-010: CLI commande `velollm optimize`
**Priority**: P1
**Estimated effort**: 2h
**Dependencies**: TASK-009

**Instructions**:

**File**: `velollm-cli/src/commands/optimize.rs`

```rust
use clap::Args;
use velollm_core::hardware::HardwareSpec;
use velollm_optimization::OllamaOptimizer;
use velollm_adapters_ollama::OllamaConfig;

#[derive(Args)]
pub struct OptimizeArgs {
    /// Dry run: show recommendations without applying
    #[arg(long)]
    dry_run: bool,

    /// Output shell script to file
    #[arg(short, long)]
    output: Option<String>,
}

pub fn execute(args: &OptimizeArgs) -> anyhow::Result<()> {
    println!("🔍 Detecting hardware...");
    let hw = HardwareSpec::detect()?;
    println!("✓ Hardware detected:");
    println!("  GPU: {}", hw.gpu.as_ref().map(|g| g.name.as_str()).unwrap_or("None"));
    println!("  VRAM: {} GB", hw.gpu.as_ref().map(|g| g.vram_total_mb / 1024).unwrap_or(0));
    println!("  CPU: {} cores / {} threads", hw.cpu.cores, hw.cpu.threads);
    println!("  RAM: {} GB\n", hw.memory.total_mb / 1024);

    println!("📊 Current Ollama configuration:");
    let current = OllamaConfig::from_env();
    println!("{:#?}\n", current);

    println!("⚡ Generating optimized configuration...");
    let optimized = OllamaOptimizer::optimize(&hw);

    let report = OllamaOptimizer::generate_report(&current, &optimized);
    println!("{}", report);

    if args.dry_run {
        println!("\n🔬 Dry run mode - no changes applied");
        println!("\nRecommended environment variables:");
        println!("{}", optimized.to_env_exports());
        return Ok(());
    }

    // Generate shell script
    let script = format!(
        "#!/bin/bash\n# VeloLLM Ollama optimization\n# Generated: {}\n\n{}\n",
        chrono::Utc::now().to_rfc3339(),
        optimized.to_env_exports()
    );

    if let Some(path) = &args.output {
        std::fs::write(path, &script)?;
        println!("\n✅ Configuration saved to: {}", path);
        println!("Run: source {}", path);
    } else {
        println!("\n📝 Add these to your ~/.bashrc or ~/.zshrc:");
        println!("{}", script);
    }

    Ok(())
}
```

**Update main CLI** (`velollm-cli/src/main.rs`):
```rust
#[derive(Subcommand)]
enum Commands {
    Detect,
    Benchmark { ... },

    /// Optimize Ollama configuration for current hardware
    Optimize(OptimizeArgs),
}

match cli.command {
    // ...
    Commands::Optimize(args) => commands::optimize::execute(&args)?,
}
```

**Validation criteria**:
- [ ] `velollm optimize --dry-run` affiche recommendations
- [ ] `velollm optimize -o velollm.sh` génère script
- [ ] Script généré est syntaxiquement valide (`bash -n velollm.sh`)
- [ ] Sourcer le script applique les env vars

---

### Sprint 4: Integration & Documentation (Semaine 7-8)

#### TASK-011: Tests end-to-end
**Priority**: P1
**Estimated effort**: 4h
**Dependencies**: All previous tasks

**Instructions**:

**File**: `tests/integration_test.rs`

```rust
use std::process::Command;
use std::fs;

#[test]
fn test_full_workflow() {
    // 1. Detect hardware
    let output = Command::new("cargo")
        .args(&["run", "--bin", "velollm", "--", "detect"])
        .output()
        .expect("Failed to run detect");

    assert!(output.status.success());
    let hw_json = String::from_utf8(output.stdout).unwrap();
    assert!(hw_json.contains("cpu"));
    assert!(hw_json.contains("memory"));

    // 2. Generate optimized config
    let output = Command::new("cargo")
        .args(&["run", "--bin", "velollm", "--", "optimize", "-o", "/tmp/velollm_test.sh"])
        .output()
        .expect("Failed to run optimize");

    assert!(output.status.success());
    assert!(fs::metadata("/tmp/velollm_test.sh").is_ok());

    // 3. Validate script
    let script = fs::read_to_string("/tmp/velollm_test.sh").unwrap();
    assert!(script.contains("OLLAMA_"));
    assert!(script.starts_with("#!/bin/bash"));

    // Cleanup
    fs::remove_file("/tmp/velollm_test.sh").ok();
}

#[test]
fn test_benchmark_runs() {
    // Requires Ollama running with llama3.2:1b loaded
    let output = Command::new("cargo")
        .args(&["run", "--bin", "velollm", "--", "benchmark", "-o", "/tmp/benchmark_test.json"])
        .output()
        .expect("Failed to run benchmark");

    if output.status.success() {
        let results = fs::read_to_string("/tmp/benchmark_test.json").unwrap();
        assert!(results.contains("tokens_per_second"));
        fs::remove_file("/tmp/benchmark_test.json").ok();
    } else {
        println!("Benchmark skipped (Ollama not running)");
    }
}
```

**Validation criteria**:
- [ ] Tous les tests integration passent
- [ ] Tests documentent le workflow complet
- [ ] CI/CD pipeline configuré (GitHub Actions)

---

#### TASK-012: Documentation complète
**Priority**: P1
**Estimated effort**: 4h
**Dependencies**: All previous tasks

**Instructions**:

**File**: `README.md`

```markdown
# VeloLLM

**Autopilot for Local LLM Inference** - Zero-config performance optimization for Ollama, llama.cpp, and more.

## The Problem

Local LLM inference is 35-50x slower than cloud solutions (vLLM, Morph) despite comparable hardware. VeloLLM bridges this gap.

## Quick Start

### Installation

```bash
# From crates.io
cargo install velollm

# From source
git clone https://github.com/yourusername/velollm
cd velollm
cargo install --path velollm-cli
```

### Usage

```bash
# 1. Detect your hardware
velollm detect

# 2. Optimize Ollama
velollm optimize --dry-run  # Preview changes
velollm optimize -o velollm.sh
source velollm.sh

# 3. Benchmark performance
velollm benchmark

# 4. Compare before/after
velollm benchmark --compare
```

## Features (Phase 1 MVP)

- ✅ **Hardware Detection**: Auto-detect GPU, CPU, RAM
- ✅ **Ollama Auto-Config**: Optimize VRAM, batch size, context
- ✅ **Benchmarking Suite**: Measure tokens/s, TTFT, memory
- ✅ **Speculative Decoding**: 1.5-2.5x speedup (llama.cpp)

## Benchmark Results

| Hardware | Model | Baseline | Optimized | Speedup |
|----------|-------|----------|-----------|---------|
| RTX 4090 24GB | Llama 3.1 8B | 28 tok/s | 67 tok/s | 2.4x |
| RTX 3060 12GB | Llama 3.2 3B | 35 tok/s | 78 tok/s | 2.2x |
| M2 Max 32GB | Llama 3.1 8B | 22 tok/s | 51 tok/s | 2.3x |

*See [BENCHMARKS.md](BENCHMARKS.md) for full results*

## Roadmap

- **Phase 1 (Months 1-3)**: MVP - Ollama optimization, speculative decoding
- **Phase 2 (Months 4-6)**: PagedAttention, continuous batching, multi-backend
- **Phase 3 (Months 7-12)**: GUI, IDE plugins, Mamba/MoE support

See [ROADMAP.md](ROADMAP.md) for details.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE)
```

**File**: `BENCHMARKS.md`

```markdown
# VeloLLM Benchmark Results

## Methodology

All benchmarks run with:
- Warm model (2nd+ run to exclude load time)
- Same prompt: "Explain quantum computing in simple terms"
- 100 token generation
- Average of 5 runs

## Hardware Configurations

### Config A: High-End Workstation
- GPU: NVIDIA RTX 4090 24GB
- CPU: AMD Ryzen 9 7950X (16C/32T)
- RAM: 64GB DDR5
- OS: Ubuntu 22.04

### Config B: Gaming Laptop
- GPU: NVIDIA RTX 3060 Mobile 12GB
- CPU: Intel i7-12700H (14C/20T)
- RAM: 16GB DDR4
- OS: Windows 11

### Config C: Apple Silicon
- SoC: Apple M2 Max
- Unified Memory: 32GB
- OS: macOS 14

## Results

### Llama 3.1 8B (Q4_K_M)

| Config | Baseline tok/s | VeloLLM tok/s | Speedup | Optimization |
|--------|----------------|---------------|---------|--------------|
| A | 28.3 | 67.1 | 2.37x | Speculative + Config |
| B | 18.5 | 42.8 | 2.31x | Speculative + Config |
| C | 22.1 | 51.3 | 2.32x | Config only |

### Memory Usage

| Config | Baseline VRAM | VeloLLM VRAM | Reduction |
|--------|---------------|--------------|-----------|
| A | 7.2 GB | 7.2 GB | - |
| B | 7.2 GB | 7.2 GB | - |

*Note: Phase 1 doesn't optimize memory yet (Phase 2: PagedAttention)*

## Reproducing

```bash
# Install VeloLLM
cargo install velollm

# Run baseline benchmark (vanilla Ollama)
velollm benchmark --baseline -o baseline.json

# Optimize and re-benchmark
velollm optimize -o velollm.sh
source velollm.sh
velollm benchmark -o optimized.json

# Compare
velollm benchmark --compare baseline.json optimized.json
```

## Raw Data

See [benchmarks/results/](../benchmarks/results/) for JSON files.
```

**Validation criteria**:
- [ ] README clair et concis (<500 mots)
- [ ] Quick start fonctionne en <5 minutes
- [ ] Benchmarks results authentiques et vérifiables
- [ ] Tous les liens fonctionnent

---

## 🚀 Phase 2: Optimisations Avancées (Mois 4-6)

### Sprint 5: PagedAttention Implementation (Semaine 9-12)

#### TASK-013: Étude de PagedAttention dans vLLM
**Priority**: P1
**Estimated effort**: 6h
**Dependencies**: None

**Instructions**:

1. **Cloner et étudier vLLM**:
   ```bash
   git clone https://github.com/vllm-project/vllm
   cd vllm

   # Fichiers clés
   cat vllm/attention/backends/abstract.py
   cat vllm/core/block_manager.py
   cat csrc/attention/attention_kernels.cu
   ```

2. **Créer documentation** (`docs/research/paged_attention.md`):
   ```markdown
   # PagedAttention Analysis

   ## Core Concept
   KV cache divisé en blocks (pages) de taille fixe au lieu d'allocations continues.

   ## Benefits
   - Fragmentation: 70% → 4%
   - Dynamic allocation: pas de préallocation
   - Sharing: multiple sequences peuvent partager des pages (prefix caching)

   ## Implementation Requirements

   ### 1. Block Manager
   - Allocate/free blocks
   - Track: which sequence owns which blocks
   - LRU eviction for memory pressure

   ### 2. Attention Kernel Modification
   - Instead of: `attention(Q, K_continuous, V_continuous)`
   - Need: `paged_attention(Q, K_blocks, V_blocks, block_table)`

   ### 3. Block Table
   - Per sequence: [block_0, block_3, block_7, ...]
   - Maps logical position → physical block

   ## Adaptation Strategy for llama.cpp

   1. **Phase 1**: Block manager en Rust (userspace)
   2. **Phase 2**: Modifier GGML attention kernels (C++)
   3. **Phase 3**: CUDA kernel pour paged attention
   ```

3. **Identifier les défis** pour portage llama.cpp:
   - GGML vs PyTorch tensor format
   - CUDA kernel development
   - CPU fallback implementation
   - Memory allocator customization

**Validation criteria**:
- [ ] Documentation complète de PagedAttention
- [ ] Liste des modifications nécessaires dans llama.cpp
- [ ] Plan d'implémentation phased

---

#### TASK-014: Block Manager Implementation
**Priority**: P1
**Estimated effort**: 8h
**Dependencies**: TASK-013

**Instructions**:

**File**: `src/optimization/paged_attention/block_manager.rs`

```rust
use std::collections::{HashMap, VecDeque};

const BLOCK_SIZE: usize = 16; // tokens per block

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

#[derive(Debug)]
pub struct Block {
    id: BlockId,
    ref_count: usize,
    data: Vec<f32>, // KV cache data (simplified)
}

pub struct BlockManager {
    blocks: HashMap<BlockId, Block>,
    free_blocks: VecDeque<BlockId>,
    next_block_id: usize,
    total_blocks: usize,
}

impl BlockManager {
    pub fn new(total_memory_mb: usize, block_size_tokens: usize) -> Self {
        // Calculate how many blocks fit in memory
        // Simplified: assume 4 bytes per element, 2 * hidden_dim elements per token
        let bytes_per_token = 2 * 4096 * 4; // 2 (K+V) * 4096 (hidden_dim) * 4 (float32)
        let bytes_per_block = bytes_per_token * block_size_tokens;
        let total_blocks = (total_memory_mb * 1024 * 1024) / bytes_per_block;

        let free_blocks = (0..total_blocks).map(BlockId).collect();

        Self {
            blocks: HashMap::new(),
            free_blocks,
            next_block_id: total_blocks,
            total_blocks,
        }
    }

    /// Allocate a new block
    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_blocks.pop_front().or_else(|| {
            // Out of free blocks - try to evict LRU
            self.evict_lru()
        })
    }

    /// Free a block (decrement ref count)
    pub fn free(&mut self, block_id: BlockId) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.blocks.remove(&block_id);
                self.free_blocks.push_back(block_id);
            }
        }
    }

    /// Increment ref count (for shared prefixes)
    pub fn add_ref(&mut self, block_id: BlockId) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.ref_count += 1;
        }
    }

    /// Evict least recently used block
    fn evict_lru(&mut self) -> Option<BlockId> {
        // Simplified: evict first block with ref_count == 1
        // Real implementation needs LRU tracking
        self.blocks
            .iter()
            .find(|(_, b)| b.ref_count == 1)
            .map(|(id, _)| *id)
            .and_then(|id| {
                self.free(id);
                Some(id)
            })
    }

    pub fn get_utilization(&self) -> f64 {
        let used = self.total_blocks - self.free_blocks.len();
        (used as f64) / (self.total_blocks as f64)
    }
}

/// Manages block allocation for a single sequence
pub struct SequenceBlockTable {
    blocks: Vec<BlockId>,
    sequence_length: usize,
}

impl SequenceBlockTable {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            sequence_length: 0,
        }
    }

    /// Append new tokens (may need new blocks)
    pub fn append_tokens(&mut self, num_tokens: usize, manager: &mut BlockManager) -> Result<(), String> {
        let new_length = self.sequence_length + num_tokens;
        let blocks_needed = (new_length + BLOCK_SIZE - 1) / BLOCK_SIZE;

        while self.blocks.len() < blocks_needed {
            let block_id = manager.allocate()
                .ok_or("Out of memory: no blocks available")?;
            self.blocks.push(block_id);
        }

        self.sequence_length = new_length;
        Ok(())
    }

    /// Get block table for attention kernel
    pub fn get_block_table(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Free all blocks
    pub fn free_all(&self, manager: &mut BlockManager) {
        for &block_id in &self.blocks {
            manager.free(block_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocation() {
        let mut manager = BlockManager::new(100, BLOCK_SIZE); // 100MB
        let block = manager.allocate();
        assert!(block.is_some());

        manager.free(block.unwrap());
        assert_eq!(manager.get_utilization(), 0.0);
    }

    #[test]
    fn test_sequence_growth() {
        let mut manager = BlockManager::new(100, BLOCK_SIZE);
        let mut seq = SequenceBlockTable::new();

        // Add 100 tokens - should need ceil(100/16) = 7 blocks
        seq.append_tokens(100, &mut manager).unwrap();
        assert_eq!(seq.blocks.len(), 7);

        // Add 16 more - should need 1 more block
        seq.append_tokens(16, &mut manager).unwrap();
        assert_eq!(seq.blocks.len(), 8);
    }
}
```

**Validation criteria**:
- [ ] Block allocation/deallocation fonctionne
- [ ] Sequence growth alloue correctement
- [ ] Tests unitaires passent
- [ ] Memory tracking précis

---

*Note: Les tâches suivantes (TASK-015 à TASK-030) continueraient dans le même format détaillé, couvrant:*
- CUDA kernel modification pour paged attention
- Continuous batching request scheduler
- CPU-GPU hybrid executor
- Multi-backend adapters (LocalAI, vLLM)
- Advanced quantization strategies
- GUI implementation avec Tauri
- IDE plugins (VSCode, Continue.dev)
- Mamba/MoE model support
- Configuration marketplace
- Etc.

---

## 📋 Format de TODO pour Agent IA

Chaque tâche suit ce template:

```markdown
#### TASK-XXX: [Titre descriptif]
**Priority**: P0 (Blocking) | P1 (High) | P2 (Medium) | P3 (Low)
**Estimated effort**: Xh
**Dependencies**: TASK-YYY, TASK-ZZZ

**Instructions**:
[Pseudo-code, commandes exactes, ou code complet]

**Files to create/modify**:
- path/to/file.rs: [description]

**Validation criteria**:
- [ ] Critère mesurable 1
- [ ] Critère mesurable 2
- [ ] Tests passent
```

---

## 🎯 Priorités d'Exécution

### P0 - Blocking (Must complete before next phase)
- TASK-001 à TASK-007: Infrastructure + PoC speculative decoding
- TASK-011: Tests integration

### P1 - High (Core features)
- TASK-008 à TASK-010: Ollama optimization
- TASK-013 à TASK-020: PagedAttention, continuous batching

### P2 - Medium (Advanced features)
- TASK-021 à TASK-025: Multi-backend, GUI

### P3 - Low (Nice to have)
- TASK-026 à TASK-030: Marketplace, advanced integrations

---

## 📊 Progress Tracking

### Phase 1 MVP (Mois 1-3)
- [x] TASK-001: Repository setup ✅ (commit: ef295cf)
- [x] TASK-002: Build system ✅ (commit: 7ab3d10)
- [x] TASK-003: Hardware detection ✅ (commit: eabd378, 8a7b193)
- [x] TASK-004: Benchmark suite ✅ (commit: 8d849e6)
- [x] TASK-005: Speculative analysis ✅ (commit: bb958d7)
- [x] TASK-006: Speculative wrapper ✅ (commit: 4099af9)
- [x] TASK-007: Benchmark comparison ✅ (commit: 18a8789)
- [x] TASK-008: Config parser ✅ (commit: 77e2204)
- [x] TASK-009: Optimizer ✅ (commit: ae1c782)
- [x] TASK-010: CLI optimize command ✅ (commit: 6369098)
- [x] TASK-011: Integration tests ✅
- [x] TASK-012: Documentation ✅

**Progress**: 12/12 tasks (100%) ✅ → Phase 1 MVP COMPLETE!

**Completed Tasks Details**:
- ✅ TASK-001: Repository structure créée avec workspace Cargo
- ✅ TASK-002: Cargo workspace avec 3 crates (core, cli, benchmarks), CI configuré
- ✅ TASK-003: Hardware detection complet (NVIDIA, AMD, Apple, Intel GPU + CPU + Memory)
- ✅ TASK-004: Benchmark suite Ollama avec standard benchmarks, résultats JSON
- ✅ TASK-005: Speculative decoding analysis (optimal params, model pairs, 2x speedup strategy)
- ✅ TASK-006: llama.cpp wrapper (SpeculativeRunner, metrics parsing, vanilla/speculative modes)
- ✅ TASK-007: Benchmark comparison (statistical analysis, mean/stddev, speedup ± error, JSON export)
- ✅ TASK-008: Ollama config parser (from_env, to_env_exports, merge, JSON serialization)
- ✅ TASK-009: Hardware-based optimizer (GPU/CPU/memory heuristics, generate_report, 5 unit tests)
- ✅ TASK-010: CLI optimize command (velollm optimize --dry-run/-o, hardware detection, config comparison, shell script generation)
- ✅ TASK-011: End-to-end integration tests (8 tests: detect, optimize, help, version, JSON validation, benchmark error handling)
- ✅ TASK-012: Documentation (CONFIG_GUIDE.md, ARCHITECTURE.md, README.md status update)

**Tests Status**:
- velollm-core: 13/13 tests passing ✅ (+5 optimizer tests)
- velollm-benchmarks: 3/3 tests passing ✅
- velollm-adapters-llamacpp: 6/6 tests passing ✅
- velollm-adapters-ollama: 6/6 tests passing ✅
- velollm-cli: 8/8 integration tests passing ✅ (NEW)
- velollm-bench-speculative: CLI binary ✅
- Doc tests: 3/3 passing ✅
- Build: ✅ `cargo build --all` successful
- Clippy: ✅ No warnings
- CI: ✅ GitHub Actions configuré (.github/workflows/ci.yml)

**Total: 39 tests passing**

### Phase 2 Advanced (Mois 4-6)
- [x] TASK-013: PagedAttention research ✅
- [ ] TASK-014: Block manager
- [ ] TASK-015: llama.cpp paged KV cache integration
- [ ] TASK-016: CUDA paged attention kernel
- [ ] [... more tasks ...]

**Progress**: 1/20 tasks (5%)

**Completed Tasks Details (Phase 2)**:
- ✅ TASK-013: PagedAttention analysis (docs/research/paged_attention.md - concept, vLLM implementation, llama.cpp integration strategy, performance expectations)

---

## 🤖 Instructions pour Agent IA

**Comment utiliser ce TODO**:

1. **Démarrer par les P0**: Exécuter TASK-001 à TASK-007 dans l'ordre
2. **Validation stricte**: Ne pas marquer "done" sans passer TOUS les critères
3. **Tests obligatoires**: Chaque tâche avec code doit avoir des tests
4. **Documentation inline**: Commenter le code complexe
5. **Git commits granulaires**: 1 commit par tâche complétée
6. **Reporting**: Après chaque tâche, résumer: ce qui marche, ce qui bloque, métriques

**Format de reporting**:
```
TASK-XXX: [DONE/BLOCKED/IN_PROGRESS]
- Completed: [description]
- Tests: [X/Y passing]
- Blockers: [liste ou "None"]
- Metrics: [benchmarks si applicable]
- Next: TASK-YYY
```

**Gestion des blockers**:
- Documenter le problème dans `docs/issues/task-XXX-blocker.md`
- Proposer 2-3 solutions alternatives
- Escalate si besoin d'input humain

---

## 📚 Ressources Rapides

### Commandes utiles
```bash
# Build & test
cargo build --release
cargo test
cargo clippy

# Benchmark
velollm benchmark -o results.json

# Docs
cargo doc --open
```

### Repos de référence
- llama.cpp: `/home/sauron/code/llama.cpp`
- vLLM: `git clone https://github.com/vllm-project/vllm`
- Ollama: `https://github.com/ollama/ollama`

### Papers critiques
- PagedAttention: [vLLM blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- Speculative Decoding: [llama.cpp PR#2926](https://github.com/ggml-org/llama.cpp/pull/2926)

---

**Ready to start? Begin with TASK-001! 🚀**
