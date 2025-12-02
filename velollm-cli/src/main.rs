use clap::{Parser, Subcommand};
use velollm_adapters_ollama::OllamaConfig;
use velollm_benchmarks::{get_standard_benchmarks, BenchmarkRunner};
use velollm_core::hardware::HardwareSpec;
use velollm_core::optimizer::{OllamaOptimizer, OptimizedConfig};

#[derive(Parser)]
#[command(name = "velollm")]
#[command(version = "0.1.0")]
#[command(about = "VeloLLM - Autopilot for Local LLM Inference", long_about = None)]
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
        /// Backend to use (ollama, llamacpp)
        #[arg(short, long, default_value = "ollama")]
        backend: String,

        /// Model to benchmark
        #[arg(short, long, default_value = "llama3.2:3b")]
        model: String,

        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Optimize configuration for current hardware
    Optimize {
        /// Dry run: show recommendations without applying
        #[arg(long)]
        dry_run: bool,

        /// Output shell script to file
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Detect => {
            println!("🔍 Detecting hardware configuration...\n");

            let hw = HardwareSpec::detect()?;

            // Pretty print hardware info
            println!("=== System Information ===");
            println!("OS: {}", hw.os);
            println!("Platform: {}", hw.platform);
            println!();

            println!("=== CPU ===");
            println!("Model: {}", hw.cpu.model);
            println!("Cores: {}", hw.cpu.cores);
            println!("Threads: {}", hw.cpu.threads);
            if let Some(freq) = hw.cpu.frequency_mhz {
                println!("Frequency: {} MHz", freq);
            }
            println!();

            println!("=== Memory ===");
            println!(
                "Total: {} MB ({:.1} GB)",
                hw.memory.total_mb,
                hw.memory.total_mb as f64 / 1024.0
            );
            println!(
                "Available: {} MB ({:.1} GB)",
                hw.memory.available_mb,
                hw.memory.available_mb as f64 / 1024.0
            );
            println!(
                "Used: {} MB ({:.1} GB)",
                hw.memory.used_mb,
                hw.memory.used_mb as f64 / 1024.0
            );
            println!();

            if let Some(ref gpu) = hw.gpu {
                println!("=== GPU ===");
                println!("Name: {}", gpu.name);
                println!("Vendor: {:?}", gpu.vendor);
                println!(
                    "VRAM Total: {} MB ({:.1} GB)",
                    gpu.vram_total_mb,
                    gpu.vram_total_mb as f64 / 1024.0
                );
                println!(
                    "VRAM Free: {} MB ({:.1} GB)",
                    gpu.vram_free_mb,
                    gpu.vram_free_mb as f64 / 1024.0
                );
                if let Some(ref driver) = gpu.driver_version {
                    println!("Driver: {}", driver);
                }
                if let Some(ref compute) = gpu.compute_capability {
                    println!("Compute Capability: {}", compute);
                }
                println!();
            } else {
                println!("=== GPU ===");
                println!("No GPU detected (CPU-only mode)");
                println!();
            }

            println!("=== JSON Output ===");
            println!("{}", serde_json::to_string_pretty(&hw)?);
        }

        Commands::Benchmark { backend, model, output } => {
            println!("🚀 VeloLLM Benchmark Suite\n");
            println!("Backend: {}", backend);
            println!("Model: {}\n", model);

            // Check if Ollama is running
            let runner = BenchmarkRunner::new(&backend);

            print!("Checking Ollama availability... ");
            if !runner.check_ollama_available().await? {
                println!("❌");
                anyhow::bail!("Ollama is not running. Please start Ollama and ensure the model '{}' is available.", model);
            }
            println!("✓\n");

            // Get standard benchmarks
            let benchmarks = get_standard_benchmarks(&model);

            println!("Running {} benchmarks...\n", benchmarks.len());
            println!("═══════════════════════════════════════════════════════\n");

            let mut results = Vec::new();
            for config in benchmarks {
                match runner.run(&config).await {
                    Ok(result) => {
                        results.push(result);
                    }
                    Err(e) => {
                        eprintln!("❌ Benchmark '{}' failed: {}", config.name, e);
                        continue;
                    }
                }
            }

            // Summary
            println!("═══════════════════════════════════════════════════════");
            println!("\n📊 Benchmark Summary\n");

            for result in &results {
                println!("{}:", result.config.name);
                println!("  Tokens/s: {:.1}", result.tokens_per_second);
                println!("  TTFT: {:.1}ms", result.time_to_first_token_ms);
                println!("  Total tokens: {}", result.total_tokens);
                println!("  Total time: {:.1}s", result.total_time_ms / 1000.0);
                println!();
            }

            // Overall average
            if !results.is_empty() {
                let avg_tps: f64 =
                    results.iter().map(|r| r.tokens_per_second).sum::<f64>() / results.len() as f64;
                let avg_ttft: f64 = results
                    .iter()
                    .map(|r| r.time_to_first_token_ms)
                    .sum::<f64>()
                    / results.len() as f64;

                println!("Overall Average:");
                println!("  Tokens/s: {:.1}", avg_tps);
                println!("  TTFT: {:.1}ms", avg_ttft);
                println!();
            }

            // Save results if output specified
            if let Some(path) = output {
                let json = serde_json::to_string_pretty(&results)?;
                std::fs::write(&path, json)?;
                println!("✅ Results saved to: {}", path);
            } else {
                println!("💡 Tip: Use -o <file> to save results to JSON");
            }
        }

        Commands::Optimize { dry_run, output } => {
            println!("⚡ VeloLLM Ollama Optimizer\n");

            // Step 1: Detect hardware
            println!("🔍 Detecting hardware configuration...");
            let hw = HardwareSpec::detect()?;

            println!("✓ Hardware detected:");
            if let Some(ref gpu) = hw.gpu {
                println!("  GPU: {} ({:.1} GB VRAM)", gpu.name, gpu.vram_total_mb as f64 / 1024.0);
            } else {
                println!("  GPU: None (CPU-only mode)");
            }
            println!("  CPU: {} cores / {} threads", hw.cpu.cores, hw.cpu.threads);
            println!("  RAM: {:.1} GB\n", hw.memory.total_mb as f64 / 1024.0);

            // Step 2: Read current configuration from environment
            println!("📊 Reading current Ollama configuration from environment...");
            let current_env = OllamaConfig::from_env();

            if current_env.is_empty() {
                println!("  No Ollama environment variables currently set (using defaults)");
            } else {
                println!("  Current configuration:");
                if let Some(val) = current_env.num_parallel {
                    println!("    OLLAMA_NUM_PARALLEL: {}", val);
                }
                if let Some(val) = current_env.num_gpu {
                    println!("    OLLAMA_NUM_GPU: {}", val);
                }
                if let Some(val) = current_env.num_batch {
                    println!("    OLLAMA_NUM_BATCH: {}", val);
                }
                if let Some(val) = current_env.num_ctx {
                    println!("    OLLAMA_NUM_CTX: {}", val);
                }
            }
            println!();

            // Step 3: Generate optimized configuration
            println!("⚙️  Generating optimized configuration for your hardware...");
            let optimized = OllamaOptimizer::optimize(&hw);

            // Convert OptimizedConfig to OllamaConfig for comparison and export
            let optimized_ollama = optimized_config_to_ollama(&optimized);

            // Convert current env to OptimizedConfig format for comparison
            let current_optimized = ollama_to_optimized_config(&current_env);

            // Step 4: Generate comparison report
            let report = OllamaOptimizer::generate_report(&current_optimized, &optimized);
            println!("\n{}", report);

            // Step 5: Show recommended configuration
            println!("📝 Recommended Ollama configuration:\n");
            println!("  OLLAMA_NUM_PARALLEL: {} (concurrent requests)", optimized.num_parallel);
            println!("  OLLAMA_NUM_GPU: {} (GPU layers to offload)", optimized.num_gpu);
            println!(
                "  OLLAMA_NUM_BATCH: {} (batch size for prompt processing)",
                optimized.num_batch
            );
            println!("  OLLAMA_NUM_CTX: {} (context window size)", optimized.num_ctx);
            println!(
                "  OLLAMA_MAX_LOADED_MODELS: {} (models to keep in memory)",
                optimized.max_loaded_models
            );
            println!("  OLLAMA_KEEP_ALIVE: \"{}\" (model retention time)", optimized.keep_alive);
            if let Some(threads) = optimized.num_thread {
                println!("  OLLAMA_NUM_THREAD: {} (CPU threads)", threads);
            }
            println!();

            if dry_run {
                println!("🔬 Dry run mode - no files created");
                println!("\n💡 To apply these settings, run without --dry-run:");
                println!("   velollm optimize -o velollm-config.sh");
                println!("   source velollm-config.sh");
                return Ok(());
            }

            // Step 6: Generate shell script
            let script = generate_shell_script(&optimized_ollama);

            if let Some(path) = output {
                std::fs::write(&path, &script)?;
                println!("✅ Configuration saved to: {}", path);
                println!("\n📌 To apply these settings:");
                println!("   source {}", path);
                println!("\n💡 Add to your shell profile for persistence:");
                println!(
                    "   echo 'source {}' >> ~/.bashrc",
                    std::fs::canonicalize(&path)?.display()
                );
            } else {
                println!("📝 Shell configuration:\n");
                println!("{}", script);
                println!("\n💡 To save to a file, use:");
                println!("   velollm optimize -o velollm-config.sh");
            }
        }
    }

    Ok(())
}

/// Convert OptimizedConfig to OllamaConfig for shell export
fn optimized_config_to_ollama(optimized: &OptimizedConfig) -> OllamaConfig {
    OllamaConfig {
        num_parallel: Some(optimized.num_parallel),
        max_loaded_models: Some(optimized.max_loaded_models),
        keep_alive: Some(optimized.keep_alive.clone()),
        num_ctx: Some(optimized.num_ctx),
        num_batch: Some(optimized.num_batch),
        num_gpu: Some(optimized.num_gpu),
        num_thread: optimized.num_thread,
        ollama_host: None,
        ollama_models: None,
        ollama_debug: None,
        ollama_flash_attention: None,
        ollama_num_gpu: Some(optimized.num_gpu),
    }
}

/// Convert OllamaConfig to OptimizedConfig for comparison
fn ollama_to_optimized_config(ollama: &OllamaConfig) -> OptimizedConfig {
    OptimizedConfig {
        num_parallel: ollama.num_parallel.unwrap_or(1),
        max_loaded_models: ollama.max_loaded_models.unwrap_or(1),
        keep_alive: ollama
            .keep_alive
            .clone()
            .unwrap_or_else(|| "5m".to_string()),
        num_ctx: ollama.num_ctx.unwrap_or(2048),
        num_batch: ollama.num_batch.unwrap_or(512),
        num_gpu: ollama.num_gpu.unwrap_or(-1),
        num_thread: ollama.num_thread,
    }
}

/// Generate shell script with proper header and exports
fn generate_shell_script(config: &OllamaConfig) -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let mut script = String::new();

    script.push_str("#!/bin/bash\n");
    script.push_str("#\n");
    script.push_str("# VeloLLM - Optimized Ollama Configuration\n");
    script.push_str(&format!("# Generated: {}\n", timestamp));
    script.push_str("#\n");
    script.push_str("# This script sets environment variables to optimize Ollama performance\n");
    script.push_str("# for your specific hardware configuration.\n");
    script.push_str("#\n");
    script.push_str("# Usage:\n");
    script.push_str("#   source velollm-config.sh\n");
    script.push_str("#\n");
    script.push_str("# To make permanent, add to your shell profile:\n");
    script.push_str("#   echo 'source /path/to/velollm-config.sh' >> ~/.bashrc\n");
    script.push_str("#\n\n");

    script.push_str(&config.to_env_exports());
    script.push('\n');

    script
}
