use clap::{Parser, Subcommand};
use velollm_core::hardware::HardwareSpec;
use velollm_benchmarks::{BenchmarkRunner, get_standard_benchmarks};

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
            println!("Total: {} MB ({:.1} GB)", hw.memory.total_mb, hw.memory.total_mb as f64 / 1024.0);
            println!("Available: {} MB ({:.1} GB)", hw.memory.available_mb, hw.memory.available_mb as f64 / 1024.0);
            println!("Used: {} MB ({:.1} GB)", hw.memory.used_mb, hw.memory.used_mb as f64 / 1024.0);
            println!();

            if let Some(ref gpu) = hw.gpu {
                println!("=== GPU ===");
                println!("Name: {}", gpu.name);
                println!("Vendor: {:?}", gpu.vendor);
                println!("VRAM Total: {} MB ({:.1} GB)", gpu.vram_total_mb, gpu.vram_total_mb as f64 / 1024.0);
                println!("VRAM Free: {} MB ({:.1} GB)", gpu.vram_free_mb, gpu.vram_free_mb as f64 / 1024.0);
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
                let avg_tps: f64 = results.iter().map(|r| r.tokens_per_second).sum::<f64>() / results.len() as f64;
                let avg_ttft: f64 = results.iter().map(|r| r.time_to_first_token_ms).sum::<f64>() / results.len() as f64;

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
            println!("Optimize command");
            if dry_run {
                println!("Dry run mode enabled");
            }
            if let Some(path) = output {
                println!("Output will be saved to: {}", path);
            }
            println!("TODO: Implement in TASK-009");
        }
    }

    Ok(())
}
