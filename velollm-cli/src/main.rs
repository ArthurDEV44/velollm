use clap::{Parser, Subcommand};
use tracing::{debug, error, info, instrument, warn, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
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

    /// Enable verbose logging (can be repeated: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware specifications
    Detect {
        /// Output as JSON only (no formatting)
        #[arg(long)]
        json: bool,
    },

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

/// Initialize tracing subscriber with appropriate log level
fn init_tracing(verbose: u8, quiet: bool) {
    let level = if quiet {
        Level::ERROR
    } else {
        match verbose {
            0 => Level::WARN,
            1 => Level::INFO,
            2 => Level::DEBUG,
            _ => Level::TRACE,
        }
    };

    let filter = EnvFilter::from_default_env()
        .add_directive(level.into())
        .add_directive("hyper=warn".parse().unwrap())
        .add_directive("reqwest=warn".parse().unwrap());

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(verbose >= 2).with_thread_ids(verbose >= 3))
        .with(filter)
        .init();

    debug!(level = %level, "Tracing initialized");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    init_tracing(cli.verbose, cli.quiet);

    let result = match cli.command {
        Commands::Detect { json } => run_detect(json).await,
        Commands::Benchmark { backend, model, output } => run_benchmark(&backend, &model, output).await,
        Commands::Optimize { dry_run, output } => run_optimize(dry_run, output).await,
    };

    if let Err(ref e) = result {
        error!(error = %e, "Command failed");
    }

    result
}

#[instrument(skip_all)]
async fn run_detect(json_only: bool) -> anyhow::Result<()> {
    info!("Starting hardware detection");

    if !json_only {
        println!("Detecting hardware configuration...\n");
    }

    let hw = HardwareSpec::detect()?;

    info!(
        os = %hw.os,
        platform = %hw.platform,
        cpu_cores = hw.cpu.cores,
        cpu_threads = hw.cpu.threads,
        memory_mb = hw.memory.total_mb,
        has_gpu = hw.gpu.is_some(),
        "Hardware detected"
    );

    if json_only {
        println!("{}", serde_json::to_string_pretty(&hw)?);
        return Ok(());
    }

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
        debug!(frequency_mhz = freq, "CPU frequency detected");
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
        info!(
            gpu_name = %gpu.name,
            gpu_vendor = ?gpu.vendor,
            vram_total_mb = gpu.vram_total_mb,
            vram_free_mb = gpu.vram_free_mb,
            "GPU detected"
        );

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
            debug!(driver = %driver, "GPU driver version");
        }
        if let Some(ref compute) = gpu.compute_capability {
            println!("Compute Capability: {}", compute);
            debug!(compute_capability = %compute, "GPU compute capability");
        }
        println!();
    } else {
        info!("No GPU detected, CPU-only mode");
        println!("=== GPU ===");
        println!("No GPU detected (CPU-only mode)");
        println!();
    }

    println!("=== JSON Output ===");
    println!("{}", serde_json::to_string_pretty(&hw)?);

    Ok(())
}

#[instrument(skip_all, fields(backend = %backend, model = %model))]
async fn run_benchmark(backend: &str, model: &str, output: Option<String>) -> anyhow::Result<()> {
    info!("Starting benchmark suite");

    println!("VeloLLM Benchmark Suite\n");
    println!("Backend: {}", backend);
    println!("Model: {}\n", model);

    // Check if Ollama is running
    let runner = BenchmarkRunner::new(backend);

    debug!("Checking Ollama availability");
    print!("Checking Ollama availability... ");
    if !runner.check_ollama_available().await? {
        println!("FAILED");
        warn!("Ollama is not running or not accessible");
        anyhow::bail!(
            "Ollama is not running. Please start Ollama and ensure the model '{}' is available.",
            model
        );
    }
    println!("OK\n");
    info!("Ollama is available");

    // Get standard benchmarks
    let benchmarks = get_standard_benchmarks(model);
    info!(benchmark_count = benchmarks.len(), "Loaded benchmark configurations");

    println!("Running {} benchmarks...\n", benchmarks.len());
    println!("-----------------------------------------------------------\n");

    let mut results = Vec::new();
    for (i, config) in benchmarks.iter().enumerate() {
        debug!(
            benchmark = %config.name,
            iteration = i + 1,
            total = benchmarks.len(),
            "Running benchmark"
        );

        match runner.run(config).await {
            Ok(result) => {
                info!(
                    benchmark = %config.name,
                    tokens_per_second = result.tokens_per_second,
                    ttft_ms = result.time_to_first_token_ms,
                    total_tokens = result.total_tokens,
                    "Benchmark completed"
                );
                results.push(result);
            }
            Err(e) => {
                error!(benchmark = %config.name, error = %e, "Benchmark failed");
                eprintln!("Benchmark '{}' failed: {}", config.name, e);
                continue;
            }
        }
    }

    // Summary
    println!("-----------------------------------------------------------");
    println!("\nBenchmark Summary\n");

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
        let avg_ttft: f64 = results.iter().map(|r| r.time_to_first_token_ms).sum::<f64>()
            / results.len() as f64;

        info!(
            avg_tokens_per_second = avg_tps,
            avg_ttft_ms = avg_ttft,
            completed_benchmarks = results.len(),
            "Benchmark suite completed"
        );

        println!("Overall Average:");
        println!("  Tokens/s: {:.1}", avg_tps);
        println!("  TTFT: {:.1}ms", avg_ttft);
        println!();
    }

    // Save results if output specified
    if let Some(path) = output {
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(&path, json)?;
        info!(output_path = %path, "Results saved to file");
        println!("Results saved to: {}", path);
    } else {
        println!("Tip: Use -o <file> to save results to JSON");
    }

    Ok(())
}

#[instrument(skip_all, fields(dry_run = dry_run))]
async fn run_optimize(dry_run: bool, output: Option<String>) -> anyhow::Result<()> {
    info!("Starting Ollama optimization");

    println!("VeloLLM Ollama Optimizer\n");

    // Step 1: Detect hardware
    debug!("Detecting hardware configuration");
    println!("Detecting hardware configuration...");
    let hw = HardwareSpec::detect()?;

    info!(
        gpu = hw.gpu.as_ref().map(|g| g.name.as_str()),
        cpu_cores = hw.cpu.cores,
        memory_gb = hw.memory.total_mb as f64 / 1024.0,
        "Hardware detected for optimization"
    );

    println!("Hardware detected:");
    if let Some(ref gpu) = hw.gpu {
        println!(
            "  GPU: {} ({:.1} GB VRAM)",
            gpu.name,
            gpu.vram_total_mb as f64 / 1024.0
        );
    } else {
        println!("  GPU: None (CPU-only mode)");
    }
    println!("  CPU: {} cores / {} threads", hw.cpu.cores, hw.cpu.threads);
    println!("  RAM: {:.1} GB\n", hw.memory.total_mb as f64 / 1024.0);

    // Step 2: Read current configuration from environment
    debug!("Reading current Ollama configuration from environment");
    println!("Reading current Ollama configuration from environment...");
    let current_env = OllamaConfig::from_env();

    if current_env.is_empty() {
        debug!("No Ollama environment variables set");
        println!("  No Ollama environment variables currently set (using defaults)");
    } else {
        debug!(config = ?current_env, "Current Ollama configuration");
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
    debug!("Generating optimized configuration");
    println!("Generating optimized configuration for your hardware...");
    let optimized = OllamaOptimizer::optimize(&hw);

    info!(
        num_parallel = optimized.num_parallel,
        num_gpu = optimized.num_gpu,
        num_batch = optimized.num_batch,
        num_ctx = optimized.num_ctx,
        max_loaded_models = optimized.max_loaded_models,
        keep_alive = %optimized.keep_alive,
        "Optimized configuration generated"
    );

    // Convert OptimizedConfig to OllamaConfig for comparison and export
    let optimized_ollama = optimized_config_to_ollama(&optimized);

    // Convert current env to OptimizedConfig format for comparison
    let current_optimized = ollama_to_optimized_config(&current_env);

    // Step 4: Generate comparison report
    let report = OllamaOptimizer::generate_report(&current_optimized, &optimized);
    println!("\n{}", report);

    // Step 5: Show recommended configuration
    println!("Recommended Ollama configuration:\n");
    println!(
        "  OLLAMA_NUM_PARALLEL: {} (concurrent requests)",
        optimized.num_parallel
    );
    println!(
        "  OLLAMA_NUM_GPU: {} (GPU layers to offload)",
        optimized.num_gpu
    );
    println!(
        "  OLLAMA_NUM_BATCH: {} (batch size for prompt processing)",
        optimized.num_batch
    );
    println!(
        "  OLLAMA_NUM_CTX: {} (context window size)",
        optimized.num_ctx
    );
    println!(
        "  OLLAMA_MAX_LOADED_MODELS: {} (models to keep in memory)",
        optimized.max_loaded_models
    );
    println!(
        "  OLLAMA_KEEP_ALIVE: \"{}\" (model retention time)",
        optimized.keep_alive
    );
    if let Some(threads) = optimized.num_thread {
        println!("  OLLAMA_NUM_THREAD: {} (CPU threads)", threads);
    }
    println!();

    if dry_run {
        info!("Dry run mode - no files created");
        println!("Dry run mode - no files created");
        println!("\nTo apply these settings, run without --dry-run:");
        println!("   velollm optimize -o velollm-config.sh");
        println!("   source velollm-config.sh");
        return Ok(());
    }

    // Step 6: Generate shell script
    let script = generate_shell_script(&optimized_ollama);

    if let Some(path) = output {
        std::fs::write(&path, &script)?;
        info!(output_path = %path, "Configuration saved to file");
        println!("Configuration saved to: {}", path);
        println!("\nTo apply these settings:");
        println!("   source {}", path);
        println!("\nAdd to your shell profile for persistence:");
        println!(
            "   echo 'source {}' >> ~/.bashrc",
            std::fs::canonicalize(&path)?.display()
        );
    } else {
        println!("Shell configuration:\n");
        println!("{}", script);
        println!("\nTo save to a file, use:");
        println!("   velollm optimize -o velollm-config.sh");
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
    }
}

/// Convert OllamaConfig to OptimizedConfig for comparison
fn ollama_to_optimized_config(ollama: &OllamaConfig) -> OptimizedConfig {
    OptimizedConfig {
        num_parallel: ollama.num_parallel.unwrap_or(1),
        max_loaded_models: ollama.max_loaded_models.unwrap_or(1),
        keep_alive: ollama.keep_alive.clone().unwrap_or_else(|| "5m".to_string()),
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
