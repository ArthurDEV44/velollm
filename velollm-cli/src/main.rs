use clap::{Parser, Subcommand};
use velollm_core::hardware::HardwareSpec;

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
        #[arg(short, long, default_value = "ollama")]
        backend: String,

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

fn main() -> anyhow::Result<()> {
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

        Commands::Benchmark { backend, output } => {
            println!("Benchmark command - backend: {}", backend);
            if let Some(path) = output {
                println!("Output will be saved to: {}", path);
            }
            println!("TODO: Implement in TASK-004");
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
