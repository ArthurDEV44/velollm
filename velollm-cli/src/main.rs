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
            let hw = HardwareSpec::detect()?;
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
