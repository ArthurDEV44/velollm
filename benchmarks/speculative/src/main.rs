use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;
use velollm_adapters_llamacpp::{SpeculativeConfig, SpeculativeRunner};

/// Benchmark tool for comparing vanilla vs speculative decoding performance
#[derive(Parser, Debug)]
#[command(name = "velollm-bench-speculative")]
#[command(about = "Compare vanilla vs speculative decoding in llama.cpp")]
struct Args {
    /// Path to llama.cpp directory
    #[arg(short = 'l', long, default_value = "/home/sauron/code/llama.cpp")]
    llama_cpp_path: String,

    /// Path to main model (GGUF format)
    #[arg(short = 'm', long)]
    main_model: String,

    /// Path to draft model (GGUF format)
    #[arg(short = 'd', long)]
    draft_model: String,

    /// Number of draft tokens per iteration
    #[arg(long, default_value = "8")]
    n_draft: u32,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value = "100")]
    n_predict: u32,

    /// Prompt to use for generation
    #[arg(
        short = 'p',
        long,
        default_value = "Explain quantum computing in simple terms"
    )]
    prompt: String,

    /// Number of iterations for each mode
    #[arg(short = 'i', long, default_value = "5")]
    iterations: usize,

    /// Output file for results (JSON)
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Skip vanilla benchmark (speculative only)
    #[arg(long)]
    skip_vanilla: bool,
}

/// Statistics for a set of measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Statistics {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
    samples: Vec<f64>,
}

impl Statistics {
    fn from_samples(samples: Vec<f64>) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self { mean, stddev, min, max, samples }
    }
}

/// Results from a benchmark run (vanilla or speculative)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkRun {
    mode: String,
    tokens_per_second: Statistics,
    time_to_first_token_ms: Statistics,
    total_time_ms: Statistics,
}

/// Complete comparison results
#[derive(Debug, Serialize, Deserialize)]
struct ComparisonResult {
    config: BenchmarkConfig,
    vanilla: Option<BenchmarkRun>,
    speculative: BenchmarkRun,
    speedup: f64,
    speedup_stddev: f64,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkConfig {
    main_model: String,
    draft_model: String,
    n_draft: u32,
    n_predict: u32,
    prompt: String,
    iterations: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== VeloLLM Speculative Decoding Benchmark ===\n");
    println!("Configuration:");
    println!("  Main model: {}", args.main_model);
    println!("  Draft model: {}", args.draft_model);
    println!("  n_draft: {}", args.n_draft);
    println!("  n_predict: {}", args.n_predict);
    println!("  Iterations: {}", args.iterations);
    println!("  Prompt: {}\n", args.prompt);

    let runner = SpeculativeRunner::new(&args.llama_cpp_path);

    // Benchmark vanilla mode (unless skipped)
    let vanilla_run = if args.skip_vanilla {
        println!("⏭️  Skipping vanilla benchmark");
        None
    } else {
        println!("🔹 Running vanilla inference benchmark...");
        let config = SpeculativeConfig {
            main_model_path: args.main_model.clone(),
            draft_model_path: String::new(),
            n_draft: 0,
            n_predict: args.n_predict,
            prompt: args.prompt.clone(),
        };

        Some(run_benchmark(&runner, &config, args.iterations, "Vanilla")?)
    };

    // Benchmark speculative mode
    println!("\n🔸 Running speculative inference benchmark...");
    let spec_config = SpeculativeConfig {
        main_model_path: args.main_model.clone(),
        draft_model_path: args.draft_model.clone(),
        n_draft: args.n_draft,
        n_predict: args.n_predict,
        prompt: args.prompt.clone(),
    };

    let speculative_run = run_benchmark(&runner, &spec_config, args.iterations, "Speculative")?;

    // Calculate speedup
    let speedup = if let Some(ref vanilla) = vanilla_run {
        speculative_run.tokens_per_second.mean / vanilla.tokens_per_second.mean
    } else {
        0.0 // No comparison if vanilla was skipped
    };

    // Speedup standard deviation (propagation of uncertainty)
    let speedup_stddev = if let Some(ref vanilla) = vanilla_run {
        let rel_var_spec = (speculative_run.tokens_per_second.stddev
            / speculative_run.tokens_per_second.mean)
            .powi(2);
        let rel_var_vanilla =
            (vanilla.tokens_per_second.stddev / vanilla.tokens_per_second.mean).powi(2);
        speedup * (rel_var_spec + rel_var_vanilla).sqrt()
    } else {
        0.0
    };

    // Print results
    println!("\n=== Benchmark Results ===\n");

    if let Some(ref vanilla) = vanilla_run {
        print_benchmark_results(vanilla);
    }

    print_benchmark_results(&speculative_run);

    if vanilla_run.is_some() {
        println!("\n=== Performance Comparison ===");
        println!("📊 Speedup: {:.2}x ± {:.2}x", speedup, speedup_stddev);

        if speedup >= 1.5 {
            println!("✅ Speculative decoding achieved target speedup (>1.5x)!");
        } else if speedup > 1.0 {
            println!("⚠️  Modest speedup. Consider:");
            println!("   - Increasing n_draft (current: {})", args.n_draft);
            println!("   - Trying different model pairs");
        } else {
            println!("❌ Speculative decoding is slower. Possible causes:");
            println!("   - Draft model too slow");
            println!("   - Low acceptance rate (models incompatible)");
            println!("   - n_draft too high");
        }
    }

    // Save results
    let result = ComparisonResult {
        config: BenchmarkConfig {
            main_model: args.main_model,
            draft_model: args.draft_model,
            n_draft: args.n_draft,
            n_predict: args.n_predict,
            prompt: args.prompt,
            iterations: args.iterations,
        },
        vanilla: vanilla_run,
        speculative: speculative_run,
        speedup,
        speedup_stddev,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    if let Some(output_path) = args.output {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(&output_path, json)?;
        println!("\n💾 Results saved to: {}", output_path.display());
    }

    Ok(())
}

fn run_benchmark(
    runner: &SpeculativeRunner,
    config: &SpeculativeConfig,
    iterations: usize,
    label: &str,
) -> Result<BenchmarkRun> {
    let mut tokens_per_second = Vec::new();
    let mut time_to_first_token = Vec::new();
    let mut total_time = Vec::new();

    for i in 0..iterations {
        print!("  Iteration {}/{} ... ", i + 1, iterations);
        std::io::Write::flush(&mut std::io::stdout())?;

        let start = Instant::now();
        let output = runner.run(config)?;
        let elapsed = start.elapsed();

        let metrics = runner
            .parse_perf_metrics(&output)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse metrics from output"))?;

        tokens_per_second.push(metrics.tokens_per_second);
        time_to_first_token.push(metrics.time_to_first_token_ms);
        total_time.push(metrics.total_time_ms);

        println!("{:.2} tok/s ({}ms)", metrics.tokens_per_second, elapsed.as_millis());
    }

    Ok(BenchmarkRun {
        mode: label.to_string(),
        tokens_per_second: Statistics::from_samples(tokens_per_second),
        time_to_first_token_ms: Statistics::from_samples(time_to_first_token),
        total_time_ms: Statistics::from_samples(total_time),
    })
}

fn print_benchmark_results(run: &BenchmarkRun) {
    println!("{} Results:", run.mode);
    println!(
        "  Tokens/s:     {:.2} ± {:.2} tok/s (min: {:.2}, max: {:.2})",
        run.tokens_per_second.mean,
        run.tokens_per_second.stddev,
        run.tokens_per_second.min,
        run.tokens_per_second.max
    );
    println!(
        "  TTFT:         {:.2} ± {:.2} ms",
        run.time_to_first_token_ms.mean, run.time_to_first_token_ms.stddev
    );
    println!(
        "  Total time:   {:.2} ± {:.2} ms",
        run.total_time_ms.mean, run.total_time_ms.stddev
    );
    println!();
}
