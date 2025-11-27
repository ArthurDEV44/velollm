use velollm_adapters_llamacpp::{PerfMetrics, SpeculativeConfig, SpeculativeRunner};

/// Example comparing vanilla vs speculative decoding performance
///
/// This example demonstrates how to use the SpeculativeRunner to compare
/// inference speeds between standard and speculative decoding.
///
/// Prerequisites:
/// - llama.cpp built in /path/to/llama.cpp
/// - Main model (e.g., llama-3.1-8b-q4_k_m.gguf)
/// - Draft model (e.g., llama-3.2-1b-q4_k_m.gguf)
///
/// Usage:
///   cargo run --example speculative_comparison
fn main() -> anyhow::Result<()> {
    // Configuration - adjust these paths for your setup
    let llama_cpp_path = "/home/sauron/code/llama.cpp";
    let main_model = "/path/to/models/llama-3.1-8b-q4_k_m.gguf";
    let draft_model = "/path/to/models/llama-3.2-1b-q4_k_m.gguf";
    let prompt = "Explain quantum computing in simple terms";
    let n_predict = 100;

    println!("=== VeloLLM Speculative Decoding Comparison ===\n");

    let runner = SpeculativeRunner::new(llama_cpp_path);

    // === Vanilla Inference ===
    println!("🔹 Running vanilla inference (no speculative decoding)...");
    let vanilla_config = SpeculativeConfig {
        main_model_path: main_model.to_string(),
        draft_model_path: String::new(), // Empty = vanilla mode
        n_draft: 0,
        n_predict,
        prompt: prompt.to_string(),
    };

    let vanilla_output = match runner.run(&vanilla_config) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("❌ Vanilla inference failed: {}", e);
            eprintln!("\nMake sure:");
            eprintln!("  1. llama.cpp is built: cd {} && make", llama_cpp_path);
            eprintln!("  2. Model path is correct: {}", main_model);
            return Err(e);
        }
    };

    let vanilla_metrics = runner
        .parse_perf_metrics(&vanilla_output)
        .expect("Failed to parse vanilla metrics");

    print_metrics("Vanilla", &vanilla_metrics);

    // === Speculative Inference ===
    println!("\n🔸 Running speculative decoding inference...");
    let speculative_config = SpeculativeConfig {
        main_model_path: main_model.to_string(),
        draft_model_path: draft_model.to_string(),
        n_draft: 8, // Generate 8 draft tokens per iteration
        n_predict,
        prompt: prompt.to_string(),
    };

    let spec_output = match runner.run(&speculative_config) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("❌ Speculative inference failed: {}", e);
            eprintln!("\nMake sure:");
            eprintln!("  1. Draft model path is correct: {}", draft_model);
            eprintln!("  2. llama-speculative binary exists in llama.cpp/build/bin/");
            return Err(e);
        }
    };

    let spec_metrics = runner
        .parse_perf_metrics(&spec_output)
        .expect("Failed to parse speculative metrics");

    print_metrics("Speculative", &spec_metrics);

    // === Comparison ===
    println!("\n=== Performance Comparison ===");
    let speedup = spec_metrics.tokens_per_second / vanilla_metrics.tokens_per_second;
    println!("📊 Speedup: {:.2}x", speedup);
    println!(
        "⚡ Tokens/s improvement: {:.2} → {:.2} ({:+.2})",
        vanilla_metrics.tokens_per_second,
        spec_metrics.tokens_per_second,
        spec_metrics.tokens_per_second - vanilla_metrics.tokens_per_second
    );
    println!(
        "⏱️  TTFT improvement: {:.2}ms → {:.2}ms ({:+.2}ms)",
        vanilla_metrics.time_to_first_token_ms,
        spec_metrics.time_to_first_token_ms,
        spec_metrics.time_to_first_token_ms - vanilla_metrics.time_to_first_token_ms
    );

    if speedup >= 1.5 {
        println!("\n✅ Speculative decoding achieved target speedup (>1.5x)!");
    } else if speedup > 1.0 {
        println!("\n⚠️  Speculative decoding provides modest speedup.");
        println!("   Consider: increasing n_draft, or trying a different model pair");
    } else {
        println!("\n❌ Speculative decoding is slower than vanilla.");
        println!("   Possible reasons:");
        println!("   - Draft model too large/slow");
        println!("   - Low acceptance rate (models too different)");
        println!("   - n_draft value too high");
    }

    // Save results to JSON
    let results = ComparisonResult {
        vanilla_tokens_per_second: vanilla_metrics.tokens_per_second,
        speculative_tokens_per_second: spec_metrics.tokens_per_second,
        speedup,
        n_draft: 8,
        prompt: prompt.to_string(),
        n_predict,
    };

    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write("speculative_comparison_results.json", json)?;
    println!("\n💾 Results saved to: speculative_comparison_results.json");

    Ok(())
}

fn print_metrics(label: &str, metrics: &PerfMetrics) {
    println!("\n  {} Results:", label);
    println!("  • Tokens/s: {:.2}", metrics.tokens_per_second);
    println!("  • TTFT: {:.2}ms", metrics.time_to_first_token_ms);
    println!("  • Total time: {:.2}ms", metrics.total_time_ms);
}

#[derive(serde::Serialize)]
struct ComparisonResult {
    vanilla_tokens_per_second: f64,
    speculative_tokens_per_second: f64,
    speedup: f64,
    n_draft: u32,
    prompt: String,
    n_predict: u32,
}
