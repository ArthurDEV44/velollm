# VeloLLM Benchmarking Guide

## Overview

VeloLLM includes a comprehensive benchmarking suite to measure LLM inference performance across different hardware and configurations.

## Quick Start

### Prerequisites

1. **Ollama installed and running**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Model downloaded**:
   ```bash
   ollama pull llama3.2:3b
   ```

### Running Benchmarks

```bash
# Run with default model (llama3.2:3b)
velollm benchmark

# Specify a different model
velollm benchmark -m llama3.1:8b

# Save results to JSON
velollm benchmark -o results.json

# Use different backend (future)
velollm benchmark -b llamacpp
```

## Standard Benchmarks

The suite includes three standard benchmarks:

### 1. Short Completion (50 tokens)
**Prompt**: "Write a hello world program in Python"
**Iterations**: 5
**Purpose**: Measure baseline generation speed

### 2. Medium Completion (150 tokens)
**Prompt**: "Explain how neural networks learn through backpropagation in detail"
**Iterations**: 3
**Purpose**: Measure sustained throughput

### 3. Code Generation (200 tokens)
**Prompt**: "Write a Rust function to compute the Fibonacci sequence using dynamic programming"
**Iterations**: 3
**Purpose**: Measure code generation performance

## Metrics Collected

### Tokens per Second (tok/s)
- **Definition**: Number of tokens generated per second
- **Higher is better**
- **Typical ranges**:
  - CPU-only: 5-20 tok/s
  - Mid-range GPU (RTX 3060): 30-60 tok/s
  - High-end GPU (RTX 4090): 80-150 tok/s

### Time to First Token (TTFT)
- **Definition**: Time from request to first token generated
- **Lower is better**
- **Components**:
  - Prompt evaluation time
  - First token generation time
- **Typical ranges**:
  - Small models (3B): 50-200ms
  - Large models (70B): 500-2000ms

### Total Time
- **Definition**: Complete request duration
- **Includes**:
  - Prompt processing
  - All token generation
  - Response formatting

## Example Output

```
🚀 VeloLLM Benchmark Suite

Backend: ollama
Model: llama3.2:3b

Checking Ollama availability... ✓

Running 3 benchmarks...

═══════════════════════════════════════════════════════

Running benchmark: short_completion (5 iterations)
  Iteration 1/5... 82.3 tok/s (612ms)
  Iteration 2/5... 85.1 tok/s (593ms)
  Iteration 3/5... 84.7 tok/s (596ms)
  Iteration 4/5... 83.9 tok/s (602ms)
  Iteration 5/5... 84.2 tok/s (599ms)
  Average: 84.0 tok/s, TTFT: 127.3ms

Running benchmark: medium_completion (3 iterations)
  Iteration 1/3... 78.5 tok/s (1913ms)
  Iteration 2/3... 79.2 tok/s (1896ms)
  Iteration 3/3... 78.9 tok/s (1903ms)
  Average: 78.9 tok/s, TTFT: 145.6ms

Running benchmark: code_generation (3 iterations)
  Iteration 1/3... 76.3 tok/s (2621ms)
  Iteration 2/3... 77.1 tok/s (2596ms)
  Iteration 3/3... 76.8 tok/s (2605ms)
  Average: 76.7 tok/s, TTFT: 152.1ms

═══════════════════════════════════════════════════════

📊 Benchmark Summary

short_completion:
  Tokens/s: 84.0
  TTFT: 127.3ms
  Total tokens: 252
  Total time: 3.0s

medium_completion:
  Tokens/s: 78.9
  TTFT: 145.6ms
  Total tokens: 447
  Total time: 5.7s

code_generation:
  Tokens/s: 76.7
  TTFT: 152.1ms
  Total tokens: 593
  Total time: 7.7s

Overall Average:
  Tokens/s: 79.9
  TTFT: 141.7ms

💡 Tip: Use -o <file> to save results to JSON
```

## JSON Output Format

```json
[
  {
    "config": {
      "name": "short_completion",
      "model": "llama3.2:3b",
      "prompt": "Write a hello world program in Python",
      "max_tokens": 50,
      "iterations": 5
    },
    "tokens_per_second": 84.0,
    "time_to_first_token_ms": 127.3,
    "total_time_ms": 3002.5,
    "total_tokens": 252,
    "prompt_eval_count": 12,
    "eval_count": 50,
    "timestamp": "2025-01-15T10:30:00Z"
  },
  ...
]
```

## Comparing Results

### Before/After Optimization

```bash
# Run baseline
velollm benchmark -o baseline.json

# Apply optimizations
velollm optimize -o velollm.sh
source velollm.sh

# Run optimized
velollm benchmark -o optimized.json

# Compare (manual)
jq '.[0].tokens_per_second' baseline.json
jq '.[0].tokens_per_second' optimized.json
```

### Across Hardware

Create a benchmark database:

```bash
# On each system
velollm detect > hardware.json
velollm benchmark -o benchmark.json

# Organize
mkdir benchmarks/rtx-4090
mv hardware.json benchmarks/rtx-4090/
mv benchmark.json benchmarks/rtx-4090/
```

## Advanced Usage

### Custom Benchmarks

Create your own benchmark config:

```rust
use velollm_benchmarks::{BenchmarkConfig, BenchmarkRunner};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        name: "custom_test".to_string(),
        model: "llama3.2:3b".to_string(),
        prompt: "Your custom prompt here".to_string(),
        max_tokens: 100,
        iterations: 5,
    };

    let runner = BenchmarkRunner::new("ollama");
    let result = runner.run(&config).await?;

    println!("Tokens/s: {}", result.tokens_per_second);
    Ok(())
}
```

### Different Models

Test multiple models:

```bash
for model in llama3.2:1b llama3.2:3b llama3.1:8b; do
    echo "=== Testing $model ==="
    velollm benchmark -m $model -o results-$model.json
done
```

## Interpreting Results

### Good Performance Indicators

✅ **High tokens/s**: Efficient GPU utilization
✅ **Low TTFT**: Fast prompt processing
✅ **Consistent across iterations**: Stable performance

### Performance Issues

❌ **Low tokens/s**: Check GPU usage, memory pressure
❌ **High TTFT**: Prompt too long or slow prompt encoder
❌ **Variance across iterations**: Thermal throttling or background processes

## Troubleshooting

### "Ollama is not running"

```bash
# Check Ollama status
systemctl status ollama  # Linux
ollama serve            # Manual start

# Verify model available
ollama list
ollama pull llama3.2:3b
```

### "Model not found"

```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2:3b
```

### Slow Performance

**Check GPU usage**:
```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi
```

**Check VRAM**:
- Ensure model fits in VRAM
- Close other GPU applications
- Try smaller model or quantization

**Check CPU usage**:
```bash
top
htop
```

### Network Errors

```bash
# Check Ollama API
curl http://localhost:11434/api/tags

# Try different port
velollm benchmark --ollama-url http://localhost:11434
```

## Performance Optimization Tips

### 1. Use Appropriate Model Size

- **<8GB VRAM**: llama3.2:1b or 3b
- **8-16GB VRAM**: llama3.1:8b (Q4 quantization)
- **>16GB VRAM**: llama3.1:13b or larger

### 2. Optimize Ollama Settings

```bash
# Increase context window
export OLLAMA_NUM_CTX=4096

# Batch size
export OLLAMA_NUM_BATCH=512

# Keep alive
export OLLAMA_KEEP_ALIVE=5m
```

### 3. GPU Offloading

```bash
# Offload all layers to GPU
export OLLAMA_NUM_GPU=99
```

### 4. Reduce Background Load

- Close browsers
- Stop other GPU applications
- Disable GPU-accelerated desktop effects

## Next Steps

- Compare your results with community benchmarks
- Experiment with different models
- Try speculative decoding (coming in Phase 2)
- Contribute your results to the VeloLLM benchmark database

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [VeloLLM ROADMAP](../../ROADMAP.md)
- [Hardware Detection Guide](hardware_detection.md)
