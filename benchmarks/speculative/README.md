# velollm-bench-speculative

Comprehensive benchmark tool for comparing vanilla vs speculative decoding performance in llama.cpp.

## Overview

This benchmark tool (TASK-007) provides rigorous statistical comparison between standard and speculative decoding modes, including:

- **Multiple iterations** with statistical analysis (mean, stddev, min, max)
- **Automated speedup calculation** with uncertainty propagation
- **JSON export** for analysis and record-keeping
- **Flexible configuration** via CLI arguments

## Features

- ✅ Statistical analysis: mean, standard deviation, min/max
- ✅ Multiple iterations for robust measurements
- ✅ Speedup calculation with error propagation
- ✅ JSON export for reproducibility
- ✅ Progress reporting during benchmark runs
- ✅ Skip vanilla mode option for faster testing
- ✅ Configurable model paths, prompts, and parameters

## Usage

### Basic Usage

```bash
cargo run -p velollm-bench-speculative -- \
  --main-model /path/to/llama-3.1-8b-q4_k_m.gguf \
  --draft-model /path/to/llama-3.2-1b-q4_k_m.gguf \
  --iterations 5 \
  --output results.json
```

### Quick Test (Skip Vanilla)

```bash
cargo run -p velollm-bench-speculative -- \
  --main-model /path/to/model.gguf \
  --draft-model /path/to/draft.gguf \
  --skip-vanilla \
  --iterations 3
```

### Custom Configuration

```bash
cargo run -p velollm-bench-speculative -- \
  -m /models/llama-3.1-8b.gguf \
  -d /models/llama-3.2-1b.gguf \
  --n-draft 10 \
  --n-predict 200 \
  --prompt "Write a Python function to sort a list" \
  --iterations 10 \
  -o benchmark_results.json
```

## CLI Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--llama-cpp-path` | `-l` | `/home/sauron/code/llama.cpp` | Path to llama.cpp directory |
| `--main-model` | `-m` | *required* | Path to main model (GGUF) |
| `--draft-model` | `-d` | *required* | Path to draft model (GGUF) |
| `--n-draft` | | `8` | Number of draft tokens per iteration |
| `--n-predict` | `-n` | `100` | Tokens to generate |
| `--prompt` | `-p` | "Explain quantum computing..." | Generation prompt |
| `--iterations` | `-i` | `5` | Number of benchmark iterations |
| `--output` | `-o` | *none* | JSON output file path |
| `--skip-vanilla` | | `false` | Skip vanilla benchmark |

## Example Output

```
=== VeloLLM Speculative Decoding Benchmark ===

Configuration:
  Main model: /models/llama-3.1-8b-q4_k_m.gguf
  Draft model: /models/llama-3.2-1b-q4_k_m.gguf
  n_draft: 8
  n_predict: 100
  Iterations: 5
  Prompt: Explain quantum computing in simple terms

🔹 Running vanilla inference benchmark...
  Iteration 1/5 ... 28.45 tok/s (3521ms)
  Iteration 2/5 ... 28.32 tok/s (3533ms)
  Iteration 3/5 ... 28.51 tok/s (3514ms)
  Iteration 4/5 ... 28.39 tok/s (3529ms)
  Iteration 5/5 ... 28.47 tok/s (3516ms)

🔸 Running speculative inference benchmark...
  Iteration 1/5 ... 67.23 tok/s (1488ms)
  Iteration 2/5 ... 66.89 tok/s (1496ms)
  Iteration 3/5 ... 67.45 tok/s (1483ms)
  Iteration 4/5 ... 67.01 tok/s (1493ms)
  Iteration 5/5 ... 67.32 tok/s (1486ms)

=== Benchmark Results ===

Vanilla Results:
  Tokens/s:     28.43 ± 0.07 tok/s (min: 28.32, max: 28.51)
  TTFT:         234.56 ± 2.34 ms
  Total time:   3522.60 ± 7.23 ms

Speculative Results:
  Tokens/s:     67.18 ± 0.21 tok/s (min: 66.89, max: 67.45)
  TTFT:         189.23 ± 3.12 ms
  Total time:   1489.20 ± 5.12 ms

=== Performance Comparison ===
📊 Speedup: 2.36x ± 0.01x
✅ Speculative decoding achieved target speedup (>1.5x)!

💾 Results saved to: results.json
```

## JSON Output Format

```json
{
  "config": {
    "main_model": "/models/llama-3.1-8b.gguf",
    "draft_model": "/models/llama-3.2-1b.gguf",
    "n_draft": 8,
    "n_predict": 100,
    "prompt": "Explain quantum computing in simple terms",
    "iterations": 5
  },
  "vanilla": {
    "mode": "Vanilla",
    "tokens_per_second": {
      "mean": 28.428,
      "stddev": 0.073,
      "min": 28.32,
      "max": 28.51,
      "samples": [28.45, 28.32, 28.51, 28.39, 28.47]
    },
    "time_to_first_token_ms": { ... },
    "total_time_ms": { ... }
  },
  "speculative": { ... },
  "speedup": 2.36,
  "speedup_stddev": 0.01,
  "timestamp": "2025-01-27T12:34:56Z"
}
```

## Statistical Methods

### Mean and Standard Deviation
- **Mean (μ)**: Average of all samples
- **Standard Deviation (σ)**: Measure of spread/variance

### Speedup Calculation
```
speedup = tokens_per_second_speculative / tokens_per_second_vanilla
```

### Error Propagation
The speedup uncertainty is calculated using standard error propagation:

```
σ_speedup = speedup × √[(σ_spec/μ_spec)² + (σ_vanilla/μ_vanilla)²]
```

This accounts for uncertainty in both measurements.

## Prerequisites

1. **llama.cpp built**: Binaries must exist in `{llama_cpp_path}/build/bin/`
2. **Models downloaded**: GGUF format main and draft models
3. **Sufficient VRAM/RAM**: Models must fit in memory

## Performance Tips

### Recommended n_draft Values

| Model Pair | Recommended n_draft | Expected Speedup |
|------------|---------------------|------------------|
| llama-3.1-8b + 3.2-1b | 8 | 2.0-2.4x |
| llama-3.1-70b + 3.2-3b | 10 | 2.2-2.6x |
| codellama-13b + 1b | 7 | 1.6-2.0x |

### Iteration Guidelines

- **Quick test**: 3 iterations (±5-10% error)
- **Standard**: 5 iterations (±3-5% error)
- **Rigorous**: 10+ iterations (±1-2% error)

## Troubleshooting

### Binary not found
```
Error: Failed to execute llama.cpp binary
```
**Solution**: Build llama.cpp with `cd /path/to/llama.cpp && make`

### Model not found
```
Error: No such file or directory
```
**Solution**: Check model paths with `ls -lh /path/to/model.gguf`

### No speedup
```
⚠️ Modest speedup. Consider:
   - Increasing n_draft (current: 8)
   - Trying different model pairs
```
**Solutions**:
- Increase `--n-draft` to 10-12
- Ensure draft model is much smaller than main model (1B draft for 8B main)
- Check models are from the same family (e.g., both Llama 3.x)

## Integration with VeloLLM

This tool fulfills TASK-007 requirements:
- ✅ Benchmark comparison vanilla vs speculative
- ✅ Multiple iterations with statistics
- ✅ JSON export for reproducibility
- ✅ Speedup measurement with target (>1.5x)

Part of the VeloLLM Phase 1 MVP pipeline.

## License

MIT - See repository root LICENSE
