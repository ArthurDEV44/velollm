# velollm-adapters-llamacpp

Rust adapter for llama.cpp with speculative decoding support.

## Overview

This crate provides a high-level Rust interface to llama.cpp's speculative decoding capabilities. It allows you to:

- Run inference with llama.cpp in vanilla or speculative mode
- Parse performance metrics from llama.cpp output
- Compare performance between different configurations
- Serialize/deserialize configurations for batch processing

## Features

- **Speculative Decoding**: Leverage llama.cpp's speculative decoding for 1.5-2.5x speedup
- **Automatic Mode Selection**: Automatically chooses between vanilla and speculative binaries
- **Performance Metrics**: Parses tokens/s, TTFT, and total time from llama.cpp output
- **Type-Safe Configuration**: Serde-based configuration management

## Prerequisites

1. **llama.cpp built**: You need a compiled llama.cpp installation
   ```bash
   cd /path/to/llama.cpp
   make
   ```

2. **Models**: GGUF format models
   - Main model (e.g., `llama-3.1-8b-q4_k_m.gguf`)
   - Draft model for speculative mode (e.g., `llama-3.2-1b-q4_k_m.gguf`)

## Usage

### Basic Usage

```rust
use velollm_adapters_llamacpp::{SpeculativeConfig, SpeculativeRunner};

let runner = SpeculativeRunner::new("/path/to/llama.cpp");

// Vanilla inference
let config = SpeculativeConfig {
    main_model_path: "/models/llama-3.1-8b.gguf".to_string(),
    draft_model_path: String::new(), // Empty = vanilla
    n_draft: 0,
    n_predict: 100,
    prompt: "Explain quantum computing".to_string(),
};

let output = runner.run(&config)?;
let metrics = runner.parse_perf_metrics(&output).unwrap();

println!("Tokens/s: {}", metrics.tokens_per_second);
```

### Speculative Decoding

```rust
// Speculative inference
let config = SpeculativeConfig {
    main_model_path: "/models/llama-3.1-8b.gguf".to_string(),
    draft_model_path: "/models/llama-3.2-1b.gguf".to_string(),
    n_draft: 8, // Draft 8 tokens per iteration
    n_predict: 100,
    prompt: "Explain quantum computing".to_string(),
};

let output = runner.run(&config)?;
let metrics = runner.parse_perf_metrics(&output).unwrap();
```

### Running the Example

```bash
# Edit paths in examples/speculative_comparison.rs first
cargo run --example speculative_comparison -p velollm-adapters-llamacpp
```

## Configuration

### SpeculativeConfig

| Field | Type | Description |
|-------|------|-------------|
| `main_model_path` | String | Path to the main (target) model |
| `draft_model_path` | String | Path to the draft model (empty for vanilla) |
| `n_draft` | u32 | Number of draft tokens (typically 5-10) |
| `n_predict` | u32 | Total tokens to generate |
| `prompt` | String | Input prompt |

### Performance Metrics

The `PerfMetrics` struct contains:
- `tokens_per_second`: Generation speed
- `time_to_first_token_ms`: Prompt processing time (TTFT)
- `total_time_ms`: Total inference time

## Optimal Model Pairs

Based on TASK-005 research:

| Main Model | Draft Model | n_draft | Expected Speedup |
|------------|-------------|---------|------------------|
| llama-3.1-8b | llama-3.2-1b | 8 | 2.0x |
| llama-3.1-70b | llama-3.2-3b | 10 | 2.5x |
| codellama-13b | codellama-1b | 7 | 1.8x |

## Binary Paths

The runner expects llama.cpp binaries in:
- Vanilla: `{llama_cpp_path}/build/bin/llama-cli`
- Speculative: `{llama_cpp_path}/build/bin/llama-speculative`

## Error Handling

Common errors:
- **Binary not found**: Build llama.cpp with `make`
- **Model not found**: Check model paths
- **Execution failed**: Check stderr output for llama.cpp errors

## Testing

```bash
cargo test -p velollm-adapters-llamacpp
```

All tests are unit tests that don't require actual llama.cpp binaries.

## Integration with VeloLLM

This adapter is part of the VeloLLM project (TASK-006). It provides the foundation for:
- TASK-007: Benchmark comparisons
- TASK-009: Automatic optimization selection
- Phase 2: Multi-backend support

## License

MIT - See LICENSE in repository root
