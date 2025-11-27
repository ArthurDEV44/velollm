# velollm-adapters-ollama

Rust adapter for parsing and managing Ollama configuration through environment variables.

## Overview

This crate provides a type-safe interface to Ollama's environment variable-based configuration system. It allows you to:

- Read current Ollama configuration from environment
- Create and modify configuration programmatically
- Generate shell export scripts for configuration
- Serialize/deserialize configurations to JSON

## Features

- ✅ **Complete Coverage**: All documented Ollama env vars
- ✅ **Type-Safe**: Rust types for all configuration values
- ✅ **Validation**: Parse errors handled gracefully
- ✅ **Shell Export**: Generate sourceable shell scripts
- ✅ **Merge Support**: Combine configurations intelligently
- ✅ **Serialization**: JSON import/export via Serde

## Supported Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_NUM_PARALLEL` | `u32` | 1 | Concurrent requests |
| `OLLAMA_MAX_LOADED_MODELS` | `u32` | 1 | Models in memory |
| `OLLAMA_KEEP_ALIVE` | `String` | "5m" | Model retention time |
| `OLLAMA_NUM_CTX` | `u32` | 2048 | Context window size |
| `OLLAMA_NUM_BATCH` | `u32` | 512 | Batch size |
| `OLLAMA_NUM_GPU` | `i32` | -1 | GPU layers to offload |
| `OLLAMA_NUM_THREAD` | `u32` | auto | CPU threads |
| `OLLAMA_HOST` | `String` | "127.0.0.1:11434" | Server address |
| `OLLAMA_MODELS` | `String` | system | Models directory |
| `OLLAMA_DEBUG` | `bool` | false | Debug logging |
| `OLLAMA_FLASH_ATTENTION` | `bool` | false | Flash attention |

## Usage

### Reading Current Configuration

```rust
use velollm_adapters_ollama::OllamaConfig;

// Read from environment
let config = OllamaConfig::from_env();

if let Some(parallel) = config.num_parallel {
    println!("Concurrent requests: {}", parallel);
}
```

### Creating Configuration

```rust
use velollm_adapters_ollama::OllamaConfig;

let mut config = OllamaConfig::default();
config.num_parallel = Some(4);
config.num_gpu = Some(999); // Offload all layers
config.keep_alive = Some("10m".to_string());

println!("{:?}", config);
```

### Generating Shell Exports

```rust
use velollm_adapters_ollama::OllamaConfig;

let mut config = OllamaConfig::default();
config.num_parallel = Some(4);
config.num_batch = Some(512);

let script = config.to_env_exports();
println!("{}", script);

// Output:
// export OLLAMA_NUM_PARALLEL=4
// export OLLAMA_NUM_BATCH=512
```

### Merging Configurations

```rust
use velollm_adapters_ollama::OllamaConfig;

let mut base = OllamaConfig::from_env();
let mut overrides = OllamaConfig::default();
overrides.num_parallel = Some(8);

base.merge(&overrides);
// base now has num_parallel=8, other values from env
```

### JSON Serialization

```rust
use velollm_adapters_ollama::OllamaConfig;

let config = OllamaConfig::from_env();

// To JSON
let json = serde_json::to_string_pretty(&config)?;
std::fs::write("ollama_config.json", json)?;

// From JSON
let loaded: OllamaConfig = serde_json::from_str(&json)?;
```

## Practical Examples

### Save Current Config

```rust
use velollm_adapters_ollama::OllamaConfig;

let config = OllamaConfig::from_env();
let json = serde_json::to_string_pretty(&config)?;
std::fs::write("ollama_backup.json", json)?;
```

### Apply Configuration

```bash
# Generate config
cargo run --example generate_config > velollm_ollama.sh

# Apply
source velollm_ollama.sh

# Restart Ollama to apply
systemctl restart ollama  # or your method
```

### Configuration Comparison

```rust
use velollm_adapters_ollama::OllamaConfig;

let before = OllamaConfig::from_env();
// ... make changes ...
let after = OllamaConfig::default();
after.num_parallel = Some(4);

if before.num_parallel != after.num_parallel {
    println!("num_parallel changed: {:?} → {:?}",
        before.num_parallel, after.num_parallel);
}
```

## Configuration Guidelines

### For Performance

```rust
// High VRAM GPU (24GB+)
config.num_parallel = Some(4);
config.num_gpu = Some(999);
config.num_batch = Some(512);
config.num_ctx = Some(8192);

// Medium VRAM GPU (12GB)
config.num_parallel = Some(2);
config.num_gpu = Some(999);
config.num_batch = Some(256);
config.num_ctx = Some(4096);

// Low VRAM GPU (8GB)
config.num_parallel = Some(1);
config.num_gpu = Some(999);
config.num_batch = Some(128);
config.num_ctx = Some(2048);
```

### For Memory Conservation

```rust
// Minimal memory footprint
config.max_loaded_models = Some(1);
config.keep_alive = Some("1m".to_string());
config.num_ctx = Some(2048);
```

### For CPU-Only

```rust
// Optimize for CPU
config.num_gpu = Some(0); // Disable GPU
config.num_thread = Some(num_cpus::get() as u32);
config.num_batch = Some(128);
```

## API Reference

### Methods

#### `OllamaConfig::from_env()`
Read configuration from environment variables.

#### `OllamaConfig::default()`
Create empty configuration (all values `None`).

#### `config.to_env_exports()`
Generate shell `export` commands.

#### `config.is_empty()`
Check if any configuration values are set.

#### `config.merge(&other)`
Merge another config, preferring values from `other`.

## Testing

```bash
cargo test -p velollm-adapters-ollama
```

All tests are unit tests and don't require Ollama to be running.

## Integration with VeloLLM

This adapter fulfills TASK-008 requirements:
- ✅ Parse Ollama environment variables
- ✅ Generate shell export scripts
- ✅ Type-safe configuration management

Used by TASK-009 (optimizer) to generate optimized configurations based on hardware.

## Environment Variable Details

### Boolean Values

Boolean env vars accept:
- `"1"` or `"true"` → `true`
- `"0"` or `"false"` → `false`
- Other values → parse error (ignored)

### Duration Strings

`OLLAMA_KEEP_ALIVE` accepts duration strings:
- `"5m"` = 5 minutes
- `"1h"` = 1 hour
- `"30s"` = 30 seconds

### GPU Layers

`OLLAMA_NUM_GPU`:
- `-1` = auto-detect
- `0` = CPU only
- `999` = all layers to GPU
- Specific number = offload N layers

## Troubleshooting

### Config Not Applied

Ollama reads env vars at startup. After changing config:
1. Generate exports: `config.to_env_exports()`
2. Source in shell
3. Restart Ollama

### Parse Errors

Invalid values are silently ignored (set to `None`):
```rust
// OLLAMA_NUM_PARALLEL=invalid
let config = OllamaConfig::from_env();
assert_eq!(config.num_parallel, None); // Parse failed
```

### GPU Not Used

If `num_gpu` is set but GPU not used:
- Check CUDA/ROCm installation
- Verify Ollama built with GPU support
- Check `nvidia-smi` or `rocm-smi` output

## References

- [Ollama FAQ - Configuration](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)
- [Ollama Environment Variables](https://github.com/ollama/ollama/blob/main/docs/faq.md)

## License

MIT - See repository root LICENSE
