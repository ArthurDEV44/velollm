# VeloLLM Architecture

This document describes the architecture, design decisions, and code organization of VeloLLM.

## Overview

VeloLLM is an **autopilot for local LLM inference**, designed to automatically detect hardware, optimize configurations, and benchmark performance. The project is built in Rust for maximum performance.

```
                           ┌────────────────────────┐
                           │      velollm CLI       │
                           │   (velollm-cli crate)  │
                           └───────────┬────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
            ▼                          ▼                          ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   velollm-core    │    │ velollm-benchmarks│    │     adapters      │
│ Hardware Detection│    │   Performance     │    │  ollama, llamacpp │
│   Optimization    │    │   Measurement     │    │                   │
└───────────────────┘    └───────────────────┘    └───────────────────┘
```

## Crate Structure

### Workspace Layout

```
velollm/
├── Cargo.toml              # Workspace configuration
├── velollm-core/           # Core library
├── velollm-cli/            # CLI binary
├── velollm-benchmarks/     # Benchmarking library
├── adapters/
│   ├── ollama/             # Ollama adapter
│   └── llamacpp/           # llama.cpp adapter
└── benchmarks/
    └── speculative/        # Speculative decoding benchmarks
```

### Crate Responsibilities

#### velollm-core

**Purpose**: Core functionality shared across all tools.

**Modules**:
- `hardware.rs`: Hardware detection (GPU, CPU, memory)
- `optimizer.rs`: Configuration optimization algorithms

**Key Types**:
```rust
// Hardware detection
pub struct HardwareSpec {
    pub gpu: Option<GpuInfo>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub os: String,
    pub platform: String,
}

// Optimization output
pub struct OptimizedConfig {
    pub num_parallel: i32,
    pub max_loaded_models: u32,
    pub keep_alive: String,
    pub num_ctx: u32,
    pub num_batch: u32,
    pub num_gpu: i32,
    pub num_thread: Option<u32>,
}
```

#### velollm-cli

**Purpose**: User-facing command-line interface.

**Commands**:
- `detect`: Show hardware information
- `benchmark`: Run performance benchmarks
- `optimize`: Generate optimized configuration

**Dependencies**:
- `velollm-core`: Hardware and optimization
- `velollm-benchmarks`: Benchmark execution
- `velollm-adapters-ollama`: Ollama configuration

#### velollm-benchmarks

**Purpose**: Performance measurement and reporting.

**Features**:
- Ollama API integration (HTTP)
- Standard benchmark suite
- Metrics collection (tokens/s, TTFT, etc.)

**Key Types**:
```rust
pub struct BenchmarkConfig {
    pub name: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub iterations: u32,
}

pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub total_tokens: u32,
}
```

#### adapters/ollama

**Purpose**: Ollama-specific configuration handling.

**Features**:
- Parse environment variables
- Generate shell exports
- Merge configurations

**Key Types**:
```rust
pub struct OllamaConfig {
    pub num_parallel: Option<u32>,
    pub max_loaded_models: Option<u32>,
    pub keep_alive: Option<String>,
    pub num_ctx: Option<u32>,
    pub num_batch: Option<u32>,
    pub num_gpu: Option<i32>,
    pub num_thread: Option<u32>,
    // ...
}
```

#### adapters/llamacpp

**Purpose**: Direct llama.cpp integration for advanced features.

**Features**:
- Speculative decoding wrapper
- Performance metrics parsing
- Binary execution management

---

## Design Decisions

### 1. Rust as Primary Language

**Decision**: Use Rust for all core functionality.

**Rationale**:
- Performance parity with C/C++ (critical for inference)
- Memory safety without garbage collection
- Excellent async support (Tokio)
- Strong type system catches errors at compile time
- Growing ecosystem for system programming

**Trade-offs**:
- Steeper learning curve
- Longer compilation times
- Less ML ecosystem support (vs Python)

### 2. Workspace Architecture

**Decision**: Use Cargo workspace with multiple crates.

**Rationale**:
- Separation of concerns (core, CLI, benchmarks)
- Independent versioning possible
- Faster incremental builds
- Clear public API boundaries

**Structure**:
```toml
[workspace]
members = [
    "velollm-core",
    "velollm-cli",
    "velollm-benchmarks",
    "adapters/llamacpp",
    "adapters/ollama",
]
```

### 3. Adapter Pattern for Backends

**Decision**: Abstract backend differences behind adapter traits.

**Rationale**:
- Support multiple backends (Ollama, llama.cpp, LocalAI, vLLM)
- Consistent interface for optimization and benchmarking
- Easy to add new backends

**Current Adapters**:
- `velollm-adapters-ollama`: Environment variable configuration
- `velollm-adapters-llamacpp`: Binary execution wrapper

**Future**: Unified trait interface:
```rust
pub trait InferenceBackend {
    async fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<String>;
    fn get_config(&self) -> &BackendConfig;
    fn set_config(&mut self, config: BackendConfig);
}
```

### 4. Hardware Detection Strategy

**Decision**: Use platform-specific commands and fallbacks.

**Strategy**:
```
GPU Detection Priority:
1. nvidia-smi (NVIDIA CUDA)
2. rocm-smi (AMD ROCm)
3. system_profiler (Apple Silicon)
4. lspci + parsing (Intel/generic)

CPU Detection:
- Linux: sysinfo crate + /proc/cpuinfo
- macOS: sysctl
- Windows: WMI

Memory Detection:
- Cross-platform: sysinfo crate
```

**Rationale**:
- Most accurate: use vendor tools when available
- Fallback: generic system tools
- Cross-platform: sysinfo crate for portability

### 5. Configuration Generation

**Decision**: Generate shell scripts rather than directly modifying environment.

**Rationale**:
- Non-invasive: user controls when to apply
- Transparent: user can inspect before sourcing
- Persistent: easy to add to shell profile
- Safe: no root/admin required

**Output Format**:
```bash
#!/bin/bash
# VeloLLM - Optimized Ollama Configuration
# Generated: 2025-01-15T12:00:00Z

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_BATCH=512
export OLLAMA_NUM_CTX=8192
```

---

## Data Flow

### Optimization Flow

```
┌─────────────┐
│ velollm     │
│ optimize    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ 1. Detect Hardware      │
│    HardwareSpec::detect │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 2. Read Current Config      │
│    OllamaConfig::from_env() │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ 3. Generate Optimized Config    │
│    OllamaOptimizer::optimize()  │
│                                 │
│    Heuristics:                  │
│    - VRAM → num_parallel        │
│    - VRAM → num_batch           │
│    - VRAM → num_ctx             │
│    - RAM → max_loaded_models    │
│    - RAM → keep_alive           │
│    - GPU presence → num_gpu     │
│    - CPU threads → num_thread   │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 4. Output                   │
│    - Dry run: show diff     │
│    - File: write shell script│
└─────────────────────────────┘
```

### Benchmark Flow

```
┌─────────────┐
│ velollm     │
│ benchmark   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│ 1. Check Ollama Available   │
│    HTTP GET /api/tags       │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 2. Load Standard Benchmarks │
│    get_standard_benchmarks()│
│    - short_completion       │
│    - medium_completion      │
│    - code_generation        │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 3. For Each Benchmark       │
│    - Send N requests        │
│    - Measure timing         │
│    - Parse API response     │
│    - Calculate metrics      │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 4. Aggregate Results        │
│    - Average tokens/s       │
│    - Average TTFT           │
│    - Total tokens           │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ 5. Output                   │
│    - Console summary        │
│    - JSON file (optional)   │
└─────────────────────────────┘
```

---

## Error Handling

### Strategy

VeloLLM uses a layered error handling approach:

1. **Library code**: `thiserror` for typed errors
2. **Application code**: `anyhow` for convenient error propagation
3. **User-facing**: Friendly messages with actionable suggestions

### Error Categories

| Category | Handling | Example |
|----------|----------|---------|
| Hardware detection failure | Graceful degradation | GPU not detected → CPU mode |
| Ollama not running | Clear error message | "Start Ollama with `ollama serve`" |
| Network errors | Retry with timeout | HTTP connection failed |
| Invalid configuration | Validation at parse time | Invalid context window size |

---

## Testing Strategy

### Test Levels

1. **Unit Tests** (`#[test]`): Test individual functions
   - Located in same file as code (`mod tests`)
   - Fast, no external dependencies

2. **Integration Tests** (`tests/`): Test complete workflows
   - Located in `velollm-cli/tests/`
   - Test CLI end-to-end
   - May require Ollama running

3. **Doc Tests** (`///`): Test documentation examples
   - Ensure examples compile and work

### Test Coverage

| Crate | Tests | Coverage |
|-------|-------|----------|
| velollm-core | 13 | Hardware detection, optimization |
| velollm-benchmarks | 3 | Config, runner creation |
| velollm-adapters-llamacpp | 6 | Metrics parsing |
| velollm-adapters-ollama | 6 | Config handling |
| velollm-cli (integration) | 8 | CLI commands |
| **Total** | **39** | All major paths |

---

## Performance Considerations

### Optimization Targets

1. **CLI startup**: < 100ms for simple commands
2. **Hardware detection**: < 500ms (cached after first run)
3. **Benchmark overhead**: < 1% of actual inference time
4. **Memory footprint**: < 10 MB for CLI tool

### Hot Paths

- **Benchmark loop**: Minimize overhead per iteration
- **Metrics parsing**: Efficient string processing
- **JSON serialization**: Use `serde_json` with efficient writers

---

## Future Architecture

### Phase 2: Multi-Backend

```
                    ┌────────────────────────┐
                    │   VeloLLM Orchestrator │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Ollama        │    │ llama.cpp     │    │ vLLM          │
│ Backend       │    │ Backend       │    │ Backend       │
│ (HTTP API)    │    │ (Binary)      │    │ (gRPC)        │
└───────────────┘    └───────────────┘    └───────────────┘
```

### Phase 3: GUI + Monitoring

```
┌────────────────────────────────────────────┐
│              Tauri Desktop App              │
│  ┌────────────────┐  ┌─────────────────┐   │
│  │   React UI     │  │  Performance    │   │
│  │   Dashboard    │  │  Charts         │   │
│  └───────┬────────┘  └───────┬─────────┘   │
│          │                   │             │
│          └─────────┬─────────┘             │
│                    │                       │
│          ┌────────▼─────────┐              │
│          │ Rust Backend     │              │
│          │ (velollm-core)   │              │
│          └──────────────────┘              │
└────────────────────────────────────────────┘
```

---

## Contributing

When adding new features:

1. **New backend**: Create adapter in `adapters/`
2. **New optimization**: Add to `velollm-core/src/optimizer.rs`
3. **New CLI command**: Add to `velollm-cli/src/main.rs`
4. **New benchmark**: Add to `velollm-benchmarks/src/lib.rs`

Always include:
- Unit tests for core logic
- Integration tests for CLI changes
- Documentation for public APIs
