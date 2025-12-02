# VeloLLM v0.1.0 - Phase 1 MVP

**Autopilot for Local LLM Inference** - First public release!

VeloLLM automatically detects your hardware and optimizes Ollama configuration for maximum performance.

---

## What's Included

### 🔍 Hardware Detection (`velollm detect`)

Automatically detects your system configuration:
- **GPU**: NVIDIA (CUDA), AMD (ROCm), Apple Silicon, Intel
- **CPU**: Model, cores, threads, frequency
- **Memory**: Total, available, used RAM
- **Platform**: OS and architecture

```bash
velollm detect
```

### ⚡ Ollama Optimization (`velollm optimize`)

Generates optimal Ollama configuration based on your hardware:
- `OLLAMA_NUM_PARALLEL`: Concurrent requests (based on VRAM)
- `OLLAMA_NUM_GPU`: GPU layer offloading
- `OLLAMA_NUM_BATCH`: Batch size for prompt processing
- `OLLAMA_NUM_CTX`: Context window size
- `OLLAMA_MAX_LOADED_MODELS`: Model caching
- `OLLAMA_KEEP_ALIVE`: Model retention time

```bash
# Preview recommendations
velollm optimize --dry-run

# Generate shell script
velollm optimize -o velollm-config.sh
source velollm-config.sh
```

### 📊 Benchmarking Suite (`velollm benchmark`)

Measure inference performance with standardized tests:
- **Tokens/s**: Generation throughput
- **TTFT**: Time to first token (latency)
- **Total time**: End-to-end performance

```bash
velollm benchmark --model llama3.2:3b -o results.json
```

---

## Quick Start

### Installation

```bash
# Clone and build
git clone https://github.com/ArthurDEV44/velollm.git
cd velollm
cargo build --release

# Or install directly
cargo install --path velollm-cli
```

### Usage

```bash
# 1. Check your hardware
velollm detect

# 2. Optimize Ollama
velollm optimize -o velollm-config.sh
source velollm-config.sh

# 3. Benchmark (requires Ollama running)
ollama serve &
velollm benchmark --model llama3.2:3b
```

---

## Benchmark Results

Tested on RTX 4070 Ti SUPER + Ryzen 7 7800X3D:

| Benchmark | Tokens/s | TTFT |
|-----------|----------|------|
| Short completion (50 tok) | 65.6 | 16ms |
| Medium completion (150 tok) | 170.5 | 22ms |
| Code generation (200 tok) | 175.4 | 22ms |
| **Average** | **137.2** | **20ms** |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed methodology.

---

## Project Status

### Phase 1 MVP - Complete ✅

| Task | Status |
|------|--------|
| Hardware detection | ✅ |
| Ollama configuration parser | ✅ |
| Hardware-based optimizer | ✅ |
| CLI commands (detect, optimize, benchmark) | ✅ |
| Benchmarking suite | ✅ |
| Speculative decoding wrapper (llama.cpp) | ✅ |
| Integration tests (39 tests) | ✅ |
| Documentation | ✅ |

### Coming in Phase 2

- PagedAttention for memory optimization
- Continuous batching for concurrent users
- Multi-backend support (llama.cpp, LocalAI, vLLM)
- CPU-GPU hybrid execution

---

## Documentation

- [README.md](README.md) - Project overview
- [ROADMAP.md](ROADMAP.md) - Development roadmap
- [BENCHMARKS.md](BENCHMARKS.md) - Performance results
- [docs/guides/CONFIG_GUIDE.md](docs/guides/CONFIG_GUIDE.md) - Configuration parameters
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture

---

## Requirements

- **Rust**: 1.70+ (for building)
- **Ollama**: For benchmarking and optimization target
- **OS**: Linux, macOS, Windows (WSL2)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE)
