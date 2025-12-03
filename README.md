# VeloLLM

**Autopilot for Local LLM Inference** - High-performance proxy and optimization toolkit for Ollama and llama.cpp.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml)

## The Problem

Local LLM inference is **19x slower** than production solutions like vLLM. VeloLLM bridges this gap by providing a high-performance Rust proxy that optimizes requests, improves tool calling reliability, and brings production-grade features to local deployments.

**Performance Gap**:
- Production (vLLM): 793 tokens/s, 80ms P99 latency
- Local (Ollama): 41 tokens/s, 673ms P99 latency

**VeloLLM Goal**: Bring vLLM-level performance to Ollama users while keeping the simplicity.

---

## What is VeloLLM?

VeloLLM is a **transparent proxy** that sits between your applications and Ollama. It intercepts API calls, optimizes them, and forwards them to Ollama. Your existing tools work without modification - just change the API endpoint.

**Key Benefits**:
- **Drop-in replacement**: Compatible with OpenAI API format
- **Tool calling improvements**: Fixes JSON formatting issues, deduplicates calls, validates arguments
- **Performance optimization**: Request batching, caching, intelligent scheduling
- **Metrics & observability**: Track tokens/s, latency, cache hit rates

**Supported Models for Tool Calling**:
- Mistral (mistral:7b, mistral-small:24b)
- Llama (llama3.2:3b, llama3.1:8b, llama3.1:70b)

---

## Quick Start

### Installation

```bash
git clone https://github.com/ArthurDEV44/velollm.git
cd velollm
cargo build --release
```

### Option 1: Use the Proxy (Recommended)

Start the VeloLLM proxy to get OpenAI API compatibility and optimizations:

```bash
# Make sure Ollama is running
ollama serve &

# Start VeloLLM proxy
cargo run -p velollm-proxy --release
```

The proxy listens on `http://localhost:8000` and forwards requests to Ollama at `http://localhost:11434`.

**Configure your applications** to use VeloLLM instead of Ollama directly:
- OpenAI SDK: Set `OPENAI_BASE_URL=http://localhost:8000/v1`
- Direct API calls: Replace `localhost:11434` with `localhost:8000`

**Available endpoints**:
- `POST /v1/chat/completions` - OpenAI-compatible chat
- `GET /v1/models` - List models (OpenAI format)
- `POST /api/chat` - Ollama native chat
- `POST /api/generate` - Ollama native generation
- `GET /api/tags` - List models (Ollama format)
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

### Option 2: Use the CLI Tools

VeloLLM also provides CLI tools for hardware detection, benchmarking, and Ollama configuration:

```bash
# Detect your hardware (GPU, CPU, RAM)
cargo run --bin velollm -- detect

# Benchmark Ollama performance
cargo run --bin velollm -- benchmark --model llama3.2:3b

# Generate optimized Ollama configuration
cargo run --bin velollm -- optimize --dry-run
cargo run --bin velollm -- optimize -o velollm-config.sh
source velollm-config.sh
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATIONS                         │
│      (Claude Code, Continue, Open WebUI, Custom Apps)       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VELOLLM PROXY                             │
│                   localhost:8000                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   OpenAI     │  │    Tool      │  │   Request    │       │
│  │   Compat     │  │   Optimizer  │  │   Batcher    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Semantic   │  │   Metrics    │  │   Streaming  │       │
│  │   Cache      │  │   Collector  │  │   SSE        │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       OLLAMA                                 │
│                   localhost:11434                            │
└─────────────────────────────────────────────────────────────┘
```

**Technology Stack**:
- **Core**: Rust (ultra-low latency, memory safe)
- **HTTP Server**: Axum + Tower (async, production-ready)
- **Async Runtime**: Tokio

---

## Features

### Implemented (Phase 1-3)

**Hardware Detection & Optimization**:
- Auto-detect GPU (NVIDIA, AMD, Apple Silicon, Intel)
- CPU and memory profiling
- Generate optimized Ollama environment variables

**Benchmarking**:
- Measure tokens/s, time-to-first-token, total latency
- Multiple benchmark profiles (short, medium, code generation)
- JSON export for comparison

**Proxy Server**:
- OpenAI API compatibility layer
- Ollama native API pass-through
- Server-Sent Events (SSE) streaming
- Health checks and metrics endpoint

**Internal Optimizations**:
- PagedAttention block manager
- Continuous batching scheduler
- CUDA paged attention kernels (optional)

### Coming Soon

**Tool Calling Enhancement** (TASK-023):
- Automatic JSON fixing for malformed responses
- Tool call deduplication
- Schema validation
- Intelligent retry on parsing failures

**Request Batching** (TASK-024):
- Group concurrent requests
- Maximize GPU utilization
- Priority-based scheduling

**Semantic Cache** (TASK-025):
- Cache similar prompts
- Embedding-based similarity matching
- Reduce redundant inference

---

## Configuration

### Environment Variables

**Proxy Configuration**:
- `VELOLLM_PORT`: Proxy listen port (default: 8000)
- `OLLAMA_HOST`: Ollama backend URL (default: http://localhost:11434)
- `VELOLLM_VERBOSE`: Enable verbose logging (default: false)

**Ollama Optimization** (generated by `velollm optimize`):
- `OLLAMA_NUM_PARALLEL`: Concurrent request handling
- `OLLAMA_NUM_GPU`: GPU layers to offload
- `OLLAMA_NUM_BATCH`: Batch size for prompt processing
- `OLLAMA_NUM_CTX`: Context window size
- `OLLAMA_MAX_LOADED_MODELS`: Models to keep in memory
- `OLLAMA_KEEP_ALIVE`: Model retention time

---

## Development

### Building

```bash
cargo build              # Debug build
cargo build --release    # Release build
```

### Testing

```bash
cargo test --all         # Run all tests
cargo clippy --all       # Lint
cargo fmt --all          # Format
make ci                  # Full CI check
```

### Project Structure

```
velollm/
├── velollm-core/       # Core library (hardware detection, optimization)
├── velollm-cli/        # CLI binary (detect, benchmark, optimize)
├── velollm-proxy/      # Proxy server binary
├── velollm-benchmarks/ # Benchmarking library
├── adapters/
│   ├── ollama/         # Ollama configuration parser
│   └── llamacpp/       # llama.cpp integration (PagedAttention, CUDA)
└── benchmarks/
    └── speculative/    # Speculative decoding experiments
```

---

## Comparison

| Feature | Ollama | vLLM | LM Studio | VeloLLM |
|---------|--------|------|-----------|---------|
| Target Use Case | Simplicity | Cloud production | Desktop GUI | Local performance |
| OpenAI API Compat | Partial | Full | Partial | Full |
| Tool Calling Fix | No | N/A | No | Yes |
| PagedAttention | No | Yes | No | Yes (local) |
| Request Batching | No | Yes | No | Yes |
| Auto-optimization | No | No | Partial | Yes |
| Language | Go | Python | Electron | Rust |
| Open Source | Yes | Yes | No | Yes |

---

## Roadmap

**Phase 1** (Complete): MVP with CLI tools
- Hardware detection, benchmarking, Ollama configuration

**Phase 2** (Complete): Advanced optimizations
- PagedAttention, continuous batching scheduler, CUDA kernels

**Phase 3** (In Progress): Intelligent proxy
- OpenAI compatibility, tool calling enhancement, caching, metrics

**Phase 4** (Planned): Ecosystem
- GUI dashboard, IDE integrations, configuration marketplace

Full details: [ROADMAP.md](ROADMAP.md) | Task tracking: [TODO.md](TODO.md)

---

## Contributing

We welcome contributions! Areas of interest:

- **Performance**: Optimize the proxy, reduce latency
- **Tool Calling**: Improve JSON fixing, add more edge cases
- **Caching**: Implement semantic cache with embeddings
- **Testing**: Add integration tests, benchmark on diverse hardware
- **Documentation**: Improve guides and API docs

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Links

- **Repository**: [github.com/ArthurDEV44/velollm](https://github.com/ArthurDEV44/velollm)
- **Issues**: [GitHub Issues](https://github.com/ArthurDEV44/velollm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ArthurDEV44/velollm/discussions)

---

**Status**: Phase 3 - Proxy development in progress

Built with Rust by the VeloLLM community.
