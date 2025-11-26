# VeloLLM

**Autopilot for Local LLM Inference** - Zero-config performance optimization for Ollama, llama.cpp, and more.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)

## The Problem

Local LLM inference is **35-50x slower** than cloud solutions (vLLM, Morph) despite comparable hardware. VeloLLM bridges this gap by bringing production-grade optimizations to local deployments.

**Current State**:
- Cloud (vLLM): 10,000+ tokens/s with speculative decoding
- Local (Ollama): 200-300 tokens/s (average user)

**VeloLLM Goal**: Close this performance gap with intelligent, automatic optimizations.

---

## Quick Start

### Installation

```bash
# From crates.io (coming soon)
cargo install velollm

# From source
git clone https://github.com/yourusername/velollm
cd velollm
cargo install --path velollm-cli
```

### Usage

```bash
# 1. Detect your hardware
velollm detect

# 2. Optimize Ollama configuration
velollm optimize --dry-run  # Preview changes
velollm optimize -o velollm.sh
source velollm.sh

# 3. Benchmark performance
velollm benchmark

# 4. Compare before/after
velollm benchmark --compare baseline.json optimized.json
```

---

## Features

### Phase 1 (MVP - Current)

- **Hardware Detection**: Auto-detect GPU (NVIDIA/AMD/Apple), CPU, RAM
- **Ollama Auto-Configuration**: Optimize VRAM usage, batch size, context window
- **Benchmarking Suite**: Measure tokens/s, time-to-first-token, memory usage
- **Speculative Decoding**: 1.5-2.5x speedup via draft model integration

### Phase 2 (Months 4-6)

- **PagedAttention**: 70% reduction in KV cache fragmentation
- **Continuous Batching**: Handle 4-8 concurrent users efficiently
- **CPU-GPU Hybrid**: Intelligent layer placement and offloading
- **Multi-Backend**: Support for llama.cpp, LocalAI, vLLM

### Phase 3 (Months 7-12)

- **GUI Dashboard**: Real-time performance monitoring
- **IDE Integrations**: VSCode, Continue.dev, Cursor
- **Mamba/MoE Support**: Next-generation model architectures
- **Configuration Marketplace**: Community-driven optimization database

See [ROADMAP.md](ROADMAP.md) for complete details.

---

## Benchmark Results

### Expected Performance (Phase 1 Targets)

| Hardware | Model | Baseline | VeloLLM | Speedup |
|----------|-------|----------|---------|---------|
| RTX 4090 24GB | Llama 3.1 8B | ~28 tok/s | 60-70 tok/s | 2.1-2.5x |
| RTX 3060 12GB | Llama 3.2 3B | ~35 tok/s | 70-85 tok/s | 2.0-2.4x |
| M2 Max 32GB | Llama 3.1 8B | ~22 tok/s | 45-55 tok/s | 2.0-2.5x |

See [BENCHMARKS.md](BENCHMARKS.md) for methodology and detailed results.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         VeloLLM Orchestration Layer             │
│  • Hardware detection                           │
│  • Auto-configuration                           │
│  • Performance profiling                        │
└────┬──────────┬──────────┬──────────┬───────────┘
     │          │          │          │
┌────▼─────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐
│  Ollama  │ │llama.cpp│ │LocalAI │ │  vLLM  │
│ Adapter  │ │ Adapter │ │ Adapter│ │ Adapter│
└──────────┘ └─────────┘ └────────┘ └────────┘
```

**Core Technologies**:
- **Backend**: Rust (performance-critical optimizations)
- **CLI/Tooling**: TypeScript/Node.js (developer experience)
- **Bindings**: Python (ML ecosystem compatibility)

---

## Development Status

**Current Phase**: Phase 1 - MVP Development

| Task | Status |
|------|--------|
| Repository setup | ✅ Complete |
| Build system | 🚧 In progress |
| Hardware detection | ⏳ Planned |
| Benchmarking suite | ⏳ Planned |
| Speculative decoding PoC | ⏳ Planned |
| Ollama optimization | ⏳ Planned |

Track progress: [TODO.md](TODO.md)

---

## Contributing

We welcome contributions! VeloLLM is in early development and needs help with:

- **Core optimizations**: PagedAttention, speculative decoding
- **Backend adapters**: Support for more inference engines
- **Benchmarking**: Testing on diverse hardware configurations
- **Documentation**: Guides, tutorials, API docs

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Roadmap

**Phase 1 (Months 1-3)**: MVP with 2-3x speedup
- Speculative decoding integration
- Ollama auto-configuration
- Baseline benchmarking

**Phase 2 (Months 4-6)**: Advanced optimizations (3-5x)
- PagedAttention implementation
- Continuous batching for local
- Multi-backend support

**Phase 3 (Months 7-12)**: Ecosystem (5-10x)
- GUI and monitoring
- IDE integrations
- Architecture alternatives (Mamba, MoE)

Full details: [ROADMAP.md](ROADMAP.md)

---

## Why VeloLLM?

### Differentiation

| Feature | Ollama | vLLM | LM Studio | VeloLLM |
|---------|--------|------|-----------|---------|
| Target | Simplicity | Cloud prod | Desktop users | Local performance |
| Speculative Decoding | ❌ | ❌ | ✅ | ✅ Auto-configured |
| PagedAttention | ❌ | ✅ | ❌ | ✅ Local-adapted |
| Continuous Batching | ❌ | ✅ | ❌ | ✅ Multi-user |
| Auto-optimization | ❌ | ❌ | Partial | ✅ Hardware-aware |
| Open Source | ✅ | ✅ | ❌ | ✅ |

### Value Proposition

**VeloLLM = "Autopilot for Local AI Inference"**

1. **Zero-config**: Detects hardware, applies optimal settings automatically
2. **Hardware-aware**: Adapts dynamically (laptop vs workstation vs server)
3. **Multi-backend**: Works with Ollama, llama.cpp, LocalAI transparently
4. **Transparent**: Detailed monitoring, metrics, optimization explanations
5. **Community-driven**: Open source, extensible, well-documented

---

## Research & References

This project builds on:

- [llama.cpp](https://github.com/ggml-org/llama.cpp): Foundation for speculative decoding
- [vLLM](https://github.com/vllm-project/vllm): PagedAttention and continuous batching
- [Ollama](https://github.com/ollama/ollama): User experience and API design
- [Mamba](https://github.com/state-spaces/mamba): Alternative architecture exploration

Key papers:
- [PagedAttention](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html): Memory optimization
- [Speculative Decoding](https://arxiv.org/abs/2211.17192): Inference acceleration

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/velollm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/velollm/discussions)
- **Email**: velollm@example.com (placeholder)

---

**Status**: Early development - contributions and feedback welcome!

Built with by the VeloLLM community.
