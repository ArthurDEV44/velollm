# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VeloLLM is an autopilot for local LLM inference optimization. It provides zero-config performance optimization for Ollama, llama.cpp, and similar tools. The goal is to close the 35-50x performance gap between local inference (200-300 tok/s) and cloud solutions (10,000+ tok/s).

**Core Technologies:**
- Backend: Rust (performance-critical optimizations)
- CLI: clap-based command-line interface
- Async runtime: Tokio

## Workspace Architecture

This is a Cargo workspace with three crates:

1. **velollm-core** (`velollm-core/`): Core library for hardware detection and optimization
   - Hardware detection (GPU, CPU, memory)
   - Optimization logic
   - Cross-platform system information gathering

2. **velollm-cli** (`velollm-cli/`): CLI application binary
   - Commands: `detect`, `benchmark`, `optimize`
   - Uses velollm-core and velollm-benchmarks

3. **velollm-benchmarks** (`velollm-benchmarks/`): Benchmarking library
   - Ollama API integration for benchmarking
   - Standard benchmark configurations
   - Performance metrics collection (tokens/s, TTFT, etc.)

## Common Development Commands

### Building
```bash
# Debug build (faster compile, slower runtime)
cargo build

# Release build (slower compile, optimized)
cargo build --release

# Or use Make shortcuts
make build        # Release build
make build-dev    # Debug build
```

### Testing
```bash
# Run all tests
cargo test --all
make test

# Run tests with output
cargo test --all -- --nocapture
make test-verbose

# Run tests for specific crate
cargo test -p velollm-core
cargo test -p velollm-benchmarks

# Run specific test by name
cargo test test_hardware_detection
cargo test hardware  # Pattern match
```

### Linting and Formatting
```bash
# Format code
cargo fmt --all
make fmt

# Check formatting (CI)
cargo fmt --all -- --check
make fmt-check

# Lint with Clippy (must pass with no warnings)
cargo clippy --all -- -D warnings
make clippy

# Run all CI checks
make ci  # Runs fmt-check, clippy, and test
```

### Running the CLI
```bash
# After building, run from target directory
./target/debug/velollm detect
./target/debug/velollm benchmark
./target/debug/velollm optimize --dry-run

# Or use Make shortcuts
make run-detect
make run-benchmark
make run-optimize

# Install to ~/.cargo/bin
cargo install --path velollm-cli
# Then use: velollm detect
```

### Documentation
```bash
# Generate and open Rust docs
cargo doc --all --no-deps --open
make doc
```

## Hardware Detection Architecture

The hardware detection system (`velollm-core/src/hardware.rs`) uses platform-specific commands:

- **Linux**: `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), `lspci` (Intel/fallback)
- **macOS**: `system_profiler` (Apple Silicon), `sysctl` (CPU/memory)
- **Windows**: `nvidia-smi.exe` (NVIDIA)

The `HardwareSpec` struct includes:
- `gpu`: Optional GPU information (vendor, VRAM, driver, compute capability)
- `cpu`: CPU info (model, cores, threads, frequency)
- `memory`: Memory info (total, available, used)
- `os`: Operating system string
- `platform`: Platform identifier (e.g., "linux-x86_64")

## Benchmarking System

The benchmarking system (`velollm-benchmarks/src/lib.rs`) currently supports:

- **Ollama backend**: HTTP API calls to localhost:11434
- Standard benchmarks: short_completion, medium_completion, code_generation
- Metrics: tokens/s, time-to-first-token (TTFT), total time, token counts

To run benchmarks:
1. Ollama must be running (`ollama serve`)
2. Model must be downloaded (e.g., `ollama pull llama3.2:3b`)
3. Run: `velollm benchmark --model llama3.2:3b --output results.json`

## Development Workflow

1. Make changes in the appropriate crate
2. Format: `cargo fmt --all`
3. Lint: `cargo clippy --all -- -D warnings`
4. Test: `cargo test --all`
5. Build and test manually: `make run-detect` or similar

## CI Requirements

All code must pass:
- `cargo fmt --all -- --check` (formatting)
- `cargo clippy --all -- -D warnings` (no warnings allowed)
- `cargo test --all` (all tests pass)

See `.github/workflows/ci.yml` for the full CI pipeline.

## Project Phases

**Current Phase**: Phase 1 - MVP Development

Phase 1 focuses on:
- Hardware detection (✅ Complete)
- Benchmarking suite (✅ Complete)
- Ollama optimization (⏳ Planned in TASK-009)
- Speculative decoding PoC (⏳ Planned)

Later phases include PagedAttention, continuous batching, multi-backend support, and GUI dashboard.

## Important Files

- `Cargo.toml` (workspace root): Workspace configuration and shared dependencies
- `Makefile`: Development shortcuts
- `TODO.md`: Task tracking and development status
- `ROADMAP.md`: Long-term project direction
- `docs/guides/DEVELOPMENT.md`: Detailed development guide
- `docs/guides/TESTING.md`: Testing strategies and platform-specific tests

## Testing Notes

- Hardware detection tests will show different results based on available hardware
- GPU tests gracefully handle systems without GPUs (returns `gpu: None`)
- Benchmark tests require Ollama to be running
- Use `-- --nocapture` to see test output for debugging

## Dependencies

Key workspace dependencies (from `Cargo.toml`):
- tokio: Async runtime
- serde/serde_json: Serialization
- clap: CLI parsing
- anyhow/thiserror: Error handling
- sysinfo/num_cpus: System information
- reqwest: HTTP client for Ollama API

## Code Style

- Use `rustfmt` defaults (configured in `.rustfmt.toml`)
- Clippy warnings treated as errors
- Prefer `anyhow::Result` for error handling in application code
- Use structured error types with `thiserror` for library code
- Write doc comments for public APIs
