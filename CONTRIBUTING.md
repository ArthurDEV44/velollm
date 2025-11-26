# Contributing to VeloLLM

Thank you for your interest in contributing to VeloLLM! This document provides guidelines and information for contributors.

---

## Getting Started

### Prerequisites

- **Rust**: 1.70+ (install via [rustup](https://rustup.rs/))
- **Node.js**: 18+ (for TypeScript tooling)
- **Git**: For version control
- **Hardware**: GPU recommended for benchmarking (but not required)

### Setting Up Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/velollm.git
cd velollm

# 2. Build the project
cargo build

# 3. Run tests
cargo test

# 4. Install development tools
cargo install cargo-watch
cargo install cargo-clippy
```

---

## Development Workflow

### 1. Pick a Task

Check [TODO.md](TODO.md) for available tasks, organized by priority:
- **P0**: Blocking tasks (infrastructure, core features)
- **P1**: High priority (optimization implementations)
- **P2**: Medium priority (advanced features)
- **P3**: Low priority (nice-to-haves)

### 2. Create a Branch

```bash
git checkout -b feature/task-XXX-description
```

Branch naming convention:
- `feature/task-XXX-name`: New features
- `fix/issue-YYY-name`: Bug fixes
- `docs/topic`: Documentation updates
- `refactor/component`: Code refactoring

### 3. Develop with Tests

**Every code change must include tests.**

```rust
// Example: src/core/hardware.rs
pub fn detect_gpu() -> Option<GpuInfo> {
    // Implementation
}

// tests/hardware_test.rs
#[test]
fn test_gpu_detection() {
    let gpu = detect_gpu();
    if gpu.is_some() {
        assert!(gpu.unwrap().vram_total_mb > 0);
    }
}
```

Run tests continuously:
```bash
cargo watch -x test
```

### 4. Code Quality

Before committing, ensure:

```bash
# Format code
cargo fmt

# Lint with clippy
cargo clippy -- -D warnings

# Run all tests
cargo test --all
```

### 5. Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

Examples:
```
feat(hardware): add AMD GPU detection via rocm-smi

Implements ROCm support for detecting AMD GPUs.
Parses rocm-smi output to extract VRAM and compute capability.

Closes #42
```

```
fix(benchmark): correct tokens/s calculation

Previous calculation didn't account for warmup iterations.
Now excludes first run from average.
```

### 6. Submit Pull Request

```bash
# Push your branch
git push origin feature/task-XXX-description

# Create PR on GitHub
# Use the PR template (auto-populated)
```

**PR Checklist**:
- [ ] Tests pass locally
- [ ] Code formatted (`cargo fmt`)
- [ ] No clippy warnings
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Linked to related issue/task

---

## Code Style

### Rust

Follow standard Rust conventions:
- Use `rustfmt` defaults
- Prefer explicit types for public APIs
- Document public functions with `///` comments
- Use `Result<T, E>` for fallible operations

```rust
/// Detects the current hardware configuration.
///
/// Returns a `HardwareSpec` containing GPU, CPU, and memory information.
///
/// # Errors
///
/// Returns an error if system information cannot be read.
///
/// # Example
///
/// ```
/// let hw = HardwareSpec::detect()?;
/// println!("GPU: {:?}", hw.gpu);
/// ```
pub fn detect() -> anyhow::Result<HardwareSpec> {
    // Implementation
}
```

### TypeScript

For TypeScript tooling:
- Use ESLint + Prettier
- Prefer `async/await` over promises
- Type everything (no `any`)

---

## Testing Guidelines

### Unit Tests

Place tests in the same file or in `tests/` directory:

```rust
// In the same file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        assert_eq!(function(), expected);
    }
}
```

### Integration Tests

Place in `tests/`:

```rust
// tests/integration_test.rs
use velollm_core::*;

#[test]
fn test_full_workflow() {
    let hw = HardwareSpec::detect().unwrap();
    let config = optimize(&hw);
    assert!(config.num_gpu.is_some());
}
```

### Benchmark Tests

For performance-critical code:

```rust
#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_operation() {
        let start = Instant::now();
        expensive_operation();
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 100, "Too slow: {:?}", elapsed);
    }
}
```

---

## Documentation

### Code Documentation

- Public APIs: Always document with `///`
- Complex logic: Inline comments explaining "why", not "what"
- Examples: Include usage examples in doc comments

### Project Documentation

When adding features, update:
- `README.md`: If it changes user-facing behavior
- `ROADMAP.md`: If it affects the roadmap
- `docs/guides/`: For tutorials and how-tos
- `docs/api/`: For API documentation

---

## Benchmarking Contributions

We need benchmarks on diverse hardware! To contribute:

### 1. Run Benchmarks

```bash
# Install VeloLLM
cargo install --path velollm-cli

# Run benchmark suite
velollm detect > hardware.json
velollm benchmark -o results.json
```

### 2. Submit Results

Create a PR with:
- `benchmarks/results/hardware-{gpu-model}.json`: Hardware specs
- `benchmarks/results/benchmark-{gpu-model}.json`: Benchmark results
- Brief description of your setup in PR

### 3. Format

```json
{
  "hardware": {
    "gpu": "RTX 4090",
    "vram_gb": 24,
    "cpu": "AMD Ryzen 9 7950X",
    "ram_gb": 64
  },
  "results": {
    "model": "llama3.1:8b",
    "baseline_tps": 28.3,
    "velollm_tps": 67.1,
    "speedup": 2.37
  }
}
```

---

## Areas Needing Help

### High Priority

1. **Hardware Detection**
   - AMD GPU support (ROCm)
   - Intel GPU support (OneAPI)
   - Apple Silicon optimization

2. **Speculative Decoding**
   - Optimal draft model selection
   - Dynamic n_draft tuning
   - Acceptance rate tracking

3. **Benchmarking**
   - More comprehensive test suite
   - Memory profiling integration
   - Automated regression detection

### Medium Priority

4. **PagedAttention**
   - CUDA kernel development
   - CPU fallback implementation
   - Block manager optimization

5. **Multi-Backend**
   - LocalAI adapter
   - vLLM local mode adapter
   - Configuration unification

### Nice to Have

6. **GUI Development**
   - Tauri app
   - Real-time metrics visualization
   - Configuration wizard

7. **Documentation**
   - Video tutorials
   - Blog posts
   - Translation (non-English)

---

## Communication

### Asking Questions

- **GitHub Discussions**: General questions, ideas
- **GitHub Issues**: Bug reports, feature requests
- **PR Comments**: Code-specific questions

### Reporting Bugs

Use the bug report template. Include:
1. Environment (OS, hardware, VeloLLM version)
2. Steps to reproduce
3. Expected vs actual behavior
4. Relevant logs/screenshots

### Feature Requests

Use the feature request template. Include:
1. Problem you're solving
2. Proposed solution
3. Alternatives considered
4. Impact on existing features

---

## Recognition

Contributors are recognized in:
- `AUTHORS` file (automatic from git history)
- Release notes for significant contributions
- README credits section

---

## Code of Conduct

Be respectful, inclusive, and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

Open a [GitHub Discussion](https://github.com/yourusername/velollm/discussions) or reach out to the maintainers.

Happy contributing! 🚀
