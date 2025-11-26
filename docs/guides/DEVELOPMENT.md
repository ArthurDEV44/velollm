# VeloLLM Development Guide

## Prerequisites

### Required

- **Rust** 1.70+ (install via [rustup](https://rustup.rs/))
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

- **Git** for version control

### Optional but Recommended

- **cargo-watch**: Auto-rebuild on file changes
  ```bash
  cargo install cargo-watch
  ```

- **cargo-tarpaulin**: Code coverage
  ```bash
  cargo install cargo-tarpaulin
  ```

## Building the Project

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/velollm
cd velollm

# Build in debug mode
cargo build

# Build in release mode (optimized)
cargo build --release

# Or use Make
make build        # Release build
make build-dev    # Debug build
```

### Workspace Structure

VeloLLM uses a Cargo workspace with three crates:

```
velollm/
├── velollm-core/          # Core library (hardware detection, optimization)
├── velollm-cli/           # CLI application
├── velollm-benchmarks/    # Benchmarking library
└── Cargo.toml             # Workspace configuration
```

## Development Workflow

### 1. Make Changes

```bash
# Edit files in src/
vim velollm-core/src/hardware.rs
```

### 2. Format Code

```bash
cargo fmt --all
# Or
make fmt
```

### 3. Check with Clippy

```bash
cargo clippy --all -- -D warnings
# Or
make clippy
```

### 4. Run Tests

```bash
cargo test --all
# Or
make test

# With output
cargo test --all -- --nocapture
# Or
make test-verbose
```

### 5. Build and Run

```bash
# Build CLI
cargo build

# Run commands
./target/debug/velollm detect
./target/debug/velollm benchmark
./target/debug/velollm optimize --dry-run

# Or use Make shortcuts
make run-detect
make run-benchmark
make run-optimize
```

## Development Commands

### Using Make

```bash
make help           # Show all available commands
make build          # Build release
make test           # Run tests
make fmt            # Format code
make clippy         # Lint code
make doc            # Generate and open docs
make ci             # Run all CI checks (fmt + clippy + test)
```

### Using Cargo Directly

```bash
# Check without building
cargo check --all

# Build specific crate
cargo build -p velollm-core

# Run tests for specific crate
cargo test -p velollm-benchmarks

# Generate documentation
cargo doc --all --no-deps --open

# Install CLI locally
cargo install --path velollm-cli
```

## Watch Mode (Auto-rebuild)

Install cargo-watch:
```bash
cargo install cargo-watch
```

Then use:
```bash
# Watch and rebuild on changes
cargo watch -x build

# Watch and run tests
cargo watch -x test

# Or use Make
make watch
make watch-test
```

## Testing

### Unit Tests

Write tests in the same file:

```rust
// In velollm-core/src/hardware.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let hw = HardwareSpec::detect().unwrap();
        assert!(hw.cpu.cores > 0);
    }
}
```

### Integration Tests

Create files in `tests/`:

```rust
// tests/integration_test.rs
use velollm_core::hardware::HardwareSpec;

#[test]
fn test_full_detection() {
    let hw = HardwareSpec::detect().unwrap();
    assert!(!hw.os.is_empty());
}
```

### Running Specific Tests

```bash
# Run all tests
cargo test

# Run tests matching a pattern
cargo test hardware

# Run tests for specific crate
cargo test -p velollm-core

# Show test output
cargo test -- --nocapture
```

## Code Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --all --out Html

# Open coverage report
open tarpaulin-report.html
```

## Debugging

### Using rust-gdb/rust-lldb

```bash
# Build with debug symbols
cargo build

# Debug with gdb (Linux)
rust-gdb target/debug/velollm

# Debug with lldb (macOS)
rust-lldb target/debug/velollm
```

### Using VSCode

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug velollm",
      "cargo": {
        "args": ["build", "--bin=velollm"]
      },
      "args": ["detect"],
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

## Performance Profiling

### Using perf (Linux)

```bash
# Build with release + debug symbols
cargo build --release

# Profile with perf
perf record --call-graph dwarf ./target/release/velollm benchmark
perf report
```

### Using Instruments (macOS)

```bash
# Build release
cargo build --release

# Open with Instruments
instruments -t "Time Profiler" ./target/release/velollm benchmark
```

## Documentation

### Generating Docs

```bash
# Generate and open documentation
cargo doc --all --no-deps --open

# Or use Make
make doc
```

### Writing Documentation

Use doc comments:

```rust
/// Detects the current hardware configuration.
///
/// # Returns
///
/// A `HardwareSpec` struct containing GPU, CPU, and memory information.
///
/// # Errors
///
/// Returns an error if system information cannot be accessed.
///
/// # Examples
///
/// ```
/// use velollm_core::hardware::HardwareSpec;
///
/// let hw = HardwareSpec::detect()?;
/// println!("CPU cores: {}", hw.cpu.cores);
/// ```
pub fn detect() -> anyhow::Result<HardwareSpec> {
    // Implementation
}
```

## Continuous Integration

CI runs on every push and PR:

- **Formatting**: `cargo fmt --check`
- **Linting**: `cargo clippy`
- **Tests**: `cargo test --all`
- **Builds**: Debug and release builds

See `.github/workflows/ci.yml` for details.

## Troubleshooting

### Cargo Build Errors

```bash
# Clean and rebuild
cargo clean
cargo build

# Or
make clean
make build
```

### Clippy Warnings

Fix all clippy warnings before committing:

```bash
# See warnings
cargo clippy --all

# Auto-fix (when possible)
cargo clippy --all --fix
```

### Test Failures

```bash
# Run specific failing test
cargo test test_name -- --nocapture --test-threads=1

# Enable logging
RUST_LOG=debug cargo test
```

## Next Steps

- Read [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines
- Check [TODO.md](../../TODO.md) for tasks to work on
- See [ROADMAP.md](../../ROADMAP.md) for project direction

## Questions?

Open a GitHub Discussion or reach out to maintainers.
