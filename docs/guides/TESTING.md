# Testing VeloLLM

## Running Tests

### All Tests

```bash
# Run all tests
cargo test --all

# With verbose output
cargo test --all -- --nocapture

# Or use Make
make test
make test-verbose
```

### Specific Crate Tests

```bash
# Test core library only
cargo test -p velollm-core

# Test CLI only
cargo test -p velollm-cli

# Test benchmarks only
cargo test -p velollm-benchmarks
```

### Specific Test Function

```bash
# Run single test
cargo test test_hardware_detection

# Run all tests matching pattern
cargo test hardware
```

## Hardware Detection Tests

### Running Hardware Tests

```bash
# Test hardware detection
cargo test -p velollm-core -- --nocapture

# This will output detected hardware to console
```

**Expected output:**
```
running 8 tests
test hardware_tests::tests::test_cpu_detection ... ok
test hardware_tests::tests::test_gpu_detection ... ok
test hardware_tests::tests::test_hardware_detection ... ok
test hardware_tests::tests::test_json_serialization ... ok
test hardware_tests::tests::test_memory_detection ... ok
test hardware_tests::tests::test_nvidia_detection_on_linux ... ok
test hardware_tests::tests::test_platform_string ... ok
test hardware_tests::tests::test_gpu_vendor_serialization ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### GPU Detection Test Results

Tests will show different results based on available hardware:

**NVIDIA GPU present:**
```
GPU detected:
  Name: NVIDIA GeForce RTX 4090
  Vendor: Nvidia
  VRAM Total: 24564 MB (24 GB)
  VRAM Free: 23012 MB (22 GB)
  Driver: 535.129.03
  Compute Capability: 8.9
```

**No GPU (CPU only):**
```
No GPU detected (running on CPU-only system)
```

**Apple Silicon:**
```
GPU detected:
  Name: Apple M2
  Vendor: Apple
  VRAM Total: 32768 MB (32 GB)
  VRAM Free: 32768 MB (32 GB)
```

## Testing the CLI

### Build and Test Locally

```bash
# Build in debug mode
cargo build

# Run detect command
./target/debug/velollm detect

# Test with Make
make run-detect
```

### Expected Output

```
🔍 Detecting hardware configuration...

=== System Information ===
OS: linux
Platform: linux-x86_64

=== CPU ===
Model: AMD Ryzen 9 7950X 16-Core Processor
Cores: 16
Threads: 32
Frequency: 4500 MHz

=== Memory ===
Total: 65536 MB (64.0 GB)
Available: 42384 MB (41.4 GB)
Used: 23152 MB (22.6 GB)

=== GPU ===
Name: NVIDIA GeForce RTX 4090
Vendor: Nvidia
VRAM Total: 24564 MB (24.0 GB)
VRAM Free: 23012 MB (22.5 GB)
Driver: 535.129.03
Compute Capability: 8.9

=== JSON Output ===
{
  "gpu": { ... },
  "cpu": { ... },
  ...
}
```

## Platform-Specific Testing

### Linux

**Prerequisites:**
- NVIDIA: `nvidia-smi` installed
- AMD: `rocm-smi` installed
- Intel: `lspci` available (usually pre-installed)

**Test commands:**
```bash
# Check if nvidia-smi is available
which nvidia-smi

# Check if rocm-smi is available
which rocm-smi

# Run tests
cargo test -p velollm-core
```

### macOS

**Prerequisites:**
- `system_profiler` (built-in)
- `sysctl` (built-in)

**Test commands:**
```bash
# Test Apple Silicon detection
cargo test -p velollm-core -- --nocapture

# Should detect M1/M2/M3 if running on Apple Silicon
```

### Windows

**Prerequisites:**
- NVIDIA: `nvidia-smi.exe` in PATH

**Test commands:**
```powershell
# Check nvidia-smi
where nvidia-smi

# Run tests
cargo test -p velollm-core
```

## Continuous Integration

Tests run automatically on:
- Every push to `main` branch
- Every pull request

See `.github/workflows/ci.yml` for configuration.

**CI runs tests on:**
- Ubuntu (Linux)
- macOS
- Windows

## Manual Testing Checklist

Before creating a PR, verify:

- [ ] `cargo test --all` passes
- [ ] `cargo clippy --all` has no warnings
- [ ] `cargo fmt --all -- --check` passes
- [ ] `velollm detect` works on your system
- [ ] JSON output is valid (test with `jq`)

```bash
# Validate JSON output
./target/debug/velollm detect | tail -n +17 | jq .
```

## Troubleshooting Tests

### Test Fails: "nvidia-smi not found"

**Cause:** NVIDIA drivers not installed or nvidia-smi not in PATH

**Solution:** This is expected on systems without NVIDIA GPUs. Test should pass with `gpu: None`.

### Test Fails: "Memory detection returns 0"

**Cause:** sysinfo crate permission issues

**Solution:** Run with appropriate permissions or check OS-specific requirements.

### Test Hangs on macOS

**Cause:** `system_profiler` can be slow on first run

**Solution:** Wait 5-10 seconds or run `system_profiler SPDisplaysDataType` manually first.

## Test Coverage

To generate code coverage:

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage
cargo tarpaulin --all --out Html

# Open report
open tarpaulin-report.html
```

**Target coverage:** >80% for core modules

## Writing New Tests

### Test Template

```rust
#[test]
fn test_new_feature() {
    // Arrange
    let expected = ...;

    // Act
    let result = function_to_test();

    // Assert
    assert_eq!(result, expected);
}
```

### Hardware Test Template

```rust
#[test]
fn test_new_hardware_detection() {
    let hw = HardwareSpec::detect().unwrap();

    // Validate results
    assert!(hw.some_field.is_some(), "Field should be detected");

    // Print for manual verification
    println!("Detected: {:?}", hw.some_field);
}
```

## Performance Testing

For performance-critical code:

```rust
#[test]
fn test_performance() {
    use std::time::Instant;

    let start = Instant::now();
    expensive_function();
    let elapsed = start.elapsed();

    // Should complete in <100ms
    assert!(elapsed.as_millis() < 100, "Too slow: {:?}", elapsed);
}
```

## Next Steps

- See [DEVELOPMENT.md](DEVELOPMENT.md) for build instructions
- See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines
