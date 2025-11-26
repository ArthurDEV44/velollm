# Hardware Detection API

## Overview

The `velollm-core::hardware` module provides comprehensive hardware detection for GPU, CPU, RAM, and OS information.

## Usage

```rust
use velollm_core::hardware::HardwareSpec;

fn main() -> anyhow::Result<()> {
    let hw = HardwareSpec::detect()?;

    println!("Platform: {}", hw.platform);
    println!("CPU: {}", hw.cpu.model);
    println!("RAM: {} GB", hw.memory.total_mb / 1024);

    if let Some(gpu) = hw.gpu {
        println!("GPU: {}", gpu.name);
        println!("VRAM: {} GB", gpu.vram_total_mb / 1024);
    }

    Ok(())
}
```

## Data Structures

### `HardwareSpec`

Main structure containing all hardware information.

```rust
pub struct HardwareSpec {
    pub gpu: Option<GpuInfo>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub os: String,
    pub platform: String,
}
```

**Fields:**
- `gpu`: GPU information (if available)
- `cpu`: CPU information
- `memory`: System memory information
- `os`: Operating system name ("linux", "macos", "windows")
- `platform`: Platform string (e.g., "linux-x86_64", "macos-aarch64")

### `GpuInfo`

GPU-specific information.

```rust
pub struct GpuInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
}
```

**Supported Vendors:**
- `GpuVendor::Nvidia`: NVIDIA GPUs (via nvidia-smi)
- `GpuVendor::Amd`: AMD GPUs (via rocm-smi)
- `GpuVendor::Apple`: Apple Silicon (M1/M2/M3)
- `GpuVendor::Intel`: Intel integrated GPUs

### `CpuInfo`

CPU-specific information.

```rust
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency_mhz: Option<u64>,
}
```

### `MemoryInfo`

System memory information.

```rust
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
    pub used_mb: u64,
}
```

## Platform Support

### Linux

**GPU Detection:**
- NVIDIA: `nvidia-smi` command
- AMD: `rocm-smi` command
- Intel: `lspci` parsing

**CPU/Memory:**
- sysinfo crate (reads /proc/cpuinfo, /proc/meminfo)

### macOS

**GPU Detection:**
- Apple Silicon: `system_profiler SPDisplaysDataType` + `sysctl hw.memsize`

**CPU/Memory:**
- sysinfo crate

### Windows

**GPU Detection:**
- NVIDIA: `nvidia-smi.exe` (if in PATH)

**CPU/Memory:**
- sysinfo crate (WMI queries)

## JSON Output

The `HardwareSpec` struct is serializable to JSON:

```bash
velollm detect
```

**Example output:**

```json
{
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "vendor": "Nvidia",
    "vram_total_mb": 24564,
    "vram_free_mb": 23012,
    "driver_version": "535.129.03",
    "compute_capability": "8.9"
  },
  "cpu": {
    "model": "AMD Ryzen 9 7950X 16-Core Processor",
    "cores": 16,
    "threads": 32,
    "frequency_mhz": 4500
  },
  "memory": {
    "total_mb": 65536,
    "available_mb": 42384,
    "used_mb": 23152
  },
  "os": "linux",
  "platform": "linux-x86_64"
}
```

## Error Handling

The `detect()` method returns `anyhow::Result<HardwareSpec>`:

- **CPU/Memory detection**: Always succeeds (falls back to safe defaults)
- **GPU detection**: Returns `None` if no GPU found or detection tools unavailable

```rust
match HardwareSpec::detect() {
    Ok(hw) => {
        // Hardware detected successfully
    }
    Err(e) => {
        eprintln!("Failed to detect hardware: {}", e);
    }
}
```

## Requirements

### External Commands

For GPU detection, the following commands must be in PATH:

- **NVIDIA**: `nvidia-smi`
- **AMD**: `rocm-smi`
- **Apple**: `system_profiler`, `sysctl` (built-in on macOS)
- **Intel (Linux)**: `lspci`

If these commands are not available, GPU detection will return `None`.

## Examples

### Detect and Print Hardware

```rust
use velollm_core::hardware::HardwareSpec;

fn main() {
    let hw = HardwareSpec::detect().unwrap();
    println!("{:#?}", hw);
}
```

### Export to JSON File

```rust
use velollm_core::hardware::HardwareSpec;
use std::fs;

fn main() -> anyhow::Result<()> {
    let hw = HardwareSpec::detect()?;
    let json = serde_json::to_string_pretty(&hw)?;
    fs::write("hardware.json", json)?;
    Ok(())
}
```

### Check GPU Availability

```rust
use velollm_core::hardware::{HardwareSpec, GpuVendor};

fn main() -> anyhow::Result<()> {
    let hw = HardwareSpec::detect()?;

    match hw.gpu {
        Some(gpu) => {
            println!("GPU available: {}", gpu.name);

            match gpu.vendor {
                GpuVendor::Nvidia => {
                    println!("NVIDIA GPU with {} GB VRAM", gpu.vram_total_mb / 1024);
                }
                GpuVendor::Apple => {
                    println!("Apple Silicon with unified memory");
                }
                _ => {
                    println!("Other GPU vendor: {:?}", gpu.vendor);
                }
            }
        }
        None => {
            println!("No GPU detected - CPU-only mode");
        }
    }

    Ok(())
}
```

## Testing

Run tests with:

```bash
cargo test -p velollm-core

# With output
cargo test -p velollm-core -- --nocapture
```

Tests validate:
- CPU detection (cores, threads, model)
- Memory detection (total, available, used)
- GPU detection (if available)
- JSON serialization/deserialization
- Platform string format

## Future Improvements

Planned enhancements:

- [ ] Windows GPU detection (AMD via DirectX)
- [ ] Intel GPU compute capability detection
- [ ] GPU memory bandwidth detection
- [ ] CPU cache size detection (L1/L2/L3)
- [ ] PCIe version/lanes for GPUs
- [ ] Multi-GPU support
- [ ] Power consumption metrics
