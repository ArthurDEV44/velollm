// Hardware detection module
//
// Detects GPU (NVIDIA/AMD/Apple), CPU, RAM, and OS information

use serde::{Deserialize, Serialize};
use std::process::Command;
use sysinfo::System;
use tracing::{debug, instrument, trace, warn};

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub gpu: Option<GpuInfo>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub os: String,
    pub platform: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Apple,
    Intel,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency_mhz: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
    pub used_mb: u64,
}

impl HardwareSpec {
    /// Detect current hardware configuration
    #[instrument(skip_all)]
    pub fn detect() -> anyhow::Result<Self> {
        debug!("Starting hardware detection");

        let gpu = detect_gpu();
        let cpu = detect_cpu()?;
        let memory = detect_memory()?;

        let spec = HardwareSpec {
            gpu,
            cpu,
            memory,
            os: std::env::consts::OS.to_string(),
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        };

        debug!(
            os = %spec.os,
            platform = %spec.platform,
            has_gpu = spec.gpu.is_some(),
            cpu_cores = spec.cpu.cores,
            memory_mb = spec.memory.total_mb,
            "Hardware detection complete"
        );

        Ok(spec)
    }
}

/// Detect GPU information
fn detect_gpu() -> Option<GpuInfo> {
    trace!("Attempting GPU detection");

    // Try NVIDIA first
    if let Some(gpu) = detect_nvidia_gpu() {
        debug!(
            name = %gpu.name,
            vendor = ?gpu.vendor,
            vram_mb = gpu.vram_total_mb,
            "NVIDIA GPU detected"
        );
        return Some(gpu);
    }

    // Try AMD
    if let Some(gpu) = detect_amd_gpu() {
        debug!(
            name = %gpu.name,
            vendor = ?gpu.vendor,
            vram_mb = gpu.vram_total_mb,
            "AMD GPU detected"
        );
        return Some(gpu);
    }

    // Try Apple Silicon
    if let Some(gpu) = detect_apple_gpu() {
        debug!(
            name = %gpu.name,
            vendor = ?gpu.vendor,
            "Apple Silicon GPU detected"
        );
        return Some(gpu);
    }

    // Try Intel (basic detection)
    if let Some(gpu) = detect_intel_gpu() {
        debug!(
            name = %gpu.name,
            vendor = ?gpu.vendor,
            "Intel GPU detected"
        );
        return Some(gpu);
    }

    debug!("No GPU detected");
    None
}

/// Detect NVIDIA GPU using nvidia-smi
pub(crate) fn detect_nvidia_gpu() -> Option<GpuInfo> {
    trace!("Trying nvidia-smi");

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        trace!("nvidia-smi command failed or not found");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    trace!(output = %stdout, "nvidia-smi output");

    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() >= 3 {
        Some(GpuInfo {
            name: parts[0].to_string(),
            vendor: GpuVendor::Nvidia,
            vram_total_mb: parts[1].parse().ok()?,
            vram_free_mb: parts[2].parse().ok()?,
            driver_version: parts.get(3).map(|s| s.to_string()),
            compute_capability: parts.get(4).map(|s| s.to_string()),
        })
    } else {
        warn!(parts = parts.len(), "Unexpected nvidia-smi output format");
        None
    }
}

/// Detect AMD GPU using rocm-smi
fn detect_amd_gpu() -> Option<GpuInfo> {
    trace!("Trying rocm-smi");

    let output = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram"])
        .output()
        .ok()?;

    if !output.status.success() {
        trace!("rocm-smi command failed or not found");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    trace!(output = %stdout, "rocm-smi output");

    // Parse rocm-smi output (format varies)
    // This is a simplified parser - real implementation would be more robust
    let name = stdout
        .lines()
        .find(|l| l.contains("Card series"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "AMD GPU".to_string());

    // Try to get VRAM info
    let vram_total = extract_vram_from_rocm(&stdout).unwrap_or(0);

    Some(GpuInfo {
        name,
        vendor: GpuVendor::Amd,
        vram_total_mb: vram_total,
        vram_free_mb: vram_total, // rocm-smi doesn't easily give free VRAM
        driver_version: None,
        compute_capability: None,
    })
}

/// Extract VRAM from rocm-smi output
fn extract_vram_from_rocm(output: &str) -> Option<u64> {
    for line in output.lines() {
        if line.contains("Total") && line.contains("MB") {
            // Extract number from line like "Total: 8192 MB"
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if part.contains("MB") && i > 0 {
                    if let Ok(vram) = parts[i - 1].parse::<u64>() {
                        return Some(vram);
                    }
                }
            }
        }
    }
    None
}

/// Detect Apple Silicon GPU using system_profiler
fn detect_apple_gpu() -> Option<GpuInfo> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("system_profiler")
            .args(&["SPDisplaysDataType", "-json"])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Try to detect Apple Silicon (M1/M2/M3)
        if stdout.contains("Apple M") {
            // Parse unified memory as VRAM
            let memory_mb = detect_unified_memory();

            // Extract chip name
            let name = if stdout.contains("M3") {
                "Apple M3".to_string()
            } else if stdout.contains("M2") {
                "Apple M2".to_string()
            } else if stdout.contains("M1") {
                "Apple M1".to_string()
            } else {
                "Apple Silicon".to_string()
            };

            return Some(GpuInfo {
                name,
                vendor: GpuVendor::Apple,
                vram_total_mb: memory_mb,
                vram_free_mb: memory_mb,
                driver_version: None,
                compute_capability: None,
            });
        }
    }

    None
}

/// Detect unified memory on Apple Silicon
#[cfg(target_os = "macos")]
fn detect_unified_memory() -> u64 {
    let output = Command::new("sysctl").arg("hw.memsize").output().ok();

    if let Some(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        if let Some(value) = stdout.split(':').nth(1) {
            if let Ok(bytes) = value.trim().parse::<u64>() {
                return bytes / (1024 * 1024); // Convert to MB
            }
        }
    }

    0
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn detect_unified_memory() -> u64 {
    0
}

/// Detect Intel GPU (basic)
fn detect_intel_gpu() -> Option<GpuInfo> {
    // Basic Intel GPU detection via system info
    // This is a placeholder - real implementation would use platform-specific APIs

    #[cfg(target_os = "linux")]
    {
        let output = Command::new("lspci").output().ok()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("VGA") && line.to_lowercase().contains("intel") {
                return Some(GpuInfo {
                    name: "Intel Integrated GPU".to_string(),
                    vendor: GpuVendor::Intel,
                    vram_total_mb: 0, // Integrated GPUs share system RAM
                    vram_free_mb: 0,
                    driver_version: None,
                    compute_capability: None,
                });
            }
        }
    }

    None
}

/// Detect CPU information
fn detect_cpu() -> anyhow::Result<CpuInfo> {
    trace!("Detecting CPU information");

    let mut sys = System::new_all();
    sys.refresh_cpu_all();

    let cpu_name = sys
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cores = num_cpus::get_physical() as u32;
    let threads = num_cpus::get() as u32;
    let frequency = sys.cpus().first().map(|cpu| cpu.frequency()).unwrap_or(0);

    let info = CpuInfo {
        model: if cpu_name.is_empty() { "Unknown CPU".to_string() } else { cpu_name },
        cores,
        threads,
        frequency_mhz: if frequency > 0 { Some(frequency) } else { None },
    };

    debug!(
        model = %info.model,
        cores = info.cores,
        threads = info.threads,
        frequency_mhz = ?info.frequency_mhz,
        "CPU detected"
    );

    Ok(info)
}

/// Detect memory information
fn detect_memory() -> anyhow::Result<MemoryInfo> {
    trace!("Detecting memory information");

    let mut sys = System::new_all();
    sys.refresh_memory();

    // sysinfo returns memory in bytes
    let total = sys.total_memory() / (1024 * 1024); // Bytes to MB
    let available = sys.available_memory() / (1024 * 1024);
    let used = sys.used_memory() / (1024 * 1024);

    let info = MemoryInfo { total_mb: total, available_mb: available, used_mb: used };

    debug!(
        total_mb = info.total_mb,
        available_mb = info.available_mb,
        used_mb = info.used_mb,
        "Memory detected"
    );

    Ok(info)
}
