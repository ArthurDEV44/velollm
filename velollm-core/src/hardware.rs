// Hardware detection module
//
// Detects GPU (NVIDIA/AMD/Apple), CPU, RAM, and OS information

use crate::command::{run_with_default_timeout, CommandResult};
use crate::error::HardwareError;
#[cfg(target_os = "macos")]
use crate::parser::AppleChip;
use crate::parser::{NvidiaSmiParser, RocmSmiParser};
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
    ///
    /// # Errors
    ///
    /// Returns `HardwareError` if CPU or memory detection fails.
    /// GPU detection failures are handled gracefully (returns `None`).
    #[instrument(skip_all)]
    pub fn detect() -> Result<Self, HardwareError> {
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

    let mut cmd = Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]);

    let output = match run_with_default_timeout(&mut cmd) {
        CommandResult::Success(output) => output,
        CommandResult::Timeout => {
            warn!("nvidia-smi timed out");
            return None;
        }
        CommandResult::SpawnError(_) => {
            trace!("nvidia-smi not found");
            return None;
        }
    };

    if !output.status.success() {
        trace!("nvidia-smi command failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    trace!(output = %stdout, "nvidia-smi output");

    // Use robust regex-based parser
    let parsed = NvidiaSmiParser::parse_csv(&stdout)?;

    Some(GpuInfo {
        name: parsed.name,
        vendor: GpuVendor::Nvidia,
        vram_total_mb: parsed.vram_total_mb,
        vram_free_mb: parsed.vram_free_mb,
        driver_version: parsed.driver_version,
        compute_capability: parsed.compute_capability,
    })
}

/// Detect AMD GPU using rocm-smi
fn detect_amd_gpu() -> Option<GpuInfo> {
    trace!("Trying rocm-smi");

    let mut cmd = Command::new("rocm-smi");
    cmd.args(["--showproductname", "--showmeminfo", "vram"]);

    let output = match run_with_default_timeout(&mut cmd) {
        CommandResult::Success(output) => output,
        CommandResult::Timeout => {
            warn!("rocm-smi timed out");
            return None;
        }
        CommandResult::SpawnError(_) => {
            trace!("rocm-smi not found");
            return None;
        }
    };

    if !output.status.success() {
        trace!("rocm-smi command failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    trace!(output = %stdout, "rocm-smi output");

    // Use robust regex-based parser
    let parsed = RocmSmiParser::parse(&stdout)?;

    Some(GpuInfo {
        name: parsed.name,
        vendor: GpuVendor::Amd,
        vram_total_mb: parsed.vram_total_mb,
        vram_free_mb: parsed.vram_free_mb.unwrap_or(parsed.vram_total_mb),
        driver_version: None,
        compute_capability: None,
    })
}

/// Detect Apple Silicon GPU using system_profiler
fn detect_apple_gpu() -> Option<GpuInfo> {
    #[cfg(target_os = "macos")]
    {
        let mut cmd = Command::new("system_profiler");
        cmd.args(["SPDisplaysDataType", "-json"]);

        let output = match run_with_default_timeout(&mut cmd) {
            CommandResult::Success(output) => output,
            CommandResult::Timeout => {
                warn!("system_profiler timed out");
                return None;
            }
            CommandResult::SpawnError(_) => {
                trace!("system_profiler not found");
                return None;
            }
        };

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Use robust regex-based parser for Apple chip detection
        if let Some(chip) = AppleChip::from_output(&stdout) {
            // Parse unified memory as VRAM
            let memory_mb = detect_unified_memory();

            return Some(GpuInfo {
                name: chip.display_name(),
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
    let mut cmd = Command::new("sysctl");
    cmd.arg("hw.memsize");

    let output = match run_with_default_timeout(&mut cmd) {
        CommandResult::Success(output) => output,
        _ => return 0,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(value) = stdout.split(':').nth(1) {
        if let Ok(bytes) = value.trim().parse::<u64>() {
            return bytes / (1024 * 1024); // Convert to MB
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
        let mut cmd = Command::new("lspci");

        let output = match run_with_default_timeout(&mut cmd) {
            CommandResult::Success(output) => output,
            CommandResult::Timeout => {
                warn!("lspci timed out");
                return None;
            }
            CommandResult::SpawnError(_) => {
                trace!("lspci not found");
                return None;
            }
        };

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
fn detect_cpu() -> Result<CpuInfo, HardwareError> {
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

    // Validate that we got meaningful CPU info
    if cores == 0 {
        return Err(HardwareError::cpu("Failed to detect CPU core count"));
    }

    let info = CpuInfo {
        model: if cpu_name.is_empty() {
            "Unknown CPU".to_string()
        } else {
            cpu_name
        },
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
fn detect_memory() -> Result<MemoryInfo, HardwareError> {
    trace!("Detecting memory information");

    let mut sys = System::new_all();
    sys.refresh_memory();

    // sysinfo returns memory in bytes
    let total = sys.total_memory() / (1024 * 1024); // Bytes to MB
    let available = sys.available_memory() / (1024 * 1024);
    let used = sys.used_memory() / (1024 * 1024);

    // Validate that we got meaningful memory info
    if total == 0 {
        return Err(HardwareError::memory("Failed to detect total memory"));
    }

    let info = MemoryInfo { total_mb: total, available_mb: available, used_mb: used };

    debug!(
        total_mb = info.total_mb,
        available_mb = info.available_mb,
        used_mb = info.used_mb,
        "Memory detected"
    );

    Ok(info)
}
