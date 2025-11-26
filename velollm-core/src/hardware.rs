// Hardware detection module
//
// Placeholder - will be implemented in TASK-003

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub gpu: Option<GpuInfo>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub os: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub cuda_version: Option<String>,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub cache_kb: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
}

impl HardwareSpec {
    /// Detect current hardware configuration
    pub fn detect() -> anyhow::Result<Self> {
        // TODO: Implement in TASK-003
        Ok(HardwareSpec {
            gpu: None,
            cpu: CpuInfo {
                model: "Unknown".to_string(),
                cores: num_cpus::get_physical() as u32,
                threads: num_cpus::get() as u32,
                cache_kb: None,
            },
            memory: MemoryInfo {
                total_mb: 0,
                available_mb: 0,
            },
            os: std::env::consts::OS.to_string(),
        })
    }
}
