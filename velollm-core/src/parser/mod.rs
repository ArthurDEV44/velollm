//! Robust parsing module for external command outputs
//!
//! This module provides regex-based parsers with fallbacks for:
//! - nvidia-smi output (NVIDIA GPU detection)
//! - rocm-smi output (AMD GPU detection)
//! - llama.cpp timing output (performance metrics)
//!
//! All parsers are designed to be resilient to format variations and
//! provide sensible defaults when parsing fails.

pub mod amd;
pub mod llama;
pub mod nvidia;

pub use amd::RocmSmiParser;
pub use llama::LlamaCppParser;
pub use nvidia::NvidiaSmiParser;

/// Result of parsing GPU memory information
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuMemoryInfo {
    /// Total VRAM in megabytes
    pub total_mb: u64,
    /// Free/available VRAM in megabytes
    pub free_mb: u64,
}

/// Result of parsing NVIDIA GPU information
#[derive(Debug, Clone, Default)]
pub struct NvidiaGpuInfo {
    /// GPU model name (e.g., "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// Total VRAM in MB
    pub vram_total_mb: u64,
    /// Free VRAM in MB
    pub vram_free_mb: u64,
    /// Driver version (e.g., "545.23.08")
    pub driver_version: Option<String>,
    /// Compute capability (e.g., "8.9")
    pub compute_capability: Option<String>,
}

/// Result of parsing AMD GPU information
#[derive(Debug, Clone, Default)]
pub struct AmdGpuInfo {
    /// GPU model name
    pub name: String,
    /// Total VRAM in MB
    pub vram_total_mb: u64,
    /// Free VRAM in MB (if available)
    pub vram_free_mb: Option<u64>,
}

/// Result of parsing Apple Silicon chip name
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppleChip {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M4,
    M4Pro,
    M4Max,
    Unknown(String),
}

impl AppleChip {
    /// Parse Apple chip name from system_profiler output
    pub fn from_output(output: &str) -> Option<Self> {
        use once_cell::sync::Lazy;
        use regex::Regex;

        // Match patterns like "Apple M1", "Apple M2 Pro", "Apple M3 Max"
        static CHIP_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"Apple\s+(M[1-4](?:\s+(?:Pro|Max|Ultra))?)")
                .expect("Invalid Apple chip regex")
        });

        CHIP_REGEX.captures(output).map(|caps| {
            let chip_name = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            match chip_name.trim() {
                "M1" => AppleChip::M1,
                "M1 Pro" => AppleChip::M1Pro,
                "M1 Max" => AppleChip::M1Max,
                "M1 Ultra" => AppleChip::M1Ultra,
                "M2" => AppleChip::M2,
                "M2 Pro" => AppleChip::M2Pro,
                "M2 Max" => AppleChip::M2Max,
                "M2 Ultra" => AppleChip::M2Ultra,
                "M3" => AppleChip::M3,
                "M3 Pro" => AppleChip::M3Pro,
                "M3 Max" => AppleChip::M3Max,
                "M4" => AppleChip::M4,
                "M4 Pro" => AppleChip::M4Pro,
                "M4 Max" => AppleChip::M4Max,
                other => AppleChip::Unknown(other.to_string()),
            }
        })
    }

    /// Get display name for this chip
    pub fn display_name(&self) -> String {
        match self {
            AppleChip::M1 => "Apple M1".to_string(),
            AppleChip::M1Pro => "Apple M1 Pro".to_string(),
            AppleChip::M1Max => "Apple M1 Max".to_string(),
            AppleChip::M1Ultra => "Apple M1 Ultra".to_string(),
            AppleChip::M2 => "Apple M2".to_string(),
            AppleChip::M2Pro => "Apple M2 Pro".to_string(),
            AppleChip::M2Max => "Apple M2 Max".to_string(),
            AppleChip::M2Ultra => "Apple M2 Ultra".to_string(),
            AppleChip::M3 => "Apple M3".to_string(),
            AppleChip::M3Pro => "Apple M3 Pro".to_string(),
            AppleChip::M3Max => "Apple M3 Max".to_string(),
            AppleChip::M4 => "Apple M4".to_string(),
            AppleChip::M4Pro => "Apple M4 Pro".to_string(),
            AppleChip::M4Max => "Apple M4 Max".to_string(),
            AppleChip::Unknown(name) => format!("Apple {}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apple_chip_detection() {
        assert_eq!(AppleChip::from_output("Chip: Apple M1"), Some(AppleChip::M1));
        assert_eq!(AppleChip::from_output("Chip: Apple M2 Pro"), Some(AppleChip::M2Pro));
        assert_eq!(AppleChip::from_output("Chip: Apple M3 Max"), Some(AppleChip::M3Max));
        assert_eq!(AppleChip::from_output("Intel Core i9"), None);
    }

    #[test]
    fn test_apple_chip_display_name() {
        assert_eq!(AppleChip::M3Pro.display_name(), "Apple M3 Pro");
        assert_eq!(AppleChip::Unknown("M5".to_string()).display_name(), "Apple M5");
    }
}
