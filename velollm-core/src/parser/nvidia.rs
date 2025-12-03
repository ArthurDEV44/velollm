//! NVIDIA GPU information parser
//!
//! Parses output from `nvidia-smi` command with multiple format support.

use once_cell::sync::Lazy;
use regex::Regex;

use super::{GpuMemoryInfo, NvidiaGpuInfo};

/// Parser for nvidia-smi command output
pub struct NvidiaSmiParser;

impl NvidiaSmiParser {
    /// Parse CSV output from nvidia-smi query format
    ///
    /// Expected input format (from `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`):
    /// ```text
    /// NVIDIA GeForce RTX 4090, 24564, 23456, 545.23.08, 8.9
    /// ```
    ///
    /// Fields: name, memory.total, memory.free, driver_version, compute_cap
    pub fn parse_csv(output: &str) -> Option<NvidiaGpuInfo> {
        let line = output.lines().next()?.trim();
        if line.is_empty() {
            return None;
        }

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        if parts.len() < 3 {
            return None;
        }

        Some(NvidiaGpuInfo {
            name: parts[0].to_string(),
            vram_total_mb: parts[1].parse().ok()?,
            vram_free_mb: parts[2].parse().ok()?,
            driver_version: parts.get(3).filter(|s| !s.is_empty()).map(|s| s.to_string()),
            compute_capability: parts.get(4).filter(|s| !s.is_empty()).map(|s| s.to_string()),
        })
    }

    /// Parse memory information from nvidia-smi default output
    ///
    /// Handles formats like:
    /// - "1234 MiB / 24564 MiB"
    /// - "1234MiB / 24564MiB"
    /// - "1234 MB / 24564 MB"
    pub fn parse_memory(output: &str) -> Option<GpuMemoryInfo> {
        // Pattern matches: "used MiB / total MiB" with flexible spacing
        static MEMORY_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"(\d+)\s*Mi?B?\s*/\s*(\d+)\s*Mi?B?").expect("Invalid memory regex")
        });

        MEMORY_REGEX.captures(output).and_then(|caps| {
            let used: u64 = caps.get(1)?.as_str().parse().ok()?;
            let total: u64 = caps.get(2)?.as_str().parse().ok()?;
            Some(GpuMemoryInfo {
                total_mb: total,
                free_mb: total.saturating_sub(used),
            })
        })
    }

    /// Parse driver version from nvidia-smi output
    ///
    /// Handles formats like:
    /// - "Driver Version: 545.23.08"
    /// - "NVIDIA-SMI 545.23.08"
    pub fn parse_driver_version(output: &str) -> Option<String> {
        static DRIVER_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"(?:Driver Version|NVIDIA-SMI)[:\s]+(\d+\.\d+(?:\.\d+)?)")
                .expect("Invalid driver regex")
        });

        DRIVER_REGEX
            .captures(output)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().to_string())
    }

    /// Parse compute capability from nvidia-smi output
    ///
    /// Handles formats like:
    /// - "Compute Cap: 8.9"
    /// - "compute capability: 8.9"
    pub fn parse_compute_capability(output: &str) -> Option<String> {
        static CC_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"[Cc]ompute\s+[Cc]ap(?:ability)?[:\s]+(\d+\.\d+)")
                .expect("Invalid compute capability regex")
        });

        CC_REGEX
            .captures(output)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().to_string())
    }

    /// Parse GPU name from nvidia-smi output
    ///
    /// Handles formats like:
    /// - "Product Name: NVIDIA GeForce RTX 4090"
    /// - "GPU 0: NVIDIA GeForce RTX 4090"
    pub fn parse_gpu_name(output: &str) -> Option<String> {
        static NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"(?:Product Name|GPU \d+)[:\s]+(.+?)(?:\s*\(|$)")
                .expect("Invalid GPU name regex")
        });

        NAME_REGEX
            .captures(output)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Test fixtures =====

    const CSV_OUTPUT_FULL: &str =
        "NVIDIA GeForce RTX 4090, 24564, 23456, 545.23.08, 8.9\n";

    const CSV_OUTPUT_MINIMAL: &str = "NVIDIA GeForce GTX 1080, 8192, 7500\n";

    const CSV_OUTPUT_MULTI_GPU: &str = "\
NVIDIA GeForce RTX 4090, 24564, 23456, 545.23.08, 8.9
NVIDIA GeForce RTX 3080, 10240, 9800, 545.23.08, 8.6
";

    const DEFAULT_OUTPUT: &str = "\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off   |                  Off |
|  0%   35C    P8              25W / 450W |    1234MiB / 24564MiB  |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
";

    const DEFAULT_OUTPUT_SPACES: &str = "\
|    1234 MiB /  24564 MiB  |
";

    // ===== CSV parsing tests =====

    #[test]
    fn test_parse_csv_full() {
        let info = NvidiaSmiParser::parse_csv(CSV_OUTPUT_FULL).unwrap();
        assert_eq!(info.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(info.vram_total_mb, 24564);
        assert_eq!(info.vram_free_mb, 23456);
        assert_eq!(info.driver_version, Some("545.23.08".to_string()));
        assert_eq!(info.compute_capability, Some("8.9".to_string()));
    }

    #[test]
    fn test_parse_csv_minimal() {
        let info = NvidiaSmiParser::parse_csv(CSV_OUTPUT_MINIMAL).unwrap();
        assert_eq!(info.name, "NVIDIA GeForce GTX 1080");
        assert_eq!(info.vram_total_mb, 8192);
        assert_eq!(info.vram_free_mb, 7500);
        assert_eq!(info.driver_version, None);
        assert_eq!(info.compute_capability, None);
    }

    #[test]
    fn test_parse_csv_multi_gpu_first() {
        // Should parse only the first GPU
        let info = NvidiaSmiParser::parse_csv(CSV_OUTPUT_MULTI_GPU).unwrap();
        assert_eq!(info.name, "NVIDIA GeForce RTX 4090");
    }

    #[test]
    fn test_parse_csv_empty() {
        assert!(NvidiaSmiParser::parse_csv("").is_none());
        assert!(NvidiaSmiParser::parse_csv("   \n  ").is_none());
    }

    #[test]
    fn test_parse_csv_invalid() {
        assert!(NvidiaSmiParser::parse_csv("not,valid").is_none());
        assert!(NvidiaSmiParser::parse_csv("name, abc, def").is_none()); // non-numeric memory
    }

    // ===== Memory parsing tests =====

    #[test]
    fn test_parse_memory_compact() {
        let mem = NvidiaSmiParser::parse_memory("1234MiB / 24564MiB").unwrap();
        assert_eq!(mem.total_mb, 24564);
        assert_eq!(mem.free_mb, 24564 - 1234);
    }

    #[test]
    fn test_parse_memory_spaced() {
        let mem = NvidiaSmiParser::parse_memory("1234 MiB / 24564 MiB").unwrap();
        assert_eq!(mem.total_mb, 24564);
        assert_eq!(mem.free_mb, 24564 - 1234);
    }

    #[test]
    fn test_parse_memory_from_default_output() {
        let mem = NvidiaSmiParser::parse_memory(DEFAULT_OUTPUT).unwrap();
        assert_eq!(mem.total_mb, 24564);
        assert_eq!(mem.free_mb, 24564 - 1234);
    }

    #[test]
    fn test_parse_memory_with_extra_spaces() {
        let mem = NvidiaSmiParser::parse_memory(DEFAULT_OUTPUT_SPACES).unwrap();
        assert_eq!(mem.total_mb, 24564);
    }

    #[test]
    fn test_parse_memory_mb_format() {
        let mem = NvidiaSmiParser::parse_memory("1000 MB / 8000 MB").unwrap();
        assert_eq!(mem.total_mb, 8000);
        assert_eq!(mem.free_mb, 7000);
    }

    // ===== Driver version tests =====

    #[test]
    fn test_parse_driver_version() {
        assert_eq!(
            NvidiaSmiParser::parse_driver_version("Driver Version: 545.23.08"),
            Some("545.23.08".to_string())
        );
        assert_eq!(
            NvidiaSmiParser::parse_driver_version("NVIDIA-SMI 545.23.08"),
            Some("545.23.08".to_string())
        );
        assert_eq!(
            NvidiaSmiParser::parse_driver_version(DEFAULT_OUTPUT),
            Some("545.23.08".to_string())
        );
    }

    #[test]
    fn test_parse_driver_version_two_parts() {
        assert_eq!(
            NvidiaSmiParser::parse_driver_version("Driver Version: 545.23"),
            Some("545.23".to_string())
        );
    }

    // ===== Compute capability tests =====

    #[test]
    fn test_parse_compute_capability() {
        assert_eq!(
            NvidiaSmiParser::parse_compute_capability("Compute Cap: 8.9"),
            Some("8.9".to_string())
        );
        assert_eq!(
            NvidiaSmiParser::parse_compute_capability("compute capability: 7.5"),
            Some("7.5".to_string())
        );
    }

    // ===== GPU name tests =====

    #[test]
    fn test_parse_gpu_name() {
        assert_eq!(
            NvidiaSmiParser::parse_gpu_name("Product Name: NVIDIA GeForce RTX 4090"),
            Some("NVIDIA GeForce RTX 4090".to_string())
        );
        assert_eq!(
            NvidiaSmiParser::parse_gpu_name("GPU 0: NVIDIA GeForce RTX 4090 (UUID: abc)"),
            Some("NVIDIA GeForce RTX 4090".to_string())
        );
    }
}
