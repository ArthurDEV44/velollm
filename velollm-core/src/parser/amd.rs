//! AMD GPU information parser
//!
//! Parses output from `rocm-smi` command with multiple format support.

use once_cell::sync::Lazy;
use regex::Regex;

use super::AmdGpuInfo;

/// Parser for rocm-smi command output
pub struct RocmSmiParser;

impl RocmSmiParser {
    /// Parse GPU information from rocm-smi output
    ///
    /// Combines multiple parsing strategies with fallbacks.
    pub fn parse(output: &str) -> Option<AmdGpuInfo> {
        let name = Self::parse_card_series(output)
            .or_else(|| Self::parse_product_name(output))
            .unwrap_or_else(|| "AMD GPU".to_string());

        let vram_total_mb = Self::parse_vram_total(output).unwrap_or(0);
        let vram_free_mb = Self::parse_vram_free(output);

        Some(AmdGpuInfo {
            name,
            vram_total_mb,
            vram_free_mb,
        })
    }

    /// Parse card series name from rocm-smi output
    ///
    /// Handles formats like:
    /// - "Card series:        Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"
    /// - "Card series: AMD Radeon RX 7900 XTX"
    pub fn parse_card_series(output: &str) -> Option<String> {
        static SERIES_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"Card series[:\s]+(.+?)(?:\n|$)").expect("Invalid card series regex")
        });

        SERIES_REGEX
            .captures(output)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Parse product name from rocm-smi output
    ///
    /// Handles formats like:
    /// - "Product Name: AMD Radeon RX 7900 XTX"
    pub fn parse_product_name(output: &str) -> Option<String> {
        static NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"Product Name[:\s]+(.+?)(?:\n|$)").expect("Invalid product name regex")
        });

        NAME_REGEX
            .captures(output)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Parse total VRAM from rocm-smi output
    ///
    /// Handles multiple formats:
    /// - "Total Memory (B): 25769803776"
    /// - "VRAM Total Memory (B):    25769803776"
    /// - "Total: 24576 MB"
    /// - "Total: 24576 MiB"
    /// - "GTT Total Memory (B):     68705394688"
    pub fn parse_vram_total(output: &str) -> Option<u64> {
        // Try VRAM-specific first (more precise)
        if let Some(bytes) = Self::parse_memory_bytes(output, "VRAM Total") {
            return Some(bytes / (1024 * 1024));
        }

        // Try generic "Total Memory" (skip GTT)
        if let Some(bytes) = Self::parse_memory_bytes_first_match(output, "Total Memory") {
            return Some(bytes / (1024 * 1024));
        }

        // Try MB/MiB format
        Self::parse_memory_mb(output, "Total")
    }

    /// Parse free VRAM from rocm-smi output
    ///
    /// Handles formats like:
    /// - "VRAM Total Used Memory (B): 1234567890"
    /// - "Free: 23000 MB"
    pub fn parse_vram_free(output: &str) -> Option<u64> {
        // Calculate free from total - used
        if let (Some(total_bytes), Some(used_bytes)) = (
            Self::parse_memory_bytes(output, "VRAM Total Memory"),
            Self::parse_memory_bytes(output, "VRAM Total Used"),
        ) {
            return Some((total_bytes.saturating_sub(used_bytes)) / (1024 * 1024));
        }

        // Try direct "Free" format
        Self::parse_memory_mb(output, "Free")
    }

    /// Parse memory in bytes from a specific field
    fn parse_memory_bytes(output: &str, field: &str) -> Option<u64> {
        let pattern = format!(r"{}[^:]*\(B\)[:\s]+(\d+)", regex::escape(field));
        let regex = Regex::new(&pattern).ok()?;

        regex
            .captures(output)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }

    /// Parse the first occurrence of memory bytes matching a pattern
    fn parse_memory_bytes_first_match(output: &str, field: &str) -> Option<u64> {
        // Skip GTT memory, get first VRAM-like match
        for line in output.lines() {
            if line.contains(field) && !line.contains("GTT") {
                static BYTES_REGEX: Lazy<Regex> = Lazy::new(|| {
                    Regex::new(r"\(B\)[:\s]+(\d+)").expect("Invalid bytes regex")
                });

                if let Some(caps) = BYTES_REGEX.captures(line) {
                    if let Some(value) = caps.get(1).and_then(|m| m.as_str().parse().ok()) {
                        return Some(value);
                    }
                }
            }
        }
        None
    }

    /// Parse memory in MB/MiB format
    fn parse_memory_mb(output: &str, field: &str) -> Option<u64> {
        let pattern = format!(r"{}[:\s]+(\d+)\s*(?:MB|MiB)", regex::escape(field));
        let regex = Regex::new(&pattern).ok()?;

        regex
            .captures(output)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Test fixtures =====

    const ROCM_SMI_OUTPUT_FULL: &str = "\
========================= ROCm System Management Interface =========================
================================== Product Info ====================================
GPU[0]      : Card series:        Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]
GPU[0]      : Card model:         0x73bf
GPU[0]      : Card vendor:        Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]      : Card SKU:           D32403
================================== Memory Info =====================================
GPU[0]      : VRAM Total Memory (B):    17179869184
GPU[0]      : VRAM Total Used Memory (B):    1073741824
GPU[0]      : GTT Total Memory (B):     68705394688
GPU[0]      : GTT Total Used Memory (B):     135270400
================================== End of ROCm SMI Log =============================
";

    const ROCM_SMI_OUTPUT_MINIMAL: &str = "\
Card series: AMD Radeon RX 7900 XTX
Total Memory (B): 25769803776
";

    const ROCM_SMI_OUTPUT_MB: &str = "\
Product Name: AMD Radeon RX 580
Total: 8192 MB
Free: 7500 MB
";

    // ===== Full parsing tests =====

    #[test]
    fn test_parse_full_output() {
        let info = RocmSmiParser::parse(ROCM_SMI_OUTPUT_FULL).unwrap();
        assert_eq!(
            info.name,
            "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"
        );
        // 17179869184 bytes = 16384 MB
        assert_eq!(info.vram_total_mb, 16384);
        // 17179869184 - 1073741824 = 16106127360 bytes = 15360 MB
        assert_eq!(info.vram_free_mb, Some(15360));
    }

    #[test]
    fn test_parse_minimal_output() {
        let info = RocmSmiParser::parse(ROCM_SMI_OUTPUT_MINIMAL).unwrap();
        assert_eq!(info.name, "AMD Radeon RX 7900 XTX");
        // 25769803776 bytes = 24576 MB
        assert_eq!(info.vram_total_mb, 24576);
    }

    #[test]
    fn test_parse_mb_format() {
        let info = RocmSmiParser::parse(ROCM_SMI_OUTPUT_MB).unwrap();
        assert_eq!(info.name, "AMD Radeon RX 580");
        assert_eq!(info.vram_total_mb, 8192);
        assert_eq!(info.vram_free_mb, Some(7500));
    }

    // ===== Individual parser tests =====

    #[test]
    fn test_parse_card_series() {
        assert_eq!(
            RocmSmiParser::parse_card_series(ROCM_SMI_OUTPUT_FULL),
            Some("Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]".to_string())
        );

        assert_eq!(
            RocmSmiParser::parse_card_series(ROCM_SMI_OUTPUT_MINIMAL),
            Some("AMD Radeon RX 7900 XTX".to_string())
        );
    }

    #[test]
    fn test_parse_product_name() {
        assert_eq!(
            RocmSmiParser::parse_product_name(ROCM_SMI_OUTPUT_MB),
            Some("AMD Radeon RX 580".to_string())
        );
    }

    #[test]
    fn test_parse_vram_total_bytes() {
        assert_eq!(
            RocmSmiParser::parse_vram_total(ROCM_SMI_OUTPUT_FULL),
            Some(16384)
        );
    }

    #[test]
    fn test_parse_vram_free() {
        assert_eq!(
            RocmSmiParser::parse_vram_free(ROCM_SMI_OUTPUT_FULL),
            Some(15360)
        );
    }

    #[test]
    fn test_parse_empty() {
        let info = RocmSmiParser::parse("").unwrap();
        assert_eq!(info.name, "AMD GPU");
        assert_eq!(info.vram_total_mb, 0);
    }

    #[test]
    fn test_skips_gtt_memory() {
        // GTT memory should be ignored when parsing VRAM
        let output = "\
GTT Total Memory (B):     68705394688
Total Memory (B): 17179869184
";
        // Should get 16384 MB (VRAM), not GTT
        assert_eq!(RocmSmiParser::parse_vram_total(output), Some(16384));
    }
}
