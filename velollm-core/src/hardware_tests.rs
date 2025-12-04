// Tests for hardware detection module

#[cfg(test)]
mod tests {
    use crate::hardware::*;

    #[test]
    fn test_hardware_detection() {
        let hw = HardwareSpec::detect();
        assert!(hw.is_ok(), "Hardware detection should succeed");

        let hw = hw.unwrap();

        // OS should always be detected
        assert!(!hw.os.is_empty(), "OS should be detected");
        assert!(!hw.platform.is_empty(), "Platform should be detected");

        println!("Detected hardware:");
        println!("OS: {}", hw.os);
        println!("Platform: {}", hw.platform);
    }

    #[test]
    fn test_cpu_detection() {
        let hw = HardwareSpec::detect().unwrap();

        // CPU info should always be available
        assert!(!hw.cpu.model.is_empty(), "CPU model should be detected");
        assert!(hw.cpu.cores > 0, "CPU cores should be > 0");
        assert!(hw.cpu.threads > 0, "CPU threads should be > 0");
        // Note: On some VMs (like GitHub Actions), threads may equal cores
        // if hyperthreading is not exposed. We only check both are positive.

        println!("CPU: {}", hw.cpu.model);
        println!("Cores: {}", hw.cpu.cores);
        println!("Threads: {}", hw.cpu.threads);
        if let Some(freq) = hw.cpu.frequency_mhz {
            println!("Frequency: {} MHz", freq);
        }
    }

    #[test]
    fn test_memory_detection() {
        let hw = HardwareSpec::detect().unwrap();

        // Memory should always be detected
        assert!(hw.memory.total_mb > 0, "Total memory should be > 0");
        assert!(
            hw.memory.available_mb <= hw.memory.total_mb,
            "Available memory should be <= total"
        );
        assert!(hw.memory.used_mb <= hw.memory.total_mb, "Used memory should be <= total");

        println!("Memory:");
        println!("  Total: {} MB ({} GB)", hw.memory.total_mb, hw.memory.total_mb / 1024);
        println!(
            "  Available: {} MB ({} GB)",
            hw.memory.available_mb,
            hw.memory.available_mb / 1024
        );
        println!("  Used: {} MB ({} GB)", hw.memory.used_mb, hw.memory.used_mb / 1024);
    }

    #[test]
    fn test_gpu_detection() {
        let hw = HardwareSpec::detect().unwrap();

        if let Some(ref gpu) = hw.gpu {
            println!("GPU detected:");
            println!("  Name: {}", gpu.name);
            println!("  Vendor: {:?}", gpu.vendor);
            println!("  VRAM Total: {} MB ({} GB)", gpu.vram_total_mb, gpu.vram_total_mb / 1024);
            println!("  VRAM Free: {} MB ({} GB)", gpu.vram_free_mb, gpu.vram_free_mb / 1024);

            if let Some(ref driver) = gpu.driver_version {
                println!("  Driver: {}", driver);
            }
            if let Some(ref compute) = gpu.compute_capability {
                println!("  Compute Capability: {}", compute);
            }

            // Validate GPU info
            assert!(!gpu.name.is_empty(), "GPU name should not be empty");

            // VRAM can be 0 for integrated GPUs
            match gpu.vendor {
                GpuVendor::Intel => {
                    // Intel integrated GPUs may report 0 VRAM
                }
                _ => {
                    assert!(
                        gpu.vram_total_mb > 0 || gpu.vram_free_mb == 0,
                        "Dedicated GPU should have VRAM info"
                    );
                }
            }
        } else {
            println!("No GPU detected (running on CPU-only system)");
        }
    }

    #[test]
    fn test_json_serialization() {
        let hw = HardwareSpec::detect().unwrap();

        let json = serde_json::to_string_pretty(&hw);
        assert!(json.is_ok(), "Should serialize to JSON");

        let json_str = json.unwrap();
        assert!(json_str.contains("cpu"), "JSON should contain CPU info");
        assert!(json_str.contains("memory"), "JSON should contain memory info");

        println!("JSON output:");
        println!("{}", json_str);

        // Test deserialization
        let deserialized: Result<HardwareSpec, _> = serde_json::from_str(&json_str);
        assert!(deserialized.is_ok(), "Should deserialize from JSON");
    }

    #[test]
    fn test_platform_string() {
        let hw = HardwareSpec::detect().unwrap();

        // Platform should contain OS and architecture
        assert!(
            hw.platform.contains("linux")
                || hw.platform.contains("macos")
                || hw.platform.contains("windows"),
            "Platform should contain OS name"
        );

        assert!(
            hw.platform.contains("x86_64")
                || hw.platform.contains("aarch64")
                || hw.platform.contains("arm"),
            "Platform should contain architecture"
        );

        println!("Platform: {}", hw.platform);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_nvidia_detection_on_linux() {
        // This test only validates the function doesn't crash
        // Actual GPU may or may not be present
        use crate::hardware::detect_nvidia_gpu;

        let gpu = detect_nvidia_gpu();
        if let Some(gpu) = gpu {
            assert_eq!(gpu.vendor, GpuVendor::Nvidia);
            println!("NVIDIA GPU: {}", gpu.name);
        } else {
            println!("No NVIDIA GPU detected");
        }
    }

    #[test]
    fn test_gpu_vendor_serialization() {
        use serde_json;

        let nvidia = GpuVendor::Nvidia;
        let json = serde_json::to_string(&nvidia).unwrap();
        assert_eq!(json, r#""Nvidia""#);

        let deserialized: GpuVendor = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, GpuVendor::Nvidia);
    }
}
