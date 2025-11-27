use crate::hardware::{HardwareSpec, GpuInfo, CpuInfo, MemoryInfo};

/// Ollama configuration optimizer based on hardware specifications
///
/// This optimizer generates optimal Ollama configurations by analyzing
/// available hardware (GPU, CPU, memory) and applying heuristics based
/// on best practices and empirical performance data.
pub struct OllamaOptimizer;

impl OllamaOptimizer {
    /// Generate optimized Ollama configuration for the given hardware
    ///
    /// # Arguments
    /// * `hw` - Hardware specification from hardware detection
    ///
    /// # Returns
    /// Optimized OllamaConfig with values tuned for the hardware
    ///
    /// # Example
    /// ```
    /// use velollm_core::hardware::HardwareSpec;
    /// use velollm_core::optimizer::OllamaOptimizer;
    ///
    /// let hw = HardwareSpec::detect().unwrap();
    /// let config = OllamaOptimizer::optimize(&hw);
    /// ```
    pub fn optimize(hw: &HardwareSpec) -> OptimizedConfig {
        let mut config = OptimizedConfig::default();

        // GPU-based optimizations (highest priority)
        if let Some(ref gpu) = hw.gpu {
            config = Self::optimize_for_gpu(config, gpu);
        } else {
            // CPU-only optimizations
            config = Self::optimize_for_cpu(config, &hw.cpu);
        }

        // Memory-based optimizations (applies to all)
        config = Self::optimize_memory(config, &hw.memory);

        config
    }

    /// Optimize configuration for GPU workloads
    fn optimize_for_gpu(mut config: OptimizedConfig, gpu: &GpuInfo) -> OptimizedConfig {
        let vram_gb = gpu.vram_total_mb / 1024;

        // num_parallel: Concurrent request handling
        // Rule: VRAM / (estimated model memory footprint)
        // Conservative estimates:
        // - Small models (1-3B): ~2GB VRAM
        // - Medium models (7-8B): ~5GB VRAM
        // - Large models (13-70B): 10GB+ VRAM
        config.num_parallel = if vram_gb < 8 {
            1 // Single request, small models only
        } else if vram_gb < 16 {
            2 // Can handle 2 small or 1 medium model
        } else if vram_gb < 24 {
            3 // Multiple medium models
        } else {
            4 // High-end GPU, multiple requests
        };

        // num_gpu: Layers to offload to GPU
        // Strategy: Offload all layers if possible
        config.num_gpu = 999; // Max value (Ollama auto-limits to model's layer count)

        // num_batch: Batch size for prompt processing
        // Larger batch = faster prompt ingestion but more VRAM usage
        // Rule: Scale with available VRAM
        config.num_batch = if vram_gb < 8 {
            128 // Conservative for low VRAM
        } else if vram_gb < 16 {
            256 // Standard for mid-range
        } else if vram_gb < 24 {
            512 // High performance
        } else {
            1024 // Maximum for high-end GPUs
        };

        // num_ctx: Context window size
        // Larger context = more VRAM usage (quadratic with attention)
        // Rule: Balance between capability and VRAM
        config.num_ctx = if vram_gb < 8 {
            2048 // Standard context
        } else if vram_gb < 16 {
            4096 // Extended context
        } else if vram_gb < 24 {
            8192 // Large context
        } else {
            16384 // Maximum for very large VRAM
        };

        config
    }

    /// Optimize configuration for CPU-only workloads
    fn optimize_for_cpu(mut config: OptimizedConfig, cpu: &CpuInfo) -> OptimizedConfig {
        // Disable GPU offloading
        config.num_gpu = 0;

        // Use all available threads
        config.num_thread = Some(cpu.threads);

        // CPU mode: smaller batches are more efficient
        config.num_batch = 128;

        // CPU mode: limit context to reduce memory pressure
        config.num_ctx = 2048;

        // CPU is slower, so only handle 1 request at a time
        config.num_parallel = 1;

        config
    }

    /// Optimize memory-related settings
    fn optimize_memory(mut config: OptimizedConfig, mem: &MemoryInfo) -> OptimizedConfig {
        let mem_gb = mem.total_mb / 1024;

        // max_loaded_models: Keep models in RAM for faster switching
        // Rule: Each model ~2-10GB RAM depending on size
        config.max_loaded_models = if mem_gb <= 16 {
            1 // Limited RAM, keep only active model
        } else if mem_gb <= 32 {
            2 // Can afford to cache one extra model
        } else if mem_gb <= 64 {
            3 // Good for multi-model workflows
        } else {
            4 // High RAM, cache multiple models
        };

        // keep_alive: Duration to keep models loaded
        // More RAM = longer keep-alive (reduces reload overhead)
        config.keep_alive = if mem_gb <= 16 {
            "5m".to_string() // Quick unload to free RAM
        } else if mem_gb <= 32 {
            "15m".to_string() // Standard duration
        } else if mem_gb <= 64 {
            "30m".to_string() // Long duration for convenience
        } else {
            "1h".to_string() // Very long for high-RAM systems
        };

        config
    }

    /// Generate a comparison report between current and optimized configs
    ///
    /// # Arguments
    /// * `current` - Current configuration (from environment or baseline)
    /// * `optimized` - Optimized configuration from optimizer
    ///
    /// # Returns
    /// Human-readable comparison report
    pub fn generate_report(current: &OptimizedConfig, optimized: &OptimizedConfig) -> String {
        let mut report = String::from("=== Ollama Configuration Optimization Report ===\n\n");

        report.push_str(&Self::compare_field(
            "num_parallel",
            current.num_parallel,
            optimized.num_parallel,
            "Concurrent requests",
            "Higher = more throughput (requires more VRAM/RAM)",
        ));

        report.push_str(&Self::compare_field(
            "num_gpu",
            current.num_gpu,
            optimized.num_gpu,
            "GPU layers",
            "999 = offload all layers to GPU",
        ));

        report.push_str(&Self::compare_field(
            "num_batch",
            current.num_batch,
            optimized.num_batch,
            "Batch size",
            "Larger = faster prompt processing",
        ));

        report.push_str(&Self::compare_field(
            "num_ctx",
            current.num_ctx,
            optimized.num_ctx,
            "Context window",
            "Larger = more context (uses more VRAM)",
        ));

        report.push_str(&Self::compare_field(
            "max_loaded_models",
            current.max_loaded_models,
            optimized.max_loaded_models,
            "Models in memory",
            "Higher = faster model switching",
        ));

        if current.keep_alive != optimized.keep_alive {
            report.push_str(&format!(
                "keep_alive: {} → {} (Model retention time)\n",
                current.keep_alive, optimized.keep_alive
            ));
        }

        if let (Some(c), Some(o)) = (current.num_thread, optimized.num_thread) {
            if c != o {
                report.push_str(&format!(
                    "num_thread: {} → {} (CPU threads)\n",
                    c, o
                ));
            }
        }

        if report.lines().count() == 2 {
            report.push_str("No changes recommended - configuration already optimal!\n");
        }

        report
    }

    fn compare_field<T: std::fmt::Display + PartialEq>(
        name: &str,
        current: T,
        optimized: T,
        description: &str,
        note: &str,
    ) -> String {
        if current != optimized {
            format!(
                "{}: {} → {} ({})\n  Note: {}\n",
                name, current, optimized, description, note
            )
        } else {
            String::new()
        }
    }
}

/// Optimized configuration result
///
/// This is a simplified version of OllamaConfig focused on the
/// parameters that the optimizer actually sets.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizedConfig {
    pub num_parallel: u32,
    pub max_loaded_models: u32,
    pub keep_alive: String,
    pub num_ctx: u32,
    pub num_batch: u32,
    pub num_gpu: i32,
    pub num_thread: Option<u32>,
}

impl Default for OptimizedConfig {
    fn default() -> Self {
        // Ollama defaults (conservative)
        Self {
            num_parallel: 1,
            max_loaded_models: 1,
            keep_alive: "5m".to_string(),
            num_ctx: 2048,
            num_batch: 512,
            num_gpu: -1, // Auto-detect
            num_thread: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuVendor, GpuInfo, CpuInfo, MemoryInfo};

    fn mock_hardware_high_end() -> HardwareSpec {
        HardwareSpec {
            gpu: Some(GpuInfo {
                name: "NVIDIA RTX 4090".to_string(),
                vendor: GpuVendor::Nvidia,
                vram_total_mb: 24576, // 24GB
                vram_free_mb: 24000,
                driver_version: Some("535.54".to_string()),
                compute_capability: Some("8.9".to_string()),
            }),
            cpu: CpuInfo {
                model: "AMD Ryzen 9 7950X".to_string(),
                cores: 16,
                threads: 32,
                frequency_mhz: Some(4500),
            },
            memory: MemoryInfo {
                total_mb: 65536, // 64GB
                available_mb: 60000,
                used_mb: 5536,
            },
            os: "Linux".to_string(),
            platform: "linux-x86_64".to_string(),
        }
    }

    fn mock_hardware_mid_range() -> HardwareSpec {
        HardwareSpec {
            gpu: Some(GpuInfo {
                name: "NVIDIA RTX 3060".to_string(),
                vendor: GpuVendor::Nvidia,
                vram_total_mb: 12288, // 12GB
                vram_free_mb: 12000,
                driver_version: Some("535.54".to_string()),
                compute_capability: Some("8.6".to_string()),
            }),
            cpu: CpuInfo {
                model: "Intel i7-12700".to_string(),
                cores: 12,
                threads: 20,
                frequency_mhz: Some(3600),
            },
            memory: MemoryInfo {
                total_mb: 32768, // 32GB
                available_mb: 28000,
                used_mb: 4768,
            },
            os: "Linux".to_string(),
            platform: "linux-x86_64".to_string(),
        }
    }

    fn mock_hardware_cpu_only() -> HardwareSpec {
        HardwareSpec {
            gpu: None,
            cpu: CpuInfo {
                model: "Intel i5-10400".to_string(),
                cores: 6,
                threads: 12,
                frequency_mhz: Some(2900),
            },
            memory: MemoryInfo {
                total_mb: 16384, // 16GB
                available_mb: 12000,
                used_mb: 4384,
            },
            os: "Linux".to_string(),
            platform: "linux-x86_64".to_string(),
        }
    }

    #[test]
    fn test_optimize_high_end() {
        let hw = mock_hardware_high_end();
        let config = OllamaOptimizer::optimize(&hw);

        // High VRAM (24GB)
        assert_eq!(config.num_parallel, 4);
        assert_eq!(config.num_gpu, 999);
        assert_eq!(config.num_batch, 1024);
        assert_eq!(config.num_ctx, 16384);

        // High RAM (64GB exactly, so <= 64 gives 3)
        assert_eq!(config.max_loaded_models, 3);
        assert_eq!(config.keep_alive, "30m");
    }

    #[test]
    fn test_optimize_mid_range() {
        let hw = mock_hardware_mid_range();
        let config = OllamaOptimizer::optimize(&hw);

        // Mid VRAM (12GB)
        assert_eq!(config.num_parallel, 2);
        assert_eq!(config.num_gpu, 999);
        assert_eq!(config.num_batch, 256);
        assert_eq!(config.num_ctx, 4096);

        // Mid RAM (32GB exactly, so <= 32 gives 2)
        assert_eq!(config.max_loaded_models, 2);
        assert_eq!(config.keep_alive, "15m");
    }

    #[test]
    fn test_optimize_cpu_only() {
        let hw = mock_hardware_cpu_only();
        let config = OllamaOptimizer::optimize(&hw);

        // CPU mode
        assert_eq!(config.num_gpu, 0);
        assert_eq!(config.num_thread, Some(12));
        assert_eq!(config.num_batch, 128);
        assert_eq!(config.num_ctx, 2048);
        assert_eq!(config.num_parallel, 1);

        // Low RAM (16GB)
        assert_eq!(config.max_loaded_models, 1);
        assert_eq!(config.keep_alive, "5m");
    }

    #[test]
    fn test_generate_report() {
        let current = OptimizedConfig::default();
        let mut optimized = OptimizedConfig::default();
        optimized.num_parallel = 4;
        optimized.num_gpu = 999;

        let report = OllamaOptimizer::generate_report(&current, &optimized);

        assert!(report.contains("num_parallel"));
        assert!(report.contains("1 → 4"));
        assert!(report.contains("num_gpu"));
        assert!(report.contains("-1 → 999"));
    }

    #[test]
    fn test_no_changes_report() {
        let config = OptimizedConfig::default();
        let report = OllamaOptimizer::generate_report(&config, &config);

        assert!(report.contains("No changes recommended"));
    }
}
