//! CUDA Paged Attention Kernel Interface
//!
//! This module provides a safe Rust interface to the VeloLLM CUDA paged attention kernel.
//! It enables efficient attention computation over non-contiguous KV cache blocks.
//!
//! # Architecture
//!
//! The CUDA kernel computes: `softmax(Q @ K^T / sqrt(d)) @ V`
//!
//! Where K and V are stored in paged blocks (non-contiguous memory) managed by
//! the BlockManager from velollm-core.
//!
//! # Example
//!
//! ```ignore
//! use velollm_adapters_llamacpp::cuda_paged::{PagedAttentionCuda, PagedAttentionConfig};
//!
//! let config = PagedAttentionConfig {
//!     num_heads: 32,
//!     num_kv_heads: 8,
//!     head_dim: 128,
//!     block_size: 16,
//!     max_context_len: 4096,
//!     dtype: DataType::Float16,
//! };
//!
//! let attention = PagedAttentionCuda::new(config)?;
//!
//! // Run attention with paged KV cache
//! attention.forward(&query, &key_cache, &value_cache, &block_tables, &context_lens, &mut output)?;
//! ```

use std::ffi::c_void;
use thiserror::Error;

/// Errors that can occur during CUDA paged attention operations
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CudaPagedError {
    #[error("CUDA paged attention not available (CUDA support not compiled)")]
    NotAvailable,

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("CUDA kernel launch failed")]
    KernelLaunchFailed,

    #[error("Out of GPU memory")]
    OutOfMemory,

    #[error("Unsupported configuration: {0}")]
    Unsupported(String),

    #[error("CUDA error: {0}")]
    CudaError(String),
}

/// Data type for attention computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    Float32 = 0,
    Float16 = 1,
}

impl DataType {
    /// Size in bytes per element
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 => 2,
        }
    }
}

/// Configuration for paged attention
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Number of query attention heads
    pub num_heads: i32,
    /// Number of KV attention heads (may differ for GQA)
    pub num_kv_heads: i32,
    /// Dimension per attention head
    pub head_dim: i32,
    /// Number of tokens per KV cache block
    pub block_size: i32,
    /// Maximum supported context length
    pub max_context_len: i32,
    /// Data type for computation
    pub dtype: DataType,
}

impl PagedAttentionConfig {
    /// Create config for Llama 3.1 8B model
    pub fn llama_8b() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 8, // GQA with 4x ratio
            head_dim: 128,
            block_size: 16,
            max_context_len: 8192,
            dtype: DataType::Float16,
        }
    }

    /// Create config for Llama 3.2 3B model
    pub fn llama_3b() -> Self {
        Self {
            num_heads: 24,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            max_context_len: 8192,
            dtype: DataType::Float16,
        }
    }

    /// Calculate maximum blocks per sequence
    pub fn max_blocks_per_seq(&self) -> i32 {
        (self.max_context_len + self.block_size - 1) / self.block_size
    }

    /// Calculate attention scale factor (1/sqrt(head_dim))
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self::llama_8b()
    }
}

// FFI declarations for the CUDA kernel
// These are only available when compiled with CUDA support
#[cfg(feature = "cuda")]
mod ffi {
    use super::*;
    use std::ffi::c_int;

    #[repr(C)]
    pub struct PagedAttentionParams {
        pub output: *mut c_void,
        pub query: *const c_void,
        pub key_cache: *const c_void,
        pub value_cache: *const c_void,
        pub block_tables: *const i32,
        pub context_lens: *const i32,
        pub num_seqs: i32,
        pub num_heads: i32,
        pub num_kv_heads: i32,
        pub head_dim: i32,
        pub block_size: i32,
        pub max_context_len: i32,
        pub max_blocks_per_seq: i32,
        pub scale: f32,
        pub dtype: c_int,
        pub stream: *mut c_void, // cudaStream_t
    }

    #[link(name = "velollm_paged_attention", kind = "static")]
    extern "C" {
        pub fn velollm_paged_attention_forward(params: *const PagedAttentionParams) -> c_int;
        pub fn velollm_get_error_string(error: c_int) -> *const std::ffi::c_char;
        pub fn velollm_paged_attention_supported(
            head_dim: c_int,
            block_size: c_int,
            dtype: c_int,
        ) -> c_int;
        pub fn velollm_get_recommended_block_size(head_dim: c_int) -> c_int;
    }
}

/// CUDA paged attention runner
///
/// Manages the configuration and execution of paged attention on GPU.
#[derive(Debug, Clone)]
pub struct PagedAttentionCuda {
    config: PagedAttentionConfig,
}

impl PagedAttentionCuda {
    /// Create a new PagedAttentionCuda instance
    ///
    /// Returns an error if the configuration is not supported.
    pub fn new(config: PagedAttentionConfig) -> Result<Self, CudaPagedError> {
        // Validate configuration
        if config.num_heads <= 0 || config.num_kv_heads <= 0 {
            return Err(CudaPagedError::InvalidArgument(
                "num_heads and num_kv_heads must be positive".to_string(),
            ));
        }

        if config.num_heads % config.num_kv_heads != 0 {
            return Err(CudaPagedError::InvalidArgument(
                "num_heads must be divisible by num_kv_heads".to_string(),
            ));
        }

        if config.head_dim <= 0 || config.block_size <= 0 {
            return Err(CudaPagedError::InvalidArgument(
                "head_dim and block_size must be positive".to_string(),
            ));
        }

        // Check if configuration is supported
        if !Self::is_supported(&config) {
            return Err(CudaPagedError::Unsupported(format!(
                "head_dim={}, block_size={}, dtype={:?} not supported",
                config.head_dim, config.block_size, config.dtype
            )));
        }

        Ok(Self { config })
    }

    /// Check if a configuration is supported by the CUDA kernel
    #[cfg(feature = "cuda")]
    pub fn is_supported(config: &PagedAttentionConfig) -> bool {
        unsafe {
            ffi::velollm_paged_attention_supported(
                config.head_dim,
                config.block_size,
                config.dtype as i32,
            ) != 0
        }
    }

    /// Check if a configuration is supported (stub when CUDA not available)
    #[cfg(not(feature = "cuda"))]
    pub fn is_supported(_config: &PagedAttentionConfig) -> bool {
        false
    }

    /// Get the recommended block size for a given head dimension
    #[cfg(feature = "cuda")]
    pub fn recommended_block_size(head_dim: i32) -> i32 {
        unsafe { ffi::velollm_get_recommended_block_size(head_dim) }
    }

    /// Get the recommended block size (stub when CUDA not available)
    #[cfg(not(feature = "cuda"))]
    pub fn recommended_block_size(_head_dim: i32) -> i32 {
        16
    }

    /// Get the configuration
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    /// Run paged attention forward pass
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - All pointers in `input` are valid CUDA device pointers
    /// - Tensor dimensions match the configuration
    /// - `input.query` has shape [num_seqs, num_heads, head_dim]
    /// - `input.key_cache` has shape [num_blocks, block_size, num_kv_heads, head_dim]
    /// - `input.value_cache` has shape [num_blocks, block_size, num_kv_heads, head_dim]
    /// - `input.block_tables` has shape [num_seqs, max_blocks_per_seq]
    /// - `input.context_lens` has shape [num_seqs]
    /// - `input.output` has shape [num_seqs, num_heads, head_dim]
    /// - Memory is properly synchronized if using non-null stream
    #[cfg(feature = "cuda")]
    pub unsafe fn forward(&self, input: &ForwardInput) -> Result<(), CudaPagedError> {
        let params = ffi::PagedAttentionParams {
            output: input.output,
            query: input.query,
            key_cache: input.key_cache,
            value_cache: input.value_cache,
            block_tables: input.block_tables,
            context_lens: input.context_lens,
            num_seqs: input.num_seqs,
            num_heads: self.config.num_heads,
            num_kv_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
            block_size: self.config.block_size,
            max_context_len: self.config.max_context_len,
            max_blocks_per_seq: self.config.max_blocks_per_seq(),
            scale: self.config.scale(),
            dtype: self.config.dtype as i32,
            stream: input.stream,
        };

        let result = ffi::velollm_paged_attention_forward(&params);

        if result == 0 {
            Ok(())
        } else {
            let error_str = ffi::velollm_get_error_string(result);
            let error_msg = if error_str.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(error_str)
                    .to_string_lossy()
                    .into_owned()
            };
            Err(CudaPagedError::CudaError(error_msg))
        }
    }

    /// Run paged attention forward pass (stub when CUDA not available)
    ///
    /// # Safety
    ///
    /// See CUDA version for safety requirements. This stub always returns NotAvailable.
    #[cfg(not(feature = "cuda"))]
    pub unsafe fn forward(&self, _input: &ForwardInput) -> Result<(), CudaPagedError> {
        Err(CudaPagedError::NotAvailable)
    }
}

/// Input tensors for paged attention forward pass
///
/// All pointers must be valid CUDA device pointers.
#[derive(Debug, Clone, Copy)]
pub struct ForwardInput {
    /// Query tensor [num_seqs, num_heads, head_dim]
    pub query: *const c_void,
    /// Key cache [num_blocks, block_size, num_kv_heads, head_dim]
    pub key_cache: *const c_void,
    /// Value cache [num_blocks, block_size, num_kv_heads, head_dim]
    pub value_cache: *const c_void,
    /// Block table mapping logical to physical blocks [num_seqs, max_blocks_per_seq]
    pub block_tables: *const i32,
    /// Context length per sequence [num_seqs]
    pub context_lens: *const i32,
    /// Number of sequences in the batch
    pub num_seqs: i32,
    /// Output tensor [num_seqs, num_heads, head_dim]
    pub output: *mut c_void,
    /// CUDA stream (null for default stream)
    pub stream: *mut c_void,
}

/// Check if CUDA paged attention is available
pub fn is_cuda_available() -> bool {
    cfg!(feature = "cuda")
}

/// GPU memory information for paged attention planning
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    /// Total GPU memory in bytes
    pub total_memory: usize,
    /// Available GPU memory in bytes
    pub available_memory: usize,
    /// Memory required per KV cache block
    pub bytes_per_block: usize,
    /// Number of blocks that can fit in available memory
    pub max_blocks: usize,
}

impl GpuMemoryInfo {
    /// Estimate GPU memory requirements for a given configuration
    pub fn estimate(config: &PagedAttentionConfig, available_memory: usize) -> Self {
        // Bytes per block = block_size * num_layers * 2 (K+V) * num_kv_heads * head_dim * dtype_size
        // Note: num_layers is typically fixed per model
        let num_layers = 32; // Typical for Llama-class models
        let bytes_per_block = config.block_size as usize
            * num_layers
            * 2  // K and V
            * config.num_kv_heads as usize
            * config.head_dim as usize
            * config.dtype.size_bytes();

        let max_blocks = available_memory / bytes_per_block;

        Self { total_memory: available_memory, available_memory, bytes_per_block, max_blocks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_llama_8b() {
        let config = PagedAttentionConfig::llama_8b();
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn test_config_scale() {
        let config = PagedAttentionConfig::llama_8b();
        let expected_scale = 1.0 / (128.0_f32).sqrt();
        assert!((config.scale() - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_max_blocks_per_seq() {
        let config = PagedAttentionConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            max_context_len: 100,
            dtype: DataType::Float16,
        };
        assert_eq!(config.max_blocks_per_seq(), 7); // ceil(100/16) = 7
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float16.size_bytes(), 2);
    }

    #[test]
    fn test_paged_attention_creation() {
        let config = PagedAttentionConfig::llama_8b();

        // Without CUDA feature, creation should succeed but forward should fail
        #[cfg(not(feature = "cuda"))]
        {
            // When CUDA is not available, is_supported returns false
            // so new() should return Unsupported error
            let result = PagedAttentionCuda::new(config);
            assert!(matches!(result, Err(CudaPagedError::Unsupported(_))));
        }
    }

    #[test]
    fn test_invalid_config() {
        let config = PagedAttentionConfig {
            num_heads: 32,
            num_kv_heads: 7, // 32 is not divisible by 7
            head_dim: 128,
            block_size: 16,
            max_context_len: 4096,
            dtype: DataType::Float16,
        };

        let result = PagedAttentionCuda::new(config);
        assert!(matches!(result, Err(CudaPagedError::InvalidArgument(_))));
    }

    #[test]
    fn test_gpu_memory_estimate() {
        let config = PagedAttentionConfig::llama_8b();
        let available_memory = 16 * 1024 * 1024 * 1024; // 16 GB

        let info = GpuMemoryInfo::estimate(&config, available_memory);

        // bytes_per_block = 16 * 32 * 2 * 8 * 128 * 2 = 2,097,152 (2 MB)
        assert_eq!(info.bytes_per_block, 2_097_152);

        // max_blocks = 16GB / 2MB = 8192
        assert_eq!(info.max_blocks, 8192);
    }

    #[test]
    fn test_recommended_block_size() {
        let size = PagedAttentionCuda::recommended_block_size(128);
        assert_eq!(size, 16);
    }

    #[test]
    fn test_is_cuda_available() {
        // This just tests the compile-time check works
        let _available = is_cuda_available();
    }
}
