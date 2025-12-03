//! Error types for VeloLLM core library
//!
//! This module provides structured error types using `thiserror` for
//! type-safe error handling across the library.
//!
//! # Error Hierarchy
//!
//! ```text
//! VeloLLMError (top-level, re-exported)
//! ├── HardwareError      - Hardware detection failures
//! ├── PagedAttentionError - Memory block management
//! └── SchedulerError     - Request scheduling
//! ```
//!
//! # Usage
//!
//! Library code should use specific error types:
//!
//! ```rust,ignore
//! use velollm_core::error::HardwareError;
//!
//! fn detect_gpu() -> Result<GpuInfo, HardwareError> {
//!     // ...
//! }
//! ```
//!
//! Application code should use `anyhow` with context:
//!
//! ```rust,ignore
//! use anyhow::Context;
//!
//! let hw = HardwareSpec::detect()
//!     .context("Failed to detect hardware")?;
//! ```

use thiserror::Error;

/// Errors that can occur during hardware detection
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HardwareError {
    /// GPU detection failed
    #[error("gpu detection failed: {0}")]
    GpuDetection(String),

    /// CPU detection failed
    #[error("cpu detection failed: {0}")]
    CpuDetection(String),

    /// Memory detection failed
    #[error("memory detection failed: {0}")]
    MemoryDetection(String),

    /// Command execution failed
    #[error("command '{command}' failed: {message}")]
    CommandFailed { command: String, message: String },

    /// Parsing error when reading hardware info
    #[error("failed to parse {location}: {message}")]
    ParseError { location: String, message: String },

    /// System information unavailable
    #[error("system information unavailable: {0}")]
    SystemInfoUnavailable(String),
}

impl HardwareError {
    /// Create a GPU detection error
    pub fn gpu(msg: impl Into<String>) -> Self {
        Self::GpuDetection(msg.into())
    }

    /// Create a CPU detection error
    pub fn cpu(msg: impl Into<String>) -> Self {
        Self::CpuDetection(msg.into())
    }

    /// Create a memory detection error
    pub fn memory(msg: impl Into<String>) -> Self {
        Self::MemoryDetection(msg.into())
    }

    /// Create a command failure error
    pub fn command(cmd: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::CommandFailed { command: cmd.into(), message: msg.into() }
    }

    /// Create a parse error
    pub fn parse(location: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::ParseError { location: location.into(), message: msg.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_error_display() {
        let err = HardwareError::gpu("nvidia-smi not found");
        assert_eq!(err.to_string(), "gpu detection failed: nvidia-smi not found");

        let err = HardwareError::command("nvidia-smi", "exit code 1");
        assert_eq!(err.to_string(), "command 'nvidia-smi' failed: exit code 1");

        let err = HardwareError::parse("nvidia-smi output", "unexpected format");
        assert_eq!(err.to_string(), "failed to parse nvidia-smi output: unexpected format");
    }

    #[test]
    fn test_hardware_error_equality() {
        let err1 = HardwareError::gpu("test");
        let err2 = HardwareError::gpu("test");
        let err3 = HardwareError::cpu("test");

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_hardware_error_helpers() {
        let err = HardwareError::gpu("test");
        assert!(matches!(err, HardwareError::GpuDetection(_)));

        let err = HardwareError::cpu("test");
        assert!(matches!(err, HardwareError::CpuDetection(_)));

        let err = HardwareError::memory("test");
        assert!(matches!(err, HardwareError::MemoryDetection(_)));
    }
}
