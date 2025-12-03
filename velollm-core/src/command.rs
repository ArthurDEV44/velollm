//! Command execution utilities with timeout support
//!
//! This module provides safe command execution with configurable timeouts
//! to prevent hanging when external tools (nvidia-smi, rocm-smi, etc.) are
//! unresponsive or the system is in a bad state.

use std::io;
use std::process::{Command, Output, Stdio};
use std::time::Duration;
use tracing::{trace, warn};
use wait_timeout::ChildExt;

/// Default timeout for command execution (5 seconds)
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of a command execution with timeout
#[derive(Debug)]
pub enum CommandResult {
    /// Command completed successfully
    Success(Output),
    /// Command timed out
    Timeout,
    /// Command failed to start (e.g., not found)
    SpawnError(io::Error),
}

impl CommandResult {
    /// Returns the output if the command succeeded, None otherwise
    pub fn output(self) -> Option<Output> {
        match self {
            CommandResult::Success(output) => Some(output),
            _ => None,
        }
    }

    /// Returns true if the command completed (regardless of exit status)
    pub fn completed(&self) -> bool {
        matches!(self, CommandResult::Success(_))
    }

    /// Returns true if the command timed out
    pub fn timed_out(&self) -> bool {
        matches!(self, CommandResult::Timeout)
    }
}

/// Execute a command with a timeout
///
/// # Arguments
/// * `cmd` - The command to execute (will be modified to capture output)
/// * `timeout` - Maximum time to wait for the command to complete
///
/// # Returns
/// * `CommandResult::Success` - Command completed within timeout
/// * `CommandResult::Timeout` - Command was killed after timeout
/// * `CommandResult::SpawnError` - Failed to start the command
///
/// # Example
/// ```no_run
/// use std::process::Command;
/// use std::time::Duration;
/// use velollm_core::command::{run_with_timeout, CommandResult};
///
/// let mut cmd = Command::new("nvidia-smi");
/// cmd.args(["--query-gpu=name", "--format=csv"]);
///
/// match run_with_timeout(&mut cmd, Duration::from_secs(5)) {
///     CommandResult::Success(output) => {
///         println!("Output: {}", String::from_utf8_lossy(&output.stdout));
///     }
///     CommandResult::Timeout => {
///         eprintln!("Command timed out");
///     }
///     CommandResult::SpawnError(e) => {
///         eprintln!("Failed to start command: {}", e);
///     }
/// }
/// ```
pub fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> CommandResult {
    // Configure to capture stdout/stderr
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // Spawn the child process
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            trace!(error = %e, "Failed to spawn command");
            return CommandResult::SpawnError(e);
        }
    };

    // Wait with timeout
    match child.wait_timeout(timeout) {
        Ok(Some(status)) => {
            // Command completed within timeout
            let stdout = child
                .stdout
                .take()
                .map(|mut s| {
                    let mut buf = Vec::new();
                    io::Read::read_to_end(&mut s, &mut buf).ok();
                    buf
                })
                .unwrap_or_default();

            let stderr = child
                .stderr
                .take()
                .map(|mut s| {
                    let mut buf = Vec::new();
                    io::Read::read_to_end(&mut s, &mut buf).ok();
                    buf
                })
                .unwrap_or_default();

            CommandResult::Success(Output {
                status,
                stdout,
                stderr,
            })
        }
        Ok(None) => {
            // Timeout - kill the process
            warn!(timeout_secs = timeout.as_secs(), "Command timed out, killing process");
            let _ = child.kill();
            let _ = child.wait(); // Reap the zombie
            CommandResult::Timeout
        }
        Err(e) => {
            // Error waiting for process
            warn!(error = %e, "Error waiting for command");
            let _ = child.kill();
            let _ = child.wait();
            CommandResult::SpawnError(e)
        }
    }
}

/// Execute a command with the default timeout
///
/// Convenience wrapper around `run_with_timeout` using `DEFAULT_TIMEOUT`.
pub fn run_with_default_timeout(cmd: &mut Command) -> CommandResult {
    run_with_timeout(cmd, DEFAULT_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_successful_command() {
        let mut cmd = Command::new("echo");
        cmd.arg("hello");

        let result = run_with_timeout(&mut cmd, Duration::from_secs(5));
        assert!(result.completed());

        if let CommandResult::Success(output) = result {
            assert!(output.status.success());
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "hello");
        }
    }

    #[test]
    fn test_command_not_found() {
        let mut cmd = Command::new("this_command_does_not_exist_12345");

        let result = run_with_timeout(&mut cmd, Duration::from_secs(1));
        assert!(matches!(result, CommandResult::SpawnError(_)));
    }

    #[test]
    fn test_timeout() {
        // Sleep for 10 seconds but timeout after 100ms
        let mut cmd = Command::new("sleep");
        cmd.arg("10");

        let result = run_with_timeout(&mut cmd, Duration::from_millis(100));
        assert!(result.timed_out());
    }

    #[test]
    fn test_command_result_methods() {
        let success = CommandResult::Success(Output {
            status: std::process::ExitStatus::default(),
            stdout: vec![],
            stderr: vec![],
        });
        assert!(success.completed());
        assert!(!success.timed_out());

        let timeout = CommandResult::Timeout;
        assert!(!timeout.completed());
        assert!(timeout.timed_out());
    }

    #[test]
    fn test_default_timeout_value() {
        assert_eq!(DEFAULT_TIMEOUT, Duration::from_secs(5));
    }
}
