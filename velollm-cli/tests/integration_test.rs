//! End-to-end integration tests for VeloLLM CLI
//!
//! These tests verify the complete workflow of VeloLLM commands:
//! - Hardware detection
//! - Configuration optimization
//! - Benchmark execution (when Ollama is available)

use std::process::Command;
use std::fs;
use std::path::PathBuf;

/// Test the `velollm detect` command
#[test]
fn test_detect_command() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "velollm", "--", "detect"])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm detect");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Print output for debugging
    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);

    // Verify expected sections are present
    assert!(stdout.contains("System Information"), "Missing system info");
    assert!(stdout.contains("CPU"), "Missing CPU section");
    assert!(stdout.contains("Memory"), "Missing Memory section");
    assert!(stdout.contains("GPU"), "Missing GPU section");
    assert!(stdout.contains("JSON Output"), "Missing JSON output");

    // Verify JSON is valid
    let json_start = stdout.find('{').expect("No JSON found");
    let json_str = &stdout[json_start..];

    // Find the end of the JSON (last closing brace)
    let json_value: serde_json::Value = serde_json::from_str(json_str.trim())
        .expect("Invalid JSON in detect output");

    // Verify required fields
    assert!(json_value.get("cpu").is_some(), "Missing cpu field in JSON");
    assert!(json_value.get("memory").is_some(), "Missing memory field in JSON");
    assert!(json_value.get("os").is_some(), "Missing os field in JSON");
    assert!(json_value.get("platform").is_some(), "Missing platform field in JSON");
}

/// Test the `velollm optimize --dry-run` command
#[test]
fn test_optimize_dry_run() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "velollm", "--", "optimize", "--dry-run"])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm optimize --dry-run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);

    // Verify expected output sections
    assert!(stdout.contains("Hardware detected"), "Missing hardware detection");
    assert!(stdout.contains("Recommended Ollama configuration"), "Missing recommendations");
    assert!(stdout.contains("OLLAMA_NUM_PARALLEL"), "Missing NUM_PARALLEL");
    assert!(stdout.contains("OLLAMA_NUM_GPU"), "Missing NUM_GPU");
    assert!(stdout.contains("OLLAMA_NUM_BATCH"), "Missing NUM_BATCH");
    assert!(stdout.contains("OLLAMA_NUM_CTX"), "Missing NUM_CTX");
    assert!(stdout.contains("Dry run mode"), "Missing dry run notice");
}

/// Test the `velollm optimize -o <file>` command
#[test]
fn test_optimize_output_file() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("velollm_test_config.sh");

    // Clean up any existing file
    let _ = fs::remove_file(&output_path);

    let output = Command::new("cargo")
        .args([
            "run", "--bin", "velollm", "--",
            "optimize",
            "-o", output_path.to_str().unwrap()
        ])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm optimize -o");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    assert!(output.status.success(), "Command failed: {}", stderr);

    // Verify file was created
    assert!(output_path.exists(), "Output file was not created");

    // Read and validate the script
    let script = fs::read_to_string(&output_path)
        .expect("Failed to read output file");

    // Verify script structure
    assert!(script.starts_with("#!/bin/bash"), "Missing shebang");
    assert!(script.contains("VeloLLM"), "Missing VeloLLM header");
    assert!(script.contains("export OLLAMA_"), "Missing export statements");

    // Verify script is syntactically valid using bash -n
    let bash_check = Command::new("bash")
        .args(["-n", output_path.to_str().unwrap()])
        .output()
        .expect("Failed to run bash -n");

    assert!(
        bash_check.status.success(),
        "Generated script has syntax errors: {}",
        String::from_utf8_lossy(&bash_check.stderr)
    );

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Test that detect command produces valid JSON that can be parsed
#[test]
fn test_detect_json_structure() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "velollm", "--", "detect"])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm detect");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Extract JSON portion
    let json_start = stdout.find('{').expect("No JSON found");
    let json_str = &stdout[json_start..];

    let json: serde_json::Value = serde_json::from_str(json_str.trim())
        .expect("Invalid JSON structure");

    // Verify CPU fields
    let cpu = json.get("cpu").expect("Missing cpu");
    assert!(cpu.get("model").is_some(), "Missing cpu.model");
    assert!(cpu.get("cores").is_some(), "Missing cpu.cores");
    assert!(cpu.get("threads").is_some(), "Missing cpu.threads");

    // Verify memory fields
    let memory = json.get("memory").expect("Missing memory");
    assert!(memory.get("total_mb").is_some(), "Missing memory.total_mb");
    assert!(memory.get("available_mb").is_some(), "Missing memory.available_mb");
    assert!(memory.get("used_mb").is_some(), "Missing memory.used_mb");

    // Verify reasonable values
    let cores = cpu["cores"].as_u64().expect("cores should be a number");
    assert!(cores > 0, "CPU cores should be > 0");

    let total_mb = memory["total_mb"].as_u64().expect("total_mb should be a number");
    assert!(total_mb > 0, "Total memory should be > 0");
}

/// Test optimize command detects hardware correctly
#[test]
fn test_optimize_hardware_detection() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "velollm", "--", "optimize", "--dry-run"])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm optimize");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show hardware info (CPU cores/threads, RAM)
    assert!(stdout.contains("cores"), "Missing cores info");
    assert!(stdout.contains("threads"), "Missing threads info");
    assert!(stdout.contains("GB"), "Missing memory info in GB");
}

/// Test CLI help output
#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "velollm", "--", "--help"])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm --help");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify help shows available commands
    assert!(stdout.contains("detect"), "Missing detect command");
    assert!(stdout.contains("benchmark"), "Missing benchmark command");
    assert!(stdout.contains("optimize"), "Missing optimize command");
    assert!(stdout.contains("VeloLLM") || stdout.contains("velollm"), "Missing program name");
}

/// Test CLI version output
#[test]
fn test_cli_version() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "velollm", "--", "--version"])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm --version");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify version is shown
    assert!(stdout.contains("velollm"), "Missing program name in version");
    assert!(stdout.contains("0.1.0"), "Missing version number");
}

/// Test that benchmark command shows proper error when Ollama is not available
/// This test doesn't require Ollama to be running
#[test]
fn test_benchmark_without_ollama() {
    // Try to run benchmark - it should either succeed (if Ollama is running)
    // or fail gracefully with a proper error message
    let output = Command::new("cargo")
        .args([
            "run", "--bin", "velollm", "--",
            "benchmark",
            "--model", "test-model",
        ])
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .output()
        .expect("Failed to run velollm benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // If Ollama is not running, it should show a helpful message
    if !output.status.success() {
        let combined = format!("{}{}", stdout, stderr);
        assert!(
            combined.contains("Ollama") || combined.contains("ollama"),
            "Error message should mention Ollama"
        );
    }
    // If it succeeds, that's also fine (Ollama is running)
}
