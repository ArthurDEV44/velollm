//! Build script for velollm-adapters-llamacpp
//!
//! When the `cuda` feature is enabled, this script compiles the CUDA paged attention kernel.

fn main() {
    // Only compile CUDA when the feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda();

    // Re-run if CUDA source files change
    println!("cargo:rerun-if-changed=cuda/paged_attention.cu");
    println!("cargo:rerun-if-changed=cuda/paged_attention.cuh");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    use std::env;
    use std::path::PathBuf;

    // Find CUDA toolkit
    let cuda_path = find_cuda_path();

    if cuda_path.is_none() {
        println!(
            "cargo:warning=CUDA toolkit not found. Paged attention kernel will not be available."
        );
        println!("cargo:warning=Install CUDA 12+ and ensure nvcc is in PATH or set CUDA_PATH.");
        return;
    }

    let cuda_path = cuda_path.unwrap();
    let nvcc = cuda_path.join("bin").join("nvcc");

    if !nvcc.exists() {
        println!("cargo:warning=nvcc not found at {:?}", nvcc);
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_src = PathBuf::from("cuda/paged_attention.cu");
    let cuda_obj = out_dir.join("paged_attention.o");
    let cuda_lib = out_dir.join("libvelollm_paged_attention.a");

    // Compile CUDA source to object file
    let status = std::process::Command::new(&nvcc)
        .args([
            "-c",
            cuda_src.to_str().unwrap(),
            "-o",
            cuda_obj.to_str().unwrap(),
            "-I",
            "cuda",
            // Optimization flags
            "-O3",
            "--use_fast_math",
            // Generate code for common architectures
            "-gencode=arch=compute_70,code=sm_70", // Volta (V100)
            "-gencode=arch=compute_75,code=sm_75", // Turing (RTX 20xx)
            "-gencode=arch=compute_80,code=sm_80", // Ampere (A100, RTX 30xx)
            "-gencode=arch=compute_86,code=sm_86", // Ampere (RTX 30xx)
            "-gencode=arch=compute_89,code=sm_89", // Ada Lovelace (RTX 40xx)
            "-gencode=arch=compute_90,code=sm_90", // Hopper (H100)
            // CUDA standard
            "--std=c++17",
            "-Xcompiler",
            "-fPIC",
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("CUDA compilation failed");
    }

    // Create static library from object file
    let status = std::process::Command::new("ar")
        .args([
            "rcs",
            cuda_lib.to_str().unwrap(),
            cuda_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute ar");

    if !status.success() {
        panic!("Failed to create static library");
    }

    // Tell cargo to link against our library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=velollm_paged_attention");

    // Link CUDA runtime
    let cuda_lib_path = cuda_path.join("lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:warning=CUDA paged attention kernel compiled successfully");
}

#[cfg(feature = "cuda")]
fn find_cuda_path() -> Option<PathBuf> {
    use std::path::PathBuf;

    // Check CUDA_PATH environment variable first
    if let Ok(path) = std::env::var("CUDA_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    // Check CUDA_HOME environment variable
    if let Ok(path) = std::env::var("CUDA_HOME") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    // Check common installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.4",
        "/opt/cuda",
    ];

    for path in &common_paths {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    // Try to find nvcc in PATH
    if let Ok(output) = std::process::Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            if let Ok(nvcc_path) = String::from_utf8(output.stdout) {
                let nvcc_path = PathBuf::from(nvcc_path.trim());
                if let Some(bin_dir) = nvcc_path.parent() {
                    if let Some(cuda_dir) = bin_dir.parent() {
                        return Some(cuda_dir.to_path_buf());
                    }
                }
            }
        }
    }

    None
}
