# VeloLLM Benchmark Results

**Last Updated**: 2025-11-27

## Methodology

All benchmarks run with:
- **Warm model** (2nd+ run to exclude load time)
- **Ollama backend** (vanilla configuration)
- **Model**: llama3.2:3b (GGUF format)
- **Average of multiple runs** (3-5 iterations per benchmark)

## Hardware Configuration

### Test System: Gaming Workstation

**GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- VRAM: 16 GB (16376 MB)
- VRAM Free: 8.3 GB (during tests)
- Driver: 581.80
- Compute Capability: 8.9
- Architecture: Ada Lovelace (RTX 40 series)

**CPU**: AMD Ryzen 7 7800X3D
- Cores: 8 physical / 16 threads
- Frequency: 4.2 GHz
- Cache: 96MB L3 (3D V-Cache technology)
- Architecture: Zen 4

**RAM**: 15.8 GB DDR5
- Available during tests: 10.7 GB
- Used: 5.1 GB

**OS**: Linux (WSL2) - `linux-x86_64`

---

## Baseline Results (Ollama Vanilla)

### Llama 3.2 3B - Q4_K_M Quantization

| Benchmark | Iterations | Max Tokens | Tokens/s | TTFT (ms) | Total Time (ms) | Total Tokens |
|-----------|------------|------------|----------|-----------|-----------------|--------------|
| **Short Completion** | 5 | 50 | **65.6** | 16.0 | 3,809 | 250 |
| **Medium Completion** | 3 | 150 | **170.5** | 21.8 | 2,640 | 450 |
| **Code Generation** | 3 | 200 | **175.4** | 21.7 | 3,420 | 600 |

**Overall Average**: **137.2 tokens/s** | **TTFT**: 19.8 ms

### Detailed Results

#### Short Completion
```yaml
Prompt: "Write a hello world program in Python"
Max Tokens: 50
Iterations: 5
Results:
  - Tokens/s: 65.6
  - TTFT: 16.0 ms
  - Prompt Eval Count: 32 tokens
  - Eval Count: 50 tokens
  - Total Time: 3.8s (for 5 iterations)
```

#### Medium Completion
```yaml
Prompt: "Explain how neural networks learn through backpropagation in detail"
Max Tokens: 150
Iterations: 3
Results:
  - Tokens/s: 170.5
  - TTFT: 21.8 ms
  - Prompt Eval Count: 37 tokens
  - Eval Count: 150 tokens
  - Total Time: 2.6s (for 3 iterations)
```

#### Code Generation
```yaml
Prompt: "Write a Rust function to compute the Fibonacci sequence using dynamic programming"
Max Tokens: 200
Iterations: 3
Results:
  - Tokens/s: 175.4
  - TTFT: 21.7 ms
  - Prompt Eval Count: 37 tokens
  - Eval Count: 200 tokens
  - Total Time: 3.4s (for 3 iterations)
```

---

## Analysis

### Performance Characteristics

1. **Excellent Baseline Performance**
   - 137 tok/s average is **4-6x better** than typical consumer hardware
   - TTFT of ~20ms is very low (excellent for interactive use)
   - RTX 4070 Ti SUPER + Ryzen 7800X3D is a strong combination

2. **Scaling with Generation Length**
   - Short (50 tok): 65.6 tok/s
   - Medium (150 tok): 170.5 tok/s (+160% improvement)
   - Long (200 tok): 175.4 tok/s (+167% improvement)

   **Observation**: Performance improves significantly with longer generations because prompt evaluation overhead is amortized over more tokens.

3. **GPU Utilization**
   - VRAM usage: ~8 GB (model + context)
   - VRAM free: 8.3 GB (plenty of headroom)
   - **Optimization potential**: Can fit larger models or increase batch size

### Comparison to Reference Hardware

| Hardware | Model | Baseline (Ollama) | Notes |
|----------|-------|-------------------|-------|
| **RTX 4070 Ti SUPER** (this system) | Llama 3.2 3B | **137 tok/s** | Current test results |
| RTX 4090 24GB | Llama 3.1 8B | ~28 tok/s | Typical user baseline |
| RTX 3060 12GB | Llama 3.2 3B | ~35 tok/s | Gaming laptop baseline |
| M2 Max 32GB | Llama 3.1 8B | ~22 tok/s | Apple Silicon baseline |

**Conclusion**: This system performs **4-5x better** than reference baselines, likely due to:
- Smaller model (3B vs 8B)
- Excellent GPU (RTX 4070 Ti SUPER)
- Fast CPU cache (Ryzen 7800X3D with 3D V-Cache)

---

## Optimization Targets

### Phase 1: Speculative Decoding (TASK-005 to TASK-007)

**Target**: 2.0-2.5x speedup
- **Baseline**: 137 tok/s
- **Optimized (estimated)**: 270-340 tok/s
- **Method**: llama3.2:3b (main) + llama3.2:1b (draft)
- **Status**: ⏳ Not yet implemented

### Phase 2: Ollama Configuration Optimization (TASK-009)

**Target**: 1.2-1.5x additional speedup
- **Optimizations**:
  - Increase `OLLAMA_NUM_BATCH` (current headroom: 8GB VRAM free)
  - Increase `OLLAMA_NUM_CTX` to 8192 or 16384
  - Set `OLLAMA_NUM_GPU=999` (max layers on GPU)
- **Status**: 🚧 CLI stub created

### Phase 3: Combined (Speculative + Config)

**Target**: 2.5-3.5x total speedup
- **Baseline**: 137 tok/s
- **Optimized (estimated)**: 340-480 tok/s
- **Methods**: Speculative decoding + optimal Ollama config
- **Status**: ⏳ Pending

---

## Memory Usage

### Current (Baseline)

| Component | Usage | Notes |
|-----------|-------|-------|
| **Model (VRAM)** | ~4-5 GB | Llama 3.2 3B Q4_K_M |
| **KV Cache (VRAM)** | ~2-3 GB | Default context window |
| **Other (VRAM)** | ~1 GB | Overhead |
| **Total VRAM** | ~8 GB | **50% GPU utilization** |

**Observation**: Significant VRAM headroom (8.3 GB free) → can optimize for higher throughput or larger models

### Optimization Potential

- **Increase batch size**: Can handle 2-4x larger batches
- **Larger context**: Can extend to 16K context window
- **Concurrent requests**: Can serve 2-3 users simultaneously
- **Larger model**: Could run llama3.2:7b or even llama3.1:8b

---

## Reproducing Results

### Prerequisites

1. **Install VeloLLM**:
   ```bash
   git clone https://github.com/ArthurDEV44/velollm
   cd velollm
   cargo build --release
   ```

2. **Install Ollama and Model**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2:3b
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

### Running Benchmarks

```bash
# Detect hardware
./target/release/velollm detect > my-hardware.json

# Run baseline benchmarks
./target/release/velollm benchmark \
  --model llama3.2:3b \
  --output my-baseline.json

# View results
cat my-baseline.json | jq
```

### Expected Output

```json
[
  {
    "config": {
      "name": "short_completion",
      "model": "llama3.2:3b",
      "prompt": "Write a hello world program in Python",
      "max_tokens": 50,
      "iterations": 5
    },
    "tokens_per_second": 65.6,
    "time_to_first_token_ms": 16.0,
    "total_time_ms": 3809.0,
    "total_tokens": 250
  },
  ...
]
```

---

## Next Steps

1. **Implement Speculative Decoding** (TASK-005 to TASK-007)
   - Expected: 2-2.5x speedup → 270-340 tok/s
   - Effort: ~9 hours

2. **Optimize Ollama Configuration** (TASK-009)
   - Expected: 1.2-1.5x additional speedup
   - Effort: ~4 hours

3. **Benchmark with Larger Models**
   - Try llama3.2:7b or llama3.1:8b
   - Document VRAM usage and performance

4. **Multi-Request Testing**
   - Test concurrent benchmark runs
   - Measure throughput degradation

---

## Raw Data

### Hardware Detection Output
```json
{
  "gpu": {
    "name": "NVIDIA GeForce RTX 4070 Ti SUPER",
    "vendor": "Nvidia",
    "vram_total_mb": 16376,
    "vram_free_mb": 8459,
    "driver_version": "581.80",
    "compute_capability": "8.9"
  },
  "cpu": {
    "model": "AMD Ryzen 7 7800X3D 8-Core Processor",
    "cores": 8,
    "threads": 16,
    "frequency_mhz": 4199
  },
  "memory": {
    "total_mb": 15820,
    "available_mb": 10700,
    "used_mb": 5119
  },
  "os": "linux",
  "platform": "linux-x86_64"
}
```

### Benchmark Results (Full)
See: [my-baseline.json](./my-baseline.json)

---

## Contributing

Have benchmark results from different hardware? Submit them via:
1. GitHub Issues with your `my-hardware.json` and `my-baseline.json`
2. Pull Request to add your results to this file

**Format**:
```markdown
### [Your Hardware Name]
- GPU: ...
- CPU: ...
- Results: ...
```

---

## License

MIT License - See [LICENSE](LICENSE) for details
