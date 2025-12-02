# VeloLLM Configuration Guide

This guide explains all Ollama configuration parameters that VeloLLM optimizes for your hardware.

## Overview

VeloLLM automatically detects your hardware and generates optimal Ollama configuration. The `velollm optimize` command creates environment variables that tune Ollama's behavior.

```bash
# Preview optimizations
velollm optimize --dry-run

# Generate and save configuration
velollm optimize -o velollm-config.sh
source velollm-config.sh
```

---

## Ollama Environment Variables

### OLLAMA_NUM_GPU

**Description**: Number of GPU layers to offload from the model.

**Values**:
- `-1` (default): Automatic detection
- `0`: CPU only
- `1-999`: Specific layer count (999 = all layers on GPU)

**VeloLLM Optimization**:
- **GPU detected**: Set to `999` (offload all layers to GPU)
- **No GPU**: Set to `0` (CPU-only mode)

**Impact**:
- Higher values = faster inference (GPU is much faster than CPU)
- Must fit within available VRAM

```bash
# Force all layers on GPU
export OLLAMA_NUM_GPU=999

# Force CPU-only mode
export OLLAMA_NUM_GPU=0
```

---

### OLLAMA_NUM_PARALLEL

**Description**: Number of concurrent requests Ollama can handle.

**Values**: `1` to `8` (default: `1`)

**VeloLLM Optimization** (based on VRAM):
| VRAM | num_parallel | Reasoning |
|------|--------------|-----------|
| < 12 GB | 1 | Limited memory for concurrent contexts |
| 12-24 GB | 2 | Medium headroom |
| > 24 GB | 4 | High-end GPU with plenty of memory |

**Impact**:
- Higher values allow serving multiple users/requests
- Each parallel request needs additional KV cache memory
- Too high may cause OOM (out of memory)

```bash
# Allow 2 concurrent requests
export OLLAMA_NUM_PARALLEL=2
```

---

### OLLAMA_NUM_BATCH

**Description**: Batch size for prompt processing (prefill phase).

**Values**: `128`, `256`, `512`, `1024` (default: `512`)

**VeloLLM Optimization** (based on VRAM):
| VRAM | num_batch | Reasoning |
|------|-----------|-----------|
| < 8 GB | 128 | Conservative to avoid OOM |
| 8-16 GB | 256 | Balanced |
| > 16 GB | 512 | Higher throughput |

**Impact**:
- Larger batch = faster prompt ingestion
- Uses more VRAM temporarily during prefill
- Primarily affects time-to-first-token (TTFT)

```bash
# Process prompts in batches of 512
export OLLAMA_NUM_BATCH=512
```

---

### OLLAMA_NUM_CTX

**Description**: Context window size (maximum tokens for prompt + response).

**Values**: `512` to `131072` (default: `2048`)

**VeloLLM Optimization** (based on VRAM):
| VRAM | num_ctx | Reasoning |
|------|---------|-----------|
| < 8 GB | 2048 | ~2K context fits in limited VRAM |
| 8-16 GB | 4096 | ~4K context for medium VRAM |
| > 16 GB | 8192 | ~8K context for high VRAM |

**Impact**:
- Larger context = handle longer conversations
- KV cache grows linearly with context size
- Formula: `KV_cache_size ≈ 2 * num_layers * num_ctx * hidden_dim * 2 bytes`

**Example Calculation (Llama 3.1 8B)**:
- Layers: 32, Hidden dim: 4096
- 4K context: ~2 GB VRAM for KV cache
- 8K context: ~4 GB VRAM for KV cache

```bash
# 8K context window
export OLLAMA_NUM_CTX=8192
```

---

### OLLAMA_NUM_THREAD

**Description**: Number of CPU threads to use for inference.

**Values**: `1` to CPU thread count (default: auto)

**VeloLLM Optimization**:
- **CPU-only mode**: Set to total thread count
- **GPU mode**: Not set (Ollama handles automatically)

**Impact**:
- Only affects CPU-bound operations (no GPU, or CPU layers)
- More threads = faster on multi-core CPUs
- Diminishing returns beyond physical core count

```bash
# Use 16 CPU threads
export OLLAMA_NUM_THREAD=16
```

---

### OLLAMA_MAX_LOADED_MODELS

**Description**: Maximum number of models to keep loaded in memory.

**Values**: `1` to unlimited (default: `1`)

**VeloLLM Optimization** (based on RAM):
| RAM | max_loaded_models | Reasoning |
|-----|-------------------|-----------|
| < 16 GB | 1 | Limited RAM for multiple models |
| 16-32 GB | 2 | Can hold 2 medium models |
| > 32 GB | 3 | Multi-model workflows |

**Impact**:
- Higher values allow faster model switching
- Each loaded model consumes RAM (even if inactive)
- Useful for multi-model workflows (e.g., main + draft model)

```bash
# Keep up to 2 models loaded
export OLLAMA_MAX_LOADED_MODELS=2
```

---

### OLLAMA_KEEP_ALIVE

**Description**: How long to keep a model loaded after the last request.

**Values**: Duration string (default: `5m`)
- `5m`: 5 minutes
- `30m`: 30 minutes
- `1h`: 1 hour
- `-1`: Forever (until unloaded manually)

**VeloLLM Optimization** (based on RAM):
| RAM | keep_alive | Reasoning |
|-----|------------|-----------|
| < 16 GB | 5m | Free memory quickly |
| 16-32 GB | 30m | Balance memory and reload time |
| > 32 GB | 1h | Keep models warm longer |

**Impact**:
- Longer = faster subsequent requests (no reload)
- Shorter = more memory available for other apps

```bash
# Keep model loaded for 30 minutes
export OLLAMA_KEEP_ALIVE=30m
```

---

### OLLAMA_HOST

**Description**: Address and port Ollama listens on.

**Values**: `host:port` (default: `127.0.0.1:11434`)

**VeloLLM**: Not automatically optimized (user-dependent).

```bash
# Listen on all interfaces
export OLLAMA_HOST=0.0.0.0:11434
```

---

### OLLAMA_MODELS

**Description**: Custom directory for model storage.

**Values**: Path to directory (default: `~/.ollama/models`)

**VeloLLM**: Not automatically optimized (user-dependent).

```bash
# Store models on fast NVMe drive
export OLLAMA_MODELS=/mnt/nvme/ollama/models
```

---

### OLLAMA_FLASH_ATTENTION

**Description**: Enable Flash Attention for faster attention computation.

**Values**: `0` (disabled) or `1` (enabled)

**Requirements**: CUDA compute capability >= 7.0 (RTX 20 series+)

**VeloLLM**: Not automatically set (experimental feature).

```bash
# Enable Flash Attention
export OLLAMA_FLASH_ATTENTION=1
```

---

## Optimization Profiles

### Low-End GPU (< 8 GB VRAM)

```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_BATCH=128
export OLLAMA_NUM_CTX=2048
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=5m
```

**Suited for**: RTX 3060 6GB, GTX 1660, RX 6600

### Mid-Range GPU (8-16 GB VRAM)

```bash
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_BATCH=256
export OLLAMA_NUM_CTX=4096
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=15m
```

**Suited for**: RTX 3070/3080, RTX 4060/4070, RX 6800

### High-End GPU (> 16 GB VRAM)

```bash
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_NUM_GPU=999
export OLLAMA_NUM_BATCH=512
export OLLAMA_NUM_CTX=8192
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=1h
```

**Suited for**: RTX 3090, RTX 4080/4090, RTX A6000, Apple M2 Max+

### CPU-Only Mode

```bash
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_NUM_GPU=0
export OLLAMA_NUM_BATCH=64
export OLLAMA_NUM_CTX=2048
export OLLAMA_NUM_THREAD=16  # Adjust to your CPU
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=5m
```

**Suited for**: Systems without GPU, or for testing

---

## Troubleshooting

### Out of Memory (OOM)

Reduce memory usage:
```bash
export OLLAMA_NUM_CTX=2048    # Reduce context
export OLLAMA_NUM_BATCH=128   # Smaller batches
export OLLAMA_NUM_PARALLEL=1  # Single request only
```

### Slow Performance

Increase throughput:
```bash
export OLLAMA_NUM_GPU=999     # All layers on GPU
export OLLAMA_NUM_BATCH=512   # Larger batches
```

### Model Loading Slow

Keep models warm:
```bash
export OLLAMA_KEEP_ALIVE=1h
export OLLAMA_MAX_LOADED_MODELS=2
```

---

## Advanced: Per-Request Overrides

Ollama API supports per-request overrides via `options`:

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Hello",
    "options": {
      "num_ctx": 4096,
      "num_batch": 512,
      "temperature": 0.7
    }
  }'
```

This allows testing different configurations without restarting Ollama.

---

## References

- [Ollama Environment Variables](https://github.com/ollama/ollama/blob/main/docs/faq.md)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [VeloLLM Hardware Detection](../api/hardware_detection.md)
