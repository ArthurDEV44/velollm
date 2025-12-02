# PagedAttention Analysis

**Research Date**: 2025-12-02
**Source**: vLLM implementation, academic papers
**Paper**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

---

## Executive Summary

PagedAttention is a memory management technique for LLM inference that treats the KV cache like virtual memory in operating systems. Instead of allocating one contiguous block per sequence, it divides the KV cache into fixed-size **blocks (pages)** that can be stored non-contiguously in memory.

**Key Results**:
- Memory waste: **60-80% → <4%**
- Throughput improvement: **2-4x** (up to 24x vs naive implementations)
- Memory sharing: **up to 55%** reduction for parallel sampling

**Key Insight**: By paginating the KV cache, we eliminate fragmentation and enable dynamic memory allocation, similar to how OS virtual memory works.

---

## The Problem: KV Cache Memory Waste

### Traditional KV Cache Allocation

In standard LLM inference, each request pre-allocates a contiguous memory block for its entire maximum sequence length:

```
Traditional Approach:
┌─────────────────────────────────────────────────────────────┐
│ Request 1: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
│            │ used ││       wasted (pre-allocated)          │ │
├─────────────────────────────────────────────────────────────┤
│ Request 2: [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
│            │   used     ││        wasted                   │ │
├─────────────────────────────────────────────────────────────┤
│ Request 3: [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
│            │u││              wasted                        │ │
└─────────────────────────────────────────────────────────────┘

Memory Utilization: 20-40% (60-80% wasted!)
```

### Types of Memory Waste

1. **Internal Fragmentation**: Pre-allocated space that is never used
   - Max sequence = 4096 tokens, actual = 500 tokens → 87% wasted

2. **External Fragmentation**: Gaps between allocations
   - Cannot fit new sequence even though total free memory is sufficient

3. **Reservation Waste**: Memory reserved but not yet needed
   - Future tokens pre-allocated upfront

### Profiling Data (from vLLM paper)

| System | Actual KV Cache Usage | Wasted Memory |
|--------|----------------------|---------------|
| HuggingFace Transformers | 20.4% | 79.6% |
| FasterTransformer | 38.4% | 61.6% |
| **vLLM (PagedAttention)** | **>96%** | **<4%** |

---

## PagedAttention Solution

### Core Concept: Virtual Memory for KV Cache

PagedAttention applies the classic OS concept of **paging** to KV cache management:

| OS Concept | PagedAttention Equivalent |
|------------|---------------------------|
| Virtual Memory | Logical KV cache |
| Physical Memory | GPU VRAM |
| Page | KV Block |
| Page Table | Block Table |
| Process | Sequence/Request |
| Byte | Token |

### How It Works

```
PagedAttention Approach:
┌──────────────────────────────────────────────────────────────────┐
│ Physical Memory (VRAM):                                          │
│ ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐    │
│ │ B0 ││ B1 ││ B2 ││ B3 ││ B4 ││ B5 ││ B6 ││ B7 ││FREE││FREE│... │
│ │Seq1││Seq2││Seq1││Seq3││Seq2││Seq1││Seq3││Seq2││    ││    │    │
│ └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘    │
└──────────────────────────────────────────────────────────────────┘

Block Tables (logical → physical mapping):
┌──────────────────────────────────────────────────────────────────┐
│ Sequence 1: [B0, B2, B5]        → 3 blocks, allocated on demand  │
│ Sequence 2: [B1, B4, B7]        → 3 blocks, non-contiguous       │
│ Sequence 3: [B3, B6]            → 2 blocks, can grow dynamically │
└──────────────────────────────────────────────────────────────────┘

Memory Utilization: >96% (waste only in last block per sequence)
```

### Block Structure

Each block contains KV cache data for a fixed number of tokens:

```
Block Size = block_size × num_layers × 2 × head_dim × num_heads × dtype_size

Example (Llama 3.1 8B, block_size=16, FP16):
- block_size: 16 tokens
- num_layers: 32
- head_dim: 128
- num_heads: 32 (but using GQA: 8 KV heads)
- dtype: FP16 (2 bytes)

Block Size = 16 × 32 × 2 × 128 × 8 × 2 = 4,194,304 bytes = 4 MB per block
```

### Block Table

The block table maps logical block indices to physical block addresses:

```python
# Per-sequence block table
class BlockTable:
    def __init__(self, max_blocks: int):
        # Maps logical_block_idx → physical_block_ptr
        self.table: List[Optional[PhysicalBlock]] = [None] * max_blocks
        self.num_blocks: int = 0

    def allocate_block(self, block_allocator: BlockAllocator) -> int:
        """Allocate new physical block for next logical block"""
        physical_block = block_allocator.allocate()
        self.table[self.num_blocks] = physical_block
        self.num_blocks += 1
        return physical_block.block_id

    def get_physical_block(self, logical_idx: int) -> PhysicalBlock:
        """Get physical block for given logical position"""
        return self.table[logical_idx]
```

---

## Memory Management

### Block Allocator

The block allocator manages the pool of physical blocks:

```python
class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.free_blocks: Deque[int] = deque(range(num_blocks))
        self.num_blocks = num_blocks
        self.block_size = block_size

    def allocate(self) -> int:
        """Allocate a free block, return block_id"""
        if not self.free_blocks:
            raise OutOfMemoryError("No free blocks available")
        return self.free_blocks.popleft()

    def free(self, block_id: int):
        """Return block to free pool"""
        self.free_blocks.append(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
```

### Dynamic Allocation

Blocks are allocated **on-demand** as tokens are generated:

```
Token Generation Flow:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Prompt arrives (100 tokens)                              │
│         - Allocate ceil(100/16) = 7 blocks                       │
│         - Block table: [B0, B1, B2, B3, B4, B5, B6]             │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: Generate 50 tokens (total: 150)                          │
│         - Need ceil(150/16) = 10 blocks                          │
│         - Allocate 3 more blocks on-demand                       │
│         - Block table: [B0, B1, B2, B3, B4, B5, B6, B8, B9, B10]│
├─────────────────────────────────────────────────────────────────┤
│ Step 3: Sequence completes (200 tokens)                          │
│         - Free all 13 blocks back to pool                        │
│         - Ready for reuse by other sequences                     │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Sharing (Copy-on-Write)

PagedAttention enables efficient memory sharing for parallel sampling:

```
Parallel Sampling (beam search, best-of-n):
┌──────────────────────────────────────────────────────────────────┐
│ Shared Prompt: "Write a poem about..."                            │
│                                                                   │
│ Traditional: 4 copies of same prompt KV cache (4x memory)        │
│                                                                   │
│ PagedAttention: Share physical blocks until divergence            │
│                                                                   │
│ Before divergence:                                                │
│   Seq1.block_table → [B0, B1, B2] (ref_count=4)                  │
│   Seq2.block_table → [B0, B1, B2] (shared)                       │
│   Seq3.block_table → [B0, B1, B2] (shared)                       │
│   Seq4.block_table → [B0, B1, B2] (shared)                       │
│                                                                   │
│ After divergence (copy-on-write):                                 │
│   Seq1.block_table → [B0, B1, B2, B3]    (B3 new)                │
│   Seq2.block_table → [B0, B1, B2, B4]    (B4 new, B0-B2 shared)  │
│   Seq3.block_table → [B0, B1, B2, B5]    (B5 new, B0-B2 shared)  │
│   Seq4.block_table → [B0, B1, B2, B6]    (B6 new, B0-B2 shared)  │
└──────────────────────────────────────────────────────────────────┘

Memory Savings: Up to 55% for complex sampling algorithms
```

---

## Attention Kernel Modification

### Standard Attention

Traditional attention requires contiguous KV cache:

```python
def standard_attention(Q, K, V):
    """
    Q: [batch, seq_len, num_heads, head_dim]
    K: [batch, kv_len, num_kv_heads, head_dim]  # CONTIGUOUS
    V: [batch, kv_len, num_kv_heads, head_dim]  # CONTIGUOUS
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)
    weights = softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output
```

### PagedAttention Kernel

PagedAttention fetches blocks from non-contiguous memory:

```python
def paged_attention(Q, K_cache, V_cache, block_tables, context_lens):
    """
    Q: [batch, num_heads, head_dim]
    K_cache: [num_blocks, block_size, num_kv_heads, head_dim]  # Block pool
    V_cache: [num_blocks, block_size, num_kv_heads, head_dim]  # Block pool
    block_tables: [batch, max_blocks]  # Logical → Physical mapping
    context_lens: [batch]  # Actual sequence lengths
    """
    output = torch.zeros_like(Q)

    for seq_idx in range(batch):
        seq_len = context_lens[seq_idx]
        num_blocks = ceil(seq_len / block_size)

        # Fetch blocks for this sequence
        for block_idx in range(num_blocks):
            physical_block = block_tables[seq_idx, block_idx]

            # Get K, V from physical block
            K_block = K_cache[physical_block]  # [block_size, num_kv_heads, head_dim]
            V_block = V_cache[physical_block]

            # Compute attention for this block
            scores = torch.matmul(Q[seq_idx], K_block.T) / sqrt(head_dim)
            # ... accumulate with softmax normalization

    return output
```

### CUDA Kernel Implementation

vLLM's actual kernel (`csrc/attention/attention_kernels.cu`):

```cuda
// Simplified version of vLLM's paged attention kernel
template<typename T, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    T* __restrict__ out,              // [num_seqs, num_heads, head_size]
    const T* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const T* __restrict__ k_cache,    // [num_blocks, block_size, num_kv_heads, head_size]
    const T* __restrict__ v_cache,    // [num_blocks, block_size, num_kv_heads, head_size]
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks]
    const int* __restrict__ context_lens,  // [num_seqs]
    const float scale,
    const int max_num_blocks_per_seq
) {
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Each thread block handles one (sequence, head) pair
    // Threads cooperatively fetch blocks and compute attention

    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;

    // Phase 1: Compute QK^T and find max for softmax stability
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

        // Fetch K block from non-contiguous location
        const T* k_block = k_cache + physical_block * BLOCK_SIZE * head_size;

        // Compute dot product Q @ K^T
        for (int token_idx = threadIdx.x; token_idx < BLOCK_SIZE; token_idx += NUM_THREADS) {
            float qk = 0.0f;
            for (int d = 0; d < head_size; ++d) {
                qk += q[...] * k_block[...];
            }
            qk *= scale;
            qk_max = fmaxf(qk_max, qk);
            // Store qk for phase 2
        }
    }

    // Reduce qk_max across threads
    __syncthreads();
    // ...

    // Phase 2: Compute softmax and accumulate V
    // ...
}
```

---

## Implementation Strategy for VeloLLM

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     VeloLLM PagedAttention                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Rust Block Manager (velollm-core)                      │
│  ├── BlockAllocator: manages physical block pool                 │
│  ├── BlockTable: per-sequence logical→physical mapping          │
│  └── MemoryPool: GPU memory management                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: llama.cpp Integration                                  │
│  ├── Modified KV cache structures                                │
│  ├── Block-aware context management                              │
│  └── Paged attention in GGML (CPU) / CUDA (GPU)                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Kernel Implementation                                  │
│  ├── CPU: GGML-based paged attention                            │
│  └── GPU: CUDA kernel for NVIDIA                                │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Rust Block Manager (TASK-014)

```rust
// velollm-core/src/paged_attention/block_manager.rs

pub const DEFAULT_BLOCK_SIZE: usize = 16;  // tokens per block

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

pub struct BlockAllocator {
    num_blocks: usize,
    block_size: usize,
    free_blocks: VecDeque<BlockId>,
    ref_counts: HashMap<BlockId, usize>,
}

impl BlockAllocator {
    pub fn new(total_memory_bytes: usize, block_size: usize, kv_cache_bytes_per_token: usize) -> Self {
        let bytes_per_block = block_size * kv_cache_bytes_per_token;
        let num_blocks = total_memory_bytes / bytes_per_block;

        Self {
            num_blocks,
            block_size,
            free_blocks: (0..num_blocks).map(BlockId).collect(),
            ref_counts: HashMap::new(),
        }
    }

    pub fn allocate(&mut self) -> Option<BlockId> {
        let block_id = self.free_blocks.pop_front()?;
        self.ref_counts.insert(block_id, 1);
        Some(block_id)
    }

    pub fn free(&mut self, block_id: BlockId) {
        if let Some(count) = self.ref_counts.get_mut(&block_id) {
            *count -= 1;
            if *count == 0 {
                self.ref_counts.remove(&block_id);
                self.free_blocks.push_back(block_id);
            }
        }
    }

    pub fn add_ref(&mut self, block_id: BlockId) {
        if let Some(count) = self.ref_counts.get_mut(&block_id) {
            *count += 1;
        }
    }
}

pub struct SequenceBlockTable {
    block_ids: Vec<BlockId>,
    num_tokens: usize,
    block_size: usize,
}

impl SequenceBlockTable {
    pub fn append_tokens(&mut self, n_tokens: usize, allocator: &mut BlockAllocator) -> Result<(), Error> {
        let new_total = self.num_tokens + n_tokens;
        let blocks_needed = (new_total + self.block_size - 1) / self.block_size;

        while self.block_ids.len() < blocks_needed {
            let block = allocator.allocate()
                .ok_or(Error::OutOfMemory)?;
            self.block_ids.push(block);
        }

        self.num_tokens = new_total;
        Ok(())
    }
}
```

#### Phase 2: llama.cpp Modifications

Key files to modify in llama.cpp:

| File | Modification |
|------|--------------|
| `llama.h` | Add paged KV cache structures |
| `llama.cpp` | Modify `llama_kv_cache_*` functions |
| `ggml.c` | Add paged attention operator |
| `ggml-cuda.cu` | CUDA kernel for paged attention |

```cpp
// llama.h additions
struct llama_kv_block {
    int block_id;
    int ref_count;
    struct ggml_tensor* k;  // [block_size, n_embd_k]
    struct ggml_tensor* v;  // [block_size, n_embd_v]
};

struct llama_kv_cache_paged {
    std::vector<llama_kv_block> blocks;
    std::vector<int> free_blocks;
    std::unordered_map<int, std::vector<int>> sequence_block_tables;
    int block_size;
};
```

#### Phase 3: CUDA Kernel

```cuda
// Simplified paged attention kernel for VeloLLM
__global__ void velollm_paged_attention_kernel(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* seq_lens,
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale
) {
    // Implementation similar to vLLM
    // ...
}
```

---

## Challenges for llama.cpp Integration

### 1. Memory Layout Differences

| Aspect | vLLM (PyTorch) | llama.cpp (GGML) |
|--------|----------------|------------------|
| Tensor format | Strided, column-major | Row-major, packed |
| Memory allocation | CUDA malloc | Custom allocator |
| Data types | FP16, BF16 | GGUF quantized (Q4, Q8) |

**Solution**: Create adapter layer that translates between formats.

### 2. Quantization Compatibility

llama.cpp uses quantized KV cache (Q4_K, Q8_0), not FP16:

```
Quantized Block Challenge:
- Q4_K: 4.5 bits per value (grouped quantization)
- Block boundaries must align with quantization groups
- May need to adjust block_size to be multiple of group_size (32)
```

**Solution**: Set `block_size = 32` or `64` to align with Q4_K groups.

### 3. CPU Fallback

vLLM is GPU-only; llama.cpp needs CPU support:

```cpp
// CPU paged attention (GGML)
void ggml_compute_paged_attention_f32(
    struct ggml_tensor* output,
    const struct ggml_tensor* q,
    const struct ggml_tensor* k_cache,  // [num_blocks, block_size, n_embd]
    const struct ggml_tensor* v_cache,
    const int* block_table,
    int context_len
) {
    // Implement in pure C with SIMD optimizations
    // Use OpenMP for parallelization
}
```

### 4. Integration with Existing API

llama.cpp API must remain backward-compatible:

```cpp
// Existing API (keep working)
llama_decode(ctx, batch);

// New paged API (opt-in)
llama_decode_paged(ctx, batch, &paged_kv_cache);
```

---

## Performance Expectations

### Memory Savings

| Scenario | Traditional | PagedAttention | Savings |
|----------|-------------|----------------|---------|
| Single 4K context | 4K × KV_size | 4K × KV_size | 0% |
| 10 concurrent 1K contexts | 10 × 4K × KV_size | 10 × 1K × KV_size | 75% |
| Variable length (avg 500) | 8 × 4K × KV_size | 8 × 500 × KV_size | 87.5% |

### Throughput Improvement

With better memory utilization:
- **More concurrent requests**: 2-4x more sequences in same VRAM
- **Higher batch sizes**: Better GPU utilization
- **Less OOM**: Graceful handling of memory pressure

Expected on RTX 4070 Ti SUPER (16GB):

| Metric | Current (Ollama) | With PagedAttention |
|--------|------------------|---------------------|
| Max concurrent 8B models | 1 | 2-3 |
| Max context per model | 4K | 8K-16K |
| Throughput (multi-user) | 137 tok/s | 200-300 tok/s |

---

## Implementation Checklist

### TASK-013: Research (This Document) ✅
- [x] Understand PagedAttention concept
- [x] Analyze vLLM implementation
- [x] Identify llama.cpp integration challenges
- [x] Document implementation strategy

### TASK-014: Block Manager (Next)
- [ ] Implement `BlockAllocator` in Rust
- [ ] Implement `SequenceBlockTable`
- [ ] Add reference counting for memory sharing
- [ ] Unit tests for allocation/deallocation
- [ ] Benchmarks for allocation performance

### TASK-015: llama.cpp Integration
- [ ] Fork llama.cpp
- [ ] Add paged KV cache structures
- [ ] Modify context management
- [ ] CPU fallback implementation

### TASK-016: CUDA Kernel
- [ ] Implement paged attention kernel
- [ ] Optimize for different GPU architectures
- [ ] Benchmark vs standard attention

---

## References

### Papers
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) - Original paper
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [vAttention](https://www.microsoft.com/en-us/research/publication/vattention/) - Microsoft's alternative

### Implementations
- [vLLM](https://github.com/vllm-project/vllm) - Reference implementation
- [vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html) - Official introduction
- [vLLM Kernel Docs](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)

### Related Projects
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Target integration
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's paged attention

---

**Status**: ✅ Analysis Complete
**Next Task**: TASK-014 - Block Manager Implementation
**Estimated Time**: 8 hours
