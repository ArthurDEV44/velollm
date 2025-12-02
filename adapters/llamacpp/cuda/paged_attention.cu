// VeloLLM Paged Attention CUDA Kernel Implementation
// SPDX-License-Identifier: MIT
//
// Implements paged attention for efficient KV cache memory management.
// Based on vLLM's PagedAttention with adaptations for llama.cpp integration.
//
// Key features:
// - Non-contiguous KV cache access via block tables
// - Two-pass softmax for numerical stability
// - Support for GQA (Grouped Query Attention)
// - FP16 and FP32 support

#include "paged_attention.cuh"
#include <float.h>
#include <math.h>

// ============================================================================
// Utility functions and macros
// ============================================================================

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            return VELOLLM_ERROR_CUDA_LAUNCH;                                  \
        }                                                                      \
    } while (0)

// Warp reduce sum
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int mask = VELOLLM_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp reduce max
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int mask = VELOLLM_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Block reduce sum using shared memory
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared_mem) {
    const int lane = threadIdx.x % VELOLLM_WARP_SIZE;
    const int warp_id = threadIdx.x / VELOLLM_WARP_SIZE;
    const int num_warps = BLOCK_SIZE / VELOLLM_WARP_SIZE;

    // First, reduce within warp
    val = warp_reduce_sum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : T(0);
        val = warp_reduce_sum(val);
    }

    return val;
}

// Block reduce max using shared memory
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_max(T val, T* shared_mem) {
    const int lane = threadIdx.x % VELOLLM_WARP_SIZE;
    const int warp_id = threadIdx.x / VELOLLM_WARP_SIZE;
    const int num_warps = BLOCK_SIZE / VELOLLM_WARP_SIZE;

    // First, reduce within warp
    val = warp_reduce_max(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }

    return val;
}

// ============================================================================
// Paged Attention Kernel - Single Query (Decoding Phase)
// ============================================================================

// This kernel handles the generation phase where we have a single query token
// attending to all previous KV cache tokens stored in paged blocks.
//
// Thread block organization:
// - Each thread block handles one (sequence, head) pair
// - Threads cooperate to fetch blocks and compute attention
//
// Memory layout:
// - Query: [num_seqs, num_heads, head_dim]
// - Key cache: [num_blocks, block_size, num_kv_heads, head_dim]
// - Value cache: [num_blocks, block_size, num_kv_heads, head_dim]
// - Block tables: [num_seqs, max_blocks_per_seq]

template <typename T, int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_kernel(
    T* __restrict__ output,              // [num_seqs, num_heads, head_dim]
    const T* __restrict__ query,         // [num_seqs, num_heads, head_dim]
    const T* __restrict__ key_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const T* __restrict__ value_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int32_t* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ context_lens,   // [num_seqs]
    const float scale,
    const int num_kv_heads,
    const int max_blocks_per_seq
) {
    // Block and thread indices
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int lane = thread_idx % VELOLLM_WARP_SIZE;
    const int warp_id = thread_idx / VELOLLM_WARP_SIZE;
    const int num_warps = NUM_THREADS / VELOLLM_WARP_SIZE;

    // GQA: map query head to KV head
    const int num_heads = gridDim.x;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;

    // Get context length for this sequence
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) {
        return;
    }

    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for reductions and intermediate results
    extern __shared__ char smem[];
    float* shared_qk = reinterpret_cast<float*>(smem);                          // [NUM_THREADS]
    float* shared_reduce = shared_qk + NUM_THREADS;                              // [NUM_THREADS / WARP_SIZE]
    T* shared_query = reinterpret_cast<T*>(shared_reduce + num_warps);          // [HEAD_DIM]
    float* shared_output = reinterpret_cast<float*>(shared_query + HEAD_DIM);   // [HEAD_DIM]

    // Load query into shared memory
    const T* q_ptr = query + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        shared_query[d] = q_ptr[d];
    }
    __syncthreads();

    // Initialize output accumulator
    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        shared_output[d] = 0.0f;
    }

    // Phase 1: Compute QK^T scores and find max for softmax stability
    float qk_max = -FLT_MAX;

    // Each warp processes tokens in parallel
    // Total tokens to process = context_len
    // Tokens per block = BLOCK_SIZE
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Get physical block index from block table
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        // Process tokens within this block
        const int block_start = block_idx * BLOCK_SIZE;
        const int block_end = min(block_start + BLOCK_SIZE, context_len);
        const int tokens_in_block = block_end - block_start;

        // Each thread computes one or more token scores
        for (int token_offset = thread_idx; token_offset < tokens_in_block; token_offset += NUM_THREADS) {
            const int token_idx = block_start + token_offset;

            // Get key for this token
            // Key cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
            const T* k_ptr = key_cache +
                             physical_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM +
                             token_offset * num_kv_heads * HEAD_DIM +
                             kv_head_idx * HEAD_DIM;

            // Compute dot product Q @ K^T
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                qk += float(shared_query[d]) * float(k_ptr[d]);
            }
            qk *= scale;

            // Store score and update max
            shared_qk[token_idx] = qk;
            qk_max = fmaxf(qk_max, qk);
        }
    }
    __syncthreads();

    // Reduce max across all threads
    qk_max = block_reduce_max<float, NUM_THREADS>(qk_max, shared_reduce);
    __syncthreads();

    // Broadcast max to all threads
    if (thread_idx == 0) {
        shared_reduce[0] = qk_max;
    }
    __syncthreads();
    qk_max = shared_reduce[0];

    // Phase 2: Compute softmax(QK^T) @ V
    float exp_sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        const int block_start = block_idx * BLOCK_SIZE;
        const int block_end = min(block_start + BLOCK_SIZE, context_len);
        const int tokens_in_block = block_end - block_start;

        for (int token_offset = thread_idx; token_offset < tokens_in_block; token_offset += NUM_THREADS) {
            const int token_idx = block_start + token_offset;

            // Compute softmax weight
            float qk = shared_qk[token_idx];
            float weight = expf(qk - qk_max);
            exp_sum += weight;

            // Get value for this token
            // Value cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
            const T* v_ptr = value_cache +
                             physical_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM +
                             token_offset * num_kv_heads * HEAD_DIM +
                             kv_head_idx * HEAD_DIM;

            // Accumulate weighted value
            // Note: This is a simplified version. Production code would use
            // more sophisticated accumulation to reduce atomic contention.
            for (int d = 0; d < HEAD_DIM; ++d) {
                atomicAdd(&shared_output[d], weight * float(v_ptr[d]));
            }
        }
    }
    __syncthreads();

    // Reduce exp_sum across threads
    exp_sum = block_reduce_sum<float, NUM_THREADS>(exp_sum, shared_reduce);
    __syncthreads();

    if (thread_idx == 0) {
        shared_reduce[0] = exp_sum;
    }
    __syncthreads();
    exp_sum = shared_reduce[0];

    // Phase 3: Normalize and write output
    T* out_ptr = output + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    const float inv_sum = 1.0f / (exp_sum + 1e-8f);

    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        out_ptr[d] = T(shared_output[d] * inv_sum);
    }
}

// ============================================================================
// Optimized Paged Attention Kernel with better memory access patterns
// ============================================================================

// This version uses a different strategy:
// - Each warp handles one block of tokens
// - Better coalesced memory access
// - Reduced shared memory usage

template <typename T, int HEAD_DIM, int BLOCK_SIZE, int NUM_WARPS>
__global__ void paged_attention_kernel_v2(
    T* __restrict__ output,              // [num_seqs, num_heads, head_dim]
    const T* __restrict__ query,         // [num_seqs, num_heads, head_dim]
    const T* __restrict__ key_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const T* __restrict__ value_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int32_t* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ context_lens,   // [num_seqs]
    const float scale,
    const int num_kv_heads,
    const int max_blocks_per_seq
) {
    const int NUM_THREADS = NUM_WARPS * VELOLLM_WARP_SIZE;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / VELOLLM_WARP_SIZE;
    const int lane = thread_idx % VELOLLM_WARP_SIZE;

    // GQA mapping
    const int num_heads = gridDim.x;
    const int kv_head_idx = head_idx * num_kv_heads / num_heads;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) {
        return;
    }

    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory layout
    extern __shared__ char smem[];
    T* shared_query = reinterpret_cast<T*>(smem);                      // [HEAD_DIM]
    float* shared_max = reinterpret_cast<float*>(shared_query + HEAD_DIM);   // [NUM_WARPS]
    float* shared_sum = shared_max + NUM_WARPS;                        // [NUM_WARPS]
    float* shared_out = shared_sum + NUM_WARPS;                        // [HEAD_DIM]

    // Load query into shared memory (coalesced)
    const T* q_ptr = query + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        shared_query[d] = q_ptr[d];
    }

    // Initialize output accumulator
    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        shared_out[d] = 0.0f;
    }
    __syncthreads();

    // Each warp computes partial results for a subset of blocks
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_out[HEAD_DIM / VELOLLM_WARP_SIZE + 1];  // Per-thread partial output

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / VELOLLM_WARP_SIZE + 1; ++d) {
        local_out[d] = 0.0f;
    }

    // Process blocks assigned to this warp
    for (int block_idx = warp_id; block_idx < num_blocks; block_idx += NUM_WARPS) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        const int block_start = block_idx * BLOCK_SIZE;
        const int block_end = min(block_start + BLOCK_SIZE, context_len);

        // Each lane processes one token in the block
        for (int token_offset = lane; token_offset < BLOCK_SIZE && (block_start + token_offset) < context_len;
             token_offset += VELOLLM_WARP_SIZE) {

            const T* k_ptr = key_cache +
                             physical_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM +
                             token_offset * num_kv_heads * HEAD_DIM +
                             kv_head_idx * HEAD_DIM;

            // Compute QK^T
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                qk += float(shared_query[d]) * float(k_ptr[d]);
            }
            qk *= scale;
            local_max = fmaxf(local_max, qk);
        }
    }

    // Reduce max within warp
    local_max = warp_reduce_max(local_max);

    // Store warp max to shared memory
    if (lane == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();

    // Find global max
    float global_max = -FLT_MAX;
    for (int w = 0; w < NUM_WARPS; ++w) {
        global_max = fmaxf(global_max, shared_max[w]);
    }
    __syncthreads();

    // Second pass: compute softmax weights and accumulate values
    for (int block_idx = warp_id; block_idx < num_blocks; block_idx += NUM_WARPS) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        const int block_start = block_idx * BLOCK_SIZE;

        for (int token_offset = lane; token_offset < BLOCK_SIZE && (block_start + token_offset) < context_len;
             token_offset += VELOLLM_WARP_SIZE) {

            const T* k_ptr = key_cache +
                             physical_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM +
                             token_offset * num_kv_heads * HEAD_DIM +
                             kv_head_idx * HEAD_DIM;

            const T* v_ptr = value_cache +
                             physical_block * BLOCK_SIZE * num_kv_heads * HEAD_DIM +
                             token_offset * num_kv_heads * HEAD_DIM +
                             kv_head_idx * HEAD_DIM;

            // Recompute QK (could cache this, but saves shared memory)
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                qk += float(shared_query[d]) * float(k_ptr[d]);
            }
            qk *= scale;

            float weight = expf(qk - global_max);
            local_sum += weight;

            // Accumulate weighted values
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += VELOLLM_WARP_SIZE) {
                if (d + lane < HEAD_DIM) {
                    local_out[d / VELOLLM_WARP_SIZE] += weight * float(v_ptr[d + lane]);
                }
            }
        }
    }

    // Reduce sum within warp
    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Reduce partial outputs across warps
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d += VELOLLM_WARP_SIZE) {
        if (d + lane < HEAD_DIM) {
            atomicAdd(&shared_out[d + lane], local_out[d / VELOLLM_WARP_SIZE]);
        }
    }
    __syncthreads();

    // Compute global sum
    float global_sum = 0.0f;
    for (int w = 0; w < NUM_WARPS; ++w) {
        global_sum += shared_sum[w];
    }

    // Normalize and write output
    T* out_ptr = output + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    const float inv_sum = 1.0f / (global_sum + 1e-8f);

    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        out_ptr[d] = T(shared_out[d] * inv_sum);
    }
}

// ============================================================================
// Kernel dispatch and C interface
// ============================================================================

template <typename T, int HEAD_DIM>
VeloLLMError launch_paged_attention_kernel(const PagedAttentionParams* params) {
    const int num_seqs = params->num_seqs;
    const int num_heads = params->num_heads;
    const int block_size = params->block_size;

    // Grid: (num_heads, num_seqs)
    dim3 grid(num_heads, num_seqs);

    // Thread block configuration
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / VELOLLM_WARP_SIZE;

    // Calculate shared memory size
    size_t smem_size =
        HEAD_DIM * sizeof(T) +              // shared_query
        NUM_WARPS * sizeof(float) +         // shared_max
        NUM_WARPS * sizeof(float) +         // shared_sum
        HEAD_DIM * sizeof(float);           // shared_out

    // Dispatch based on block size
    if (block_size == 16) {
        paged_attention_kernel_v2<T, HEAD_DIM, 16, NUM_WARPS><<<grid, NUM_THREADS, smem_size, params->stream>>>(
            static_cast<T*>(params->output),
            static_cast<const T*>(params->query),
            static_cast<const T*>(params->key_cache),
            static_cast<const T*>(params->value_cache),
            params->block_tables,
            params->context_lens,
            params->scale,
            params->num_kv_heads,
            params->max_blocks_per_seq
        );
    } else if (block_size == 32) {
        paged_attention_kernel_v2<T, HEAD_DIM, 32, NUM_WARPS><<<grid, NUM_THREADS, smem_size, params->stream>>>(
            static_cast<T*>(params->output),
            static_cast<const T*>(params->query),
            static_cast<const T*>(params->key_cache),
            static_cast<const T*>(params->value_cache),
            params->block_tables,
            params->context_lens,
            params->scale,
            params->num_kv_heads,
            params->max_blocks_per_seq
        );
    } else {
        return VELOLLM_ERROR_UNSUPPORTED;
    }

    CUDA_CHECK(cudaGetLastError());
    return VELOLLM_SUCCESS;
}

template <typename T>
VeloLLMError dispatch_head_dim(const PagedAttentionParams* params) {
    switch (params->head_dim) {
        case 64:
            return launch_paged_attention_kernel<T, 64>(params);
        case 80:
            return launch_paged_attention_kernel<T, 80>(params);
        case 96:
            return launch_paged_attention_kernel<T, 96>(params);
        case 128:
            return launch_paged_attention_kernel<T, 128>(params);
        case 256:
            return launch_paged_attention_kernel<T, 256>(params);
        default:
            return VELOLLM_ERROR_UNSUPPORTED;
    }
}

extern "C" {

VeloLLMError velollm_paged_attention_forward(const PagedAttentionParams* params) {
    if (params == nullptr ||
        params->output == nullptr ||
        params->query == nullptr ||
        params->key_cache == nullptr ||
        params->value_cache == nullptr ||
        params->block_tables == nullptr ||
        params->context_lens == nullptr) {
        return VELOLLM_ERROR_INVALID_ARGUMENT;
    }

    if (params->num_seqs <= 0 ||
        params->num_heads <= 0 ||
        params->num_kv_heads <= 0 ||
        params->head_dim <= 0 ||
        params->block_size <= 0) {
        return VELOLLM_ERROR_INVALID_ARGUMENT;
    }

    // Check GQA compatibility
    if (params->num_heads % params->num_kv_heads != 0) {
        return VELOLLM_ERROR_INVALID_ARGUMENT;
    }

    switch (params->dtype) {
        case VELOLLM_DTYPE_FP32:
            return dispatch_head_dim<float>(params);
        case VELOLLM_DTYPE_FP16:
            return dispatch_head_dim<half>(params);
        default:
            return VELOLLM_ERROR_UNSUPPORTED;
    }
}

const char* velollm_get_error_string(VeloLLMError error) {
    switch (error) {
        case VELOLLM_SUCCESS:
            return "Success";
        case VELOLLM_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case VELOLLM_ERROR_CUDA_LAUNCH:
            return "CUDA kernel launch failed";
        case VELOLLM_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case VELOLLM_ERROR_UNSUPPORTED:
            return "Unsupported configuration";
        default:
            return "Unknown error";
    }
}

int velollm_paged_attention_supported(
    int head_dim,
    int block_size,
    VeloLLMDType dtype
) {
    // Check supported head dimensions
    if (head_dim != 64 && head_dim != 80 && head_dim != 96 &&
        head_dim != 128 && head_dim != 256) {
        return 0;
    }

    // Check supported block sizes
    if (block_size != 16 && block_size != 32) {
        return 0;
    }

    // Check supported data types
    if (dtype != VELOLLM_DTYPE_FP32 && dtype != VELOLLM_DTYPE_FP16) {
        return 0;
    }

    return 1;
}

int velollm_get_recommended_block_size(int head_dim) {
    // For most head dimensions, 16 is optimal
    // Matches default BlockManager configuration
    (void)head_dim;
    return 16;
}

} // extern "C"
