// VeloLLM Paged Attention CUDA Kernel Header
// SPDX-License-Identifier: MIT
//
// This header defines the interface for paged attention CUDA kernels.
// Designed to integrate with VeloLLM's BlockManager for efficient KV cache management.

#ifndef VELOLLM_PAGED_ATTENTION_CUH
#define VELOLLM_PAGED_ATTENTION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define VELOLLM_WARP_SIZE 32
#define VELOLLM_DEFAULT_BLOCK_SIZE 16  // Tokens per KV cache block
#define VELOLLM_MAX_HEAD_DIM 256

// Error codes
typedef enum {
    VELOLLM_SUCCESS = 0,
    VELOLLM_ERROR_INVALID_ARGUMENT = 1,
    VELOLLM_ERROR_CUDA_LAUNCH = 2,
    VELOLLM_ERROR_OUT_OF_MEMORY = 3,
    VELOLLM_ERROR_UNSUPPORTED = 4,
} VeloLLMError;

// Data type enum for kernel dispatch
typedef enum {
    VELOLLM_DTYPE_FP32 = 0,
    VELOLLM_DTYPE_FP16 = 1,
} VeloLLMDType;

// Paged attention parameters
typedef struct {
    // Output tensor [num_seqs, num_heads, head_dim]
    void* output;

    // Query tensor [num_seqs, num_heads, head_dim]
    const void* query;

    // Key cache [num_blocks, block_size, num_kv_heads, head_dim]
    const void* key_cache;

    // Value cache [num_blocks, block_size, num_kv_heads, head_dim]
    const void* value_cache;

    // Block tables [num_seqs, max_blocks_per_seq]
    // Maps logical block index to physical block index
    const int32_t* block_tables;

    // Context lengths [num_seqs]
    // Actual sequence length for each request
    const int32_t* context_lens;

    // Dimensions
    int32_t num_seqs;           // Number of sequences in batch
    int32_t num_heads;          // Number of query heads
    int32_t num_kv_heads;       // Number of KV heads (may differ for GQA)
    int32_t head_dim;           // Dimension per head
    int32_t block_size;         // Tokens per block (default: 16)
    int32_t max_context_len;    // Maximum context length
    int32_t max_blocks_per_seq; // Maximum blocks per sequence

    // Attention scale (typically 1/sqrt(head_dim))
    float scale;

    // Data type
    VeloLLMDType dtype;

    // CUDA stream
    cudaStream_t stream;
} PagedAttentionParams;

// Launch paged attention kernel
// Returns VELOLLM_SUCCESS on success, error code otherwise
VeloLLMError velollm_paged_attention_forward(const PagedAttentionParams* params);

// Get error string for error code
const char* velollm_get_error_string(VeloLLMError error);

// Check if paged attention is supported for given configuration
// Returns 1 if supported, 0 otherwise
int velollm_paged_attention_supported(
    int head_dim,
    int block_size,
    VeloLLMDType dtype
);

// Utility: Get recommended block size for given head dimension
int velollm_get_recommended_block_size(int head_dim);

#ifdef __cplusplus
}
#endif

#endif // VELOLLM_PAGED_ATTENTION_CUH
