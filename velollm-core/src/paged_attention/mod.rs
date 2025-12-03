//! PagedAttention Memory Management
//!
//! This module implements PagedAttention-style memory management for KV cache,
//! inspired by vLLM's approach. Instead of allocating contiguous memory per sequence,
//! we divide the KV cache into fixed-size blocks that can be allocated dynamically.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      BlockManager                                │
//! │  ┌─────────────────┐  ┌──────────────────────────────────────┐  │
//! │  │  BlockAllocator │  │     SequenceBlockTables              │  │
//! │  │  ┌─────────────┐│  │  ┌────────┐ ┌────────┐ ┌────────┐   │  │
//! │  │  │ Free Blocks ││  │  │ Seq 1  │ │ Seq 2  │ │ Seq 3  │   │  │
//! │  │  │ [B5,B6,B7]  ││  │  │[B0,B2] │ │[B1,B3] │ │[B4]    │   │  │
//! │  │  └─────────────┘│  │  └────────┘ └────────┘ └────────┘   │  │
//! │  └─────────────────┘  └──────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Benefits
//!
//! - **Memory Efficiency**: <4% waste vs 60-80% with traditional allocation
//! - **Dynamic Allocation**: Blocks allocated on-demand as tokens are generated
//! - **Memory Sharing**: Copy-on-write for parallel sampling (beam search, etc.)
//!
//! # Example
//!
//! ```rust
//! use velollm_core::paged_attention::{BlockManager, BlockManagerConfig};
//!
//! // Create a block manager with 1GB of memory, 16 tokens per block
//! let config = BlockManagerConfig {
//!     total_memory_bytes: 1024 * 1024 * 1024,  // 1 GB
//!     block_size: 16,
//!     kv_head_dim: 128,
//!     num_kv_heads: 8,
//!     num_layers: 32,
//!     dtype_bytes: 2,  // FP16
//! };
//!
//! let mut manager = BlockManager::new(config);
//!
//! // Allocate blocks for a new sequence
//! let seq_id = manager.create_sequence().unwrap();
//!
//! // Add tokens (blocks allocated automatically)
//! manager.append_tokens(seq_id, 100).unwrap();
//!
//! // Get block table for attention kernel
//! let block_table = manager.get_block_table(seq_id).unwrap();
//! ```

mod block_allocator;
mod block_table;

pub use block_allocator::{BlockAllocator, BlockId};
pub use block_table::SequenceBlockTable;

use std::collections::HashMap;
use thiserror::Error;

/// Default number of tokens per block
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Errors that can occur during paged attention operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PagedAttentionError {
    #[error("out of memory: no free blocks available")]
    OutOfMemory,

    #[error("sequence {0} not found")]
    SequenceNotFound(u64),

    #[error("block {0} not found")]
    BlockNotFound(usize),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Configuration for the BlockManager
#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    /// Total memory available for KV cache (in bytes)
    pub total_memory_bytes: usize,

    /// Number of tokens per block
    pub block_size: usize,

    /// Dimension of each attention head
    pub kv_head_dim: usize,

    /// Number of KV heads (may differ from query heads in GQA)
    pub num_kv_heads: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Bytes per value (2 for FP16, 4 for FP32)
    pub dtype_bytes: usize,
}

impl BlockManagerConfig {
    /// Calculate bytes needed for KV cache per token
    pub fn bytes_per_token(&self) -> usize {
        // K + V for each layer
        2 * self.num_layers * self.num_kv_heads * self.kv_head_dim * self.dtype_bytes
    }

    /// Calculate bytes per block
    pub fn bytes_per_block(&self) -> usize {
        self.block_size * self.bytes_per_token()
    }

    /// Calculate total number of blocks that fit in memory
    pub fn num_blocks(&self) -> usize {
        self.total_memory_bytes / self.bytes_per_block()
    }

    /// Create config for a specific model
    pub fn for_model(
        total_memory_bytes: usize,
        num_layers: usize,
        num_kv_heads: usize,
        kv_head_dim: usize,
    ) -> Self {
        Self {
            total_memory_bytes,
            block_size: DEFAULT_BLOCK_SIZE,
            kv_head_dim,
            num_kv_heads,
            num_layers,
            dtype_bytes: 2, // FP16 by default
        }
    }

    /// Create config for Llama 3.1 8B
    pub fn llama_8b(total_memory_bytes: usize) -> Self {
        Self::for_model(
            total_memory_bytes,
            32,  // layers
            8,   // GQA: 8 KV heads
            128, // head_dim
        )
    }

    /// Create config for Llama 3.2 3B
    pub fn llama_3b(total_memory_bytes: usize) -> Self {
        Self::for_model(
            total_memory_bytes,
            28,  // layers
            8,   // GQA: 8 KV heads
            128, // head_dim
        )
    }
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        // Default: 1GB, Llama-like model
        Self::llama_8b(1024 * 1024 * 1024)
    }
}

/// Main orchestrator for paged attention memory management
///
/// Manages multiple sequences, each with their own block table,
/// and coordinates block allocation/deallocation.
pub struct BlockManager {
    /// Configuration
    config: BlockManagerConfig,

    /// Block allocator (manages physical blocks)
    allocator: BlockAllocator,

    /// Block tables for each sequence
    sequences: HashMap<u64, SequenceBlockTable>,

    /// Next sequence ID to assign
    next_seq_id: u64,
}

impl BlockManager {
    /// Create a new BlockManager with the given configuration
    pub fn new(config: BlockManagerConfig) -> Self {
        let num_blocks = config.num_blocks();
        let allocator = BlockAllocator::new(num_blocks);

        Self { config, allocator, sequences: HashMap::new(), next_seq_id: 0 }
    }

    /// Create a new sequence and return its ID
    pub fn create_sequence(&mut self) -> Result<u64, PagedAttentionError> {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let block_table = SequenceBlockTable::new(self.config.block_size);
        self.sequences.insert(seq_id, block_table);

        Ok(seq_id)
    }

    /// Remove a sequence and free its blocks
    pub fn remove_sequence(&mut self, seq_id: u64) -> Result<(), PagedAttentionError> {
        let block_table = self
            .sequences
            .remove(&seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound(seq_id))?;

        // Free all blocks
        for block_id in block_table.block_ids() {
            self.allocator.free(*block_id);
        }

        Ok(())
    }

    /// Append tokens to a sequence, allocating blocks as needed
    pub fn append_tokens(
        &mut self,
        seq_id: u64,
        num_tokens: usize,
    ) -> Result<(), PagedAttentionError> {
        let block_table = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound(seq_id))?;

        block_table.append_tokens(num_tokens, &mut self.allocator)
    }

    /// Get the block table for a sequence (for passing to attention kernel)
    pub fn get_block_table(&self, seq_id: u64) -> Result<&[BlockId], PagedAttentionError> {
        let block_table = self
            .sequences
            .get(&seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound(seq_id))?;

        Ok(block_table.block_ids())
    }

    /// Get the number of tokens in a sequence
    pub fn get_sequence_length(&self, seq_id: u64) -> Result<usize, PagedAttentionError> {
        let block_table = self
            .sequences
            .get(&seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound(seq_id))?;

        Ok(block_table.num_tokens())
    }

    /// Fork a sequence (for parallel sampling)
    /// Creates a new sequence that shares blocks with the parent (copy-on-write)
    pub fn fork_sequence(&mut self, parent_seq_id: u64) -> Result<u64, PagedAttentionError> {
        let parent_table = self
            .sequences
            .get(&parent_seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound(parent_seq_id))?;

        // Clone the block table (shallow copy - blocks are shared)
        let child_table = parent_table.fork(&mut self.allocator);

        let child_seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        self.sequences.insert(child_seq_id, child_table);

        Ok(child_seq_id)
    }

    /// Get memory utilization (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        self.allocator.utilization()
    }

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free_blocks()
    }

    /// Get total number of blocks
    pub fn num_total_blocks(&self) -> usize {
        self.allocator.num_total_blocks()
    }

    /// Get number of active sequences
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get the configuration
    pub fn config(&self) -> &BlockManagerConfig {
        &self.config
    }

    /// Check if there's enough memory for a given number of tokens
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        let blocks_needed = num_tokens.div_ceil(self.config.block_size);
        self.allocator.num_free_blocks() >= blocks_needed
    }

    /// Get statistics about current memory usage
    pub fn stats(&self) -> BlockManagerStats {
        let total_tokens: usize = self.sequences.values().map(|s| s.num_tokens()).sum();

        BlockManagerStats {
            total_blocks: self.allocator.num_total_blocks(),
            used_blocks: self.allocator.num_total_blocks() - self.allocator.num_free_blocks(),
            free_blocks: self.allocator.num_free_blocks(),
            num_sequences: self.sequences.len(),
            total_tokens,
            memory_utilization: self.allocator.utilization(),
            bytes_per_block: self.config.bytes_per_block(),
        }
    }
}

/// Statistics about BlockManager memory usage
#[derive(Debug, Clone)]
pub struct BlockManagerStats {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub free_blocks: usize,
    pub num_sequences: usize,
    pub total_tokens: usize,
    pub memory_utilization: f64,
    pub bytes_per_block: usize,
}

impl std::fmt::Display for BlockManagerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BlockManager: {}/{} blocks used ({:.1}%), {} sequences, {} tokens",
            self.used_blocks,
            self.total_blocks,
            self.memory_utilization * 100.0,
            self.num_sequences,
            self.total_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BlockManagerConfig {
        BlockManagerConfig {
            total_memory_bytes: 1024 * 1024, // 1 MB
            block_size: 16,
            kv_head_dim: 64,
            num_kv_heads: 4,
            num_layers: 8,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn test_config_calculations() {
        let config = test_config();

        // bytes_per_token = 2 * 8 * 4 * 64 * 2 = 8192 bytes
        assert_eq!(config.bytes_per_token(), 8192);

        // bytes_per_block = 16 * 8192 = 131072 bytes
        assert_eq!(config.bytes_per_block(), 131072);

        // num_blocks = 1MB / 131072 = 8 blocks
        assert_eq!(config.num_blocks(), 8);
    }

    #[test]
    fn test_create_sequence() {
        let config = test_config();
        let mut manager = BlockManager::new(config);

        let seq_id = manager.create_sequence().unwrap();
        assert_eq!(seq_id, 0);
        assert_eq!(manager.num_sequences(), 1);

        let seq_id2 = manager.create_sequence().unwrap();
        assert_eq!(seq_id2, 1);
        assert_eq!(manager.num_sequences(), 2);
    }

    #[test]
    fn test_append_tokens() {
        let config = test_config();
        let mut manager = BlockManager::new(config);

        let seq_id = manager.create_sequence().unwrap();

        // Append 10 tokens (fits in 1 block of 16)
        manager.append_tokens(seq_id, 10).unwrap();
        assert_eq!(manager.get_sequence_length(seq_id).unwrap(), 10);

        let block_table = manager.get_block_table(seq_id).unwrap();
        assert_eq!(block_table.len(), 1);

        // Append 10 more tokens (now need 2 blocks)
        manager.append_tokens(seq_id, 10).unwrap();
        assert_eq!(manager.get_sequence_length(seq_id).unwrap(), 20);

        let block_table = manager.get_block_table(seq_id).unwrap();
        assert_eq!(block_table.len(), 2);
    }

    #[test]
    fn test_remove_sequence() {
        let config = test_config();
        let mut manager = BlockManager::new(config);

        let seq_id = manager.create_sequence().unwrap();
        manager.append_tokens(seq_id, 32).unwrap(); // 2 blocks

        let free_before = manager.num_free_blocks();
        manager.remove_sequence(seq_id).unwrap();
        let free_after = manager.num_free_blocks();

        assert_eq!(free_after - free_before, 2);
        assert_eq!(manager.num_sequences(), 0);
    }

    #[test]
    fn test_out_of_memory() {
        let config = test_config(); // 8 blocks available
        let mut manager = BlockManager::new(config);

        let seq_id = manager.create_sequence().unwrap();

        // Try to allocate more than available (8 blocks * 16 tokens = 128 tokens max)
        let result = manager.append_tokens(seq_id, 200);
        assert!(matches!(result, Err(PagedAttentionError::OutOfMemory)));
    }

    #[test]
    fn test_fork_sequence() {
        let config = test_config();
        let mut manager = BlockManager::new(config);

        let parent_id = manager.create_sequence().unwrap();
        manager.append_tokens(parent_id, 32).unwrap();

        let child_id = manager.fork_sequence(parent_id).unwrap();

        // Both should have same length
        assert_eq!(
            manager.get_sequence_length(parent_id).unwrap(),
            manager.get_sequence_length(child_id).unwrap()
        );

        // Block tables should share blocks (same block IDs)
        let parent_blocks = manager.get_block_table(parent_id).unwrap().to_vec();
        let child_blocks = manager.get_block_table(child_id).unwrap().to_vec();
        assert_eq!(parent_blocks, child_blocks);
    }

    #[test]
    fn test_memory_utilization() {
        let config = test_config(); // 8 blocks
        let mut manager = BlockManager::new(config);

        assert_eq!(manager.memory_utilization(), 0.0);

        let seq_id = manager.create_sequence().unwrap();
        manager.append_tokens(seq_id, 16).unwrap(); // 1 block

        assert!((manager.memory_utilization() - 0.125).abs() < 0.01); // 1/8

        manager.append_tokens(seq_id, 48).unwrap(); // 3 more blocks (4 total)
        assert!((manager.memory_utilization() - 0.5).abs() < 0.01); // 4/8
    }

    #[test]
    fn test_can_allocate() {
        let config = test_config(); // 8 blocks, 16 tokens/block
        let mut manager = BlockManager::new(config);

        assert!(manager.can_allocate(128)); // 8 blocks exactly
        assert!(!manager.can_allocate(129)); // Would need 9 blocks

        let seq_id = manager.create_sequence().unwrap();
        manager.append_tokens(seq_id, 64).unwrap(); // Use 4 blocks

        assert!(manager.can_allocate(64)); // 4 blocks left
        assert!(!manager.can_allocate(65)); // Would need 5 blocks
    }

    #[test]
    fn test_stats() {
        let config = test_config();
        let mut manager = BlockManager::new(config);

        let seq1 = manager.create_sequence().unwrap();
        manager.append_tokens(seq1, 20).unwrap();

        let seq2 = manager.create_sequence().unwrap();
        manager.append_tokens(seq2, 10).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.num_sequences, 2);
        assert_eq!(stats.total_tokens, 30);
        assert_eq!(stats.used_blocks, 3); // 2 + 1
        assert_eq!(stats.free_blocks, 5); // 8 - 3
    }

    #[test]
    fn test_llama_config() {
        // 8GB VRAM for Llama 8B
        let config = BlockManagerConfig::llama_8b(8 * 1024 * 1024 * 1024);

        // bytes_per_token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes
        assert_eq!(config.bytes_per_token(), 131_072);

        // bytes_per_block = 16 * 131,072 = 2,097,152 bytes (2 MB)
        assert_eq!(config.bytes_per_block(), 2_097_152);

        // num_blocks = 8GB / 2MB = 4096 blocks
        assert_eq!(config.num_blocks(), 4096);
    }
}
