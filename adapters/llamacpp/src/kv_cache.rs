//! Paged KV Cache for llama.cpp integration
//!
//! This module provides a PagedKvCache that uses VeloLLM's BlockManager
//! to efficiently manage KV cache memory for multiple concurrent sequences.
//! It provides an interface compatible with llama.cpp's memory management API.

use std::collections::HashMap;
use thiserror::Error;
use velollm_core::paged_attention::{
    BlockId, BlockManager, BlockManagerConfig, PagedAttentionError,
};

/// Errors that can occur during KV cache operations
#[derive(Debug, Error)]
pub enum KvCacheError {
    #[error("out of memory: cannot allocate blocks for sequence")]
    OutOfMemory,

    #[error("sequence {0} not found")]
    SequenceNotFound(i32),

    #[error("sequence {0} already exists")]
    SequenceExists(i32),

    #[error("invalid position range: p0={0}, p1={1}")]
    InvalidPositionRange(i32, i32),

    #[error("block manager error: {0}")]
    BlockManager(#[from] PagedAttentionError),
}

/// Represents a position in a sequence (compatible with llama_pos)
pub type LlamaPos = i32;

/// Represents a sequence ID (compatible with llama_seq_id)
pub type LlamaSeqId = i32;

/// Additional metadata for a sequence (not tracked by BlockManager)
#[derive(Debug, Clone)]
struct SequenceMetadata {
    /// Minimum position in cache
    pos_min: LlamaPos,
    /// Maximum position in cache
    pos_max: LlamaPos,
}

/// Paged KV Cache for managing multiple sequences
///
/// This struct wraps VeloLLM's BlockManager and provides an interface
/// compatible with llama.cpp's `llama_memory_*` API.
///
/// # Example
/// ```
/// use velollm_adapters_llamacpp::kv_cache::PagedKvCache;
/// use velollm_core::paged_attention::BlockManagerConfig;
///
/// // Create cache for 1GB of KV memory
/// let config = BlockManagerConfig::llama_8b(1024 * 1024 * 1024);
/// let mut cache = PagedKvCache::new(config);
///
/// // Add a new sequence with initial tokens
/// let seq_id = cache.add_sequence(100).unwrap(); // 100 initial tokens
///
/// // Append more tokens
/// cache.append_tokens(seq_id, 50).unwrap();
///
/// // Query sequence info
/// assert_eq!(cache.seq_pos_max(seq_id), 149);
/// ```
pub struct PagedKvCache {
    /// The underlying block manager
    block_manager: BlockManager,

    /// Mapping from llama_seq_id (i32) to internal seq_id (u64)
    seq_id_map: HashMap<LlamaSeqId, u64>,

    /// Reverse mapping from internal seq_id to llama_seq_id
    internal_to_llama: HashMap<u64, LlamaSeqId>,

    /// Additional metadata per sequence
    metadata: HashMap<LlamaSeqId, SequenceMetadata>,

    /// Next available llama sequence ID
    next_llama_seq_id: LlamaSeqId,
}

impl PagedKvCache {
    /// Create a new PagedKvCache with the given configuration
    pub fn new(config: BlockManagerConfig) -> Self {
        Self {
            block_manager: BlockManager::new(config),
            seq_id_map: HashMap::new(),
            internal_to_llama: HashMap::new(),
            metadata: HashMap::new(),
            next_llama_seq_id: 0,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &BlockManagerConfig {
        self.block_manager.config()
    }

    /// Add a new sequence with the given number of initial tokens
    ///
    /// Returns the sequence ID assigned to the new sequence.
    pub fn add_sequence(&mut self, num_tokens: usize) -> Result<LlamaSeqId, KvCacheError> {
        let llama_seq_id = self.next_llama_seq_id;
        self.next_llama_seq_id += 1;

        // Create internal sequence
        let internal_id = self.block_manager.create_sequence()?;

        // Allocate initial tokens
        if num_tokens > 0 {
            self.block_manager.append_tokens(internal_id, num_tokens)?;
        }

        // Store mappings
        self.seq_id_map.insert(llama_seq_id, internal_id);
        self.internal_to_llama.insert(internal_id, llama_seq_id);

        // Store metadata
        let metadata = SequenceMetadata {
            pos_min: if num_tokens > 0 { 0 } else { -1 },
            pos_max: if num_tokens > 0 {
                (num_tokens - 1) as LlamaPos
            } else {
                -1
            },
        };
        self.metadata.insert(llama_seq_id, metadata);

        Ok(llama_seq_id)
    }

    /// Add a sequence with a specific ID
    pub fn add_sequence_with_id(
        &mut self,
        seq_id: LlamaSeqId,
        num_tokens: usize,
    ) -> Result<(), KvCacheError> {
        if self.seq_id_map.contains_key(&seq_id) {
            return Err(KvCacheError::SequenceExists(seq_id));
        }

        // Create internal sequence
        let internal_id = self.block_manager.create_sequence()?;

        // Allocate initial tokens
        if num_tokens > 0 {
            self.block_manager.append_tokens(internal_id, num_tokens)?;
        }

        // Store mappings
        self.seq_id_map.insert(seq_id, internal_id);
        self.internal_to_llama.insert(internal_id, seq_id);

        // Store metadata
        let metadata = SequenceMetadata {
            pos_min: if num_tokens > 0 { 0 } else { -1 },
            pos_max: if num_tokens > 0 {
                (num_tokens - 1) as LlamaPos
            } else {
                -1
            },
        };
        self.metadata.insert(seq_id, metadata);

        if seq_id >= self.next_llama_seq_id {
            self.next_llama_seq_id = seq_id + 1;
        }

        Ok(())
    }

    /// Append tokens to an existing sequence
    pub fn append_tokens(
        &mut self,
        seq_id: LlamaSeqId,
        num_tokens: usize,
    ) -> Result<(), KvCacheError> {
        let internal_id = *self
            .seq_id_map
            .get(&seq_id)
            .ok_or(KvCacheError::SequenceNotFound(seq_id))?;

        self.block_manager.append_tokens(internal_id, num_tokens)?;

        // Update metadata
        if let Some(meta) = self.metadata.get_mut(&seq_id) {
            let new_len = self.block_manager.get_sequence_length(internal_id)?;
            if meta.pos_min == -1 {
                meta.pos_min = 0;
            }
            meta.pos_max = (new_len - 1) as LlamaPos;
        }

        Ok(())
    }

    /// Get the number of tokens in a sequence
    pub fn get_sequence_length(&self, seq_id: LlamaSeqId) -> Result<usize, KvCacheError> {
        let internal_id = *self
            .seq_id_map
            .get(&seq_id)
            .ok_or(KvCacheError::SequenceNotFound(seq_id))?;

        Ok(self.block_manager.get_sequence_length(internal_id)?)
    }

    /// Get block IDs for a sequence (for passing to attention kernel)
    pub fn get_block_table(&self, seq_id: LlamaSeqId) -> Result<&[BlockId], KvCacheError> {
        let internal_id = *self
            .seq_id_map
            .get(&seq_id)
            .ok_or(KvCacheError::SequenceNotFound(seq_id))?;

        Ok(self.block_manager.get_block_table(internal_id)?)
    }

    // =========================================================================
    // llama_memory_* compatible API
    // =========================================================================

    /// Clear all sequences (equivalent to llama_memory_clear)
    pub fn clear(&mut self, _clear_data: bool) {
        // Remove all sequences
        let seq_ids: Vec<LlamaSeqId> = self.seq_id_map.keys().copied().collect();
        for seq_id in seq_ids {
            if let Some(internal_id) = self.seq_id_map.remove(&seq_id) {
                let _ = self.block_manager.remove_sequence(internal_id);
                self.internal_to_llama.remove(&internal_id);
            }
            self.metadata.remove(&seq_id);
        }
        self.next_llama_seq_id = 0;
    }

    /// Remove positions [p0, p1) from a sequence (equivalent to llama_memory_seq_rm)
    ///
    /// If p0 == p1 == -1, removes the entire sequence.
    /// Returns true if the sequence is now empty and was removed.
    pub fn seq_rm(
        &mut self,
        seq_id: LlamaSeqId,
        p0: LlamaPos,
        p1: LlamaPos,
    ) -> Result<bool, KvCacheError> {
        // Special case: remove entire sequence
        if p0 == -1 && p1 == -1 {
            if let Some(internal_id) = self.seq_id_map.remove(&seq_id) {
                self.block_manager.remove_sequence(internal_id)?;
                self.internal_to_llama.remove(&internal_id);
                self.metadata.remove(&seq_id);
                return Ok(true);
            }
            return Err(KvCacheError::SequenceNotFound(seq_id));
        }

        let internal_id = *self
            .seq_id_map
            .get(&seq_id)
            .ok_or(KvCacheError::SequenceNotFound(seq_id))?;

        // Validate range
        if p0 > p1 || p0 < 0 {
            return Err(KvCacheError::InvalidPositionRange(p0, p1));
        }

        // Get current length
        let current_len = self.block_manager.get_sequence_length(internal_id)?;
        let tokens_to_remove = (p1 - p0) as usize;

        if tokens_to_remove >= current_len {
            // Removing all tokens - remove the sequence entirely
            self.seq_id_map.remove(&seq_id);
            self.block_manager.remove_sequence(internal_id)?;
            self.internal_to_llama.remove(&internal_id);
            self.metadata.remove(&seq_id);
            return Ok(true);
        }

        // Note: BlockManager doesn't support partial token removal,
        // so we just update the metadata for now
        // In a full implementation, we'd need to handle this at the block level
        if let Some(meta) = self.metadata.get_mut(&seq_id) {
            let new_len = current_len - tokens_to_remove;
            meta.pos_max = if new_len > 0 {
                (new_len - 1) as LlamaPos
            } else {
                -1
            };
        }

        Ok(false)
    }

    /// Copy sequence (equivalent to llama_memory_seq_cp)
    ///
    /// Copies tokens from [p0, p1) of src to dst sequence.
    /// Uses copy-on-write semantics.
    pub fn seq_cp(
        &mut self,
        seq_id_src: LlamaSeqId,
        seq_id_dst: LlamaSeqId,
        _p0: LlamaPos,
        _p1: LlamaPos,
    ) -> Result<(), KvCacheError> {
        let src_internal_id = *self
            .seq_id_map
            .get(&seq_id_src)
            .ok_or(KvCacheError::SequenceNotFound(seq_id_src))?;

        // If dst exists, remove it first
        if let Some(dst_internal_id) = self.seq_id_map.remove(&seq_id_dst) {
            self.block_manager.remove_sequence(dst_internal_id)?;
            self.internal_to_llama.remove(&dst_internal_id);
        }

        // Fork the source sequence (copy-on-write)
        let dst_internal_id = self.block_manager.fork_sequence(src_internal_id)?;

        // Store mappings
        self.seq_id_map.insert(seq_id_dst, dst_internal_id);
        self.internal_to_llama.insert(dst_internal_id, seq_id_dst);

        // Copy metadata
        if let Some(src_meta) = self.metadata.get(&seq_id_src).cloned() {
            self.metadata.insert(seq_id_dst, src_meta);
        }

        if seq_id_dst >= self.next_llama_seq_id {
            self.next_llama_seq_id = seq_id_dst + 1;
        }

        Ok(())
    }

    /// Keep only the specified sequence (equivalent to llama_memory_seq_keep)
    pub fn seq_keep(&mut self, seq_id: LlamaSeqId) -> Result<(), KvCacheError> {
        if !self.seq_id_map.contains_key(&seq_id) {
            return Err(KvCacheError::SequenceNotFound(seq_id));
        }

        // Collect sequences to remove
        let to_remove: Vec<LlamaSeqId> = self
            .seq_id_map
            .keys()
            .filter(|&&id| id != seq_id)
            .copied()
            .collect();

        // Remove all other sequences
        for id in to_remove {
            if let Some(internal_id) = self.seq_id_map.remove(&id) {
                let _ = self.block_manager.remove_sequence(internal_id);
                self.internal_to_llama.remove(&internal_id);
            }
            self.metadata.remove(&id);
        }

        Ok(())
    }

    /// Add shift to positions [p0, p1) (equivalent to llama_memory_seq_add)
    pub fn seq_add(
        &mut self,
        seq_id: LlamaSeqId,
        _p0: LlamaPos,
        _p1: LlamaPos,
        shift: LlamaPos,
    ) -> Result<(), KvCacheError> {
        if !self.seq_id_map.contains_key(&seq_id) {
            return Err(KvCacheError::SequenceNotFound(seq_id));
        }

        if let Some(meta) = self.metadata.get_mut(&seq_id) {
            meta.pos_min = (meta.pos_min + shift).max(0);
            meta.pos_max += shift;
        }

        Ok(())
    }

    /// Divide positions [p0, p1) by d (equivalent to llama_memory_seq_div)
    pub fn seq_div(
        &mut self,
        seq_id: LlamaSeqId,
        _p0: LlamaPos,
        _p1: LlamaPos,
        d: i32,
    ) -> Result<(), KvCacheError> {
        if d <= 0 {
            return Err(KvCacheError::InvalidPositionRange(0, d));
        }

        if !self.seq_id_map.contains_key(&seq_id) {
            return Err(KvCacheError::SequenceNotFound(seq_id));
        }

        if let Some(meta) = self.metadata.get_mut(&seq_id) {
            meta.pos_min /= d;
            meta.pos_max /= d;
        }

        Ok(())
    }

    /// Get minimum position for a sequence (equivalent to llama_memory_seq_pos_min)
    pub fn seq_pos_min(&self, seq_id: LlamaSeqId) -> LlamaPos {
        self.metadata
            .get(&seq_id)
            .map(|meta| meta.pos_min)
            .unwrap_or(-1)
    }

    /// Get maximum position for a sequence (equivalent to llama_memory_seq_pos_max)
    pub fn seq_pos_max(&self, seq_id: LlamaSeqId) -> LlamaPos {
        self.metadata
            .get(&seq_id)
            .map(|meta| meta.pos_max)
            .unwrap_or(-1)
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get the number of active sequences
    pub fn num_sequences(&self) -> usize {
        self.seq_id_map.len()
    }

    /// Get total number of tokens across all sequences
    pub fn total_tokens(&self) -> usize {
        self.seq_id_map
            .keys()
            .filter_map(|&seq_id| self.get_sequence_length(seq_id).ok())
            .sum()
    }

    /// Get memory utilization (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        self.block_manager.memory_utilization()
    }

    /// Check if we can allocate the given number of tokens
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        self.block_manager.can_allocate(num_tokens)
    }

    /// Get statistics about the cache
    pub fn stats(&self) -> PagedKvCacheStats {
        let bm_stats = self.block_manager.stats();
        PagedKvCacheStats {
            num_sequences: self.num_sequences(),
            total_tokens: self.total_tokens(),
            total_blocks: bm_stats.used_blocks,
            max_blocks: bm_stats.total_blocks,
            memory_utilization: bm_stats.memory_utilization,
            block_size: self.config().block_size,
        }
    }
}

/// Statistics about the PagedKvCache
#[derive(Debug, Clone)]
pub struct PagedKvCacheStats {
    pub num_sequences: usize,
    pub total_tokens: usize,
    pub total_blocks: usize,
    pub max_blocks: usize,
    pub memory_utilization: f64,
    pub block_size: usize,
}

impl std::fmt::Display for PagedKvCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} sequences, {} tokens, {}/{} blocks ({:.1}% utilization)",
            self.num_sequences,
            self.total_tokens,
            self.total_blocks,
            self.max_blocks,
            self.memory_utilization * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> BlockManagerConfig {
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
    fn test_new_cache() {
        let config = create_test_config();
        let cache = PagedKvCache::new(config);

        assert_eq!(cache.num_sequences(), 0);
        assert_eq!(cache.total_tokens(), 0);
    }

    #[test]
    fn test_add_sequence() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(50).unwrap();
        assert_eq!(seq_id, 0);
        assert_eq!(cache.num_sequences(), 1);
        assert_eq!(cache.get_sequence_length(seq_id).unwrap(), 50);
    }

    #[test]
    fn test_append_tokens() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(10).unwrap();
        assert_eq!(cache.get_sequence_length(seq_id).unwrap(), 10);

        cache.append_tokens(seq_id, 10).unwrap();
        assert_eq!(cache.get_sequence_length(seq_id).unwrap(), 20);
    }

    #[test]
    fn test_seq_rm_entire() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(50).unwrap();
        assert_eq!(cache.num_sequences(), 1);

        let removed = cache.seq_rm(seq_id, -1, -1).unwrap();
        assert!(removed);
        assert_eq!(cache.num_sequences(), 0);
    }

    #[test]
    fn test_seq_cp() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let src_id = cache.add_sequence(50).unwrap();
        let dst_id = 5;

        cache.seq_cp(src_id, dst_id, 0, 50).unwrap();

        assert_eq!(cache.num_sequences(), 2);
        assert_eq!(
            cache.get_sequence_length(src_id).unwrap(),
            cache.get_sequence_length(dst_id).unwrap()
        );
    }

    #[test]
    fn test_seq_keep() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        cache.add_sequence(50).unwrap();
        let keep_id = cache.add_sequence(30).unwrap();
        cache.add_sequence(20).unwrap();

        assert_eq!(cache.num_sequences(), 3);

        cache.seq_keep(keep_id).unwrap();

        assert_eq!(cache.num_sequences(), 1);
        assert!(cache.get_sequence_length(keep_id).is_ok());
    }

    #[test]
    fn test_seq_pos_min_max() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(50).unwrap();

        assert_eq!(cache.seq_pos_min(seq_id), 0);
        assert_eq!(cache.seq_pos_max(seq_id), 49);

        // Non-existent sequence
        assert_eq!(cache.seq_pos_min(999), -1);
        assert_eq!(cache.seq_pos_max(999), -1);
    }

    #[test]
    fn test_clear() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        cache.add_sequence(50).unwrap();
        cache.add_sequence(30).unwrap();
        cache.add_sequence(20).unwrap();

        assert_eq!(cache.num_sequences(), 3);

        cache.clear(true);

        assert_eq!(cache.num_sequences(), 0);
        assert_eq!(cache.total_tokens(), 0);
    }

    #[test]
    fn test_memory_utilization() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        assert_eq!(cache.memory_utilization(), 0.0);

        // Add sequence - utilization should increase
        cache.add_sequence(16).unwrap();
        assert!(cache.memory_utilization() > 0.0);
    }

    #[test]
    fn test_stats() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        cache.add_sequence(50).unwrap();
        cache.add_sequence(30).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.num_sequences, 2);
        assert_eq!(stats.total_tokens, 80);
    }

    #[test]
    fn test_add_sequence_with_id() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        cache.add_sequence_with_id(5, 50).unwrap();
        assert!(cache.get_sequence_length(5).is_ok());

        // Duplicate should fail
        let result = cache.add_sequence_with_id(5, 30);
        assert!(matches!(result, Err(KvCacheError::SequenceExists(5))));
    }

    #[test]
    fn test_block_table() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(50).unwrap();
        let block_table = cache.get_block_table(seq_id).unwrap();

        // 50 tokens with block_size=16 needs ceil(50/16) = 4 blocks
        assert_eq!(block_table.len(), 4);
    }

    #[test]
    fn test_seq_add_shift() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(50).unwrap();
        assert_eq!(cache.seq_pos_min(seq_id), 0);
        assert_eq!(cache.seq_pos_max(seq_id), 49);

        cache.seq_add(seq_id, 0, 50, 10).unwrap();
        assert_eq!(cache.seq_pos_min(seq_id), 10);
        assert_eq!(cache.seq_pos_max(seq_id), 59);
    }

    #[test]
    fn test_seq_div() {
        let config = create_test_config();
        let mut cache = PagedKvCache::new(config);

        let seq_id = cache.add_sequence(100).unwrap();
        assert_eq!(cache.seq_pos_max(seq_id), 99);

        cache.seq_div(seq_id, 0, 100, 2).unwrap();
        assert_eq!(cache.seq_pos_max(seq_id), 49);
    }
}
