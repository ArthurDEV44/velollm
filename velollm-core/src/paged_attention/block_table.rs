//! Sequence Block Table for PagedAttention
//!
//! Manages the mapping from logical block indices to physical block IDs
//! for a single sequence. Handles dynamic growth as tokens are added.

use super::block_allocator::{BlockAllocator, BlockId};
use super::PagedAttentionError;

/// Manages block allocation for a single sequence
///
/// The block table tracks:
/// - Which physical blocks are assigned to this sequence
/// - How many tokens are stored
/// - Logical → Physical block mapping
///
/// Blocks are allocated on-demand as tokens are appended.
#[derive(Debug, Clone)]
pub struct SequenceBlockTable {
    /// Physical block IDs in logical order
    block_ids: Vec<BlockId>,

    /// Number of tokens in this sequence
    num_tokens: usize,

    /// Tokens per block
    block_size: usize,
}

impl SequenceBlockTable {
    /// Create a new empty block table
    pub fn new(block_size: usize) -> Self {
        Self {
            block_ids: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Get the number of tokens in this sequence
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get the number of blocks allocated
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the block IDs (for passing to attention kernel)
    pub fn block_ids(&self) -> &[BlockId] {
        &self.block_ids
    }

    /// Get the physical block ID for a logical block index
    pub fn get_block(&self, logical_idx: usize) -> Option<BlockId> {
        self.block_ids.get(logical_idx).copied()
    }

    /// Get the physical block ID for a token position
    pub fn get_block_for_token(&self, token_pos: usize) -> Option<BlockId> {
        let logical_idx = token_pos / self.block_size;
        self.get_block(logical_idx)
    }

    /// Get the offset within a block for a token position
    pub fn get_offset_in_block(&self, token_pos: usize) -> usize {
        token_pos % self.block_size
    }

    /// Calculate how many blocks are needed for a given number of tokens
    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        if num_tokens == 0 {
            return 0;
        }
        num_tokens.div_ceil(self.block_size)
    }

    /// Append tokens to this sequence, allocating blocks as needed
    pub fn append_tokens(
        &mut self,
        num_tokens: usize,
        allocator: &mut BlockAllocator,
    ) -> Result<(), PagedAttentionError> {
        let new_total = self.num_tokens + num_tokens;
        let blocks_needed = self.blocks_needed(new_total);

        // Allocate new blocks if needed
        while self.block_ids.len() < blocks_needed {
            let block_id = allocator.allocate().ok_or(PagedAttentionError::OutOfMemory)?;
            self.block_ids.push(block_id);
        }

        self.num_tokens = new_total;
        Ok(())
    }

    /// Set a specific number of tokens (may shrink or grow)
    pub fn set_num_tokens(
        &mut self,
        num_tokens: usize,
        allocator: &mut BlockAllocator,
    ) -> Result<(), PagedAttentionError> {
        if num_tokens > self.num_tokens {
            // Growing
            self.append_tokens(num_tokens - self.num_tokens, allocator)
        } else if num_tokens < self.num_tokens {
            // Shrinking - free excess blocks
            let blocks_needed = self.blocks_needed(num_tokens);
            while self.block_ids.len() > blocks_needed {
                if let Some(block_id) = self.block_ids.pop() {
                    allocator.free(block_id);
                }
            }
            self.num_tokens = num_tokens;
            Ok(())
        } else {
            // No change
            Ok(())
        }
    }

    /// Free all blocks and reset to empty
    pub fn clear(&mut self, allocator: &mut BlockAllocator) {
        for block_id in self.block_ids.drain(..) {
            allocator.free(block_id);
        }
        self.num_tokens = 0;
    }

    /// Fork this block table (copy-on-write semantics)
    ///
    /// Creates a new block table that shares all blocks with this one.
    /// The caller must handle actual data copy when blocks diverge.
    pub fn fork(&self, allocator: &mut BlockAllocator) -> Self {
        // Increment reference count for all shared blocks
        for &block_id in &self.block_ids {
            allocator.add_ref(block_id);
        }

        Self {
            block_ids: self.block_ids.clone(),
            num_tokens: self.num_tokens,
            block_size: self.block_size,
        }
    }

    /// Check if the last block has space for more tokens
    pub fn has_space_in_last_block(&self) -> bool {
        if self.block_ids.is_empty() {
            return false;
        }
        !self.num_tokens.is_multiple_of(self.block_size)
    }

    /// Get how many tokens can fit in the last block
    pub fn space_in_last_block(&self) -> usize {
        if self.block_ids.is_empty() {
            return 0;
        }
        let used_in_last = self.num_tokens % self.block_size;
        if used_in_last == 0 && self.num_tokens > 0 {
            0 // Last block is full
        } else {
            self.block_size - used_in_last
        }
    }

    /// Get statistics about this block table
    pub fn stats(&self) -> BlockTableStats {
        let capacity = self.block_ids.len() * self.block_size;
        let utilization = if capacity > 0 {
            self.num_tokens as f64 / capacity as f64
        } else {
            0.0
        };

        BlockTableStats {
            num_tokens: self.num_tokens,
            num_blocks: self.block_ids.len(),
            block_size: self.block_size,
            capacity,
            utilization,
        }
    }
}

/// Statistics about a block table
#[derive(Debug, Clone)]
pub struct BlockTableStats {
    pub num_tokens: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub capacity: usize,
    pub utilization: f64,
}

impl std::fmt::Display for BlockTableStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tokens in {} blocks ({:.1}% utilization)",
            self.num_tokens,
            self.num_blocks,
            self.utilization * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (SequenceBlockTable, BlockAllocator) {
        let table = SequenceBlockTable::new(16);
        let allocator = BlockAllocator::new(10);
        (table, allocator)
    }

    #[test]
    fn test_new_block_table() {
        let (table, _) = setup();
        assert_eq!(table.num_tokens(), 0);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.block_size(), 16);
    }

    #[test]
    fn test_append_tokens_single_block() {
        let (mut table, mut allocator) = setup();

        table.append_tokens(10, &mut allocator).unwrap();

        assert_eq!(table.num_tokens(), 10);
        assert_eq!(table.num_blocks(), 1);
    }

    #[test]
    fn test_append_tokens_multiple_blocks() {
        let (mut table, mut allocator) = setup();

        // 50 tokens with block_size=16 needs ceil(50/16) = 4 blocks
        table.append_tokens(50, &mut allocator).unwrap();

        assert_eq!(table.num_tokens(), 50);
        assert_eq!(table.num_blocks(), 4);
    }

    #[test]
    fn test_append_tokens_exact_blocks() {
        let (mut table, mut allocator) = setup();

        // 32 tokens with block_size=16 needs exactly 2 blocks
        table.append_tokens(32, &mut allocator).unwrap();

        assert_eq!(table.num_tokens(), 32);
        assert_eq!(table.num_blocks(), 2);
    }

    #[test]
    fn test_incremental_append() {
        let (mut table, mut allocator) = setup();

        table.append_tokens(10, &mut allocator).unwrap();
        assert_eq!(table.num_blocks(), 1);

        table.append_tokens(5, &mut allocator).unwrap();
        assert_eq!(table.num_tokens(), 15);
        assert_eq!(table.num_blocks(), 1); // Still fits in 1 block

        table.append_tokens(5, &mut allocator).unwrap();
        assert_eq!(table.num_tokens(), 20);
        assert_eq!(table.num_blocks(), 2); // Now needs 2 blocks
    }

    #[test]
    fn test_get_block_for_token() {
        let (mut table, mut allocator) = setup();
        table.append_tokens(40, &mut allocator).unwrap();

        // Token 0-15 in block 0
        let block0 = table.get_block_for_token(0).unwrap();
        assert_eq!(table.get_block_for_token(15).unwrap(), block0);

        // Token 16-31 in block 1
        let block1 = table.get_block_for_token(16).unwrap();
        assert_ne!(block0, block1);
        assert_eq!(table.get_block_for_token(31).unwrap(), block1);

        // Token 32-39 in block 2
        let block2 = table.get_block_for_token(32).unwrap();
        assert_ne!(block1, block2);
    }

    #[test]
    fn test_get_offset_in_block() {
        let (table, _) = setup();

        assert_eq!(table.get_offset_in_block(0), 0);
        assert_eq!(table.get_offset_in_block(5), 5);
        assert_eq!(table.get_offset_in_block(15), 15);
        assert_eq!(table.get_offset_in_block(16), 0); // First token of second block
        assert_eq!(table.get_offset_in_block(20), 4);
    }

    #[test]
    fn test_set_num_tokens_grow() {
        let (mut table, mut allocator) = setup();

        table.set_num_tokens(30, &mut allocator).unwrap();
        assert_eq!(table.num_tokens(), 30);
        assert_eq!(table.num_blocks(), 2);
    }

    #[test]
    fn test_set_num_tokens_shrink() {
        let (mut table, mut allocator) = setup();

        table.append_tokens(50, &mut allocator).unwrap();
        assert_eq!(table.num_blocks(), 4);

        let free_before = allocator.num_free_blocks();

        table.set_num_tokens(20, &mut allocator).unwrap();
        assert_eq!(table.num_tokens(), 20);
        assert_eq!(table.num_blocks(), 2);

        // Should have freed 2 blocks
        assert_eq!(allocator.num_free_blocks(), free_before + 2);
    }

    #[test]
    fn test_clear() {
        let (mut table, mut allocator) = setup();

        table.append_tokens(50, &mut allocator).unwrap();
        let free_before = allocator.num_free_blocks();

        table.clear(&mut allocator);

        assert_eq!(table.num_tokens(), 0);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(allocator.num_free_blocks(), free_before + 4);
    }

    #[test]
    fn test_fork() {
        let (mut table, mut allocator) = setup();

        table.append_tokens(32, &mut allocator).unwrap();
        let block0 = table.get_block(0).unwrap();
        let block1 = table.get_block(1).unwrap();

        let forked = table.fork(&mut allocator);

        // Forked table has same content
        assert_eq!(forked.num_tokens(), 32);
        assert_eq!(forked.num_blocks(), 2);

        // Blocks are shared
        assert_eq!(forked.get_block(0).unwrap(), block0);
        assert_eq!(forked.get_block(1).unwrap(), block1);

        // Reference counts incremented
        assert_eq!(allocator.ref_count(block0), 2);
        assert_eq!(allocator.ref_count(block1), 2);
    }

    #[test]
    fn test_space_in_last_block() {
        let (mut table, mut allocator) = setup();

        // Empty table
        assert_eq!(table.space_in_last_block(), 0);
        assert!(!table.has_space_in_last_block());

        // 10 tokens in 16-token block
        table.append_tokens(10, &mut allocator).unwrap();
        assert_eq!(table.space_in_last_block(), 6);
        assert!(table.has_space_in_last_block());

        // 16 tokens - block is full
        table.append_tokens(6, &mut allocator).unwrap();
        assert_eq!(table.space_in_last_block(), 0);
        assert!(!table.has_space_in_last_block());

        // 17 tokens - new block with 15 spaces
        table.append_tokens(1, &mut allocator).unwrap();
        assert_eq!(table.space_in_last_block(), 15);
        assert!(table.has_space_in_last_block());
    }

    #[test]
    fn test_blocks_needed() {
        let (table, _) = setup();

        assert_eq!(table.blocks_needed(0), 0);
        assert_eq!(table.blocks_needed(1), 1);
        assert_eq!(table.blocks_needed(16), 1);
        assert_eq!(table.blocks_needed(17), 2);
        assert_eq!(table.blocks_needed(32), 2);
        assert_eq!(table.blocks_needed(33), 3);
    }

    #[test]
    fn test_stats() {
        let (mut table, mut allocator) = setup();

        table.append_tokens(25, &mut allocator).unwrap();
        let stats = table.stats();

        assert_eq!(stats.num_tokens, 25);
        assert_eq!(stats.num_blocks, 2);
        assert_eq!(stats.block_size, 16);
        assert_eq!(stats.capacity, 32);
        assert!((stats.utilization - 0.78125).abs() < 0.01); // 25/32
    }

    #[test]
    fn test_out_of_memory() {
        let mut table = SequenceBlockTable::new(16);
        let mut allocator = BlockAllocator::new(2); // Only 2 blocks

        // Can allocate 32 tokens
        table.append_tokens(32, &mut allocator).unwrap();

        // Can't allocate more
        let result = table.append_tokens(1, &mut allocator);
        assert!(matches!(result, Err(PagedAttentionError::OutOfMemory)));
    }
}
