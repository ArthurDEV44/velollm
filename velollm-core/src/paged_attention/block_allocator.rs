//! Block Allocator for PagedAttention
//!
//! Manages a pool of physical memory blocks for KV cache storage.
//! Blocks can be allocated, freed, and reference-counted for memory sharing.

use std::collections::{HashMap, VecDeque};

use super::PagedAttentionError;

/// Unique identifier for a physical memory block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

impl BlockId {
    /// Get the raw block ID value
    pub fn value(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block({})", self.0)
    }
}

/// Manages allocation and deallocation of physical memory blocks
///
/// The allocator maintains a pool of blocks identified by BlockId.
/// It supports:
/// - Allocation from a free list
/// - Reference counting for shared blocks (copy-on-write)
/// - LRU-based eviction when memory is exhausted
pub struct BlockAllocator {
    /// Total number of blocks in the pool
    num_blocks: usize,

    /// Queue of free block IDs (FIFO for allocation)
    free_blocks: VecDeque<BlockId>,

    /// Reference count for each allocated block
    ref_counts: HashMap<BlockId, usize>,
}

impl BlockAllocator {
    /// Create a new block allocator with the given number of blocks
    pub fn new(num_blocks: usize) -> Self {
        let free_blocks = (0..num_blocks).map(BlockId).collect();

        Self { num_blocks, free_blocks, ref_counts: HashMap::new() }
    }

    /// Allocate a new block from the free pool
    ///
    /// Returns `None` if no blocks are available.
    pub fn allocate(&mut self) -> Option<BlockId> {
        let block_id = self.free_blocks.pop_front()?;
        self.ref_counts.insert(block_id, 1);
        Some(block_id)
    }

    /// Free a block (decrement reference count)
    ///
    /// The block is only returned to the free pool when its reference count reaches 0.
    pub fn free(&mut self, block_id: BlockId) {
        if let Some(count) = self.ref_counts.get_mut(&block_id) {
            *count -= 1;
            if *count == 0 {
                self.ref_counts.remove(&block_id);
                self.free_blocks.push_back(block_id);
            }
        }
    }

    /// Increment the reference count for a block (for copy-on-write sharing)
    pub fn add_ref(&mut self, block_id: BlockId) {
        if let Some(count) = self.ref_counts.get_mut(&block_id) {
            *count += 1;
        }
    }

    /// Get the reference count for a block
    pub fn ref_count(&self, block_id: BlockId) -> usize {
        self.ref_counts.get(&block_id).copied().unwrap_or(0)
    }

    /// Check if a block is shared (ref_count > 1)
    pub fn is_shared(&self, block_id: BlockId) -> bool {
        self.ref_count(block_id) > 1
    }

    /// Get the number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get the total number of blocks
    pub fn num_total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get the number of allocated blocks
    pub fn num_allocated_blocks(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Get memory utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.num_allocated_blocks() as f64 / self.num_blocks as f64
    }

    /// Check if a specific number of blocks can be allocated
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.free_blocks.len() >= num_blocks
    }

    /// Allocate multiple blocks at once
    ///
    /// Returns `Err` if not enough blocks are available (no partial allocation).
    pub fn allocate_many(&mut self, count: usize) -> Result<Vec<BlockId>, PagedAttentionError> {
        if !self.can_allocate(count) {
            return Err(PagedAttentionError::OutOfMemory);
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            // Safe to unwrap because we checked can_allocate
            blocks.push(self.allocate().unwrap());
        }

        Ok(blocks)
    }

    /// Free multiple blocks at once
    pub fn free_many(&mut self, blocks: &[BlockId]) {
        for block_id in blocks {
            self.free(*block_id);
        }
    }

    /// Copy a block for copy-on-write
    ///
    /// If the block is shared, allocates a new block and returns it.
    /// If the block is not shared, returns the same block.
    ///
    /// Note: The actual data copy must be handled by the caller.
    pub fn copy_on_write(&mut self, block_id: BlockId) -> Result<BlockId, PagedAttentionError> {
        if !self.is_shared(block_id) {
            // Not shared, can modify in place
            return Ok(block_id);
        }

        // Allocate new block
        let new_block = self.allocate().ok_or(PagedAttentionError::OutOfMemory)?;

        // Decrement old block's ref count
        self.free(block_id);

        Ok(new_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_allocator() {
        let allocator = BlockAllocator::new(10);
        assert_eq!(allocator.num_total_blocks(), 10);
        assert_eq!(allocator.num_free_blocks(), 10);
        assert_eq!(allocator.num_allocated_blocks(), 0);
        assert_eq!(allocator.utilization(), 0.0);
    }

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = BlockAllocator::new(5);

        // Allocate a block
        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free_blocks(), 4);
        assert_eq!(allocator.num_allocated_blocks(), 1);

        // Free the block
        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 5);
        assert_eq!(allocator.num_allocated_blocks(), 0);
    }

    #[test]
    fn test_allocate_all() {
        let mut allocator = BlockAllocator::new(3);

        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let b3 = allocator.allocate().unwrap();

        assert_eq!(allocator.num_free_blocks(), 0);
        assert!(allocator.allocate().is_none());

        // Free one and allocate again
        allocator.free(b2);
        let b4 = allocator.allocate().unwrap();
        assert_eq!(b4, b2); // Should get the same block back (FIFO)

        // Clean up
        allocator.free(b1);
        allocator.free(b3);
        allocator.free(b4);
    }

    #[test]
    fn test_reference_counting() {
        let mut allocator = BlockAllocator::new(5);

        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.ref_count(block), 1);
        assert!(!allocator.is_shared(block));

        // Add reference (simulating fork)
        allocator.add_ref(block);
        assert_eq!(allocator.ref_count(block), 2);
        assert!(allocator.is_shared(block));

        // Free once - should not return to pool
        allocator.free(block);
        assert_eq!(allocator.ref_count(block), 1);
        assert_eq!(allocator.num_free_blocks(), 4);

        // Free again - should return to pool
        allocator.free(block);
        assert_eq!(allocator.ref_count(block), 0);
        assert_eq!(allocator.num_free_blocks(), 5);
    }

    #[test]
    fn test_allocate_many() {
        let mut allocator = BlockAllocator::new(5);

        let blocks = allocator.allocate_many(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(allocator.num_free_blocks(), 2);

        // Try to allocate too many
        let result = allocator.allocate_many(3);
        assert!(matches!(result, Err(PagedAttentionError::OutOfMemory)));

        // Original allocation should still be valid
        assert_eq!(allocator.num_free_blocks(), 2);

        allocator.free_many(&blocks);
        assert_eq!(allocator.num_free_blocks(), 5);
    }

    #[test]
    fn test_copy_on_write_not_shared() {
        let mut allocator = BlockAllocator::new(5);

        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.ref_count(block), 1);

        // CoW on non-shared block returns same block
        let cow_block = allocator.copy_on_write(block).unwrap();
        assert_eq!(cow_block, block);
        assert_eq!(allocator.num_allocated_blocks(), 1);
    }

    #[test]
    fn test_copy_on_write_shared() {
        let mut allocator = BlockAllocator::new(5);

        let block = allocator.allocate().unwrap();
        allocator.add_ref(block); // Simulate sharing

        assert!(allocator.is_shared(block));

        // CoW on shared block allocates new block
        let cow_block = allocator.copy_on_write(block).unwrap();
        assert_ne!(cow_block, block);
        assert_eq!(allocator.num_allocated_blocks(), 2);

        // Original block's ref count decremented
        assert_eq!(allocator.ref_count(block), 1);
        // New block has ref count 1
        assert_eq!(allocator.ref_count(cow_block), 1);
    }

    #[test]
    fn test_utilization() {
        let mut allocator = BlockAllocator::new(4);

        assert_eq!(allocator.utilization(), 0.0);

        allocator.allocate().unwrap();
        assert!((allocator.utilization() - 0.25).abs() < 0.01);

        allocator.allocate().unwrap();
        assert!((allocator.utilization() - 0.50).abs() < 0.01);

        allocator.allocate().unwrap();
        allocator.allocate().unwrap();
        assert!((allocator.utilization() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_can_allocate() {
        let mut allocator = BlockAllocator::new(5);

        assert!(allocator.can_allocate(5));
        assert!(!allocator.can_allocate(6));

        allocator.allocate_many(3).unwrap();
        assert!(allocator.can_allocate(2));
        assert!(!allocator.can_allocate(3));
    }

    #[test]
    fn test_block_id_display() {
        let block = BlockId(42);
        assert_eq!(format!("{}", block), "Block(42)");
        assert_eq!(block.value(), 42);
    }
}
