//! Continuous Batching Scheduler
//!
//! This module implements a continuous batching scheduler for LLM inference,
//! inspired by vLLM's approach. Instead of waiting for entire batches to complete,
//! the scheduler dynamically adds and removes requests at each iteration.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              Request Queue                       │
//! │  [Req1: prompt] [Req2: gen step 5]              │
//! │  [Req3: gen step 2] [Req4: prompt]              │
//! └───────────────┬─────────────────────────────────┘
//!                 │
//!         ┌───────▼───────┐
//!         │   Scheduler   │ ← Dynamic batch assembly
//!         └───────┬───────┘
//!                 │
//!         ┌───────▼───────┐
//!         │   Inference   │
//!         │    Engine     │
//!         └───────────────┘
//! ```
//!
//! # Benefits
//!
//! - **Higher Throughput**: GPU utilization stays high even with variable-length sequences
//! - **Lower Latency**: New requests don't wait for long-running sequences
//! - **Memory Efficiency**: Combined with PagedAttention for optimal memory usage
//!
//! # Example
//!
//! ```rust
//! use velollm_core::scheduler::{Scheduler, SchedulerConfig, Request};
//! use velollm_core::paged_attention::{BlockManager, BlockManagerConfig};
//!
//! // Create block manager and scheduler
//! let block_config = BlockManagerConfig::default();
//! let scheduler_config = SchedulerConfig::default();
//! let mut scheduler = Scheduler::new(block_config, scheduler_config);
//!
//! // Add requests
//! let req = Request::new(vec![1, 2, 3], 100);
//! scheduler.add_request(req);
//!
//! // Schedule a batch for execution
//! let output = scheduler.schedule();
//! println!("Running {} sequences", output.running_sequences.len());
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::paged_attention::{BlockManager, BlockManagerConfig, PagedAttentionError};

/// Configuration for the scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences to run in parallel
    pub max_batch_size: usize,

    /// Maximum number of tokens per scheduling step
    pub max_tokens_per_step: usize,

    /// Maximum number of tokens in the waiting queue before rejecting new requests
    pub max_waiting_tokens: usize,

    /// Memory utilization threshold for preemption (0.0 to 1.0)
    pub preemption_threshold: f64,

    /// Enable priority boosting for requests waiting too long
    pub enable_priority_boost: bool,

    /// Time in milliseconds after which a waiting request gets priority boost
    pub priority_boost_delay_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_tokens_per_step: 4096,
            max_waiting_tokens: 8192,
            preemption_threshold: 0.95,
            enable_priority_boost: true,
            priority_boost_delay_ms: 5000,
        }
    }
}

impl SchedulerConfig {
    /// Create config optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_step: 1024,
            max_waiting_tokens: 2048,
            preemption_threshold: 0.90,
            enable_priority_boost: true,
            priority_boost_delay_ms: 2000,
        }
    }

    /// Create config optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 64,
            max_tokens_per_step: 8192,
            max_waiting_tokens: 16384,
            preemption_threshold: 0.98,
            enable_priority_boost: false,
            priority_boost_delay_ms: 10000,
        }
    }
}

/// State of a request in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Request is waiting to be scheduled
    Waiting,
    /// Request is currently running (in the active batch)
    Running,
    /// Request was preempted (will be rescheduled)
    Preempted,
    /// Request has finished generation
    Finished,
}

impl std::fmt::Display for RequestState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestState::Waiting => write!(f, "Waiting"),
            RequestState::Running => write!(f, "Running"),
            RequestState::Preempted => write!(f, "Preempted"),
            RequestState::Finished => write!(f, "Finished"),
        }
    }
}

/// A request in the scheduler
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier
    pub id: u64,

    /// Input prompt token IDs
    pub prompt_tokens: Vec<u32>,

    /// Maximum number of new tokens to generate
    pub max_new_tokens: u32,

    /// Tokens generated so far
    pub generated_tokens: Vec<u32>,

    /// Current state of the request
    pub state: RequestState,

    /// Sequence ID in the BlockManager (assigned when running)
    pub sequence_id: Option<u64>,

    /// Time when the request was added
    pub arrival_time: Instant,

    /// Priority (lower = higher priority)
    pub priority: u32,
}

impl Request {
    /// Create a new request
    pub fn new(prompt_tokens: Vec<u32>, max_new_tokens: u32) -> Self {
        Self {
            id: 0, // Will be assigned by scheduler
            prompt_tokens,
            max_new_tokens,
            generated_tokens: Vec::new(),
            state: RequestState::Waiting,
            sequence_id: None,
            arrival_time: Instant::now(),
            priority: 0,
        }
    }

    /// Create a new request with custom priority
    pub fn with_priority(prompt_tokens: Vec<u32>, max_new_tokens: u32, priority: u32) -> Self {
        let mut request = Self::new(prompt_tokens, max_new_tokens);
        request.priority = priority;
        request
    }

    /// Get the total number of tokens (prompt + generated)
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    /// Get the number of remaining tokens to generate
    pub fn remaining_tokens(&self) -> usize {
        self.max_new_tokens.saturating_sub(self.generated_tokens.len() as u32) as usize
    }

    /// Check if the request has finished generation
    pub fn is_finished(&self) -> bool {
        self.generated_tokens.len() >= self.max_new_tokens as usize
    }

    /// Get time spent waiting in milliseconds
    pub fn waiting_time_ms(&self) -> u64 {
        self.arrival_time.elapsed().as_millis() as u64
    }
}

/// Output from a scheduling step
#[derive(Debug, Default)]
pub struct SchedulerOutput {
    /// Sequences scheduled for prefill (new prompts)
    pub prefill_sequences: Vec<u64>,

    /// Sequences scheduled for decode (generation step)
    pub decode_sequences: Vec<u64>,

    /// All running sequences (prefill + decode)
    pub running_sequences: Vec<u64>,

    /// Sequences that were preempted
    pub preempted_sequences: Vec<u64>,

    /// Total number of tokens in this batch
    pub num_tokens: usize,

    /// Number of requests still waiting
    pub num_waiting: usize,
}

impl SchedulerOutput {
    /// Check if there's any work to do
    pub fn has_work(&self) -> bool {
        !self.running_sequences.is_empty()
    }

    /// Get total number of sequences
    pub fn num_sequences(&self) -> usize {
        self.running_sequences.len()
    }
}

/// Errors that can occur during scheduling
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerError {
    /// Request queue is full
    QueueFull,
    /// Request not found
    RequestNotFound(u64),
    /// Sequence not found
    SequenceNotFound(u64),
    /// Memory allocation failed
    OutOfMemory,
    /// Invalid state transition
    InvalidState { from: RequestState, to: RequestState },
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::QueueFull => write!(f, "Request queue is full"),
            SchedulerError::RequestNotFound(id) => write!(f, "Request {} not found", id),
            SchedulerError::SequenceNotFound(id) => write!(f, "Sequence {} not found", id),
            SchedulerError::OutOfMemory => write!(f, "Out of memory"),
            SchedulerError::InvalidState { from, to } => {
                write!(f, "Invalid state transition from {} to {}", from, to)
            }
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<PagedAttentionError> for SchedulerError {
    fn from(err: PagedAttentionError) -> Self {
        match err {
            PagedAttentionError::OutOfMemory => SchedulerError::OutOfMemory,
            PagedAttentionError::SequenceNotFound(id) => SchedulerError::SequenceNotFound(id),
            _ => SchedulerError::OutOfMemory,
        }
    }
}

/// Continuous batching scheduler
///
/// Manages request lifecycle and coordinates with BlockManager for memory allocation.
pub struct Scheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Block manager for KV cache memory
    block_manager: BlockManager,

    /// Queue of waiting requests (ordered by priority/arrival)
    waiting_queue: VecDeque<Request>,

    /// Currently running requests (keyed by request ID)
    running: HashMap<u64, Request>,

    /// Mapping from request ID to sequence ID
    request_to_sequence: HashMap<u64, u64>,

    /// Next request ID to assign
    next_request_id: u64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(block_config: BlockManagerConfig, config: SchedulerConfig) -> Self {
        Self {
            config,
            block_manager: BlockManager::new(block_config),
            waiting_queue: VecDeque::new(),
            running: HashMap::new(),
            request_to_sequence: HashMap::new(),
            next_request_id: 0,
        }
    }

    /// Create scheduler with default configuration
    pub fn with_defaults(block_config: BlockManagerConfig) -> Self {
        Self::new(block_config, SchedulerConfig::default())
    }

    /// Add a new request to the waiting queue
    pub fn add_request(&mut self, mut request: Request) -> Result<u64, SchedulerError> {
        // Check queue capacity
        let waiting_tokens: usize = self.waiting_queue.iter().map(|r| r.total_tokens()).sum();
        if waiting_tokens + request.prompt_tokens.len() > self.config.max_waiting_tokens {
            return Err(SchedulerError::QueueFull);
        }

        // Assign ID and set state
        request.id = self.next_request_id;
        self.next_request_id += 1;
        request.state = RequestState::Waiting;
        request.arrival_time = Instant::now();

        let request_id = request.id;

        // Insert in priority order
        let insert_pos = self.find_insert_position(&request);
        self.waiting_queue.insert(insert_pos, request);

        Ok(request_id)
    }

    /// Find the position to insert a request based on priority
    fn find_insert_position(&self, request: &Request) -> usize {
        // Lower priority value = higher priority
        for (i, existing) in self.waiting_queue.iter().enumerate() {
            if request.priority < existing.priority {
                return i;
            }
        }
        self.waiting_queue.len()
    }

    /// Schedule the next batch for execution
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();

        // Apply priority boost to long-waiting requests
        if self.config.enable_priority_boost {
            self.apply_priority_boost();
        }

        // Check for preemption if memory is tight
        if self.block_manager.memory_utilization() > self.config.preemption_threshold {
            self.preempt_lowest_priority(&mut output);
        }

        // Schedule running requests (decode step)
        let mut tokens_in_batch = 0;
        for request in self.running.values() {
            if request.state == RequestState::Running {
                if let Some(&seq_id) = self.request_to_sequence.get(&request.id) {
                    output.decode_sequences.push(seq_id);
                    output.running_sequences.push(seq_id);
                    tokens_in_batch += 1; // 1 token per decode step
                }
            }
        }

        // Try to add waiting requests (prefill)
        while !self.waiting_queue.is_empty()
            && self.running.len() < self.config.max_batch_size
            && tokens_in_batch < self.config.max_tokens_per_step
        {
            let request = match self.waiting_queue.front() {
                Some(r) => r,
                None => break,
            };

            // Check if we have enough memory for this request
            let tokens_needed = request.total_tokens() + request.remaining_tokens();
            if !self.block_manager.can_allocate(tokens_needed) {
                break;
            }

            // Check if adding prompt tokens exceeds step limit
            if tokens_in_batch + request.prompt_tokens.len() > self.config.max_tokens_per_step {
                break;
            }

            // Pop from waiting queue
            let mut request = self.waiting_queue.pop_front().unwrap();

            // Allocate sequence in block manager
            match self.block_manager.create_sequence() {
                Ok(seq_id) => {
                    // Allocate blocks for prompt
                    if self.block_manager.append_tokens(seq_id, request.prompt_tokens.len()).is_err() {
                        // Cleanup and skip
                        let _ = self.block_manager.remove_sequence(seq_id);
                        self.waiting_queue.push_front(request);
                        break;
                    }

                    request.state = RequestState::Running;
                    request.sequence_id = Some(seq_id);

                    tokens_in_batch += request.prompt_tokens.len();
                    output.prefill_sequences.push(seq_id);
                    output.running_sequences.push(seq_id);

                    self.request_to_sequence.insert(request.id, seq_id);
                    self.running.insert(request.id, request);
                }
                Err(_) => {
                    // Can't allocate, put back in queue
                    self.waiting_queue.push_front(request);
                    break;
                }
            }
        }

        output.num_tokens = tokens_in_batch;
        output.num_waiting = self.waiting_queue.len();

        output
    }

    /// Apply priority boost to long-waiting requests
    fn apply_priority_boost(&mut self) {
        for request in self.waiting_queue.iter_mut() {
            if request.waiting_time_ms() > self.config.priority_boost_delay_ms {
                // Boost priority (lower value = higher priority)
                request.priority = request.priority.saturating_sub(1);
            }
        }
    }

    /// Preempt the lowest priority running request
    fn preempt_lowest_priority(&mut self, output: &mut SchedulerOutput) {
        // Find the running request with lowest priority (highest priority value)
        let lowest_priority_id = self
            .running
            .values()
            .filter(|r| r.state == RequestState::Running)
            .max_by_key(|r| r.priority)
            .map(|r| r.id);

        if let Some(request_id) = lowest_priority_id {
            self.preempt_request(request_id, output);
        }
    }

    /// Preempt a specific request
    fn preempt_request(&mut self, request_id: u64, output: &mut SchedulerOutput) {
        if let Some(mut request) = self.running.remove(&request_id) {
            // Free sequence blocks
            if let Some(seq_id) = request.sequence_id.take() {
                let _ = self.block_manager.remove_sequence(seq_id);
                self.request_to_sequence.remove(&request_id);
                output.preempted_sequences.push(seq_id);
            }

            request.state = RequestState::Preempted;
            // Put back at front of waiting queue with boosted priority
            request.priority = request.priority.saturating_sub(2);
            self.waiting_queue.push_front(request);
        }
    }

    /// Update scheduler state after an inference step
    ///
    /// Call this after each inference step to:
    /// - Mark completed requests as finished
    /// - Update token counts for running requests
    /// - Free memory for completed sequences
    pub fn update(
        &mut self,
        generated: &HashMap<u64, Vec<u32>>,
        completed: &[u64],
    ) -> Result<(), SchedulerError> {
        // Update generated tokens for running requests
        for (&request_id, tokens) in generated {
            if let Some(request) = self.running.get_mut(&request_id) {
                request.generated_tokens.extend(tokens);

                // Allocate blocks for new tokens
                if let Some(seq_id) = request.sequence_id {
                    self.block_manager.append_tokens(seq_id, tokens.len())?;
                }
            }
        }

        // Mark completed requests as finished
        for &request_id in completed {
            if let Some(mut request) = self.running.remove(&request_id) {
                request.state = RequestState::Finished;

                // Free sequence blocks
                if let Some(seq_id) = request.sequence_id.take() {
                    self.block_manager.remove_sequence(seq_id)?;
                    self.request_to_sequence.remove(&request_id);
                }
            }
        }

        Ok(())
    }

    /// Abort a request (remove it from the system)
    pub fn abort_request(&mut self, request_id: u64) -> Result<(), SchedulerError> {
        // Check waiting queue
        if let Some(pos) = self.waiting_queue.iter().position(|r| r.id == request_id) {
            self.waiting_queue.remove(pos);
            return Ok(());
        }

        // Check running requests
        if let Some(request) = self.running.remove(&request_id) {
            if let Some(seq_id) = request.sequence_id {
                self.block_manager.remove_sequence(seq_id)?;
                self.request_to_sequence.remove(&request_id);
            }
            return Ok(());
        }

        Err(SchedulerError::RequestNotFound(request_id))
    }

    /// Get the number of waiting requests
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Get the number of running requests
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Get the total number of active requests (waiting + running)
    pub fn num_active(&self) -> usize {
        self.waiting_queue.len() + self.running.len()
    }

    /// Check if the scheduler is idle (no requests)
    pub fn is_idle(&self) -> bool {
        self.waiting_queue.is_empty() && self.running.is_empty()
    }

    /// Get memory utilization
    pub fn memory_utilization(&self) -> f64 {
        self.block_manager.memory_utilization()
    }

    /// Get the block manager (for advanced use cases)
    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    /// Get the configuration
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Get statistics about the scheduler
    pub fn stats(&self) -> SchedulerStats {
        let waiting_tokens: usize = self.waiting_queue.iter().map(|r| r.total_tokens()).sum();
        let running_tokens: usize = self.running.values().map(|r| r.total_tokens()).sum();

        SchedulerStats {
            num_waiting: self.waiting_queue.len(),
            num_running: self.running.len(),
            waiting_tokens,
            running_tokens,
            memory_utilization: self.block_manager.memory_utilization(),
            block_manager_stats: self.block_manager.stats(),
        }
    }

    /// Get a request by ID
    pub fn get_request(&self, request_id: u64) -> Option<&Request> {
        self.waiting_queue
            .iter()
            .find(|r| r.id == request_id)
            .or_else(|| self.running.get(&request_id))
    }
}

/// Statistics about the scheduler
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub num_waiting: usize,
    pub num_running: usize,
    pub waiting_tokens: usize,
    pub running_tokens: usize,
    pub memory_utilization: f64,
    pub block_manager_stats: crate::paged_attention::BlockManagerStats,
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Scheduler: {} waiting, {} running, {} tokens, {:.1}% memory",
            self.num_waiting,
            self.num_running,
            self.waiting_tokens + self.running_tokens,
            self.memory_utilization * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_block_config() -> BlockManagerConfig {
        BlockManagerConfig {
            total_memory_bytes: 1024 * 1024, // 1 MB
            block_size: 16,
            kv_head_dim: 64,
            num_kv_heads: 4,
            num_layers: 8,
            dtype_bytes: 2,
        }
    }

    fn test_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 4,
            max_tokens_per_step: 256,
            max_waiting_tokens: 1024,
            preemption_threshold: 0.95,
            enable_priority_boost: true,
            priority_boost_delay_ms: 100,
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::new(test_block_config(), test_scheduler_config());
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn test_add_request() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        let request = Request::new(vec![1, 2, 3, 4, 5], 10);
        let id = scheduler.add_request(request).unwrap();

        assert_eq!(id, 0);
        assert_eq!(scheduler.num_waiting(), 1);
        assert!(!scheduler.is_idle());
    }

    #[test]
    fn test_schedule_single_request() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        let request = Request::new(vec![1, 2, 3, 4, 5], 10);
        scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();

        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.running_sequences.len(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 1);
    }

    #[test]
    fn test_schedule_multiple_requests() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        for i in 0..3 {
            let request = Request::new(vec![i as u32; 10], 20);
            scheduler.add_request(request).unwrap();
        }

        assert_eq!(scheduler.num_waiting(), 3);

        let output = scheduler.schedule();

        assert_eq!(output.prefill_sequences.len(), 3);
        assert_eq!(scheduler.num_running(), 3);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn test_request_completion() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        let request = Request::new(vec![1, 2, 3], 5);
        let request_id = scheduler.add_request(request).unwrap();

        // Schedule the request
        scheduler.schedule();
        assert_eq!(scheduler.num_running(), 1);

        // Simulate generation
        let mut generated = HashMap::new();
        generated.insert(request_id, vec![10, 11, 12, 13, 14]);

        // Mark as completed
        scheduler.update(&generated, &[request_id]).unwrap();

        assert_eq!(scheduler.num_running(), 0);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn test_abort_request() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        let request = Request::new(vec![1, 2, 3], 10);
        let id = scheduler.add_request(request).unwrap();

        // Abort while waiting
        scheduler.abort_request(id).unwrap();
        assert_eq!(scheduler.num_waiting(), 0);

        // Add and schedule
        let request2 = Request::new(vec![1, 2, 3], 10);
        let id2 = scheduler.add_request(request2).unwrap();
        scheduler.schedule();

        // Abort while running
        scheduler.abort_request(id2).unwrap();
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        // Add low priority first
        let low = Request::with_priority(vec![1, 2, 3], 10, 10);
        scheduler.add_request(low).unwrap();

        // Add high priority second
        let high = Request::with_priority(vec![4, 5, 6], 10, 1);
        let high_id = scheduler.add_request(high).unwrap();

        // High priority should be at front
        let front = scheduler.waiting_queue.front().unwrap();
        assert_eq!(front.id, high_id);
    }

    #[test]
    fn test_max_batch_size() {
        let mut config = test_scheduler_config();
        config.max_batch_size = 2;
        let mut scheduler = Scheduler::new(test_block_config(), config);

        // Add 4 requests
        for _ in 0..4 {
            let request = Request::new(vec![1, 2, 3], 10);
            scheduler.add_request(request).unwrap();
        }

        let output = scheduler.schedule();

        // Only 2 should be running
        assert_eq!(output.running_sequences.len(), 2);
        assert_eq!(scheduler.num_running(), 2);
        assert_eq!(scheduler.num_waiting(), 2);
    }

    #[test]
    fn test_queue_full() {
        let mut config = test_scheduler_config();
        config.max_waiting_tokens = 10;
        let mut scheduler = Scheduler::new(test_block_config(), config);

        // Add request that fits
        let request1 = Request::new(vec![1, 2, 3, 4, 5], 10);
        scheduler.add_request(request1).unwrap();

        // Add request that exceeds limit
        let request2 = Request::new(vec![1, 2, 3, 4, 5, 6], 10);
        let result = scheduler.add_request(request2);

        assert!(matches!(result, Err(SchedulerError::QueueFull)));
    }

    #[test]
    fn test_continuous_batching() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        // Add first request and schedule
        let req1 = Request::new(vec![1, 2, 3], 5);
        let id1 = scheduler.add_request(req1).unwrap();
        scheduler.schedule();

        // Add second request while first is running
        let req2 = Request::new(vec![4, 5, 6], 5);
        let _id2 = scheduler.add_request(req2).unwrap();

        // Schedule again - should add second to batch
        let output = scheduler.schedule();

        assert_eq!(scheduler.num_running(), 2);
        // First request is now decode, second is prefill
        assert_eq!(output.decode_sequences.len(), 1);
        assert_eq!(output.prefill_sequences.len(), 1);

        // Complete first request
        let mut generated = HashMap::new();
        generated.insert(id1, vec![10, 11, 12, 13, 14]);
        scheduler.update(&generated, &[id1]).unwrap();

        assert_eq!(scheduler.num_running(), 1);
    }

    #[test]
    fn test_decode_step() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        let request = Request::new(vec![1, 2, 3], 10);
        let id = scheduler.add_request(request).unwrap();

        // First schedule = prefill
        let output1 = scheduler.schedule();
        assert_eq!(output1.prefill_sequences.len(), 1);
        assert_eq!(output1.decode_sequences.len(), 0);

        // Second schedule = decode (no new requests)
        let output2 = scheduler.schedule();
        assert_eq!(output2.prefill_sequences.len(), 0);
        assert_eq!(output2.decode_sequences.len(), 1);

        // Simulate generating a token
        let mut generated = HashMap::new();
        generated.insert(id, vec![10]);
        scheduler.update(&generated, &[]).unwrap();

        // Check the request state
        let request = scheduler.get_request(id).unwrap();
        assert_eq!(request.generated_tokens.len(), 1);
        assert_eq!(request.generated_tokens[0], 10);
    }

    #[test]
    fn test_stats() {
        let mut scheduler = Scheduler::new(test_block_config(), test_scheduler_config());

        let request = Request::new(vec![1, 2, 3, 4, 5], 10);
        scheduler.add_request(request).unwrap();

        let stats = scheduler.stats();
        assert_eq!(stats.num_waiting, 1);
        assert_eq!(stats.num_running, 0);
        assert_eq!(stats.waiting_tokens, 5);

        scheduler.schedule();

        let stats = scheduler.stats();
        assert_eq!(stats.num_waiting, 0);
        assert_eq!(stats.num_running, 1);
        assert_eq!(stats.running_tokens, 5);
    }

    #[test]
    fn test_request_methods() {
        let mut request = Request::new(vec![1, 2, 3], 10);

        assert_eq!(request.total_tokens(), 3);
        assert_eq!(request.remaining_tokens(), 10);
        assert!(!request.is_finished());

        request.generated_tokens = vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        assert_eq!(request.total_tokens(), 13);
        assert_eq!(request.remaining_tokens(), 0);
        assert!(request.is_finished());
    }

    #[test]
    fn test_scheduler_output() {
        let output = SchedulerOutput::default();
        assert!(!output.has_work());
        assert_eq!(output.num_sequences(), 0);

        let output = SchedulerOutput {
            running_sequences: vec![1, 2, 3],
            ..Default::default()
        };
        assert!(output.has_work());
        assert_eq!(output.num_sequences(), 3);
    }

    #[test]
    fn test_config_presets() {
        let default = SchedulerConfig::default();
        assert_eq!(default.max_batch_size, 32);

        let low_latency = SchedulerConfig::low_latency();
        assert_eq!(low_latency.max_batch_size, 8);
        assert!(low_latency.enable_priority_boost);

        let high_throughput = SchedulerConfig::high_throughput();
        assert_eq!(high_throughput.max_batch_size, 64);
        assert!(!high_throughput.enable_priority_boost);
    }
}
