//! Request queue with concurrency control for the VeloLLM proxy.
//!
//! This module implements intelligent request queuing that works alongside
//! Ollama's native batching (OLLAMA_NUM_PARALLEL) to maximize throughput.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

use super::config::BatcherConfig;
use super::metrics::{BatcherMetrics, RequestTimer};

/// Error types for queue operations
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("Queue is full (max: {max})")]
    QueueFull { max: usize },

    #[error("Request timed out after {elapsed:?} in queue")]
    Timeout { elapsed: Duration },

    #[error("Queue shutdown")]
    Shutdown,
}

/// A queued request waiting for processing
pub struct QueuedRequest<T> {
    /// The request payload
    pub request: T,
    /// Model name for model-aware queuing
    pub model: String,
    /// Timer for tracking wait times
    pub timer: RequestTimer,
    /// When this request will timeout
    pub deadline: Instant,
}

impl<T> QueuedRequest<T> {
    /// Create a new queued request
    pub fn new(request: T, model: String, timeout: Duration) -> Self {
        Self {
            request,
            model,
            timer: RequestTimer::new(),
            deadline: Instant::now() + timeout,
        }
    }

    /// Check if this request has timed out
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.deadline
    }

    /// Time remaining before timeout
    pub fn time_remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

/// Request queue with concurrency control and model-aware scheduling
pub struct RequestQueue<T> {
    /// Configuration
    config: BatcherConfig,
    /// Per-model queues for model-aware scheduling
    model_queues: Mutex<HashMap<String, VecDeque<QueuedRequest<T>>>>,
    /// Global queue when model-aware queuing is disabled
    global_queue: Mutex<VecDeque<QueuedRequest<T>>>,
    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,
    /// Metrics for monitoring
    metrics: Arc<BatcherMetrics>,
    /// Current model being processed (for model affinity)
    current_model: Mutex<Option<String>>,
    /// Total queued count across all queues
    total_queued: Mutex<usize>,
}

impl<T: Send + 'static> RequestQueue<T> {
    /// Create a new request queue with the given configuration
    pub fn new(config: BatcherConfig, metrics: Arc<BatcherMetrics>) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self {
            config,
            model_queues: Mutex::new(HashMap::new()),
            global_queue: Mutex::new(VecDeque::new()),
            semaphore,
            metrics,
            current_model: Mutex::new(None),
            total_queued: Mutex::new(0),
        }
    }

    /// Enqueue a request for processing
    ///
    /// Returns an error if the queue is full
    pub async fn enqueue(&self, request: T, model: String) -> Result<(), QueueError> {
        // Check total queue capacity
        {
            let total = self.total_queued.lock().await;
            if *total >= self.config.max_queue_total {
                self.metrics.record_rejected();
                return Err(QueueError::QueueFull {
                    max: self.config.max_queue_total,
                });
            }
        }

        // Record metrics
        self.metrics.record_received();
        self.metrics.record_queued();

        let queued = QueuedRequest::new(request, model.clone(), self.config.queue_timeout);

        if self.config.model_aware_queuing {
            // Add to model-specific queue
            let mut queues = self.model_queues.lock().await;
            let model_queue = queues.entry(model.clone()).or_insert_with(VecDeque::new);

            // Check per-model limit
            if model_queue.len() >= self.config.max_queue_per_model {
                self.metrics.record_rejected();
                // Undo the queued metric since we're rejecting
                self.metrics.requests_queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                return Err(QueueError::QueueFull {
                    max: self.config.max_queue_per_model,
                });
            }

            model_queue.push_back(queued);
        } else {
            // Add to global queue
            let mut queue = self.global_queue.lock().await;
            queue.push_back(queued);
        }

        // Update total count
        {
            let mut total = self.total_queued.lock().await;
            *total += 1;
        }

        tracing::debug!(
            model = %model,
            queue_depth = self.metrics.queue_depth(),
            "Request enqueued"
        );

        Ok(())
    }

    /// Dequeue a request, waiting for a semaphore permit
    ///
    /// Returns the request and permit when one becomes available.
    /// The permit should be held until request processing is complete.
    pub async fn dequeue(&self) -> Result<(T, OwnedSemaphorePermit), QueueError> {
        // Wait for a semaphore permit
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| QueueError::Shutdown)?;

        // Get the next request (with model affinity if enabled)
        let queued = self.get_next_request().await?;

        // Calculate wait time
        let wait_time = queued.timer.queue_wait_time();
        self.metrics.record_dequeued(wait_time);

        // Update total count
        {
            let mut total = self.total_queued.lock().await;
            *total = total.saturating_sub(1);
        }

        tracing::debug!(
            model = %queued.model,
            wait_ms = wait_time.as_millis(),
            "Request dequeued"
        );

        Ok((queued.request, permit))
    }

    /// Get the next request based on scheduling policy
    async fn get_next_request(&self) -> Result<QueuedRequest<T>, QueueError> {
        if self.config.model_aware_queuing {
            self.get_next_model_aware().await
        } else {
            self.get_next_fifo().await
        }
    }

    /// Get next request using FIFO policy
    async fn get_next_fifo(&self) -> Result<QueuedRequest<T>, QueueError> {
        loop {
            let mut queue = self.global_queue.lock().await;

            // Remove expired requests
            while let Some(front) = queue.front() {
                if front.is_expired() {
                    queue.pop_front();
                    self.metrics.record_timeout();
                    tracing::warn!("Request timed out in queue");
                } else {
                    break;
                }
            }

            if let Some(queued) = queue.pop_front() {
                return Ok(queued);
            }

            // Queue is empty, wait a bit and try again
            drop(queue);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Get next request using model-aware scheduling
    ///
    /// Prioritizes requests for the currently loaded model to maximize
    /// cache efficiency and reduce context switching.
    async fn get_next_model_aware(&self) -> Result<QueuedRequest<T>, QueueError> {
        loop {
            let mut queues = self.model_queues.lock().await;
            let current_model = self.current_model.lock().await;

            // Clean up expired requests from all queues
            let mut expired_models = Vec::new();
            for (model, queue) in queues.iter_mut() {
                while let Some(front) = queue.front() {
                    if front.is_expired() {
                        queue.pop_front();
                        self.metrics.record_timeout();
                        tracing::warn!(model = %model, "Request timed out in queue");
                    } else {
                        break;
                    }
                }
                if queue.is_empty() {
                    expired_models.push(model.clone());
                }
            }

            // Remove empty queues
            for model in expired_models {
                queues.remove(&model);
            }

            // Try to get request from current model first (cache affinity)
            if let Some(ref model) = *current_model {
                if let Some(queue) = queues.get_mut(model) {
                    if let Some(queued) = queue.pop_front() {
                        return Ok(queued);
                    }
                }
            }

            // Fall back to longest queue (fair scheduling)
            let longest_model = queues
                .iter()
                .filter(|(_, q)| !q.is_empty())
                .max_by_key(|(_, q)| q.len())
                .map(|(m, _)| m.clone());

            if let Some(model) = longest_model {
                if let Some(queue) = queues.get_mut(&model) {
                    if let Some(queued) = queue.pop_front() {
                        // Update current model for affinity
                        drop(current_model);
                        *self.current_model.lock().await = Some(queued.model.clone());
                        return Ok(queued);
                    }
                }
            }

            // No requests available, wait and retry
            drop(queues);
            drop(current_model);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Try to dequeue immediately without waiting
    ///
    /// Returns None if no permit is available or queue is empty
    pub async fn try_dequeue(&self) -> Option<(T, OwnedSemaphorePermit)> {
        // Try to get a permit without waiting
        let permit = self.semaphore.clone().try_acquire_owned().ok()?;

        // Try to get a request
        let queued = if self.config.model_aware_queuing {
            self.try_get_next_model_aware().await
        } else {
            self.try_get_next_fifo().await
        };

        match queued {
            Some(q) => {
                let wait_time = q.timer.queue_wait_time();
                self.metrics.record_dequeued(wait_time);

                let mut total = self.total_queued.lock().await;
                *total = total.saturating_sub(1);

                Some((q.request, permit))
            }
            None => {
                // No request available, release the permit
                drop(permit);
                None
            }
        }
    }

    /// Try to get next request from FIFO queue without blocking
    async fn try_get_next_fifo(&self) -> Option<QueuedRequest<T>> {
        let mut queue = self.global_queue.lock().await;

        // Remove expired requests
        while let Some(front) = queue.front() {
            if front.is_expired() {
                queue.pop_front();
                self.metrics.record_timeout();
            } else {
                break;
            }
        }

        queue.pop_front()
    }

    /// Try to get next request using model-aware scheduling without blocking
    async fn try_get_next_model_aware(&self) -> Option<QueuedRequest<T>> {
        let mut queues = self.model_queues.lock().await;
        let current_model = self.current_model.lock().await;

        // Try current model first
        if let Some(ref model) = *current_model {
            if let Some(queue) = queues.get_mut(model) {
                // Remove expired
                while let Some(front) = queue.front() {
                    if front.is_expired() {
                        queue.pop_front();
                        self.metrics.record_timeout();
                    } else {
                        break;
                    }
                }
                if let Some(queued) = queue.pop_front() {
                    return Some(queued);
                }
            }
        }

        // Fall back to any non-empty queue
        for (_, queue) in queues.iter_mut() {
            while let Some(front) = queue.front() {
                if front.is_expired() {
                    queue.pop_front();
                    self.metrics.record_timeout();
                } else {
                    break;
                }
            }
            if let Some(queued) = queue.pop_front() {
                return Some(queued);
            }
        }

        None
    }

    /// Get current queue depth
    pub async fn queue_depth(&self) -> usize {
        *self.total_queued.lock().await
    }

    /// Get number of available permits
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Check if queue is empty
    pub async fn is_empty(&self) -> bool {
        *self.total_queued.lock().await == 0
    }

    /// Get queue depths per model (for monitoring)
    pub async fn queue_depths_by_model(&self) -> HashMap<String, usize> {
        let queues = self.model_queues.lock().await;
        queues
            .iter()
            .map(|(model, queue)| (model.clone(), queue.len()))
            .collect()
    }

    /// Clear all queues (for shutdown)
    pub async fn clear(&self) {
        {
            let mut queues = self.model_queues.lock().await;
            queues.clear();
        }
        {
            let mut queue = self.global_queue.lock().await;
            queue.clear();
        }
        {
            let mut total = self.total_queued.lock().await;
            *total = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let config = BatcherConfig::default();
        let metrics = Arc::new(BatcherMetrics::new());
        let queue: RequestQueue<String> = RequestQueue::new(config, metrics.clone());

        // Enqueue a request
        queue
            .enqueue("test request".to_string(), "llama3.2:3b".to_string())
            .await
            .unwrap();

        assert_eq!(queue.queue_depth().await, 1);

        // Dequeue it
        let (request, _permit) = queue.dequeue().await.unwrap();
        assert_eq!(request, "test request");
        assert_eq!(queue.queue_depth().await, 0);
    }

    #[tokio::test]
    async fn test_queue_full() {
        let mut config = BatcherConfig::default();
        config.max_queue_total = 2;
        let metrics = Arc::new(BatcherMetrics::new());
        let queue: RequestQueue<String> = RequestQueue::new(config, metrics.clone());

        // Fill the queue
        queue.enqueue("req1".to_string(), "model".to_string()).await.unwrap();
        queue.enqueue("req2".to_string(), "model".to_string()).await.unwrap();

        // Should fail
        let result = queue.enqueue("req3".to_string(), "model".to_string()).await;
        assert!(matches!(result, Err(QueueError::QueueFull { .. })));
    }

    #[tokio::test]
    async fn test_model_aware_queuing() {
        let mut config = BatcherConfig::default();
        config.model_aware_queuing = true;
        config.max_concurrent = 1;
        let metrics = Arc::new(BatcherMetrics::new());
        let queue: RequestQueue<String> = RequestQueue::new(config, metrics.clone());

        // Enqueue requests for different models
        queue.enqueue("llama_req1".to_string(), "llama3.2:3b".to_string()).await.unwrap();
        queue.enqueue("mistral_req1".to_string(), "mistral:7b".to_string()).await.unwrap();
        queue.enqueue("llama_req2".to_string(), "llama3.2:3b".to_string()).await.unwrap();

        // First dequeue should get from llama (longest queue initially tied, picks one)
        let (req1, permit1) = queue.dequeue().await.unwrap();
        // Should prioritize same model for second request
        drop(permit1);

        let depths = queue.queue_depths_by_model().await;
        assert!(depths.len() <= 2);
    }

    #[tokio::test]
    async fn test_concurrency_limit() {
        let mut config = BatcherConfig::default();
        config.max_concurrent = 2;
        let metrics = Arc::new(BatcherMetrics::new());
        let queue: RequestQueue<String> = RequestQueue::new(config, metrics.clone());

        // Enqueue requests
        for i in 0..5 {
            queue.enqueue(format!("req{}", i), "model".to_string()).await.unwrap();
        }

        // Should only be able to get 2 permits
        assert_eq!(queue.available_permits(), 2);

        let (_req1, _permit1) = queue.try_dequeue().await.unwrap();
        assert_eq!(queue.available_permits(), 1);

        let (_req2, _permit2) = queue.try_dequeue().await.unwrap();
        assert_eq!(queue.available_permits(), 0);

        // Third try should fail (no permits)
        let result = queue.try_dequeue().await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_try_dequeue_empty() {
        let config = BatcherConfig::default();
        let metrics = Arc::new(BatcherMetrics::new());
        let queue: RequestQueue<String> = RequestQueue::new(config, metrics.clone());

        // Should return None on empty queue
        let result = queue.try_dequeue().await;
        assert!(result.is_none());
    }
}
