//! Prefetch task queue.

use std::collections::VecDeque;
use std::time::Instant;

use tokio::sync::Mutex;

use super::predictor::{PredictedQuery, QueryType};
use crate::types::ollama::Message;

/// A prefetch task to be executed
#[derive(Debug, Clone)]
pub struct PrefetchTask {
    /// Messages context (original conversation + predicted follow-up)
    pub messages: Vec<Message>,
    /// Model to use for generation
    pub model: String,
    /// Predicted query that triggered this task
    pub prediction: PredictedQuery,
    /// When this task was created
    pub created_at: Instant,
}

impl PrefetchTask {
    /// Create a new prefetch task
    pub fn new(messages: Vec<Message>, model: String, prediction: PredictedQuery) -> Self {
        Self { messages, model, prediction, created_at: Instant::now() }
    }

    /// Get the age of this task
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

/// Queue for prefetch tasks with priority by confidence
pub struct PrefetchQueue {
    /// Tasks ordered by confidence (highest first)
    tasks: Mutex<VecDeque<PrefetchTask>>,
    /// Maximum queue size
    max_size: usize,
}

impl PrefetchQueue {
    /// Create a new prefetch queue
    pub fn new(max_size: usize) -> Self {
        Self { tasks: Mutex::new(VecDeque::with_capacity(max_size)), max_size }
    }

    /// Add a task to the queue
    ///
    /// Returns true if task was added, false if queue is full
    pub async fn push(&self, task: PrefetchTask) -> bool {
        let mut tasks = self.tasks.lock().await;

        if tasks.len() >= self.max_size {
            // Queue is full, check if new task has higher priority
            if let Some(last) = tasks.back() {
                if task.prediction.confidence > last.prediction.confidence {
                    // Replace lowest priority task
                    tasks.pop_back();
                } else {
                    return false;
                }
            }
        }

        // Insert in priority order (highest confidence first)
        let pos = tasks
            .iter()
            .position(|t| t.prediction.confidence < task.prediction.confidence)
            .unwrap_or(tasks.len());

        tasks.insert(pos, task);
        true
    }

    /// Pop the highest priority task
    pub async fn pop(&self) -> Option<PrefetchTask> {
        let mut tasks = self.tasks.lock().await;
        tasks.pop_front()
    }

    /// Get current queue size
    pub async fn len(&self) -> usize {
        let tasks = self.tasks.lock().await;
        tasks.len()
    }

    /// Check if queue is empty
    pub async fn is_empty(&self) -> bool {
        let tasks = self.tasks.lock().await;
        tasks.is_empty()
    }

    /// Remove tasks older than the given age
    pub async fn remove_expired(&self, max_age: std::time::Duration) -> usize {
        let mut tasks = self.tasks.lock().await;
        let before = tasks.len();

        tasks.retain(|t| t.age() < max_age);

        before - tasks.len()
    }

    /// Get tasks for a specific query type
    pub async fn tasks_for_type(&self, query_type: QueryType) -> Vec<PrefetchTask> {
        let tasks = self.tasks.lock().await;
        tasks
            .iter()
            .filter(|t| t.prediction.query_type == query_type)
            .cloned()
            .collect()
    }

    /// Clear all tasks
    pub async fn clear(&self) {
        let mut tasks = self.tasks.lock().await;
        tasks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefetch::predictor::QueryType;

    fn make_task(confidence: f32) -> PrefetchTask {
        PrefetchTask {
            messages: vec![Message {
                role: "user".to_string(),
                content: "test".to_string(),
                images: None,
                tool_calls: None,
            }],
            model: "test-model".to_string(),
            prediction: PredictedQuery {
                query: "test query".to_string(),
                confidence,
                query_type: QueryType::General,
            },
            created_at: Instant::now(),
        }
    }

    #[tokio::test]
    async fn test_push_and_pop() {
        let queue = PrefetchQueue::new(10);

        queue.push(make_task(0.5)).await;
        queue.push(make_task(0.8)).await;
        queue.push(make_task(0.3)).await;

        assert_eq!(queue.len().await, 3);

        // Should pop in priority order (highest first)
        let task1 = queue.pop().await.unwrap();
        assert!((task1.prediction.confidence - 0.8).abs() < 0.001);

        let task2 = queue.pop().await.unwrap();
        assert!((task2.prediction.confidence - 0.5).abs() < 0.001);

        let task3 = queue.pop().await.unwrap();
        assert!((task3.prediction.confidence - 0.3).abs() < 0.001);

        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn test_queue_full() {
        let queue = PrefetchQueue::new(2);

        assert!(queue.push(make_task(0.5)).await);
        assert!(queue.push(make_task(0.6)).await);

        // Queue is full, low priority task should be rejected
        assert!(!queue.push(make_task(0.4)).await);

        // High priority task should replace lowest
        assert!(queue.push(make_task(0.9)).await);
        assert_eq!(queue.len().await, 2);

        // Check highest priority tasks remain
        let task = queue.pop().await.unwrap();
        assert!((task.prediction.confidence - 0.9).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_clear() {
        let queue = PrefetchQueue::new(10);

        queue.push(make_task(0.5)).await;
        queue.push(make_task(0.6)).await;

        queue.clear().await;
        assert!(queue.is_empty().await);
    }
}
