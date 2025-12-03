//! Exact match cache using LRU eviction and TTL expiration.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use xxhash_rust::xxh3::xxh3_64;

use crate::types::openai::{ChatCompletionRequest, ChatCompletionResponse};

/// A cached response with timestamp for TTL checking
struct CachedEntry {
    response: ChatCompletionResponse,
    created_at: Instant,
}

impl CachedEntry {
    fn new(response: ChatCompletionResponse) -> Self {
        Self {
            response,
            created_at: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Exact match cache using hash-based lookup
///
/// This cache provides O(1) lookup for identical queries using xxh3 hashing.
/// It uses LRU eviction when the cache is full and TTL-based expiration.
pub struct ExactCache {
    cache: LruCache<u64, CachedEntry>,
    ttl: Duration,
}

impl ExactCache {
    /// Create a new exact cache with the given capacity and TTL
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            cache: LruCache::new(capacity),
            ttl,
        }
    }

    /// Hash a request to create a cache key
    ///
    /// The hash includes:
    /// - Model name
    /// - All message contents and roles
    /// - Temperature and max_tokens (if set)
    fn hash_request(&self, request: &ChatCompletionRequest) -> u64 {
        // Create a canonical representation for hashing
        let mut hasher_input = String::new();

        // Include model
        hasher_input.push_str(&request.model);
        hasher_input.push('\0');

        // Include messages
        for msg in &request.messages {
            hasher_input.push_str(&format!("{:?}", msg.role));
            hasher_input.push(':');
            if let Some(ref content) = msg.content {
                hasher_input.push_str(content);
            }
            hasher_input.push('\n');
        }

        // Include parameters that affect output
        if let Some(temp) = request.temperature {
            hasher_input.push_str(&format!("temp:{:.2}", temp));
        }
        if let Some(max_tokens) = request.max_tokens {
            hasher_input.push_str(&format!("max:{}", max_tokens));
        }
        if let Some(top_p) = request.top_p {
            hasher_input.push_str(&format!("top_p:{:.2}", top_p));
        }

        // Hash using xxh3 (very fast)
        xxh3_64(hasher_input.as_bytes())
    }

    /// Try to get a cached response for the request
    pub fn get(&mut self, request: &ChatCompletionRequest) -> Option<ChatCompletionResponse> {
        let key = self.hash_request(request);

        // Check if we have this entry
        if let Some(entry) = self.cache.get(&key) {
            if entry.is_expired(self.ttl) {
                // Entry expired, remove it
                self.cache.pop(&key);
                return None;
            }
            return Some(entry.response.clone());
        }

        None
    }

    /// Store a response in the cache
    pub fn put(&mut self, request: &ChatCompletionRequest, response: ChatCompletionResponse) {
        let key = self.hash_request(request);
        self.cache.put(key, CachedEntry::new(response));
    }

    /// Get the current number of entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Remove expired entries (call periodically for cleanup)
    pub fn evict_expired(&mut self) -> usize {
        let ttl = self.ttl;
        let mut expired_keys = Vec::new();

        // Find expired entries
        for (key, entry) in self.cache.iter() {
            if entry.is_expired(ttl) {
                expired_keys.push(*key);
            }
        }

        // Remove them
        let count = expired_keys.len();
        for key in expired_keys {
            self.cache.pop(&key);
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ChatMessage, Choice, Role, Usage};

    fn create_request(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "llama3.2:3b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(content.to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            n: None,
            user: None,
            seed: None,
            response_format: None,
        }
    }

    fn create_response(content: &str) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "llama3.2:3b".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: Some(content.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            system_fingerprint: None,
        }
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = ExactCache::new(100, Duration::from_secs(3600));

        let request = create_request("Hello");
        let response = create_response("Hi there!");

        cache.put(&request, response.clone());

        let cached = cache.get(&request);
        assert!(cached.is_some());
        assert_eq!(
            cached.unwrap().choices[0].message.content,
            Some("Hi there!".to_string())
        );
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ExactCache::new(100, Duration::from_secs(3600));

        let request = create_request("Hello");
        assert!(cache.get(&request).is_none());
    }

    #[test]
    fn test_different_requests() {
        let mut cache = ExactCache::new(100, Duration::from_secs(3600));

        let request1 = create_request("Hello");
        let request2 = create_request("Goodbye");
        let response1 = create_response("Hi!");
        let response2 = create_response("Bye!");

        cache.put(&request1, response1);
        cache.put(&request2, response2);

        let cached1 = cache.get(&request1);
        let cached2 = cache.get(&request2);

        assert!(cached1.is_some());
        assert!(cached2.is_some());
        assert_eq!(
            cached1.unwrap().choices[0].message.content,
            Some("Hi!".to_string())
        );
        assert_eq!(
            cached2.unwrap().choices[0].message.content,
            Some("Bye!".to_string())
        );
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = ExactCache::new(2, Duration::from_secs(3600));

        let request1 = create_request("One");
        let request2 = create_request("Two");
        let request3 = create_request("Three");

        cache.put(&request1, create_response("1"));
        cache.put(&request2, create_response("2"));
        cache.put(&request3, create_response("3")); // Should evict request1

        assert!(cache.get(&request1).is_none()); // Evicted
        assert!(cache.get(&request2).is_some());
        assert!(cache.get(&request3).is_some());
    }

    #[test]
    fn test_ttl_expiration() {
        let mut cache = ExactCache::new(100, Duration::from_millis(1));

        let request = create_request("Hello");
        cache.put(&request, create_response("Hi!"));

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(10));

        assert!(cache.get(&request).is_none());
    }

    #[test]
    fn test_same_content_different_model() {
        let mut cache = ExactCache::new(100, Duration::from_secs(3600));

        let mut request1 = create_request("Hello");
        request1.model = "llama3.2:3b".to_string();

        let mut request2 = create_request("Hello");
        request2.model = "mistral:7b".to_string();

        cache.put(&request1, create_response("Llama says hi"));
        cache.put(&request2, create_response("Mistral says hi"));

        let cached1 = cache.get(&request1);
        let cached2 = cache.get(&request2);

        assert_ne!(
            cached1.unwrap().choices[0].message.content,
            cached2.unwrap().choices[0].message.content
        );
    }

    #[test]
    fn test_temperature_affects_hash() {
        let mut cache = ExactCache::new(100, Duration::from_secs(3600));

        let mut request1 = create_request("Hello");
        request1.temperature = Some(0.0);

        let mut request2 = create_request("Hello");
        request2.temperature = Some(1.0);

        cache.put(&request1, create_response("Deterministic"));
        cache.put(&request2, create_response("Creative"));

        // Different temperatures should have different cache entries
        assert_eq!(cache.len(), 2);
    }
}
