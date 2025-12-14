//! Pattern-based query predictor for speculative prefetch.

use crate::types::ollama::Message;

/// Types of queries we can detect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Asking for explanation or definition
    Clarification,
    /// Asking for code implementation
    Code,
    /// Comparing two or more things
    Comparison,
    /// Debugging or troubleshooting
    Debug,
    /// General question
    General,
}

/// A predicted follow-up query
#[derive(Debug, Clone)]
pub struct PredictedQuery {
    /// The predicted follow-up message
    pub query: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Type of query detected
    pub query_type: QueryType,
}

/// Pattern for detecting query types
struct QueryPattern {
    /// Keywords that indicate this pattern
    keywords: &'static [&'static str],
    /// Query type this pattern detects
    query_type: QueryType,
    /// Follow-up templates with placeholders
    followups: &'static [&'static str],
    /// Base confidence for this pattern
    base_confidence: f32,
}

/// Pattern-based query predictor
pub struct QueryPredictor {
    /// Patterns for detecting query types
    patterns: Vec<QueryPattern>,
}

impl Default for QueryPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPredictor {
    /// Create a new query predictor with default patterns
    pub fn new() -> Self {
        let patterns = vec![
            // Clarification patterns
            QueryPattern {
                keywords: &[
                    "what is",
                    "explain",
                    "how does",
                    "what does",
                    "define",
                    "describe",
                ],
                query_type: QueryType::Clarification,
                followups: &[
                    "Can you give me an example?",
                    "Why is this important?",
                    "How is this different from",
                    "What are the main use cases?",
                ],
                base_confidence: 0.75,
            },
            // Code patterns
            QueryPattern {
                keywords: &[
                    "write",
                    "implement",
                    "create a function",
                    "code",
                    "program",
                    "script",
                    "function that",
                ],
                query_type: QueryType::Code,
                followups: &[
                    "Can you add error handling?",
                    "How would I test this?",
                    "Can you optimize this?",
                    "Add comments to explain",
                ],
                base_confidence: 0.8,
            },
            // Comparison patterns
            QueryPattern {
                keywords: &[
                    "vs",
                    "versus",
                    "difference between",
                    "compare",
                    "which is better",
                    "pros and cons",
                ],
                query_type: QueryType::Comparison,
                followups: &[
                    "Which one should I use for my project?",
                    "What are the performance differences?",
                    "Which is easier to learn?",
                ],
                base_confidence: 0.7,
            },
            // Debug patterns
            QueryPattern {
                keywords: &[
                    "error",
                    "bug",
                    "doesn't work",
                    "not working",
                    "failed",
                    "exception",
                    "crash",
                    "fix",
                ],
                query_type: QueryType::Debug,
                followups: &[
                    "What could be causing this?",
                    "How can I debug this further?",
                    "What logs should I check?",
                    "Is there a workaround?",
                ],
                base_confidence: 0.85,
            },
        ];

        Self { patterns }
    }

    /// Predict follow-up queries based on conversation context
    pub fn predict(&self, messages: &[Message], max_predictions: usize) -> Vec<PredictedQuery> {
        if messages.is_empty() || max_predictions == 0 {
            return Vec::new();
        }

        // Get the last user message for analysis
        let last_user_msg = messages.iter().rev().find(|m| m.role == "user");

        let Some(msg) = last_user_msg else {
            return Vec::new();
        };

        let user_query = &msg.content;

        let query_lower = user_query.to_lowercase();
        let mut predictions = Vec::new();

        // Check each pattern
        for pattern in &self.patterns {
            if self.matches_pattern(&query_lower, pattern) {
                // Generate predictions from this pattern's followups
                for (i, followup) in pattern.followups.iter().enumerate() {
                    if predictions.len() >= max_predictions {
                        break;
                    }

                    // Adjust confidence based on position (earlier = higher confidence)
                    let position_factor = 1.0 - (i as f32 * 0.1);
                    let confidence = pattern.base_confidence * position_factor;

                    predictions.push(PredictedQuery {
                        query: followup.to_string(),
                        confidence,
                        query_type: pattern.query_type,
                    });
                }
            }
        }

        // Sort by confidence and take top N
        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        predictions.truncate(max_predictions);

        predictions
    }

    /// Check if a query matches a pattern
    fn matches_pattern(&self, query: &str, pattern: &QueryPattern) -> bool {
        pattern.keywords.iter().any(|kw| query.contains(kw))
    }

    /// Detect the type of a query
    pub fn detect_query_type(&self, query: &str) -> QueryType {
        let query_lower = query.to_lowercase();

        for pattern in &self.patterns {
            if self.matches_pattern(&query_lower, pattern) {
                return pattern.query_type;
            }
        }

        QueryType::General
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    #[test]
    fn test_clarification_detection() {
        let predictor = QueryPredictor::new();
        let messages = vec![make_message("user", "What is a closure in Rust?")];

        let predictions = predictor.predict(&messages, 2);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].query_type, QueryType::Clarification);
        assert!(predictions[0].confidence >= 0.7);
    }

    #[test]
    fn test_code_detection() {
        let predictor = QueryPredictor::new();
        let messages = vec![make_message("user", "Write a function to sort an array")];

        let predictions = predictor.predict(&messages, 2);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].query_type, QueryType::Code);
    }

    #[test]
    fn test_comparison_detection() {
        let predictor = QueryPredictor::new();
        let messages = vec![make_message("user", "Compare Vec vs array in Rust")];

        let predictions = predictor.predict(&messages, 2);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].query_type, QueryType::Comparison);
    }

    #[test]
    fn test_debug_detection() {
        let predictor = QueryPredictor::new();
        let messages = vec![make_message(
            "user",
            "I'm getting an error when I try to compile",
        )];

        let predictions = predictor.predict(&messages, 2);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].query_type, QueryType::Debug);
    }

    #[test]
    fn test_max_predictions() {
        let predictor = QueryPredictor::new();
        let messages = vec![make_message("user", "What is Rust?")];

        let predictions = predictor.predict(&messages, 1);
        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_empty_messages() {
        let predictor = QueryPredictor::new();
        let predictions = predictor.predict(&[], 5);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_no_user_message() {
        let predictor = QueryPredictor::new();
        let messages = vec![make_message("system", "You are a helpful assistant")];

        let predictions = predictor.predict(&messages, 5);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_detect_query_type() {
        let predictor = QueryPredictor::new();

        assert_eq!(predictor.detect_query_type("What is Rust?"), QueryType::Clarification);
        assert_eq!(predictor.detect_query_type("Write a function"), QueryType::Code);
        assert_eq!(predictor.detect_query_type("Compare A vs B"), QueryType::Comparison);
        assert_eq!(predictor.detect_query_type("I have an error"), QueryType::Debug);
        assert_eq!(predictor.detect_query_type("Hello world"), QueryType::General);
    }
}
