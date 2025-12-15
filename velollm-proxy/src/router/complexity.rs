//! Complexity analysis for request routing.
//!
//! This module analyzes chat requests to determine their complexity,
//! which is used to route requests to appropriately-sized models.

use crate::types::openai::{ChatCompletionRequest, Role};

/// Complexity score with breakdown by factor.
#[derive(Debug, Clone, Default)]
pub struct ComplexityScore {
    /// Overall complexity score (0.0 to 1.0).
    pub total: f32,

    /// Breakdown by factor (for debugging and metrics).
    pub factors: ComplexityFactors,
}

/// Individual complexity factors.
#[derive(Debug, Clone, Default)]
pub struct ComplexityFactors {
    /// Score from prompt length (longer = more complex).
    pub length: f32,

    /// Score from tool usage (tools = more complex).
    pub tools: f32,

    /// Score from conversation depth (more turns = more complex).
    pub conversation: f32,

    /// Score from system prompt complexity.
    pub system: f32,

    /// Score from detected task type (coding, reasoning, etc.).
    pub task_type: f32,
}

/// Complexity analyzer for chat requests.
#[derive(Debug, Clone)]
pub struct ComplexityAnalyzer {
    /// Weight for length factor.
    pub length_weight: f32,

    /// Weight for tools factor.
    pub tools_weight: f32,

    /// Weight for conversation factor.
    pub conversation_weight: f32,

    /// Weight for system prompt factor.
    pub system_weight: f32,

    /// Weight for task type factor.
    pub task_type_weight: f32,

    /// Character count considered "long" for a prompt.
    pub long_prompt_chars: usize,

    /// Number of messages considered "deep" conversation.
    pub deep_conversation_turns: usize,
}

impl Default for ComplexityAnalyzer {
    fn default() -> Self {
        Self {
            length_weight: 0.15,
            tools_weight: 0.20,
            conversation_weight: 0.10,
            system_weight: 0.15,
            task_type_weight: 0.40, // Task type is the most important factor
            long_prompt_chars: 2000,
            deep_conversation_turns: 10,
        }
    }
}

impl ComplexityAnalyzer {
    /// Analyze the complexity of a chat request.
    pub fn analyze(&self, request: &ChatCompletionRequest) -> ComplexityScore {
        let factors = ComplexityFactors {
            length: self.analyze_length(request),
            tools: self.analyze_tools(request),
            conversation: self.analyze_conversation(request),
            system: self.analyze_system_prompt(request),
            task_type: self.analyze_task_type(request),
        };

        let total = factors.length * self.length_weight
            + factors.tools * self.tools_weight
            + factors.conversation * self.conversation_weight
            + factors.system * self.system_weight
            + factors.task_type * self.task_type_weight;

        ComplexityScore { total: total.clamp(0.0, 1.0), factors }
    }

    /// Analyze prompt length complexity.
    fn analyze_length(&self, request: &ChatCompletionRequest) -> f32 {
        let total_chars: usize = request
            .messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len())
            .sum();

        // Sigmoid-like scaling
        let ratio = total_chars as f32 / self.long_prompt_chars as f32;
        (ratio / (1.0 + ratio)).min(1.0)
    }

    /// Analyze tool usage complexity.
    fn analyze_tools(&self, request: &ChatCompletionRequest) -> f32 {
        match &request.tools {
            None => 0.0,
            Some(tools) if tools.is_empty() => 0.0,
            Some(tools) => {
                // More tools = more complex
                let tool_count = tools.len();
                let base_score = 0.5; // Having tools at all adds complexity

                // Additional complexity for many tools
                let tool_bonus = (tool_count as f32 / 10.0).min(0.5);

                // Check for complex tool schemas
                let schema_complexity: f32 = tools
                    .iter()
                    .map(|t| {
                        let params = &t.function.parameters;
                        // Count nested properties
                        if let Some(obj) = params.as_object() {
                            let props = obj.get("properties").and_then(|p| p.as_object());
                            let prop_count = props.map(|p| p.len()).unwrap_or(0);
                            (prop_count as f32 / 20.0).min(0.3)
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>()
                    / tools.len().max(1) as f32;

                (base_score + tool_bonus + schema_complexity).min(1.0)
            }
        }
    }

    /// Analyze conversation depth complexity.
    fn analyze_conversation(&self, request: &ChatCompletionRequest) -> f32 {
        let user_messages = request
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .count();

        // More turns = more context to track
        let turn_ratio = user_messages as f32 / self.deep_conversation_turns as f32;

        // Also consider tool call/response pairs
        let tool_responses = request
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::Tool))
            .count();

        let tool_bonus = (tool_responses as f32 / 5.0).min(0.3);

        (turn_ratio + tool_bonus).min(1.0)
    }

    /// Analyze system prompt complexity.
    fn analyze_system_prompt(&self, request: &ChatCompletionRequest) -> f32 {
        let system_message = request
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::System));

        match system_message {
            None => 0.0,
            Some(msg) => {
                let content = msg.content.as_deref().unwrap_or("");
                let length_score = (content.len() as f32 / 2000.0).min(0.5);

                // Check for complex instructions
                let complexity_keywords = [
                    "step by step",
                    "analyze",
                    "evaluate",
                    "compare",
                    "reasoning",
                    "chain of thought",
                    "json",
                    "format",
                    "structured",
                    "expert",
                ];

                let keyword_score = complexity_keywords
                    .iter()
                    .filter(|kw| content.to_lowercase().contains(*kw))
                    .count() as f32
                    / complexity_keywords.len() as f32;

                (length_score + keyword_score * 0.5).min(1.0)
            }
        }
    }

    /// Analyze task type complexity based on content.
    fn analyze_task_type(&self, request: &ChatCompletionRequest) -> f32 {
        // Get the last user message (the actual query)
        let last_user_message = request
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::User))
            .and_then(|m| m.content.as_deref())
            .unwrap_or("");

        let lower = last_user_message.to_lowercase();

        // High complexity tasks
        let high_complexity_patterns = [
            "write code",
            "implement",
            "debug",
            "fix this",
            "refactor",
            "analyze",
            "explain why",
            "compare and contrast",
            "evaluate",
            "design",
            "architect",
            "optimize",
            "algorithm",
            "data structure",
            "regex",
            "sql query",
            "performance",
        ];

        // Medium complexity tasks
        let medium_complexity_patterns = [
            "explain",
            "how does",
            "what is",
            "summarize",
            "translate",
            "convert",
            "list",
            "describe",
            "help me",
        ];

        // Low complexity tasks
        let low_complexity_patterns = [
            "hello",
            "hi",
            "thanks",
            "thank you",
            "yes",
            "no",
            "ok",
            "okay",
            "bye",
            "goodbye",
        ];

        // Check for code blocks (indicates technical task)
        let has_code = last_user_message.contains("```");

        // Score based on patterns
        let high_matches = high_complexity_patterns
            .iter()
            .filter(|p| lower.contains(*p))
            .count();
        let medium_matches = medium_complexity_patterns
            .iter()
            .filter(|p| lower.contains(*p))
            .count();
        let low_matches = low_complexity_patterns
            .iter()
            .filter(|p| lower.contains(*p))
            .count();

        if low_matches > 0 && high_matches == 0 && medium_matches == 0 {
            return 0.1;
        }

        let mut score = 0.2; // Base score

        if high_matches > 0 {
            // High complexity tasks get significant boost
            score += 0.5 + (high_matches as f32 * 0.1).min(0.3);
        }

        if medium_matches > 0 {
            score += 0.3;
        }

        if has_code {
            score += 0.3;
        }

        // Boost for longer prompts (more detailed instructions)
        let prompt_len = last_user_message.len();
        if prompt_len > 100 {
            score += 0.1;
        }
        if prompt_len > 200 {
            score += 0.1;
        }

        score.min(1.0)
    }
}

/// Model tier based on complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTier {
    /// Small model for simple queries.
    Small,
    /// Medium model for moderate complexity.
    Medium,
    /// Large model for complex queries.
    Large,
}

impl ModelTier {
    /// Determine tier from complexity score and thresholds.
    pub fn from_score(score: f32, small_threshold: f32, large_threshold: f32) -> Self {
        if score < small_threshold {
            ModelTier::Small
        } else if score > large_threshold {
            ModelTier::Large
        } else {
            ModelTier::Medium
        }
    }
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelTier::Small => write!(f, "small"),
            ModelTier::Medium => write!(f, "medium"),
            ModelTier::Large => write!(f, "large"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ChatMessage, FunctionDef, Tool};

    fn make_request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        }
    }

    fn user_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn system_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::System,
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_simple_greeting_low_complexity() {
        let analyzer = ComplexityAnalyzer::default();
        let request = make_request(vec![user_message("Hello!")]);

        let score = analyzer.analyze(&request);
        assert!(score.total < 0.3, "Simple greeting should be low complexity");
    }

    #[test]
    fn test_coding_task_high_complexity() {
        let analyzer = ComplexityAnalyzer::default();
        let request = make_request(vec![user_message(
            "Write code to implement a binary search tree with insert, delete, and search operations",
        )]);

        let score = analyzer.analyze(&request);
        // Coding tasks should be at least medium complexity (above small threshold of 0.3)
        assert!(
            score.total > 0.3,
            "Coding task should be at least medium complexity: {}",
            score.total
        );
        // Task type factor should be significant
        assert!(
            score.factors.task_type > 0.5,
            "Task type factor should detect coding task: {}",
            score.factors.task_type
        );
    }

    #[test]
    fn test_tools_increase_complexity() {
        let analyzer = ComplexityAnalyzer::default();

        let request_no_tools = make_request(vec![user_message("What's the weather?")]);

        let mut request_with_tools = make_request(vec![user_message("What's the weather?")]);
        request_with_tools.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }),
            },
        }]);

        let score_no_tools = analyzer.analyze(&request_no_tools);
        let score_with_tools = analyzer.analyze(&request_with_tools);

        assert!(
            score_with_tools.total > score_no_tools.total,
            "Tools should increase complexity"
        );
    }

    #[test]
    fn test_long_conversation_increases_complexity() {
        let analyzer = ComplexityAnalyzer::default();

        let short_conv = make_request(vec![user_message("Hello")]);

        let long_conv = make_request(vec![
            system_message("You are a helpful assistant."),
            user_message("What is Rust?"),
            ChatMessage {
                role: Role::Assistant,
                content: Some("Rust is a programming language...".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            user_message("How does ownership work?"),
            ChatMessage {
                role: Role::Assistant,
                content: Some("Ownership in Rust...".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            user_message("Can you show me an example?"),
        ]);

        let short_score = analyzer.analyze(&short_conv);
        let long_score = analyzer.analyze(&long_conv);

        assert!(
            long_score.total > short_score.total,
            "Longer conversations should be more complex"
        );
    }

    #[test]
    fn test_model_tier_from_score() {
        let small = 0.3;
        let large = 0.7;

        assert_eq!(ModelTier::from_score(0.1, small, large), ModelTier::Small);
        assert_eq!(ModelTier::from_score(0.5, small, large), ModelTier::Medium);
        assert_eq!(ModelTier::from_score(0.9, small, large), ModelTier::Large);
    }

    #[test]
    fn test_system_prompt_complexity() {
        let analyzer = ComplexityAnalyzer::default();

        let simple_system = make_request(vec![
            system_message("You are a helpful assistant."),
            user_message("Hello"),
        ]);

        let complex_system = make_request(vec![
            system_message(
                "You are an expert software engineer. Analyze code step by step. \
                 Use chain of thought reasoning. Format your response as JSON. \
                 Evaluate all edge cases and compare different approaches.",
            ),
            user_message("Hello"),
        ]);

        let simple_score = analyzer.analyze(&simple_system);
        let complex_score = analyzer.analyze(&complex_system);

        assert!(
            complex_score.factors.system > simple_score.factors.system,
            "Complex system prompts should increase complexity"
        );
    }
}
