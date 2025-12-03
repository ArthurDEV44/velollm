//! Ollama API types.
//!
//! Based on the official Ollama API documentation:
//! https://github.com/ollama/ollama/blob/main/docs/api.md
//!
//! Focused on Mistral and Llama models for tool calling support.

use serde::{Deserialize, Serialize};

/// Request body for POST /api/generate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Model name (required)
    pub model: String,

    /// The prompt to generate a response for
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// Text after the model response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// Base64-encoded images for multimodal models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    /// Format of the response: "json" or JSON schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,

    /// Model-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ModelOptions>,

    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Custom prompt template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,

    /// Conversation context from previous response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,

    /// Enable streaming (default: true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Disable prompt formatting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,

    /// How long to keep model loaded (e.g., "5m", "1h")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Response from POST /api/generate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Model name
    pub model: String,

    /// Timestamp
    pub created_at: String,

    /// Generated text (streaming: partial, final: complete)
    #[serde(default)]
    pub response: String,

    /// Whether generation is complete
    pub done: bool,

    /// Reason for completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,

    /// Conversation context for follow-up requests
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,

    /// Total generation time in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,

    /// Time loading the model in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,

    /// Number of tokens in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,

    /// Time evaluating the prompt in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,

    /// Number of tokens in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,

    /// Time generating the response in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Request body for POST /api/chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model name (required)
    pub model: String,

    /// Chat messages
    pub messages: Vec<Message>,

    /// Format of the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,

    /// Model-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ModelOptions>,

    /// Enable streaming (default: true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// How long to keep model loaded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,

    /// Tools available to the model (for function calling)
    /// Supported by Mistral and Llama 3.1+
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// Response from POST /api/chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Model name
    pub model: String,

    /// Timestamp
    pub created_at: String,

    /// The assistant's message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,

    /// Whether generation is complete
    pub done: bool,

    /// Reason for completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,

    /// Total generation time in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,

    /// Time loading the model in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,

    /// Number of tokens in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,

    /// Time evaluating the prompt in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,

    /// Number of tokens in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,

    /// Time generating the response in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role: "system", "user", "assistant", or "tool"
    pub role: String,

    /// Message content
    pub content: String,

    /// Images for multimodal (base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    /// Tool calls made by the assistant (Mistral/Llama format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Tool definition for function calling
/// Compatible with Mistral and Llama 3.1+
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type of tool (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,

    /// Description of what the function does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the function parameters
    pub parameters: serde_json::Value,
}

/// Tool call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Function call details
    pub function: FunctionCall,
}

/// Function call within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,

    /// Arguments as JSON object
    pub arguments: serde_json::Value,
}

/// Model-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOptions {
    /// Number of tokens to predict (-1 for infinite, -2 for fill context)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,

    /// Temperature (0.0-2.0, default: 0.8)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-k sampling (default: 40)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Top-p (nucleus) sampling (default: 0.9)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Repetition penalty (default: 1.1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,

    /// Seed for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Number of context tokens (default: 2048)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,

    /// Number of batch tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<i32>,

    /// Number of GPU layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
}

/// Response from GET /api/tags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagsResponse {
    /// List of available models
    pub models: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,

    /// When the model was last modified
    pub modified_at: String,

    /// Model size in bytes
    pub size: u64,

    /// Model digest
    pub digest: String,

    /// Model details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<ModelDetails>,
}

/// Model details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    /// Format (e.g., "gguf")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Model family (e.g., "llama", "mistral")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,

    /// Parameter size (e.g., "7B")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_size: Option<String>,

    /// Quantization level (e.g., "Q4_0")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_level: Option<String>,
}

impl ChatRequest {
    /// Create a new chat request with a single user message
    pub fn new(model: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: vec![Message {
                role: "user".to_string(),
                content: content.into(),
                images: None,
                tool_calls: None,
            }],
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
            tools: None,
        }
    }

    /// Check if this request uses tool calling
    pub fn has_tools(&self) -> bool {
        self.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false)
    }
}

impl GenerateRequest {
    /// Create a new generate request
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: Some(prompt.into()),
            suffix: None,
            images: None,
            format: None,
            options: None,
            system: None,
            template: None,
            context: None,
            stream: Some(false),
            raw: None,
            keep_alive: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialization() {
        let request = ChatRequest::new("llama3.2:3b", "Hello!");
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("llama3.2:3b"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_chat_request_with_tools() {
        let mut request = ChatRequest::new("mistral:7b", "What's the weather?");
        request.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the current weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }),
            },
        }]);

        assert!(request.has_tools());
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("get_weather"));
    }

    #[test]
    fn test_generate_request() {
        let request = GenerateRequest::new("llama3.2:3b", "Write a haiku");
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Write a haiku"));
    }
}
