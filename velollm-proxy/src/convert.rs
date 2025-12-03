//! Conversion layer between OpenAI and Ollama API formats.
//!
//! This module handles bidirectional conversion to enable OpenAI-compatible
//! clients to work with Ollama backend.
//!
//! Focused on Mistral and Llama models for reliable tool calling support.

use crate::types::ollama;
use crate::types::openai;
use tracing::debug;
use uuid::Uuid;

/// Convert OpenAI ChatCompletionRequest to Ollama ChatRequest
pub fn openai_to_ollama_chat(request: &openai::ChatCompletionRequest) -> ollama::ChatRequest {
    debug!(
        model = %request.model,
        messages = request.messages.len(),
        "Converting OpenAI request to Ollama format"
    );

    ollama::ChatRequest {
        model: request.model.clone(),
        messages: request
            .messages
            .iter()
            .map(convert_message_to_ollama)
            .collect(),
        format: request.response_format.as_ref().and_then(|rf| {
            if rf.format_type == "json_object" {
                Some(serde_json::json!("json"))
            } else {
                None
            }
        }),
        options: Some(ollama::ModelOptions {
            num_predict: request.max_tokens.map(|n| n as i32),
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            stop: request.stop.clone(),
            ..Default::default()
        }),
        stream: Some(request.stream),
        keep_alive: None,
        tools: request
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(convert_tool_to_ollama).collect()),
    }
}

/// Convert a single message from OpenAI to Ollama format
fn convert_message_to_ollama(msg: &openai::ChatMessage) -> ollama::Message {
    ollama::Message {
        role: msg.role.to_string(),
        content: msg.content.clone().unwrap_or_default(),
        images: None,
        tool_calls: msg
            .tool_calls
            .as_ref()
            .map(|calls| calls.iter().map(convert_tool_call_to_ollama).collect()),
    }
}

/// Convert tool definition from OpenAI to Ollama format
fn convert_tool_to_ollama(tool: &openai::Tool) -> ollama::Tool {
    ollama::Tool {
        tool_type: tool.tool_type.clone(),
        function: ollama::FunctionDefinition {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
        },
    }
}

/// Convert tool call from OpenAI to Ollama format
fn convert_tool_call_to_ollama(call: &openai::ToolCall) -> ollama::ToolCall {
    // Parse the arguments string to JSON Value
    let arguments =
        serde_json::from_str(&call.function.arguments).unwrap_or_else(|_| serde_json::json!({}));

    ollama::ToolCall {
        function: ollama::FunctionCall { name: call.function.name.clone(), arguments },
    }
}

/// Convert Ollama ChatResponse to OpenAI ChatCompletionResponse
pub fn ollama_to_openai_chat(
    response: &ollama::ChatResponse,
    model: &str,
) -> openai::ChatCompletionResponse {
    debug!(
        model = %model,
        done = response.done,
        "Converting Ollama response to OpenAI format"
    );

    let message = response.message.as_ref();

    // Determine finish reason
    let finish_reason = if message.and_then(|m| m.tool_calls.as_ref()).is_some() {
        "tool_calls"
    } else {
        response.done_reason.as_deref().unwrap_or("stop")
    };

    // Convert message
    let openai_message =
        message
            .map(convert_message_to_openai)
            .unwrap_or_else(|| openai::ChatMessage {
                role: openai::Role::Assistant,
                content: Some(String::new()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });

    // Calculate token counts
    let prompt_tokens = response.prompt_eval_count.unwrap_or(0);
    let completion_tokens = response.eval_count.unwrap_or(0);

    openai::ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: model.to_string(),
        choices: vec![openai::Choice {
            index: 0,
            message: openai_message,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage: openai::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        system_fingerprint: None,
    }
}

/// Convert a single message from Ollama to OpenAI format
fn convert_message_to_openai(msg: &ollama::Message) -> openai::ChatMessage {
    let role = match msg.role.as_str() {
        "system" => openai::Role::System,
        "user" => openai::Role::User,
        "assistant" => openai::Role::Assistant,
        "tool" => openai::Role::Tool,
        _ => openai::Role::User,
    };

    openai::ChatMessage {
        role,
        content: if msg.content.is_empty() {
            None
        } else {
            Some(msg.content.clone())
        },
        name: None,
        tool_calls: msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(i, call)| convert_tool_call_to_openai(call, i))
                .collect()
        }),
        tool_call_id: None,
    }
}

/// Convert tool call from Ollama to OpenAI format
fn convert_tool_call_to_openai(call: &ollama::ToolCall, _index: usize) -> openai::ToolCall {
    openai::ToolCall {
        id: format!("call_{}", &Uuid::new_v4().to_string().replace("-", "")[..24]),
        call_type: "function".to_string(),
        function: openai::FunctionCall {
            name: call.function.name.clone(),
            arguments: serde_json::to_string(&call.function.arguments).unwrap_or_default(),
        },
    }
}

/// Convert Ollama streaming chunk to OpenAI SSE format
pub fn ollama_chunk_to_openai_sse(
    chunk: &ollama::ChatResponse,
    model: &str,
    chunk_id: &str,
) -> openai::ChatCompletionChunk {
    let delta = if let Some(msg) = &chunk.message {
        openai::Delta {
            role: Some(openai::Role::Assistant),
            content: if msg.content.is_empty() {
                None
            } else {
                Some(msg.content.clone())
            },
            tool_calls: msg.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .enumerate()
                    .map(|(i, call)| convert_tool_call_to_openai(call, i))
                    .collect()
            }),
        }
    } else {
        openai::Delta::default()
    };

    let finish_reason = if chunk.done {
        Some(
            chunk
                .done_reason
                .clone()
                .unwrap_or_else(|| "stop".to_string()),
        )
    } else {
        None
    };

    openai::ChatCompletionChunk {
        id: chunk_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: model.to_string(),
        choices: vec![openai::ChunkChoice { index: 0, delta, finish_reason }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_to_ollama_basic() {
        let request = openai::ChatCompletionRequest {
            model: "llama3.2:3b".to_string(),
            messages: vec![openai::ChatMessage {
                role: openai::Role::User,
                content: Some("Hello!".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(0.7),
            top_p: None,
            n: None,
            stream: false,
            stop: None,
            max_tokens: Some(100),
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let ollama_req = openai_to_ollama_chat(&request);

        assert_eq!(ollama_req.model, "llama3.2:3b");
        assert_eq!(ollama_req.messages.len(), 1);
        assert_eq!(ollama_req.messages[0].role, "user");
        assert_eq!(ollama_req.messages[0].content, "Hello!");
        assert_eq!(ollama_req.options.as_ref().unwrap().temperature, Some(0.7));
        assert_eq!(ollama_req.options.as_ref().unwrap().num_predict, Some(100));
    }

    #[test]
    fn test_ollama_to_openai_basic() {
        let response = ollama::ChatResponse {
            model: "llama3.2:3b".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            message: Some(ollama::Message {
                role: "assistant".to_string(),
                content: "Hello! How can I help?".to_string(),
                images: None,
                tool_calls: None,
            }),
            done: true,
            done_reason: Some("stop".to_string()),
            total_duration: Some(1000000000),
            load_duration: None,
            prompt_eval_count: Some(10),
            prompt_eval_duration: None,
            eval_count: Some(15),
            eval_duration: None,
        };

        let openai_resp = ollama_to_openai_chat(&response, "llama3.2:3b");

        assert_eq!(openai_resp.object, "chat.completion");
        assert_eq!(openai_resp.model, "llama3.2:3b");
        assert_eq!(openai_resp.choices.len(), 1);
        assert_eq!(openai_resp.choices[0].message.role, openai::Role::Assistant);
        assert_eq!(
            openai_resp.choices[0].message.content,
            Some("Hello! How can I help?".to_string())
        );
        assert_eq!(openai_resp.usage.prompt_tokens, 10);
        assert_eq!(openai_resp.usage.completion_tokens, 15);
    }

    #[test]
    fn test_tool_call_conversion() {
        let ollama_call = ollama::ToolCall {
            function: ollama::FunctionCall {
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"location": "Paris"}),
            },
        };

        let openai_call = convert_tool_call_to_openai(&ollama_call, 0);

        assert_eq!(openai_call.function.name, "get_weather");
        assert!(openai_call.function.arguments.contains("Paris"));
        assert!(openai_call.id.starts_with("call_"));
    }
}
