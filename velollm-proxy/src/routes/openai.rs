//! OpenAI-compatible API routes.
//!
//! These routes provide OpenAI API compatibility, allowing existing
//! OpenAI clients and SDKs to work with VeloLLM proxy.
//!
//! Focused on Mistral and Llama models for reliable tool calling.

use axum::{
    extract::State,
    response::{IntoResponse, Response, Sse},
    Json,
};
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::convert::{ollama_chunk_to_openai_sse, ollama_to_openai_chat, openai_to_ollama_chat};
use crate::error::ProxyError;
use crate::metrics::RequestTimer;
use crate::optimizer::ToolOptimizer;
use crate::state::AppState;
use crate::types::ollama::ChatResponse as OllamaChatResponse;
use crate::types::openai::{ChatCompletionRequest, ChatCompletionResponse, Model, ModelsResponse};

/// Chat completions endpoint
///
/// POST /v1/chat/completions
///
/// This endpoint provides OpenAI-compatible chat completions with
/// enhanced tool calling support for Mistral and Llama models.
///
/// Tool calling enhancements (TASK-023):
/// - JSON repair for malformed arguments
/// - Deduplication of duplicate tool calls
/// - Validation against JSON Schema
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ProxyError> {
    info!(
        model = %request.model,
        messages = request.messages.len(),
        stream = request.stream,
        has_tools = request.has_tools(),
        "Handling POST /v1/chat/completions"
    );

    // Validate model is Mistral or Llama for tool calling
    let has_tools = request.has_tools();
    if has_tools {
        if !ToolOptimizer::is_model_supported(&request.model) {
            warn!(
                model = %request.model,
                "Tool calling requested but model is not supported. \
                 Only Mistral and Llama models are supported for reliable tool calling. \
                 Recommended: mistral:7b, mistral-small, llama3.1:8b, llama3.2:3b"
            );
            // We don't reject the request, just warn - the user might know what they're doing
        } else {
            let model_type = ToolOptimizer::get_model_type(&request.model);
            info!(
                model = %request.model,
                model_type = ?model_type,
                "Using optimized tool calling for supported model"
            );
        }

        // Register tools with the optimizer for validation
        if let Some(ref tools) = request.tools {
            let mut optimizer = state.tool_optimizer.lock().await;
            if let Err(e) = optimizer.register_tools(tools) {
                warn!(error = %e, "Failed to register tools, validation disabled");
            }
        }
    }

    // Update stats
    {
        let mut stats = state.stats.lock().await;
        stats.requests_total += 1;
    }

    // Convert to Ollama format
    let ollama_request = openai_to_ollama_chat(&request);
    let model = request.model.clone();

    // Start Prometheus request timer
    let timer = RequestTimer::new(&model);

    if request.stream {
        // Streaming response
        // Note: Tool call optimization for streaming is limited because we need
        // to accumulate the full response before processing tool calls
        let stream = match state.proxy.chat_stream(&ollama_request).await {
            Ok(s) => s,
            Err(e) => {
                timer.record_failure(&e.to_string());
                return Err(e);
            }
        };
        let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        // For streaming, we record success immediately (tokens counted separately)
        // In a real implementation, we'd accumulate token count from stream
        timer.record_success(0);

        // Convert Ollama stream to OpenAI SSE format
        let sse_stream = convert_stream_to_sse(stream, model.clone(), chunk_id);

        Ok(Sse::new(sse_stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(Duration::from_secs(15))
                    .text("keep-alive"),
            )
            .into_response())
    } else {
        // Non-streaming response
        let ollama_response = match state.proxy.chat(&ollama_request).await {
            Ok(r) => r,
            Err(e) => {
                timer.record_failure(&e.to_string());
                return Err(e);
            }
        };

        // Convert to OpenAI format
        let mut openai_response = ollama_to_openai_chat(&ollama_response, &model);

        // Apply tool calling optimizations if tools were used
        if has_tools {
            openai_response = optimize_tool_calls(&state, openai_response).await;
        }

        let tokens = openai_response.usage.completion_tokens as u64;

        // Update stats
        {
            let mut stats = state.stats.lock().await;
            stats.requests_success += 1;
            stats.tokens_generated += tokens;
        }

        // Record Prometheus metrics
        timer.record_success(tokens);

        Ok(Json(openai_response).into_response())
    }
}

/// Apply tool calling optimizations to the response
///
/// This performs:
/// - JSON repair for malformed arguments
/// - Deduplication of identical tool calls
/// - Validation against JSON Schema
async fn optimize_tool_calls(
    state: &Arc<AppState>,
    mut response: ChatCompletionResponse,
) -> ChatCompletionResponse {
    // Check if there are tool calls to optimize
    let has_tool_calls = response.choices.iter().any(|c| {
        c.message
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty())
    });

    if !has_tool_calls {
        return response;
    }

    // Process tool calls through the optimizer
    let mut optimizer = state.tool_optimizer.lock().await;

    for choice in &mut response.choices {
        if let Some(tool_calls) = choice.message.tool_calls.take() {
            let optimized = optimizer.process_tool_calls(tool_calls);
            if !optimized.is_empty() {
                choice.message.tool_calls = Some(optimized);
            }
        }
    }

    // Log optimization stats
    let stats = optimizer.stats();
    if stats.json_repairs > 0 || stats.duplicates_removed > 0 {
        info!(
            repairs = stats.json_repairs,
            duplicates = stats.duplicates_removed,
            validations = stats.validation_successes,
            failures = stats.validation_failures,
            "Tool call optimization applied"
        );
    }

    response
}

/// Convert Ollama byte stream to OpenAI SSE events
fn convert_stream_to_sse(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: String,
    chunk_id: String,
) -> impl Stream<Item = Result<axum::response::sse::Event, Infallible>> + Send {
    let model = Arc::new(model);
    let chunk_id = Arc::new(chunk_id);

    stream
        .filter_map(move |result| {
            let model = Arc::clone(&model);
            let chunk_id = Arc::clone(&chunk_id);

            async move {
                match result {
                    Ok(bytes) => {
                        // Parse Ollama NDJSON chunk
                        let text = String::from_utf8_lossy(&bytes);
                        for line in text.lines() {
                            if line.trim().is_empty() {
                                continue;
                            }
                            match serde_json::from_str::<OllamaChatResponse>(line) {
                                Ok(ollama_chunk) => {
                                    let openai_chunk =
                                        ollama_chunk_to_openai_sse(&ollama_chunk, &model, &chunk_id);
                                    let data = serde_json::to_string(&openai_chunk).ok()?;

                                    return Some(Ok(axum::response::sse::Event::default().data(data)));
                                }
                                Err(e) => {
                                    debug!(error = %e, line = %line, "Failed to parse Ollama chunk");
                                }
                            }
                        }
                        None
                    }
                    Err(e) => {
                        error!(error = %e, "Stream error");
                        None
                    }
                }
            }
        })
        .chain(stream::once(async {
            // Send [DONE] marker at the end
            Ok(axum::response::sse::Event::default().data("[DONE]"))
        }))
}

/// List models endpoint
///
/// GET /v1/models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ModelsResponse>, ProxyError> {
    debug!("Handling GET /v1/models");

    let ollama_tags = state.proxy.list_models().await?;

    // Convert to OpenAI format
    let models: Vec<Model> = ollama_tags
        .models
        .into_iter()
        .map(|m| Model {
            id: m.name.clone(),
            object: "model".to_string(),
            created: chrono::DateTime::parse_from_rfc3339(&m.modified_at)
                .map(|dt| dt.timestamp() as u64)
                .unwrap_or(0),
            owned_by: m
                .details
                .and_then(|d| d.family)
                .unwrap_or_else(|| "ollama".to_string()),
        })
        .collect();

    Ok(Json(ModelsResponse { object: "list".to_string(), data: models }))
}

/// Get model info
///
/// GET /v1/models/{model}
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<Model>, ProxyError> {
    debug!(model_id = %model_id, "Handling GET /v1/models/{{model}}");

    let ollama_tags = state.proxy.list_models().await?;

    let model = ollama_tags
        .models
        .into_iter()
        .find(|m| m.name == model_id)
        .map(|m| Model {
            id: m.name.clone(),
            object: "model".to_string(),
            created: chrono::DateTime::parse_from_rfc3339(&m.modified_at)
                .map(|dt| dt.timestamp() as u64)
                .unwrap_or(0),
            owned_by: m
                .details
                .and_then(|d| d.family)
                .unwrap_or_else(|| "ollama".to_string()),
        })
        .ok_or_else(|| ProxyError::InvalidRequest(format!("Model '{}' not found", model_id)))?;

    Ok(Json(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ChatMessage, Role};

    #[test]
    fn test_request_validation() {
        let request = ChatCompletionRequest {
            model: "llama3.2:3b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some("Hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
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
        };

        assert!(!request.has_tools());
    }
}
