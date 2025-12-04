//! Ollama-native API routes.
//!
//! These routes provide direct pass-through to Ollama's native API,
//! with optional optimization when enabled.

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use std::sync::Arc;
use tracing::{debug, info};

use crate::error::ProxyError;
use crate::state::AppState;
use crate::types::ollama::{ChatRequest, GenerateRequest, TagsResponse};

/// List available models
///
/// GET /api/tags
pub async fn tags(State(state): State<Arc<AppState>>) -> Result<Json<TagsResponse>, ProxyError> {
    debug!("Handling GET /api/tags");

    let response = state.proxy.list_models().await?;

    // Update stats
    {
        let mut stats = state.stats.lock().await;
        stats.requests_total += 1;
        stats.requests_success += 1;
    }

    Ok(Json(response))
}

/// Generate completion
///
/// POST /api/generate
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<GenerateRequest>,
) -> Result<Response, ProxyError> {
    info!(model = %request.model, "Handling POST /api/generate");

    // Update stats
    {
        let mut stats = state.stats.lock().await;
        stats.requests_total += 1;
    }

    let is_streaming = request.stream.unwrap_or(true);

    if is_streaming {
        // Streaming response
        request.stream = Some(true);
        let stream = state.proxy.generate_stream(&request).await?;

        let body = Body::from_stream(stream.map(|result| result.map_err(std::io::Error::other)));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/x-ndjson")
            .header("Transfer-Encoding", "chunked")
            .body(body)
            .unwrap())
    } else {
        // Non-streaming response
        request.stream = Some(false);
        let response = state.proxy.generate(&request).await?;

        // Update stats with token count
        {
            let mut stats = state.stats.lock().await;
            stats.requests_success += 1;
            if let Some(tokens) = response.eval_count {
                stats.tokens_generated += tokens as u64;
            }
        }

        Ok(Json(response).into_response())
    }
}

/// Chat completion
///
/// POST /api/chat
pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<ChatRequest>,
) -> Result<Response, ProxyError> {
    info!(
        model = %request.model,
        messages = request.messages.len(),
        has_tools = request.has_tools(),
        "Handling POST /api/chat"
    );

    // Update stats
    {
        let mut stats = state.stats.lock().await;
        stats.requests_total += 1;
    }

    let is_streaming = request.stream.unwrap_or(true);

    if is_streaming {
        // Streaming response
        request.stream = Some(true);
        let stream = state.proxy.chat_stream(&request).await?;

        let body = Body::from_stream(stream.map(|result| result.map_err(std::io::Error::other)));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/x-ndjson")
            .header("Transfer-Encoding", "chunked")
            .body(body)
            .unwrap())
    } else {
        // Non-streaming response
        request.stream = Some(false);
        let response = state.proxy.chat(&request).await?;

        // Update stats with token count
        {
            let mut stats = state.stats.lock().await;
            stats.requests_success += 1;
            if let Some(tokens) = response.eval_count {
                stats.tokens_generated += tokens as u64;
            }
        }

        Ok(Json(response).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ollama::Message;

    #[test]
    fn test_chat_request_with_tools() {
        let request = ChatRequest {
            model: "mistral:7b".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                images: None,
                tool_calls: None,
            }],
            format: None,
            options: None,
            stream: Some(false),
            keep_alive: None,
            tools: None,
        };

        assert!(!request.has_tools());
    }
}
