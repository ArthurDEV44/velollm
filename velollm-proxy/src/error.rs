//! Error types for the VeloLLM proxy.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Proxy error types
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// Failed to connect to Ollama backend
    #[error("Failed to connect to Ollama: {0}")]
    OllamaConnection(String),

    /// Ollama returned an error
    #[error("Ollama error: {0}")]
    OllamaError(String),

    /// Request parsing error
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP client error
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Internal server error
    #[error("Internal error: {0}")]
    #[allow(dead_code)]
    Internal(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            ProxyError::OllamaConnection(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::OllamaError(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ProxyError::Serialization(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ProxyError::Http(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({
            "error": {
                "message": error_message,
                "type": format!("{:?}", self).split('(').next().unwrap_or("Unknown"),
            }
        }));

        (status, body).into_response()
    }
}
