//! Health check and metrics endpoints.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;
use std::sync::Arc;

use crate::state::AppState;

/// Health check endpoint
///
/// GET /health
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.proxy.health_check().await {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({
                "status": "healthy",
                "ollama": "connected",
                "version": env!("CARGO_PKG_VERSION")
            })),
        ),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "ollama": "disconnected",
                "error": e.to_string(),
                "version": env!("CARGO_PKG_VERSION")
            })),
        ),
    }
}

/// Metrics endpoint (placeholder for future Prometheus metrics)
///
/// GET /metrics
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.stats.lock().await;
    Json(json!({
        "requests_total": stats.requests_total,
        "requests_success": stats.requests_success,
        "requests_failed": stats.requests_failed,
        "tokens_generated": stats.tokens_generated,
        "avg_tokens_per_second": stats.avg_tokens_per_second(),
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
    }))
}

/// Ready check (for Kubernetes)
///
/// GET /ready
pub async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.proxy.health_check().await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Live check (for Kubernetes)
///
/// GET /live
pub async fn live() -> impl IntoResponse {
    StatusCode::OK
}
