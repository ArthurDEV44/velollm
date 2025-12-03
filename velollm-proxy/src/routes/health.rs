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

/// Metrics endpoint with proxy and batcher statistics
///
/// GET /metrics
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.stats.lock().await;
    let batcher_snapshot = state.batcher_metrics.snapshot();
    let queue_depth = state.request_queue.queue_depth().await;

    Json(json!({
        "proxy": {
            "requests_total": stats.requests_total,
            "requests_success": stats.requests_success,
            "requests_failed": stats.requests_failed,
            "tokens_generated": stats.tokens_generated,
            "avg_tokens_per_second": stats.avg_tokens_per_second(),
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses
        },
        "batcher": {
            "config": {
                "max_concurrent": state.batcher_config.max_concurrent,
                "max_queue_total": state.batcher_config.max_queue_total,
                "max_queue_per_model": state.batcher_config.max_queue_per_model,
                "model_aware_queuing": state.batcher_config.model_aware_queuing
            },
            "current": {
                "queue_depth": queue_depth,
                "available_permits": state.request_queue.available_permits()
            },
            "totals": {
                "requests_received": batcher_snapshot.requests_received,
                "requests_completed": batcher_snapshot.requests_completed,
                "requests_timed_out": batcher_snapshot.requests_timed_out,
                "requests_rejected": batcher_snapshot.requests_rejected,
                "max_queue_depth": batcher_snapshot.max_queue_depth
            },
            "performance": {
                "avg_queue_wait_ms": batcher_snapshot.avg_queue_wait_ms,
                "avg_processing_ms": batcher_snapshot.avg_processing_ms,
                "avg_batch_size": batcher_snapshot.avg_batch_size
            }
        }
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
