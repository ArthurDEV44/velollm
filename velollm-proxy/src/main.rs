// Allow dead_code for now - these functions will be used in upcoming tasks
#![allow(dead_code)]

//! VeloLLM Proxy - High-performance proxy for local LLM inference optimization.
//!
//! This proxy sits between your application and Ollama, providing:
//! - OpenAI API compatibility
//! - Request optimization
//! - Tool calling improvements (focused on Mistral and Llama)
//! - Performance metrics
//!
//! ## Quick Start
//!
//! ```bash
//! # Start with defaults (port 8000, Ollama at localhost:11434)
//! velollm-proxy
//!
//! # Custom configuration
//! OLLAMA_HOST=http://192.168.1.100:11434 VELOLLM_PORT=9000 velollm-proxy
//! ```
//!
//! ## Usage with OpenAI clients
//!
//! ```python
//! from openai import OpenAI
//!
//! client = OpenAI(
//!     base_url="http://localhost:8000/v1",
//!     api_key="not-needed"
//! )
//!
//! response = client.chat.completions.create(
//!     model="llama3.2:3b",  # or "mistral:7b"
//!     messages=[{"role": "user", "content": "Hello!"}]
//! )
//! ```

use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod batcher;
mod convert;
mod error;
mod optimizer;
mod proxy;
mod routes;
mod state;
mod types;

use state::{AppState, ProxyConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("velollm_proxy=info,tower_http=info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    // Load configuration
    let config = ProxyConfig::from_env();

    info!(
        port = config.port,
        ollama_url = %config.ollama_url,
        "Starting VeloLLM Proxy v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Create application state
    let state = Arc::new(AppState::new(config.clone()));

    // Check Ollama connectivity
    match state.proxy.health_check().await {
        Ok(_) => info!("Connected to Ollama at {}", config.ollama_url),
        Err(e) => {
            warn!(
                "Could not connect to Ollama at {}: {}. \
                 Proxy will start anyway and retry on requests.",
                config.ollama_url, e
            );
        }
    }

    // Build router
    let app = Router::new()
        // Health endpoints
        .route("/health", get(routes::health))
        .route("/ready", get(routes::ready))
        .route("/live", get(routes::live))
        .route("/metrics", get(routes::metrics))
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/models/:model", get(routes::get_model))
        // Ollama-native endpoints
        .route("/api/generate", post(routes::generate))
        .route("/api/chat", post(routes::chat))
        .route("/api/tags", get(routes::tags))
        // Middleware
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    // Start server
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("VeloLLM Proxy listening on http://{}", addr);
    info!("OpenAI API: http://{}/v1/chat/completions", addr);
    info!("Ollama API: http://{}/api/chat", addr);
    info!("Health:     http://{}/health", addr);

    println!();
    println!("==================================================");
    println!("  VeloLLM Proxy v{}", env!("CARGO_PKG_VERSION"));
    println!("==================================================");
    println!("  Listening on: http://{}", addr);
    println!("  Ollama backend: {}", config.ollama_url);
    println!();
    println!("  Batcher configuration:");
    println!(
        "    Max concurrent: {}",
        state.batcher_config.max_concurrent
    );
    println!(
        "    Max queue: {} (per model: {})",
        state.batcher_config.max_queue_total, state.batcher_config.max_queue_per_model
    );
    println!(
        "    Model-aware queuing: {}",
        if state.batcher_config.model_aware_queuing {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!();
    println!("  Endpoints:");
    println!("    OpenAI: POST /v1/chat/completions");
    println!("    Ollama: POST /api/chat, POST /api/generate");
    println!("    Models: GET  /v1/models, GET /api/tags");
    println!("    Health: GET  /health, /ready, /live, /metrics");
    println!();
    println!("  Supported models for tool calling:");
    println!("    - Mistral: mistral:7b, mistral:latest");
    println!("    - Llama:   llama3.2:3b, llama3.1:8b, llama3.1:70b");
    println!("==================================================");
    println!();

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env() {
        // Test default config
        let config = ProxyConfig::default();
        assert_eq!(config.port, 8000);
        assert_eq!(config.ollama_url, "http://localhost:11434");
    }
}
