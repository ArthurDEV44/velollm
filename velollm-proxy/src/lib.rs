//! VeloLLM Proxy Library
//!
//! High-performance proxy for local LLM inference optimization.
//!
//! This library provides the core functionality for the VeloLLM proxy,
//! which can be used either as a standalone binary or integrated into
//! the `velollm` CLI.

#![allow(dead_code)]

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

pub mod batcher;
pub mod cache;
pub mod compressor;
pub mod convert;
pub mod error;
pub mod metrics;
pub mod optimizer;
pub mod prefetch;
pub mod proxy;
pub mod router;
pub mod routes;
pub mod state;
pub mod types;

pub use state::{AppState, ProxyConfig};

/// Server configuration for the proxy
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Ollama backend URL
    pub ollama_url: String,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Whether to print the banner on startup
    pub print_banner: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8000,
            ollama_url: "http://localhost:11434".to_string(),
            max_concurrent: 4,
            print_banner: true,
        }
    }
}

impl From<ServerConfig> for ProxyConfig {
    fn from(config: ServerConfig) -> Self {
        ProxyConfig { port: config.port, ollama_url: config.ollama_url, verbose: false }
    }
}

/// Initialize Prometheus metrics registry.
/// Should be called once before starting the server.
pub fn init_metrics() {
    if let Err(e) = metrics::register_metrics() {
        warn!("Failed to register Prometheus metrics: {}", e);
    }
}

/// Run the VeloLLM proxy server.
///
/// This function starts the HTTP server and blocks until it's shut down.
///
/// # Arguments
/// * `config` - Server configuration
///
/// # Example
/// ```no_run
/// use velollm_proxy::{run_server, ServerConfig};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = ServerConfig {
///         port: 8000,
///         ollama_url: "http://localhost:11434".to_string(),
///         ..Default::default()
///     };
///     run_server(config).await
/// }
/// ```
pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    // Initialize Prometheus metrics
    init_metrics();

    let proxy_config = ProxyConfig::from(config.clone());

    info!(
        port = proxy_config.port,
        ollama_url = %proxy_config.ollama_url,
        "Starting VeloLLM Proxy v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Create application state
    let state = Arc::new(AppState::new(proxy_config.clone()));

    // Check Ollama connectivity
    match state.proxy.health_check().await {
        Ok(_) => info!("Connected to Ollama at {}", proxy_config.ollama_url),
        Err(e) => {
            warn!(
                "Could not connect to Ollama at {}: {}. \
                 Proxy will start anyway and retry on requests.",
                proxy_config.ollama_url, e
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
        .route("/metrics/prometheus", get(routes::metrics_prometheus))
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

    if config.print_banner {
        print_banner(&config, &state);
    }

    axum::serve(listener, app).await?;

    Ok(())
}

/// Print the startup banner
fn print_banner(config: &ServerConfig, state: &Arc<AppState>) {
    let addr = format!("0.0.0.0:{}", config.port);

    println!();
    println!("==================================================");
    println!("  VeloLLM Proxy v{}", env!("CARGO_PKG_VERSION"));
    println!("==================================================");
    println!("  Listening on: http://{}", addr);
    println!("  Ollama backend: {}", config.ollama_url);
    println!();
    println!("  Batcher configuration:");
    println!("    Max concurrent: {}", state.batcher_config.max_concurrent);
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
    println!("  Cache configuration:");
    println!(
        "    Exact cache: {} entries, TTL {}s",
        state.cache_config.exact_cache_size,
        state.cache_config.exact_cache_ttl.as_secs()
    );
    #[cfg(feature = "semantic-cache")]
    if state.cache_config.semantic_cache_enabled {
        println!(
            "    Semantic cache: {} entries, threshold {:.0}%",
            state.cache_config.semantic_cache_size,
            state.cache_config.similarity_threshold * 100.0
        );
    }
    #[cfg(not(feature = "semantic-cache"))]
    println!("    Semantic cache: disabled (enable with --features semantic-cache)");
    println!();
    println!("  Compression configuration:");
    if state.compressor_config.enabled {
        println!(
            "    Enabled: yes (max: {} tokens, target: {} tokens)",
            state.compressor_config.max_context_tokens,
            state.compressor_config.target_context_tokens
        );
        println!(
            "    Dedup: {}, Summarization: {}, System cache: {}",
            if state.compressor_config.dedup_enabled {
                "on"
            } else {
                "off"
            },
            if state.compressor_config.summarization_enabled {
                "on"
            } else {
                "off"
            },
            if state.compressor_config.system_prompt_cache_enabled {
                "on"
            } else {
                "off"
            }
        );
    } else {
        println!("    Enabled: no (set VELOLLM_COMPRESSION_ENABLED=true to enable)");
    }
    println!();
    println!("  Prefetch configuration:");
    if state.prefetch_config.enabled {
        println!(
            "    Enabled: yes (max predictions: {}, confidence: {:.0}%)",
            state.prefetch_config.max_predictions,
            state.prefetch_config.min_confidence * 100.0
        );
        println!(
            "    Cache TTL: {}s, Queue size: {}",
            state.prefetch_config.cache_ttl_secs, state.prefetch_config.max_queue_size
        );
    } else {
        println!("    Enabled: no (set VELOLLM_PREFETCH_ENABLED=true to enable)");
    }
    println!();
    println!("  Router configuration:");
    if state.router_config.enabled {
        println!(
            "    Enabled: yes (thresholds: small < {:.0}%, large > {:.0}%)",
            state.router_config.small_threshold * 100.0,
            state.router_config.large_threshold * 100.0
        );
        println!(
            "    Models: {} / {} / {}",
            state.router_config.small_model,
            state.router_config.medium_model,
            state.router_config.large_model
        );
        println!("    Use model=\"auto\" to enable automatic routing");
    } else {
        println!("    Enabled: no (set VELOLLM_ROUTER_ENABLED=true to enable)");
    }
    println!();
    println!("  Endpoints:");
    println!("    OpenAI: POST /v1/chat/completions");
    println!("    Ollama: POST /api/chat, POST /api/generate");
    println!("    Models: GET  /v1/models, GET /api/tags");
    println!("    Health: GET  /health, /ready, /live, /metrics");
    println!("    Prometheus: GET /metrics/prometheus");
    println!();
    println!("  Supported models for tool calling:");
    println!("    - Mistral: mistral:7b, mistral:latest");
    println!("    - Llama:   llama3.2:3b, llama3.1:8b, llama3.1:70b");
    println!("==================================================");
    println!();
}
