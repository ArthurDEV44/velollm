//! VeloLLM Proxy Binary
//!
//! Standalone binary for the VeloLLM proxy server.
//! For library usage, see [`velollm_proxy`].

use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use velollm_proxy::{run_server, ServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("velollm_proxy=info,tower_http=info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    // Load configuration from environment
    let port = std::env::var("VELOLLM_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let ollama_url = std::env::var("OLLAMA_HOST")
        .or_else(|_| std::env::var("OLLAMA_URL"))
        .unwrap_or_else(|_| "http://localhost:11434".to_string());

    let config = ServerConfig { port, ollama_url, print_banner: true, ..Default::default() };

    run_server(config).await
}

#[cfg(test)]
mod tests {
    use velollm_proxy::state::ProxyConfig;

    #[test]
    fn test_config_from_env() {
        // Test default config
        let config = ProxyConfig::default();
        assert_eq!(config.port, 8000);
        assert_eq!(config.ollama_url, "http://localhost:11434");
    }
}
