//! Proxy layer for forwarding requests to Ollama.
//!
//! This module handles all communication with the Ollama backend,
//! including request forwarding and response streaming.

use crate::error::ProxyError;
use crate::types::ollama::{
    ChatRequest, ChatResponse, GenerateRequest, GenerateResponse, TagsResponse,
};
use bytes::Bytes;
use futures::Stream;
use reqwest::Client;
use std::pin::Pin;
use tracing::{debug, error, info, instrument};

/// Type alias for streaming response
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

/// Ollama proxy client
#[derive(Clone)]
pub struct OllamaProxy {
    /// HTTP client
    client: Client,
    /// Ollama base URL
    base_url: String,
}

impl OllamaProxy {
    /// Create a new Ollama proxy
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        info!(url = %base_url, "Creating Ollama proxy");

        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(300)) // 5 min timeout for long generations
                .build()
                .expect("Failed to create HTTP client"),
            base_url,
        }
    }

    /// Get the Ollama base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Check if Ollama is available
    #[instrument(skip(self))]
    pub async fn health_check(&self) -> Result<(), ProxyError> {
        let url = format!("{}/api/tags", self.base_url);
        debug!(url = %url, "Checking Ollama health");

        match self.client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                info!("Ollama is healthy");
                Ok(())
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                error!(status = %status, body = %body, "Ollama returned error");
                Err(ProxyError::OllamaError(format!("Ollama returned status {}: {}", status, body)))
            }
            Err(e) => {
                error!(error = %e, "Failed to connect to Ollama");
                Err(ProxyError::OllamaConnection(e.to_string()))
            }
        }
    }

    /// List available models
    #[instrument(skip(self))]
    pub async fn list_models(&self) -> Result<TagsResponse, ProxyError> {
        let url = format!("{}/api/tags", self.base_url);
        debug!(url = %url, "Listing models");

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::OllamaError(body));
        }

        let tags: TagsResponse = response.json().await?;
        debug!(count = tags.models.len(), "Found models");
        Ok(tags)
    }

    /// Send a generate request (non-streaming)
    #[instrument(skip(self, request), fields(model = %request.model))]
    pub async fn generate(
        &self,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse, ProxyError> {
        let url = format!("{}/api/generate", self.base_url);
        debug!(url = %url, "Sending generate request");

        let response = self.client.post(&url).json(request).send().await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::OllamaError(body));
        }

        let result: GenerateResponse = response.json().await?;
        debug!(tokens = ?result.eval_count, "Generate complete");
        Ok(result)
    }

    /// Send a generate request with streaming
    #[instrument(skip(self, request), fields(model = %request.model))]
    pub async fn generate_stream(
        &self,
        request: &GenerateRequest,
    ) -> Result<ByteStream, ProxyError> {
        let url = format!("{}/api/generate", self.base_url);
        debug!(url = %url, "Sending streaming generate request");

        let response = self.client.post(&url).json(request).send().await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::OllamaError(body));
        }

        Ok(Box::pin(response.bytes_stream()))
    }

    /// Send a chat request (non-streaming)
    #[instrument(skip(self, request), fields(model = %request.model, messages = request.messages.len()))]
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProxyError> {
        let url = format!("{}/api/chat", self.base_url);
        debug!(url = %url, "Sending chat request");

        let response = self.client.post(&url).json(request).send().await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::OllamaError(body));
        }

        let result: ChatResponse = response.json().await?;
        debug!(tokens = ?result.eval_count, "Chat complete");
        Ok(result)
    }

    /// Send a chat request with streaming
    #[instrument(skip(self, request), fields(model = %request.model, messages = request.messages.len()))]
    pub async fn chat_stream(&self, request: &ChatRequest) -> Result<ByteStream, ProxyError> {
        let url = format!("{}/api/chat", self.base_url);
        debug!(url = %url, "Sending streaming chat request");

        let response = self.client.post(&url).json(request).send().await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::OllamaError(body));
        }

        Ok(Box::pin(response.bytes_stream()))
    }

    /// Forward a raw request to Ollama
    #[instrument(skip(self, body))]
    pub async fn forward_raw(
        &self,
        endpoint: &str,
        method: reqwest::Method,
        body: Option<serde_json::Value>,
    ) -> Result<reqwest::Response, ProxyError> {
        let url = format!("{}{}", self.base_url, endpoint);
        debug!(url = %url, method = %method, "Forwarding raw request");

        let mut builder = self.client.request(method, &url);
        if let Some(body) = body {
            builder = builder.json(&body);
        }

        let response = builder.send().await?;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_creation() {
        let proxy = OllamaProxy::new("http://localhost:11434");
        assert_eq!(proxy.base_url(), "http://localhost:11434");
    }

    #[test]
    fn test_proxy_with_custom_url() {
        let proxy = OllamaProxy::new("http://192.168.1.100:11434");
        assert_eq!(proxy.base_url(), "http://192.168.1.100:11434");
    }
}
