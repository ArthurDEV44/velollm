# VeloLLM - Phase 3: Proxy Intelligent

Ce document définit les tâches pour transformer VeloLLM en proxy d'optimisation pour l'inférence LLM locale.

**Vision**: Un proxy Rust ultra-performant qui s'intercale entre les applications et Ollama/llama.cpp pour optimiser automatiquement l'inférence, améliorer le tool-calling, et maximiser les performances.

**Positionnement**: "vLLM performance for Ollama users" - garder la simplicité d'Ollama avec les performances de vLLM.

---

## 📊 Historique des Phases

| Phase | Description | Status | Documentation |
|-------|-------------|--------|---------------|
| Phase 1 | MVP (CLI detect/benchmark/optimize) | ✅ 100% | [TODO_MVP.md](TODO_MVP.md) |
| Phase 2 | PagedAttention, Scheduler, CUDA kernels | ✅ 83% | [TODO_PHASE2.md](TODO_PHASE2.md) |
| **Phase 3** | **Proxy Intelligent** | 🚧 0% | Ce fichier |

---

## 🎯 Phase 3: Proxy Intelligent (Current)

### Architecture Cible

```
Client App  ──▶  VeloLLM Proxy (localhost:8000)  ──▶  Ollama (localhost:11434)
                        │
                 ┌──────┴──────┐
                 │ Optimise:   │
                 │ • Batching  │
                 │ • Tools     │
                 │ • Cache     │
                 │ • Streaming │
                 └─────────────┘
```

---

### Sprint 7: Core Proxy Infrastructure (Semaines 17-20)

#### TASK-021: HTTP Server Foundation
**Priority**: P0 (Critical Path)
**Estimated effort**: 8h
**Dependencies**: None

**Objectif**: Créer le serveur HTTP de base avec axum/tower.

**Instructions**:

1. **Créer le crate proxy**:
   ```bash
   mkdir -p velollm-proxy/src
   ```

2. **Ajouter au workspace** (`Cargo.toml` racine):
   ```toml
   members = [
       # ... existing
       "velollm-proxy",
   ]
   ```

3. **Créer `velollm-proxy/Cargo.toml`**:
   ```toml
   [package]
   name = "velollm-proxy"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   axum = { version = "0.7", features = ["macros"] }
   tokio = { workspace = true, features = ["full"] }
   tower = "0.4"
   tower-http = { version = "0.5", features = ["cors", "trace", "timeout"] }
   hyper = { version = "1.0", features = ["full"] }
   serde = { workspace = true }
   serde_json = { workspace = true }
   tracing = { workspace = true }
   reqwest = { workspace = true, features = ["json", "stream"] }
   futures = "0.3"
   bytes = "1.0"
   async-stream = "0.3"

   velollm-core = { path = "../velollm-core" }
   ```

4. **Implémenter le serveur de base** (`velollm-proxy/src/main.rs`):
   ```rust
   use axum::{
       Router,
       routing::{get, post},
       extract::State,
       response::IntoResponse,
       Json,
   };
   use std::sync::Arc;
   use tower_http::trace::TraceLayer;

   mod config;
   mod routes;
   mod proxy;
   mod error;

   #[derive(Clone)]
   pub struct AppState {
       pub ollama_url: String,
       pub client: reqwest::Client,
   }

   #[tokio::main]
   async fn main() -> anyhow::Result<()> {
       tracing_subscriber::fmt::init();

       let state = Arc::new(AppState {
           ollama_url: std::env::var("OLLAMA_HOST")
               .unwrap_or_else(|_| "http://localhost:11434".to_string()),
           client: reqwest::Client::new(),
       });

       let app = Router::new()
           // OpenAI-compatible endpoints
           .route("/v1/chat/completions", post(routes::chat_completions))
           .route("/v1/completions", post(routes::completions))
           .route("/v1/models", get(routes::list_models))
           // Ollama-native endpoints
           .route("/api/generate", post(routes::ollama_generate))
           .route("/api/chat", post(routes::ollama_chat))
           .route("/api/tags", get(routes::ollama_tags))
           // Health & metrics
           .route("/health", get(routes::health))
           .route("/metrics", get(routes::metrics))
           .layer(TraceLayer::new_for_http())
           .with_state(state);

       let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
       tracing::info!("VeloLLM proxy listening on http://0.0.0.0:8000");

       axum::serve(listener, app).await?;
       Ok(())
   }
   ```

5. **Implémenter le proxy transparent** (`velollm-proxy/src/proxy.rs`):
   ```rust
   use reqwest::Client;
   use bytes::Bytes;
   use futures::Stream;

   pub struct OllamaProxy {
       client: Client,
       base_url: String,
   }

   impl OllamaProxy {
       pub fn new(base_url: String) -> Self {
           Self {
               client: Client::new(),
               base_url,
           }
       }

       /// Forward request to Ollama and return response
       pub async fn forward<T: serde::Serialize>(
           &self,
           endpoint: &str,
           body: &T,
       ) -> Result<reqwest::Response, ProxyError> {
           let url = format!("{}{}", self.base_url, endpoint);
           self.client
               .post(&url)
               .json(body)
               .send()
               .await
               .map_err(ProxyError::from)
       }

       /// Forward with streaming response
       pub async fn forward_stream<T: serde::Serialize>(
           &self,
           endpoint: &str,
           body: &T,
       ) -> Result<impl Stream<Item = Result<Bytes, reqwest::Error>>, ProxyError> {
           let url = format!("{}{}", self.base_url, endpoint);
           let response = self.client
               .post(&url)
               .json(body)
               .send()
               .await?;

           Ok(response.bytes_stream())
       }
   }
   ```

**Validation criteria**:
- [ ] `cargo build -p velollm-proxy` compile sans erreur
- [ ] Le serveur démarre sur le port 8000
- [ ] `/health` retourne 200 OK
- [ ] Les requêtes sont forwardées vers Ollama
- [ ] Le streaming fonctionne (Server-Sent Events)

**Tests**:
```bash
# Démarrer le proxy
cargo run -p velollm-proxy

# Tester le health check
curl http://localhost:8000/health

# Tester le forwarding (Ollama doit tourner)
curl http://localhost:8000/api/tags

# Tester une génération
curl http://localhost:8000/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello",
  "stream": false
}'
```

---

#### TASK-022: OpenAI API Compatibility Layer
**Priority**: P0 (Critical Path)
**Estimated effort**: 10h
**Dependencies**: TASK-021

**Objectif**: Implémenter la compatibilité API OpenAI pour que les clients existants fonctionnent sans modification.

**Instructions**:

1. **Créer les types OpenAI** (`velollm-proxy/src/types/openai.rs`):
   ```rust
   use serde::{Deserialize, Serialize};

   #[derive(Debug, Deserialize)]
   pub struct ChatCompletionRequest {
       pub model: String,
       pub messages: Vec<Message>,
       #[serde(default)]
       pub temperature: Option<f32>,
       #[serde(default)]
       pub max_tokens: Option<u32>,
       #[serde(default)]
       pub stream: bool,
       #[serde(default)]
       pub tools: Option<Vec<Tool>>,
       #[serde(default)]
       pub tool_choice: Option<ToolChoice>,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Message {
       pub role: Role,
       pub content: Option<String>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub tool_calls: Option<Vec<ToolCall>>,
       #[serde(skip_serializing_if = "Option::is_none")]
       pub tool_call_id: Option<String>,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   #[serde(rename_all = "lowercase")]
   pub enum Role {
       System,
       User,
       Assistant,
       Tool,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Tool {
       #[serde(rename = "type")]
       pub tool_type: String, // "function"
       pub function: FunctionDef,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct FunctionDef {
       pub name: String,
       pub description: Option<String>,
       pub parameters: serde_json::Value,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct ToolCall {
       pub id: String,
       #[serde(rename = "type")]
       pub call_type: String,
       pub function: FunctionCall,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct FunctionCall {
       pub name: String,
       pub arguments: String,
   }

   #[derive(Debug, Serialize)]
   pub struct ChatCompletionResponse {
       pub id: String,
       pub object: String,
       pub created: u64,
       pub model: String,
       pub choices: Vec<Choice>,
       pub usage: Usage,
   }

   #[derive(Debug, Serialize)]
   pub struct Choice {
       pub index: u32,
       pub message: Message,
       pub finish_reason: String,
   }

   #[derive(Debug, Serialize)]
   pub struct Usage {
       pub prompt_tokens: u32,
       pub completion_tokens: u32,
       pub total_tokens: u32,
   }
   ```

2. **Implémenter le convertisseur OpenAI ↔ Ollama** (`velollm-proxy/src/convert.rs`):
   ```rust
   use crate::types::{openai, ollama};

   pub fn openai_to_ollama(req: openai::ChatCompletionRequest) -> ollama::ChatRequest {
       ollama::ChatRequest {
           model: req.model,
           messages: req.messages.into_iter().map(convert_message).collect(),
           stream: Some(req.stream),
           options: Some(ollama::Options {
               temperature: req.temperature,
               num_predict: req.max_tokens.map(|n| n as i32),
               ..Default::default()
           }),
           tools: req.tools.map(|tools| {
               tools.into_iter().map(convert_tool).collect()
           }),
           ..Default::default()
       }
   }

   pub fn ollama_to_openai(resp: ollama::ChatResponse, model: &str) -> openai::ChatCompletionResponse {
       // ... conversion logic
   }
   ```

3. **Implémenter le handler** (`velollm-proxy/src/routes/chat.rs`):
   ```rust
   pub async fn chat_completions(
       State(state): State<Arc<AppState>>,
       Json(request): Json<ChatCompletionRequest>,
   ) -> Result<impl IntoResponse, ProxyError> {
       let ollama_request = convert::openai_to_ollama(request.clone());

       if request.stream {
           // Return SSE stream
           let stream = state.proxy.forward_stream("/api/chat", &ollama_request).await?;
           Ok(Sse::new(stream.map(|chunk| {
               // Convert each chunk to OpenAI format
           })))
       } else {
           // Return single response
           let response = state.proxy.forward("/api/chat", &ollama_request).await?;
           let ollama_resp: ollama::ChatResponse = response.json().await?;
           Ok(Json(convert::ollama_to_openai(ollama_resp, &request.model)))
       }
   }
   ```

**Validation criteria**:
- [ ] `/v1/chat/completions` accepte le format OpenAI
- [ ] Les réponses sont au format OpenAI
- [ ] Le streaming SSE fonctionne
- [ ] Compatible avec les SDK OpenAI (Python, Node.js)

**Tests**:
```python
# test_openai_compat.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

#### TASK-023: Tool Calling Enhancement
**Priority**: P0 (Critical Path)
**Estimated effort**: 12h
**Dependencies**: TASK-022

**Objectif**: Améliorer la fiabilité et performance du tool-calling via le proxy.

**Problèmes à résoudre**:
1. JSON malformé dans les réponses tool
2. Tool calls dupliqués
3. Mauvaise détection du début/fin des tools
4. Pas de validation des arguments

**Instructions**:

1. **Créer le module tool optimizer** (`velollm-proxy/src/optimizer/tools.rs`):
   ```rust
   use serde_json::Value;
   use crate::types::openai::{Tool, ToolCall, FunctionCall};

   pub struct ToolOptimizer {
       /// JSON schema validators for each tool
       validators: HashMap<String, jsonschema::JSONSchema>,
       /// Retry configuration
       max_retries: u32,
   }

   impl ToolOptimizer {
       pub fn new() -> Self {
           Self {
               validators: HashMap::new(),
               max_retries: 3,
           }
       }

       /// Register tools and compile their JSON schemas
       pub fn register_tools(&mut self, tools: &[Tool]) {
           for tool in tools {
               if let Ok(schema) = jsonschema::JSONSchema::compile(&tool.function.parameters) {
                   self.validators.insert(tool.function.name.clone(), schema);
               }
           }
       }

       /// Fix common JSON issues in tool call arguments
       pub fn fix_json(&self, raw: &str) -> Result<String, ToolError> {
           // 1. Try parsing as-is
           if serde_json::from_str::<Value>(raw).is_ok() {
               return Ok(raw.to_string());
           }

           // 2. Try fixing common issues
           let fixed = raw
               // Remove markdown code blocks
               .trim_start_matches("```json")
               .trim_start_matches("```")
               .trim_end_matches("```")
               // Fix trailing commas
               .replace(",}", "}")
               .replace(",]", "]")
               // Fix unquoted keys (simple cases)
               .trim();

           if serde_json::from_str::<Value>(fixed).is_ok() {
               return Ok(fixed.to_string());
           }

           // 3. Try extracting JSON from mixed content
           if let Some(json) = self.extract_json(raw) {
               return Ok(json);
           }

           Err(ToolError::InvalidJson(raw.to_string()))
       }

       /// Validate tool call arguments against schema
       pub fn validate(&self, tool_name: &str, args: &Value) -> Result<(), ToolError> {
           if let Some(schema) = self.validators.get(tool_name) {
               if schema.is_valid(args) {
                   Ok(())
               } else {
                   let errors: Vec<_> = schema.iter_errors(args).collect();
                   Err(ToolError::ValidationFailed(errors))
               }
           } else {
               // No schema, accept anything
               Ok(())
           }
       }

       /// Process and fix tool calls from model response
       pub fn process_tool_calls(&self, calls: Vec<ToolCall>) -> Vec<ToolCall> {
           let mut seen_ids = HashSet::new();
           let mut result = Vec::new();

           for mut call in calls {
               // Deduplicate by ID
               if seen_ids.contains(&call.id) {
                   continue;
               }
               seen_ids.insert(call.id.clone());

               // Fix JSON arguments
               if let Ok(fixed) = self.fix_json(&call.function.arguments) {
                   call.function.arguments = fixed;
               }

               result.push(call);
           }

           result
       }

       /// Extract JSON object from mixed text content
       fn extract_json(&self, text: &str) -> Option<String> {
           // Find first { and matching }
           let start = text.find('{')?;
           let mut depth = 0;
           let mut end = start;

           for (i, c) in text[start..].char_indices() {
               match c {
                   '{' => depth += 1,
                   '}' => {
                       depth -= 1;
                       if depth == 0 {
                           end = start + i + 1;
                           break;
                       }
                   }
                   _ => {}
               }
           }

           let json_str = &text[start..end];
           if serde_json::from_str::<Value>(json_str).is_ok() {
               Some(json_str.to_string())
           } else {
               None
           }
       }
   }
   ```

2. **Implémenter le retry intelligent** (`velollm-proxy/src/optimizer/retry.rs`):
   ```rust
   pub struct ToolRetryStrategy {
       pub max_attempts: u32,
       pub backoff_ms: u64,
   }

   impl ToolRetryStrategy {
       pub async fn execute_with_retry<F, T, E>(
           &self,
           mut operation: F,
           tool_optimizer: &ToolOptimizer,
       ) -> Result<T, E>
       where
           F: FnMut() -> Future<Output = Result<T, E>>,
       {
           let mut attempts = 0;
           loop {
               match operation().await {
                   Ok(result) => return Ok(result),
                   Err(e) if attempts < self.max_attempts => {
                       attempts += 1;
                       tokio::time::sleep(Duration::from_millis(self.backoff_ms)).await;
                       // Could modify prompt to ask for better formatting
                   }
                   Err(e) => return Err(e),
               }
           }
       }
   }
   ```

3. **Intégrer dans le flux de requête**:
   ```rust
   // Dans routes/chat.rs
   pub async fn chat_completions(
       State(state): State<Arc<AppState>>,
       Json(mut request): Json<ChatCompletionRequest>,
   ) -> Result<impl IntoResponse, ProxyError> {
       // Register tools for validation
       if let Some(ref tools) = request.tools {
           state.tool_optimizer.lock().await.register_tools(tools);
       }

       let response = forward_to_ollama(&state, &request).await?;

       // Post-process tool calls
       if let Some(ref mut choices) = response.choices {
           for choice in choices {
               if let Some(ref mut tool_calls) = choice.message.tool_calls {
                   *tool_calls = state.tool_optimizer
                       .lock().await
                       .process_tool_calls(std::mem::take(tool_calls));
               }
           }
       }

       Ok(Json(response))
   }
   ```

**Validation criteria**:
- [ ] JSON malformé est automatiquement corrigé
- [ ] Tool calls dupliqués sont filtrés
- [ ] Arguments sont validés contre le schema
- [ ] Retry automatique sur échec de parsing
- [ ] Tests avec cas limites (markdown dans JSON, trailing commas, etc.)

---

#### TASK-024: Request Batching & Queuing
**Priority**: P1
**Estimated effort**: 10h
**Dependencies**: TASK-021, TASK-017 (Scheduler de Phase 2)

**Objectif**: Implémenter le batching intelligent des requêtes pour maximiser le throughput.

**Instructions**:

1. **Créer le request batcher** (`velollm-proxy/src/batcher.rs`):
   ```rust
   use tokio::sync::{mpsc, oneshot};
   use std::time::Duration;

   pub struct RequestBatcher {
       /// Pending requests waiting to be batched
       pending: Vec<PendingRequest>,
       /// Maximum batch size
       max_batch_size: usize,
       /// Maximum wait time before flushing
       max_wait: Duration,
       /// Sender to scheduler
       scheduler_tx: mpsc::Sender<Batch>,
   }

   struct PendingRequest {
       request: ChatCompletionRequest,
       response_tx: oneshot::Sender<ChatCompletionResponse>,
       arrived_at: Instant,
   }

   impl RequestBatcher {
       pub async fn add_request(
           &mut self,
           request: ChatCompletionRequest,
       ) -> oneshot::Receiver<ChatCompletionResponse> {
           let (tx, rx) = oneshot::channel();

           self.pending.push(PendingRequest {
               request,
               response_tx: tx,
               arrived_at: Instant::now(),
           });

           // Check if we should flush
           if self.should_flush() {
               self.flush().await;
           }

           rx
       }

       fn should_flush(&self) -> bool {
           self.pending.len() >= self.max_batch_size
               || self.pending.first()
                   .map(|r| r.arrived_at.elapsed() > self.max_wait)
                   .unwrap_or(false)
       }

       async fn flush(&mut self) {
           if self.pending.is_empty() {
               return;
           }

           let batch = std::mem::take(&mut self.pending);
           // Send to scheduler for processing
           self.scheduler_tx.send(Batch { requests: batch }).await.ok();
       }
   }
   ```

2. **Intégrer avec le Scheduler de Phase 2**:
   ```rust
   use velollm_core::scheduler::{Scheduler, Request, SchedulerConfig};

   pub struct ProxyScheduler {
       scheduler: Scheduler,
       batcher: RequestBatcher,
   }

   impl ProxyScheduler {
       pub async fn process_batch(&mut self, batch: Batch) {
           // Add all requests to scheduler
           for pending in batch.requests {
               let request = Request::new(
                   self.next_id(),
                   tokenize(&pending.request),
                   pending.request.max_tokens.unwrap_or(512),
               );
               self.scheduler.add_request(request);
           }

           // Schedule and process
           loop {
               let output = self.scheduler.schedule();
               if !output.has_work() {
                   break;
               }

               // Process batch through Ollama
               // ...
           }
       }
   }
   ```

**Validation criteria**:
- [ ] Requêtes sont groupées par modèle
- [ ] Batch flush après max_batch_size ou timeout
- [ ] Throughput augmente avec la charge
- [ ] Latence reste acceptable (P99 < 2x single request)

---

### Sprint 8: Caching & Performance (Semaines 21-24)

#### TASK-025: Semantic Cache
**Priority**: P1
**Estimated effort**: 12h
**Dependencies**: TASK-021

**Objectif**: Cache intelligent basé sur la similarité sémantique des prompts.

**Instructions**:

1. **Créer le module cache** (`velollm-proxy/src/cache/mod.rs`):
   ```rust
   pub mod exact;
   pub mod semantic;
   pub mod kv;

   pub use exact::ExactCache;
   pub use semantic::SemanticCache;
   ```

2. **Implémenter le cache exact** (`velollm-proxy/src/cache/exact.rs`):
   ```rust
   use lru::LruCache;
   use std::hash::{Hash, Hasher};
   use xxhash_rust::xxh3::xxh3_64;

   pub struct ExactCache {
       cache: LruCache<u64, CachedResponse>,
       max_age: Duration,
   }

   impl ExactCache {
       pub fn get(&mut self, request: &ChatCompletionRequest) -> Option<&CachedResponse> {
           let key = self.hash_request(request);
           self.cache.get(&key).filter(|r| r.is_fresh(self.max_age))
       }

       pub fn insert(&mut self, request: &ChatCompletionRequest, response: ChatCompletionResponse) {
           let key = self.hash_request(request);
           self.cache.put(key, CachedResponse::new(response));
       }

       fn hash_request(&self, req: &ChatCompletionRequest) -> u64 {
           let json = serde_json::to_string(req).unwrap_or_default();
           xxh3_64(json.as_bytes())
       }
   }
   ```

3. **Implémenter le cache sémantique** (`velollm-proxy/src/cache/semantic.rs`):
   ```rust
   use ort::{Session, Value}; // ONNX Runtime for embeddings

   pub struct SemanticCache {
       /// Embedding model (e.g., all-MiniLM-L6-v2)
       embedder: Session,
       /// Vector store (in-memory for now)
       vectors: Vec<(Vec<f32>, CachedResponse)>,
       /// Similarity threshold
       threshold: f32,
   }

   impl SemanticCache {
       pub async fn get(&self, request: &ChatCompletionRequest) -> Option<&CachedResponse> {
           let query_text = self.extract_query_text(request);
           let query_embedding = self.embed(&query_text)?;

           // Find most similar cached response
           let mut best_match: Option<(f32, &CachedResponse)> = None;
           for (embedding, response) in &self.vectors {
               let similarity = cosine_similarity(&query_embedding, embedding);
               if similarity > self.threshold {
                   if best_match.map(|(s, _)| similarity > s).unwrap_or(true) {
                       best_match = Some((similarity, response));
                   }
               }
           }

           best_match.map(|(_, r)| r)
       }

       fn embed(&self, text: &str) -> Option<Vec<f32>> {
           // Run through ONNX model
           // ...
       }

       fn extract_query_text(&self, req: &ChatCompletionRequest) -> String {
           req.messages
               .iter()
               .filter(|m| matches!(m.role, Role::User))
               .filter_map(|m| m.content.as_ref())
               .collect::<Vec<_>>()
               .join("\n")
       }
   }

   fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
       let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
       let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
       let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
       dot / (norm_a * norm_b)
   }
   ```

**Validation criteria**:
- [ ] Cache exact avec TTL configurable
- [ ] Cache sémantique avec threshold configurable
- [ ] Hit rate > 30% sur workloads répétitifs
- [ ] Latence cache hit < 5ms

---

#### TASK-026: Metrics & Observability
**Priority**: P1
**Estimated effort**: 6h
**Dependencies**: TASK-021

**Objectif**: Métriques Prometheus et dashboard de monitoring.

**Metrics à collecter**:
- `velollm_requests_total` (counter): Total requests by model, status
- `velollm_request_duration_seconds` (histogram): Request latency
- `velollm_tokens_per_second` (gauge): Current throughput
- `velollm_cache_hits_total` (counter): Cache hit/miss
- `velollm_queue_size` (gauge): Pending requests
- `velollm_active_batches` (gauge): Concurrent batches

---

#### TASK-027: CLI Integration
**Priority**: P2
**Estimated effort**: 4h
**Dependencies**: TASK-021

**Objectif**: Ajouter la commande `velollm serve` pour démarrer le proxy.

**Instructions**:

1. **Ajouter la commande dans velollm-cli**:
   ```rust
   #[derive(Subcommand)]
   enum Commands {
       // ... existing commands
       /// Start the VeloLLM proxy server
       Serve {
           /// Port to listen on
           #[arg(short, long, default_value = "8000")]
           port: u16,
           /// Ollama URL
           #[arg(long, default_value = "http://localhost:11434")]
           ollama_url: String,
           /// Enable semantic cache
           #[arg(long)]
           semantic_cache: bool,
           /// Max batch size
           #[arg(long, default_value = "8")]
           max_batch_size: usize,
       },
   }
   ```

2. **Usage**:
   ```bash
   # Start proxy with defaults
   velollm serve

   # Custom configuration
   velollm serve --port 9000 --semantic-cache --max-batch-size 16

   # Then use with any OpenAI-compatible client
   export OPENAI_BASE_URL=http://localhost:8000/v1
   ```

---

### Sprint 9: Advanced Optimizations (Semaines 25-28)

#### TASK-028: Prompt Compression
**Priority**: P2
**Estimated effort**: 8h
**Dependencies**: TASK-022

**Objectif**: Réduire la taille des prompts sans perdre d'information.

**Techniques**:
- Token deduplication dans le contexte
- Summarization des messages anciens
- Compression des system prompts répétés

---

#### TASK-029: Speculative Prefetch
**Priority**: P2
**Estimated effort**: 10h
**Dependencies**: TASK-024

**Objectif**: Prédire les prochaines requêtes et pré-générer les réponses.

---

#### TASK-030: Multi-Model Load Balancing
**Priority**: P2
**Estimated effort**: 8h
**Dependencies**: TASK-024

**Objectif**: Router les requêtes vers le modèle optimal selon la complexité.

---

## 📊 Progress Tracking

### Phase 3: Proxy Intelligent
- [x] TASK-021: HTTP Server Foundation ✅
- [x] TASK-022: OpenAI API Compatibility Layer ✅
- [x] TASK-023: Tool Calling Enhancement ✅
- [x] TASK-024: Request Batching & Queuing ✅
- [x] TASK-025: Semantic Cache ✅
- [x] TASK-026: Metrics & Observability ✅
- [x] TASK-027: CLI Integration ✅
- [x] TASK-028: Prompt Compression ✅
- [x] TASK-029: Speculative Prefetch ✅
- [x] TASK-030: Multi-Model Load Balancing ✅

**Progress**: 10/10 tasks (100%) 🎉

---

## 🚀 Quick Start (After Implementation)

```bash
# 1. Start Ollama
ollama serve &

# 2. Start VeloLLM Proxy
velollm serve --port 8000

# 3. Use with any OpenAI-compatible client
export OPENAI_BASE_URL=http://localhost:8000/v1

# Python example
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 📚 Resources

- [axum documentation](https://docs.rs/axum)
- [OpenAI API reference](https://platform.openai.com/docs/api-reference)
- [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [vLLM architecture](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [OptiLLM (reference)](https://github.com/codelion/optillm)

---

**Phase 3 Complete! 🎉 Ready for Phase 4: GUI Dashboard & IDE Integrations**
