# Testing VeloLLM Proxy

This guide explains how to manually test the VeloLLM proxy after building it.

## Prerequisites

1. **Ollama must be running** on `localhost:11434`
2. **At least one supported model** must be available (Mistral or Llama)

```bash
# Start Ollama (if not already running)
ollama serve &

# Pull a supported model for tool calling
ollama pull llama3.2:3b
# or
ollama pull mistral:7b
```

## Building the Proxy

```bash
# Development build (faster compilation)
cargo build -p velollm-proxy

# Release build (optimized, recommended for testing)
cargo build --release -p velollm-proxy
```

## Starting the Proxy

### Option 1: Foreground (see logs directly)

```bash
./target/release/velollm-proxy
```

### Option 2: Background with log file

```bash
./target/release/velollm-proxy > /tmp/velollm-proxy.log 2>&1 &
PROXY_PID=$!
echo "Proxy started with PID: $PROXY_PID"

# View logs
tail -f /tmp/velollm-proxy.log
```

### Troubleshooting: Port Already in Use

If you get `Error: Address already in use (os error 98)`:

```bash
# Find what's using port 8000
ss -tlnp | grep 8000

# Kill the process using the port
fuser -k 8000/tcp

# Or kill by process name
pkill -f velollm-proxy

# Wait and retry
sleep 2
./target/release/velollm-proxy
```

## Testing Endpoints

### 1. Health Check

```bash
curl -s http://localhost:8000/health
```

Expected response:
```json
{"ollama":"connected","status":"healthy","version":"0.1.0"}
```

### 2. List Models (OpenAI Format)

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### 3. List Models (Ollama Format)

```bash
curl -s http://localhost:8000/api/tags | python3 -m json.tool
```

### 4. Simple Chat Completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

### 5. Tool Calling (Supported Model)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "max_tokens": 200
  }' | python3 -m json.tool
```

Expected: Response with `tool_calls` array containing function call.

### 6. Tool Calling with Mistral

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "messages": [{"role": "user", "content": "Calculate 5 + 3"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "Perform arithmetic",
          "parameters": {
            "type": "object",
            "properties": {
              "operation": {"type": "string", "enum": ["add", "subtract"]},
              "a": {"type": "number"},
              "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
          }
        }
      }
    ]
  }' | python3 -m json.tool
```

## Verifying Tool Optimization

Check the proxy logs for optimization messages:

```
INFO Using optimized tool calling for supported model model=llama3.2:3b model_type=Some(Llama)
INFO Registered tools for optimization count=1
INFO Processing tool calls count=1
INFO Tool calls processed original=1 processed=1 repairs=0 duplicates=0
```

### Testing Unsupported Model Warning

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi3:mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}]
  }'
```

Check logs for warning:
```
WARN Tool calling requested but model is not supported...
```

## Supported Models for Tool Calling

| Family | Models |
|--------|--------|
| **Mistral** | `mistral:7b`, `mistral:latest`, `mistral-small:*`, `mistral-nemo` |
| **Llama** | `llama3.1:*`, `llama3.2:*`, `llama3.3:*` |

**Not supported**: Qwen, Phi, Gemma, CodeLlama (will show warning but still work)

## Stopping the Proxy

```bash
# If running in foreground: Ctrl+C

# If running in background:
pkill -f velollm-proxy
# or
kill $PROXY_PID
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VELOLLM_PORT` | `8000` | Port to listen on |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama backend URL |
| `VELOLLM_VERBOSE` | `false` | Enable verbose logging |

Example:
```bash
VELOLLM_PORT=9000 OLLAMA_HOST=http://192.168.1.100:11434 ./target/release/velollm-proxy
```

## Running Automated Tests

```bash
# Run all proxy tests
cargo test -p velollm-proxy

# Run with output
cargo test -p velollm-proxy -- --nocapture

# Run specific test
cargo test -p velollm-proxy test_tool_call
```

---

# Tester le Proxy VeloLLM (Français)

Ce guide explique comment tester manuellement le proxy VeloLLM après sa compilation.

## Prérequis

1. **Ollama doit tourner** sur `localhost:11434`
2. **Au moins un modèle supporté** doit être disponible (Mistral ou Llama)

```bash
# Démarrer Ollama (si pas déjà lancé)
ollama serve &

# Télécharger un modèle supporté pour le tool calling
ollama pull llama3.2:3b
# ou
ollama pull mistral:7b
```

## Compiler le Proxy

```bash
# Build développement (compilation rapide)
cargo build -p velollm-proxy

# Build release (optimisé, recommandé pour les tests)
cargo build --release -p velollm-proxy
```

## Démarrer le Proxy

### Option 1: Premier plan (logs visibles)

```bash
./target/release/velollm-proxy
```

### Option 2: Arrière-plan avec fichier de log

```bash
./target/release/velollm-proxy > /tmp/velollm-proxy.log 2>&1 &
PROXY_PID=$!
echo "Proxy démarré avec PID: $PROXY_PID"

# Voir les logs
tail -f /tmp/velollm-proxy.log
```

### Dépannage: Port déjà utilisé

Si vous obtenez `Error: Address already in use (os error 98)`:

```bash
# Trouver ce qui utilise le port 8000
ss -tlnp | grep 8000

# Tuer le processus utilisant le port
fuser -k 8000/tcp

# Ou tuer par nom de processus
pkill -f velollm-proxy

# Attendre et réessayer
sleep 2
./target/release/velollm-proxy
```

## Tester les Endpoints

### 1. Vérification de santé

```bash
curl -s http://localhost:8000/health
```

Réponse attendue:
```json
{"ollama":"connected","status":"healthy","version":"0.1.0"}
```

### 2. Liste des modèles (Format OpenAI)

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### 3. Liste des modèles (Format Ollama)

```bash
curl -s http://localhost:8000/api/tags | python3 -m json.tool
```

### 4. Chat simple

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Dis bonjour en une phrase."}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

### 5. Tool Calling (Modèle supporté)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Quelle est la météo à Paris?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Obtenir la météo actuelle",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "Nom de la ville"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "max_tokens": 200
  }' | python3 -m json.tool
```

Attendu: Réponse avec tableau `tool_calls` contenant l'appel de fonction.

### 6. Tool Calling avec Mistral

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "messages": [{"role": "user", "content": "Calcule 5 + 3"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "Effectuer des opérations arithmétiques",
          "parameters": {
            "type": "object",
            "properties": {
              "operation": {"type": "string", "enum": ["add", "subtract"]},
              "a": {"type": "number"},
              "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
          }
        }
      }
    ]
  }' | python3 -m json.tool
```

## Vérifier l'optimisation des tools

Vérifiez les logs du proxy pour les messages d'optimisation:

```
INFO Using optimized tool calling for supported model model=llama3.2:3b model_type=Some(Llama)
INFO Registered tools for optimization count=1
INFO Processing tool calls count=1
INFO Tool calls processed original=1 processed=1 repairs=0 duplicates=0
```

### Tester l'avertissement pour modèle non supporté

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi3:mini",
    "messages": [{"role": "user", "content": "Bonjour"}],
    "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}]
  }'
```

Vérifiez les logs pour l'avertissement:
```
WARN Tool calling requested but model is not supported...
```

## Modèles supportés pour le Tool Calling

| Famille | Modèles |
|---------|---------|
| **Mistral** | `mistral:7b`, `mistral:latest`, `mistral-small:*`, `mistral-nemo` |
| **Llama** | `llama3.1:*`, `llama3.2:*`, `llama3.3:*` |

**Non supportés**: Qwen, Phi, Gemma, CodeLlama (afficheront un avertissement mais fonctionneront quand même)

## Arrêter le Proxy

```bash
# Si en premier plan: Ctrl+C

# Si en arrière-plan:
pkill -f velollm-proxy
# ou
kill $PROXY_PID
```

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `VELOLLM_PORT` | `8000` | Port d'écoute |
| `OLLAMA_HOST` | `http://localhost:11434` | URL du backend Ollama |
| `VELOLLM_VERBOSE` | `false` | Activer les logs verbeux |

Exemple:
```bash
VELOLLM_PORT=9000 OLLAMA_HOST=http://192.168.1.100:11434 ./target/release/velollm-proxy
```

## Lancer les tests automatisés

```bash
# Lancer tous les tests du proxy
cargo test -p velollm-proxy

# Lancer avec affichage
cargo test -p velollm-proxy -- --nocapture

# Lancer un test spécifique
cargo test -p velollm-proxy test_tool_call
```
