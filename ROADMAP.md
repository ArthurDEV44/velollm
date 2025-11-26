# VeloLLM - Roadmap de Développement

## Vision

VeloLLM est un **autopilot pour l'inférence locale d'IA**, visant à combler l'écart de performance 35-50x entre les solutions cloud optimisées et les déploiements locaux.

**Objectif principal**: Apporter les optimisations de niveau production (vLLM, Morph) aux utilisateurs locaux avec une configuration automatique intelligente.

---

## Phase 1: MVP - Fondations & Validation (Mois 1-3)

### Objectifs Clés
- Valider la faisabilité technique des optimisations principales
- Créer un wrapper Ollama intelligent avec auto-configuration
- Démontrer un speedup mesurable (2x minimum)

### Livrables

#### 1.1 Validation Technique (Semaines 1-2)

**Objectif**: Prouver que les optimisations fonctionnent en local

##### Speculative Decoding PoC
- [ ] Fork llama.cpp et identifier le code de speculative decoding existant
  - Fichiers clés: `common/speculative.cpp`, `examples/speculative/`
- [ ] Créer un benchmark comparatif Ollama vanilla vs speculative
  - Paire de test: Llama 3.1 8B (main) + Llama 3.2 1B (draft)
  - Objectif: 1.8x-2.5x speedup sur génération de texte
- [ ] Documenter les paramètres optimaux découverts
  - Draft tokens: 5-10
  - Sampling strategy: top-k=10
  - Context overlap requirements

##### KV Cache Quantization
- [ ] Analyser l'implémentation actuelle du KV cache dans llama.cpp
  - Fichiers: `ggml/src/ggml-backend.cpp`, structures de données
- [ ] Implémenter quantization 16-bit → 4-bit du KV cache
  - Réduction attendue: 4x de la mémoire
- [ ] Mesurer l'impact sur la qualité (perplexity tests)
  - Seuil acceptable: <2% de dégradation

##### Hardware Detection
- [ ] Système de détection automatique
  - GPU: type, VRAM disponible, compute capability
  - CPU: cores, threads, cache L3
  - RAM système: capacité, bande passante
- [ ] Base de données de configurations optimales par hardware
  - Format JSON: `{gpu_model: {vram: X, optimal_batch: Y, ...}}`

#### 1.2 Wrapper Ollama Intelligent (Semaines 3-6)

**Objectif**: Tool qui optimise automatiquement Ollama sans modification

##### Auto-Configuration Engine
- [ ] Scanner les paramètres Ollama actuels
  - Lire `~/.ollama/config.json` ou équivalent
  - Parser `ollama ps` pour les modèles chargés
- [ ] Appliquer les configurations optimales
  - `OLLAMA_NUM_PARALLEL`: basé sur VRAM disponible
  - `OLLAMA_MAX_LOADED_MODELS`: mémoire management
  - `OLLAMA_KEEP_ALIVE`: stratégie de warming intelligente
  - Context window optimization: `num_ctx` basé sur use-case
- [ ] Mode dry-run pour preview des changements
  - Afficher: paramètres actuels → recommandés → gain estimé

##### CLI de Base
```bash
# Installation
npm install -g velollm
# ou
cargo install velollm

# Commandes essentielles
velollm detect              # Affiche hardware détecté
velollm optimize            # Applique auto-config à Ollama
velollm benchmark <model>   # Mesure performance avant/après
velollm serve <model>       # Lance serveur optimisé
```

#### 1.3 Benchmarking Suite (Semaines 7-8)

**Objectif**: Prouver les gains avec données mesurables

##### Metrics Tracker
- [ ] Implémentation des mesures clés
  - **Tokens/s**: débit de génération
  - **Time to First Token (TTFT)**: latence initiale
  - **Memory Usage**: VRAM + RAM consommées
  - **Throughput**: requêtes/minute (multi-request)
- [ ] Comparaison automatique
  - Baseline: Ollama vanilla
  - Optimized: VeloLLM config
  - Target: 2-3x speedup minimum

##### Test Suite Standard
```yaml
benchmarks:
  - name: "Short completion"
    prompt_length: 100 tokens
    completion_length: 50 tokens
    iterations: 100

  - name: "Long conversation"
    prompt_length: 2000 tokens
    completion_length: 500 tokens
    iterations: 20

  - name: "Code generation"
    prompt_length: 500 tokens
    completion_length: 200 tokens
    iterations: 50
```

##### Hardware Coverage
- [ ] Tests sur configurations représentatives
  - **Gaming laptop**: RTX 3060 Mobile, 16GB RAM
  - **Workstation**: RTX 4090, 64GB RAM
  - **MacBook Pro**: M2 Max, 32GB unified memory
  - **CPU only**: 32 cores, 128GB RAM

#### 1.4 Documentation MVP

- [ ] **README.md**: Quick start, installation, basic usage
- [ ] **BENCHMARKS.md**: Résultats mesurés par hardware
- [ ] **CONFIG_GUIDE.md**: Explication des paramètres optimisés
- [ ] **ARCHITECTURE.md**: Design decisions, code organization

### Critères de Succès Phase 1

✅ **Performance**: 2x speedup démontré sur au moins 3 configurations hardware
✅ **Usability**: Installation en <5 minutes, optimisation en 1 commande
✅ **Compatibility**: Fonctionne avec Ollama existant sans modification
✅ **Documentation**: Un nouveau utilisateur peut reproduire les benchmarks

---

## Phase 2: Optimisations Avancées (Mois 4-6)

### Objectifs Clés
- Implémenter les techniques d'optimisation avancées
- Support multi-backend (Ollama, llama.cpp direct, LocalAI)
- Atteindre 3-5x speedup

### Livrables

#### 2.1 PagedAttention pour Local

**Objectif**: Réduire la fragmentation mémoire du KV cache de 90%

##### Implémentation Core
- [ ] Étudier l'implémentation vLLM de PagedAttention
  - Repo: `vllm-project/vllm`, fichiers `attention/backends/`
- [ ] Adapter à llama.cpp
  - Paging strategy: blocks de 16-32 tokens
  - Memory allocator: custom pool manager
- [ ] Gestion dynamique des pages
  - Allocation à la demande
  - Défragmentation en arrière-plan
  - Eviction LRU pour contextes longs

##### Performance Targets
- [ ] Réduction mémoire: 70% → 10% de fragmentation
- [ ] Augmentation batch size supporté: 2-4x
- [ ] Pas de régression de vitesse (<5%)

#### 2.2 Continuous Batching Local

**Objectif**: Traiter plusieurs requêtes simultanées sans idle GPU

##### Architecture
```
┌─────────────────────────────────────────────┐
│          Request Queue                       │
│  [Req1: prompt] [Req2: gen step 5]          │
│  [Req3: gen step 2] [Req4: prompt]          │
└───────────────┬─────────────────────────────┘
                │
        ┌───────▼───────┐
        │   Scheduler   │ ← Dynamic batch assembly
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │  llama.cpp    │
        │   Backend     │
        └───────────────┘
```

##### Implémentation
- [ ] Request queue avec priorités
  - FIFO pour équité
  - Priority boosting pour latence
- [ ] Dynamic batch assembly
  - Mixer: nouveaux prompts + continuations en cours
  - Max batch size: auto-adapté à VRAM
- [ ] Iteration-level batching
  - Retirer les requêtes terminées du batch
  - Ajouter nouvelles sans attendre

##### Use Cases
- API locale multi-utilisateurs (famille/équipe)
- IDE plugins avec multiples requêtes simultanées
- Agent workflows avec parallélisation

#### 2.3 CPU-GPU Hybrid Execution

**Objectif**: Exploiter la RAM système pour réduire la pression VRAM

##### Stratégies d'Offloading
- [ ] **Layer-wise offloading**
  - Auto-détection: layers sur GPU vs CPU
  - Critère: temps de transfert < temps de calcul
- [ ] **KV cache splitting**
  - Keys quantifiés (4-bit) sur GPU
  - Values (FP16) sur CPU RAM
  - Reconstruction à la volée
- [ ] **Prefetching intelligent**
  - Anticiper les layers nécessaires
  - Pipeline: compute GPU pendant transfer CPU→GPU

##### Scheduler Adaptatif
```python
# Pseudo-code du scheduler
def place_layer(layer_idx, layer_size, compute_cost):
    if gpu_vram_free > layer_size:
        if gpu_compute_time < cpu_compute_time * 0.8:
            return GPU

    if cpu_ram_free > layer_size:
        return CPU

    return OFFLOAD  # Swap to disk as last resort
```

#### 2.4 Multi-Backend Support

**Objectif**: Fonctionner avec n'importe quel backend local

##### Adapters Architecture
```
┌──────────────────────────────────────────────┐
│         VeloLLM Orchestration Layer          │
└────┬──────────┬──────────┬──────────┬────────┘
     │          │          │          │
┌────▼─────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐
│  Ollama  │ │llama.cpp│ │LocalAI │ │  vLLM  │
│ Adapter  │ │ Adapter │ │ Adapter│ │ Adapter│
└──────────┘ └─────────┘ └────────┘ └────────┘
```

##### Implémentation
- [ ] **Interface abstraite commune**
  - `load_model()`, `generate()`, `unload_model()`
  - Unified config format
- [ ] **Ollama Adapter**
  - API: `/api/generate`, `/api/chat`
  - Config injection: env vars + API params
- [ ] **llama.cpp Direct Adapter**
  - Binary exec: `llama-cli`, `llama-server`
  - Config: command-line args
- [ ] **LocalAI Adapter**
  - OpenAI-compatible API
  - Model configuration via YAML
- [ ] **vLLM Local Adapter**
  - `vllm serve` en mode local
  - Inherits PagedAttention, continuous batching

##### Auto-Selection Logic
```yaml
backend_selection:
  if: "ollama_running"
    use: ollama_adapter
  elif: "llama_cpp_installed"
    use: llamacpp_adapter
  elif: "vllm_installed AND gpu_vram > 8GB"
    use: vllm_adapter
  else:
    install: "ollama"  # Fallback simple
```

#### 2.5 Advanced Quantization

**Objectif**: Adaptation dynamique de la précision

##### Techniques Implémentées
- [ ] **GPTQ/AWQ Support**
  - Intégration avec llama.cpp quantization
  - Auto-download de quantized models si disponibles
- [ ] **Mixed Precision Inference**
  - Attention layers: FP16 (critique pour qualité)
  - FFN layers: INT4 (tolérant à quantization)
  - Embeddings: INT8 (bon compromis)
- [ ] **Dynamic Precision Switching**
  ```python
  if memory_pressure > 90%:
      downgrade_precision()  # FP16 → INT8 → INT4

  if quality_metric < threshold:
      upgrade_precision()    # INT4 → INT8 → FP16
  ```

##### Quality Monitoring
- [ ] Perplexity tracking en temps réel
- [ ] Automatic rollback si dégradation >5%
- [ ] User-configurable quality/speed trade-off

### Critères de Succès Phase 2

✅ **Performance**: 3-5x speedup vs Ollama vanilla
✅ **Memory**: 50% réduction de VRAM usage via PagedAttention
✅ **Concurrency**: 4-8 utilisateurs simultanés sans dégradation
✅ **Flexibility**: 3+ backends supportés avec config unifiée

---

## Phase 3: Écosystème & Production-Ready (Mois 7-12)

### Objectifs Clés
- Interface graphique intuitive
- Intégrations avec outils populaires
- Support architectures alternatives (Mamba, MoE)
- Community building & marketplace

### Livrables

#### 3.1 GUI & Monitoring Dashboard

**Objectif**: Expérience utilisateur de niveau production

##### Desktop App (Tauri + React)
```
┌─────────────────────────────────────────────────┐
│  VeloLLM Dashboard                    [≡] [○] [X]│
├─────────────────────────────────────────────────┤
│  📊 Performance      🔧 Config      📚 Models    │
├─────────────────────────────────────────────────┤
│                                                  │
│  ⚡ Real-time Metrics                           │
│  ┌──────────────────────────────────────────┐  │
│  │  Tokens/s: 87.3  ▁▂▃▅▇█▇▅▃▂▁             │  │
│  │  VRAM: 6.2/24 GB ████░░░░░░░░             │  │
│  │  Active Requests: 3                       │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  🎯 Active Models                               │
│  ┌──────────────────────────────────────────┐  │
│  │  ● llama3.1:8b    87 tok/s    [Optimized]│  │
│  │    ↳ Draft: llama3.2:1b (Spec. Decoding) │  │
│  │  ○ codellama:13b            [Unloaded]   │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  💡 Recommendations                             │
│  • Enable PagedAttention for 2x batch size      │
│  • Download llama3.2:1b for speculative boost   │
│                                                  │
└─────────────────────────────────────────────────┘
```

##### Features
- [ ] **Live Performance Charts**
  - Tokens/s over time
  - Memory usage (VRAM, RAM)
  - Request latency distribution
- [ ] **Model Management**
  - One-click model download
  - Auto-download optimal draft models
  - Disk space monitoring
- [ ] **Configuration UI**
  - Preset profiles: "Max Speed", "Balanced", "Max Quality"
  - Advanced: sliders pour tous les paramètres
  - Import/export configurations
- [ ] **Benchmark Runner**
  - Built-in benchmark suite
  - Compare: before/after optimization
  - Export reports (PDF, JSON)

#### 3.2 IDE Integrations

**Objectif**: VeloLLM comme backend pour coding assistants

##### VSCode Extension
- [ ] Extension marketplace: "VeloLLM for VSCode"
- [ ] Features:
  - Auto-détection de VeloLLM local
  - Sélection de modèles optimisés
  - Performance overlay dans status bar
- [ ] Compatible avec: Continue.dev, Cody, Cursor (via API)

##### API Universelle
```typescript
// OpenAI-compatible API
POST /v1/chat/completions
{
  "model": "llama3.1:8b",
  "messages": [...],
  "velollm": {
    "enable_speculative": true,
    "draft_model": "auto",
    "optimize_latency": true
  }
}
```

##### Plugins Supportés
- [ ] **Continue.dev**: Custom provider configuration
- [ ] **LangChain**: VeloLLM LLM wrapper
- [ ] **LlamaIndex**: Custom connector
- [ ] **Open WebUI**: Backend option

#### 3.3 Support Architectures Alternatives

**Objectif**: Tirer parti des modèles next-gen

##### Mamba / State Space Models
- [ ] **Detection automatique**
  - Identifier les modèles Mamba vs Transformer
  - File: `config.json` → `"architecture": "mamba"`
- [ ] **Optimisations spécifiques**
  - Pas de KV cache (constant memory)
  - Linear scaling pour longues séquences
- [ ] **Fallback intelligent**
  ```python
  if sequence_length > 8192:
      if mamba_model_available:
          switch_to_mamba()  # Better for long context
  ```

##### Mixture of Experts (MoE)
- [ ] **Expert Loading Strategy**
  - Profiling: identifier les experts fréquents
  - Lazy loading: charger experts à la demande
  - LRU cache: garder top-K experts en VRAM
- [ ] **Memory Optimization**
  - Shared expert parameters
  - Quantization agressive des experts rares
- [ ] **Model Support**
  - Mixtral 8x7B optimisé
  - DeepSeek MoE variants

#### 3.4 Community & Marketplace

**Objectif**: Ecosystem-driven optimization

##### Configuration Registry
```yaml
# Crowd-sourced optimal configs
configs:
  - hardware: "RTX 4090 24GB"
    model: "llama3.1:70b"
    config:
      quantization: "Q4_K_M"
      batch_size: 512
      speculative: true
      draft_model: "llama3.2:3b"
      kv_cache_quantization: 4
    benchmark:
      tokens_per_sec: 45.3
      submitted_by: "@user123"
      verified: true
```

##### Features
- [ ] **Config Sharing**
  - `velollm config publish`
  - Automatic hardware tagging
  - Upvote/downvote system
- [ ] **Draft Model Registry**
  - Optimal pairings: (main, draft) → speedup
  - Community testing & validation
- [ ] **Benchmark Leaderboard**
  - Top configurations par hardware
  - Filtrage: GPU type, RAM, OS

#### 3.5 Advanced Features

##### Cloud-Local Hybrid (Inspiré de Morph)
```yaml
routing_policy:
  - condition: "prompt_length < 1000 AND latency_critical"
    target: "local"

  - condition: "prompt_length > 8000 OR complexity_high"
    target: "cloud"  # Optionnel, user-configured

  - condition: "privacy_sensitive"
    target: "local"  # Force local
```

##### Implémentation
- [ ] Request router avec heuristiques
- [ ] Cloud providers support (optionnel)
  - OpenAI, Anthropic, etc. pour fallback
  - User consent requis
- [ ] Privacy-preserving: local by default

##### Smart Caching & Prefetching
```python
# Learning user patterns
if time.hour == 9 and user_role == "developer":
    preload("codellama:13b")
    preload_draft("codellama:1b")
    warm_kv_cache(common_code_snippets)

if conversation_context.includes("SQL"):
    preload("sqlcoder:7b")
```

- [ ] Usage pattern learning
- [ ] Context-aware model warming
- [ ] Conversation history analysis

### Critères de Succès Phase 3

✅ **UX**: GUI utilisable par non-tech users
✅ **Ecosystem**: 5+ integrations majeures (VSCode, LangChain, etc.)
✅ **Community**: 100+ shared configurations in registry
✅ **Performance**: 5-10x speedup sur cas d'usage production
✅ **Adoption**: 1000+ users actifs, 50+ contributors

---

## Métriques de Succès Globales

### Performance Targets

| Metric                    | Baseline (Ollama) | Phase 1 | Phase 2 | Phase 3 |
|---------------------------|-------------------|---------|---------|---------|
| **Tokens/s (8B model)**   | 20-30             | 40-60   | 60-100  | 100-150 |
| **TTFT (ms)**             | 200-500           | 100-200 | <100    | <50     |
| **Memory (8B FP16)**      | 16GB              | 12GB    | 8GB     | 6GB     |
| **Concurrent Users**      | 1                 | 1-2     | 4-8     | 8-16    |
| **Context Length (8GB)**  | 4K                | 8K      | 16K     | 32K     |

### Adoption Metrics

- **Phase 1**: 100+ GitHub stars, 10+ early adopters
- **Phase 2**: 1K+ stars, featured in 2+ tech blogs
- **Phase 3**: 5K+ stars, integration with major tools

---

## Stack Technique

### Core
- **Backend optimizations**: Rust (performance critique)
- **CLI & Tooling**: TypeScript/Node.js (developer UX)
- **Python bindings**: Pour ML community (LangChain, etc.)

### GUI
- **Desktop**: Tauri (Rust backend + React frontend)
- **Web Dashboard**: React + Recharts pour metrics

### Backend Adapters
- **llama.cpp**: C++ (direct fork/patches)
- **Communication**: gRPC pour high-performance IPC
- **Configuration**: YAML + JSON schemas

---

## Risques & Mitigations

### Risques Techniques

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Incompatibilité llama.cpp versions | High | Medium | Version pinning, automated tests |
| PagedAttention complexité | High | High | Start with simple paging, iterate |
| Performance overhead layers | Medium | Medium | Extensive profiling, zero-copy designs |
| Multi-backend support bugs | Medium | High | Comprehensive integration tests |

### Risques Écosystème

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Ollama API changes | Medium | Medium | Adapter pattern, version matrix |
| Community adoption slow | High | Medium | Early demos, benchmark transparency |
| Competition (LM Studio, etc.) | Medium | Low | Differentiate on open-source + perf |

---

## Prochaines Actions Immédiates

### Semaine 1-2: Setup & Validation

1. **Repository Setup**
   ```bash
   mkdir velollm && cd velollm
   git init
   # Structure: /src /benchmarks /docs /adapters
   ```

2. **Benchmark Baseline**
   - Mesurer Ollama vanilla sur 3 hardwares
   - Documenter: tokens/s, TTFT, memory

3. **Speculative Decoding PoC**
   - Fork llama.cpp
   - Test: Llama 3.1 8B + 3.2 1B
   - Target: 1.5x+ speedup minimum

4. **Community Engagement**
   - README avec vision claire
   - Issues templates pour contributions
   - Discord/discussions pour early feedback

### Semaine 3-4: MVP Development

5. **Hardware Detection**
   - Script multi-platform (Linux, macOS, Windows)
   - Output JSON avec specs complètes

6. **Ollama Auto-Config**
   - Parser config actuelle
   - Appliquer optimizations
   - `velollm optimize --dry-run`

7. **First Benchmark Report**
   - Publier résultats mesurés
   - Before/after comparisons
   - Invite community testing

---

## Conclusion

Cette roadmap est **ambitieuse mais réalisable** grâce à:

1. **Technologies matures**: Toutes les briques existent (llama.cpp, vLLM research)
2. **Timing parfait**: Explosion de l'IA locale, demande forte pour performance
3. **Gap évident**: 35-50x de différence à combler
4. **Approche lean**: MVP en 3 mois, validation rapide

**Positionnement unique**: "Autopilot pour l'inférence locale" - zero-config, multi-backend, hardware-aware.

**Next step**: Créer le premier benchmark comparatif pour valider l'approche. 🚀
