# VeloLLM

**Pilote automatique pour l'inférence LLM locale** - Proxy haute performance et boîte à outils d'optimisation pour Ollama et llama.cpp.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml)

---

## Le Problème

L'inférence LLM locale est **19x plus lente** que les solutions de production comme vLLM. VeloLLM comble cet écart en fournissant un proxy Rust haute performance qui optimise les requêtes, améliore la fiabilité du tool-calling, et apporte des fonctionnalités de niveau production aux déploiements locaux.

| Métrique | Production (vLLM) | Local (Ollama) | Écart |
|----------|-------------------|----------------|-------|
| Débit | 793 tokens/s | 41 tokens/s | 19x |
| Latence P99 | 80ms | 673ms | 8x |

**Objectif VeloLLM** : Apporter les performances de vLLM aux utilisateurs d'Ollama tout en conservant la simplicité.

---

## Qu'est-ce que VeloLLM ?

VeloLLM est un **proxy transparent** qui se place entre vos applications et Ollama. Il intercepte les appels API, applique des optimisations intelligentes, et les transfère à Ollama. Vos outils existants fonctionnent sans modification - changez simplement l'endpoint API.

### Avantages Clés

- **Remplacement direct** : Compatibilité complète avec l'API OpenAI
- **Amélioration du tool-calling** : Correction JSON, déduplication, validation de schéma
- **Optimisation des performances** : Batching des requêtes, cache intelligent, ordonnancement continu
- **Métriques & observabilité** : Suivi des tokens/s, latence, taux de cache hit
- **Gestion avancée de la mémoire** : PagedAttention pour un cache KV efficace

### Modèles Supportés pour le Tool Calling

- Mistral (mistral:7b, mistral-small:24b)
- Llama (llama3.2:3b, llama3.1:8b, llama3.1:70b)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VOS APPLICATIONS                                 │
│         Claude Code  │  Continue  │  Open WebUI  │  Apps Custom         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROXY VELOLLM :8000                                 │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      COUCHE API                                  │    │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │    │
│  │  │   Compatibilité  │ │   API Native     │ │    Streaming     │ │    │
│  │  │     OpenAI       │ │     Ollama       │ │       SSE        │ │    │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   COUCHE OPTIMISATION                            │    │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │    │
│  │  │   Optimiseur     │ │    Batcher de    │ │      Cache       │ │    │
│  │  │     Tools        │ │    Requêtes      │ │    Sémantique    │ │    │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  COUCHE ORDONNANCEMENT                           │    │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │    │
│  │  │    Scheduler     │ │   Gestionnaire   │ │  PagedAttention  │ │    │
│  │  │ Batching Continu │ │    de Blocs      │ │                  │ │    │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  OBSERVABILITÉ  │  Collecteur de Métriques  │  Prometheus       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKEND D'INFÉRENCE                                 │
│              Ollama :11434  │  llama.cpp  │  LocalAI                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flux de Requête

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Application │      │    Proxy     │      │    Cache     │
│              │      │   VeloLLM    │      │  Sémantique  │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       │  POST /v1/chat      │                     │
       │────────────────────>│                     │
       │                     │                     │
       │                     │  Vérifier cache     │
       │                     │────────────────────>│
       │                     │                     │
       │                     │    Cache Hit?       │
       │                     │<────────────────────│
       │                     │                     │
       │         ┌───────────┴───────────┐         │
       │         │                       │         │
       │    [Cache Hit]            [Cache Miss]    │
       │         │                       │         │
       │         ▼                       ▼         │
       │  ┌─────────────┐      ┌─────────────────┐ │
       │  │  Réponse    │      │    Batcher      │ │
       │  │   cachée    │      │  de Requêtes    │ │
       │  └──────┬──────┘      └────────┬────────┘ │
       │         │                      │          │
       │         │                      ▼          │
       │         │             ┌─────────────────┐ │
       │         │             │   Scheduler     │ │
       │         │             └────────┬────────┘ │
       │         │                      │          │
       │         │                      ▼          │
       │         │             ┌─────────────────┐ │
       │         │             │    Ollama       │ │
       │         │             └────────┬────────┘ │
       │         │                      │          │
       │         │    Streamer réponse  │          │
       │         │<─────────────────────┘          │
       │         │                                 │
       │<────────┴─────────────────────────────────│
       │     Retourner réponse                     │
       │                                           │
```

---

## Fonctionnalités

### Implémentées

#### Phase 1 : MVP (Complète)

| Fonctionnalité | Description |
|----------------|-------------|
| **Détection Matérielle** | Détection auto GPU (NVIDIA, AMD, Apple Silicon, Intel), CPU et mémoire |
| **Suite de Benchmarks** | Mesure tokens/s, TTFT, latence totale avec plusieurs profils |
| **Auto-Configuration Ollama** | Génération de variables d'environnement optimisées selon le matériel |
| **Analyse Décodage Spéculatif** | Recherche et recommandations de paramètres pour modèles brouillons |

#### Phase 2 : Optimisations Avancées (83% Complète)

| Fonctionnalité | Description |
|----------------|-------------|
| **Gestionnaire de Blocs PagedAttention** | Cache KV efficace en mémoire avec blocs de 16 tokens, comptage de références et CoW |
| **Intégration Cache KV llama.cpp** | Wrapper de cache paginé compatible avec l'API llama_memory_* |
| **Kernel CUDA Paged Attention** | Attention accélérée GPU avec support FP16/FP32 et GQA |
| **Scheduler Batching Continu** | Ordonnancement dynamique des requêtes avec priorité et préemption |

#### Phase 3 : Proxy Intelligent (50% Complète)

| Fonctionnalité | Description |
|----------------|-------------|
| **Serveur HTTP** | Serveur basé sur Axum avec middleware Tower |
| **Compatibilité API OpenAI** | Support complet pour `/v1/chat/completions`, `/v1/models` |
| **Amélioration Tool Calling** | Correction JSON automatique, déduplication, validation de schéma |
| **Batching des Requêtes** | Groupement des requêtes concurrentes, ordonnancement par priorité |
| **Cache Sémantique** | Matching par similarité d'embeddings, cache exact + sémantique |

### À Venir

| Fonctionnalité | Statut |
|----------------|--------|
| Métriques & Observabilité | Planifié |
| Intégration CLI (`velollm serve`) | Planifié |
| Compression de Prompts | Planifié |
| Prefetch Spéculatif | Planifié |
| Équilibrage Multi-Modèles | Planifié |

---

## Gestion Mémoire : PagedAttention

VeloLLM implémente PagedAttention pour une gestion efficace du cache KV, inspiré de vLLM.

```
    MÉMOIRE VIRTUELLE                TABLES DE BLOCS              MÉMOIRE GPU PHYSIQUE
    (Logique)                                                     (Blocs de 16 tokens)

┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│   Séquence 1    │              │  S1: [0, 3, 5]  │         ┌───>│    Bloc 0       │
│   "Bonjour,     │─────────────>│                 │─────────┤    ├─────────────────┤
│   comment..."   │              └─────────────────┘         │┌──>│    Bloc 1       │
└─────────────────┘                                          ││   ├─────────────────┤
                                                             ││┌─>│    Bloc 2       │
┌─────────────────┐              ┌─────────────────┐         │││  ├─────────────────┤
│   Séquence 2    │              │  S2: [1, 4]     │────────┐│││  │    Bloc 3       │<─┐
│   "Écris un     │─────────────>│                 │        │└┼┼─>├─────────────────┤  │
│   poème..."     │              └─────────────────┘        │ ││  │    Bloc 4       │<─┼─┐
└─────────────────┘                                         └─┼┼─>├─────────────────┤  │ │
                                                              ││  │    Bloc 5       │<─┘ │
┌─────────────────┐              ┌─────────────────┐          ││  ├─────────────────┤    │
│   Séquence 3    │              │  S3: [2, 6, 7]  │──────────┼┼─>│    Bloc 6       │    │
│   "Résume ce    │─────────────>│                 │──────────┘│  ├─────────────────┤    │
│   texte..."     │              └─────────────────┘           └─>│    Bloc 7       │    │
└─────────────────┘                                               └─────────────────┘    │
                                                                                         │
                                     Allocation non-contiguë ────────────────────────────┘
```

**Avantages** :
- **Réduction de 70%** de la fragmentation mémoire
- **Allocation dynamique** : Les séquences grandissent sans pré-allocation
- **Partage mémoire** : Copy-on-Write pour beam search et sampling parallèle
- **Préemption efficace** : Swap des séquences sans perdre le contexte

---

## Optimisation du Tool Calling

VeloLLM corrige les problèmes courants de tool calling qui surviennent avec les modèles locaux.

```
  RÉPONSE DU MODÈLE                    OPTIMISEUR TOOL                    APPEL PROPRE
  (Malformée)                                                              (Valide)

┌────────────────────┐           ┌────────────────────┐           ┌────────────────────┐
│ ```json            │           │                    │           │ {                  │
│ {name: 'weather',  │           │  ┌──────────────┐  │           │   "name":          │
│  args: {           │──────────>│  │ Correcteur   │  │           │     "get_weather", │
│    city: 'Paris',, │           │  │    JSON      │  │           │   "arguments": {   │
│  }}                │           │  └──────┬───────┘  │           │     "city": "Paris"│
│ ```                │           │         │          │           │   }                │
└────────────────────┘           │         ▼          │           │ }                  │
                                 │  ┌──────────────┐  │           └────────────────────┘
                                 │  │ Validateur   │  │
                                 │  │  de Schéma   │  │
                                 │  └──────┬───────┘  │
                                 │         │          │
                                 │         ▼          │
                                 │  ┌──────────────┐  │──────────>
                                 │  │ Dédupliqueur │  │
                                 │  └──────────────┘  │
                                 │                    │
                                 └────────────────────┘
```

**Corrections appliquées** :
- Suppression des blocs de code markdown
- Correction des virgules finales
- Quotation des clés non quotées
- Extraction du JSON depuis contenu mixte
- Validation contre les schémas de fonctions
- Déduplication des appels répétés

---

## Système de Cache

VeloLLM implémente un système de cache à deux niveaux pour des performances optimales.

```
                              REQUÊTE ENTRANTE
                                     │
                                     ▼
               ┌─────────────────────────────────────────┐
               │          NIVEAU 1 : CACHE EXACT         │
               │                                         │
               │  ┌─────────────┐    ┌─────────────────┐ │
               │  │  Hash XXH3  │───>│   Cache LRU     │ │
               │  │  (< 1µs)    │    │  (1000 entrées) │ │
               │  └─────────────┘    └────────┬────────┘ │
               │                              │          │
               └──────────────────────────────┼──────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                         [Cache Hit]                     [Cache Miss]
                              │                               │
                              ▼                               ▼
                    ┌─────────────────┐     ┌─────────────────────────────────┐
                    │  Réponse < 1ms  │     │    NIVEAU 2 : CACHE SÉMANTIQUE  │
                    └─────────────────┘     │                                 │
                                            │  ┌──────────────────────────┐   │
                                            │  │   Modèle d'Embedding     │   │
                                            │  │   (all-MiniLM-L6-v2)     │   │
                                            │  └────────────┬─────────────┘   │
                                            │               │                 │
                                            │               ▼                 │
                                            │  ┌──────────────────────────┐   │
                                            │  │   Recherche Similarité   │   │
                                            │  │   (cosine > 0.95)        │   │
                                            │  └────────────┬─────────────┘   │
                                            │               │                 │
                                            └───────────────┼─────────────────┘
                                                            │
                                            ┌───────────────┴───────────────┐
                                            │                               │
                                       [Match trouvé]                  [Pas de match]
                                            │                               │
                                            ▼                               ▼
                                  ┌─────────────────┐             ┌─────────────────┐
                                  │  Réponse < 5ms  │             │ Forward Ollama  │
                                  └─────────────────┘             └─────────────────┘
```

**Objectifs de performance** :
- Cache hit exact : < 1ms de latence
- Cache hit sémantique : < 5ms de latence
- Taux de cache hit : > 30% sur workloads répétitifs

---

## Structure du Projet

```
velollm/
│
├── velollm-core/                 # Bibliothèque core
│   ├── src/
│   │   ├── hardware.rs           # Détection matérielle
│   │   ├── optimizer.rs          # Optimisation config
│   │   ├── paged_attention/      # PagedAttention
│   │   │   ├── mod.rs
│   │   │   ├── block_allocator.rs
│   │   │   └── block_table.rs
│   │   └── scheduler.rs          # Batching continu
│   └── Cargo.toml
│
├── velollm-cli/                  # Binaire CLI
│   ├── src/
│   │   └── main.rs               # detect, benchmark, optimize
│   └── Cargo.toml
│
├── velollm-proxy/                # Serveur proxy
│   ├── src/
│   │   ├── main.rs               # Point d'entrée
│   │   ├── routes/               # Handlers HTTP
│   │   ├── optimizer/            # Tool calling, batching
│   │   └── cache/                # Cache exact + sémantique
│   └── Cargo.toml
│
├── velollm-benchmarks/           # Suite de benchmarks
│   └── src/lib.rs
│
├── adapters/
│   ├── ollama/                   # Config Ollama
│   │   └── src/lib.rs
│   └── llamacpp/                 # Intégration llama.cpp
│       ├── src/
│       │   ├── lib.rs
│       │   ├── kv_cache.rs       # Cache KV paginé
│       │   └── cuda_paged.rs     # Wrapper CUDA
│       └── cuda/
│           ├── paged_attention.cu
│           └── paged_attention.cuh
│
└── docs/
    ├── fr/                       # Documentation française
    └── research/                 # Notes de recherche
```

**Dépendances entre crates** :

```
                    ┌─────────────────┐
                    │   velollm-cli   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────┐
│ velollm-benchmarks│  │ velollm-core│  │  velollm-proxy  │
└────────┬────────┘  └──────┬──────┘  └────────┬────────┘
         │                  │                   │
         │                  ├───────────────────┘
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│ adapters/ollama │  │adapters/llamacpp│
└─────────────────┘  └─────────────────┘
```

---

## Statut de Développement

### Progression par Phase

```
PROGRESSION DU PROJET
═══════════════════════════════════════════════════════════════════════

Phase 1 : MVP                    ████████████████████████████████  100%
                                 12/12 tâches complètes

Phase 2 : Optimisations          ██████████████████████████░░░░░░   83%
                                 5/6 tâches actives (2 en attente)

Phase 3 : Proxy Intelligent      ████████████████░░░░░░░░░░░░░░░░   50%
                                 5/10 tâches complètes

───────────────────────────────────────────────────────────────────────
TOTAL                            ██████████████████████░░░░░░░░░░   76%
                                 22/29 tâches
```

| Phase | Statut | Progression |
|-------|--------|-------------|
| Phase 1 : MVP | Complète | 12/12 (100%) |
| Phase 2 : Optimisations Avancées | En cours | 5/6 actives (83%) |
| Phase 3 : Proxy Intelligent | En cours | 5/10 (50%) |

### Couverture de Tests

```
TESTS PAR CRATE
═══════════════════════════════════════════════════════════════════════

velollm-core              ████████████████████████████████████████  63
velollm-adapters-llamacpp ██████████████████████░░░░░░░░░░░░░░░░░░  29
velollm-cli (intégration) █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8
Doc tests                 █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8
velollm-adapters-ollama   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6
velollm-benchmarks        ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3

───────────────────────────────────────────────────────────────────────
TOTAL                                                               117
```

---

## Comparaison

```
                    │ Ollama │  vLLM  │LM Studio│ VeloLLM │
════════════════════╪════════╪════════╪═════════╪═════════╡
Cas d'usage         │Simpli- │  Prod  │   GUI   │  Perf   │
                    │  cité  │ Cloud  │ Desktop │ Locale  │
────────────────────┼────────┼────────┼─────────┼─────────┤
API OpenAI          │Partiel │Complet │ Partiel │ Complet │
────────────────────┼────────┼────────┼─────────┼─────────┤
Fix Tool Calling    │   ✗    │  N/A   │    ✗    │    ✓    │
────────────────────┼────────┼────────┼─────────┼─────────┤
PagedAttention      │   ✗    │   ✓    │    ✗    │    ✓    │
────────────────────┼────────┼────────┼─────────┼─────────┤
Request Batching    │   ✗    │   ✓    │    ✗    │    ✓    │
────────────────────┼────────┼────────┼─────────┼─────────┤
Cache Sémantique    │   ✗    │   ✗    │    ✗    │    ✓    │
────────────────────┼────────┼────────┼─────────┼─────────┤
Auto-optimisation   │   ✗    │   ✗    │ Partiel │    ✓    │
────────────────────┼────────┼────────┼─────────┼─────────┤
Langage             │   Go   │ Python │Electron │  Rust   │
────────────────────┼────────┼────────┼─────────┼─────────┤
Open Source         │   ✓    │   ✓    │    ✗    │    ✓    │
════════════════════╧════════╧════════╧═════════╧═════════╛
```

---

## Feuille de Route

```
2024                                                                    2025
 Jan   Fév   Mar   Avr   Mai   Jun   Jul   Aoû   Sep   Oct   Nov   Déc   Jan
  │     │     │     │     │     │     │     │     │     │     │     │     │
  │     │     │     │     │     │     │     │     │     │     │     │     │
PHASE 1 : MVP
  ├─────────────────────────────┤
  │  ████████████████████████   │  [TERMINÉ]
  │  Hardware Detection         │
  │  Benchmarking               │
  │  Ollama Config              │
  │                             │
PHASE 2 : OPTIMISATIONS AVANCÉES
                    ├───────────────────────────────┤
                    │  ████████████████████████░░░  │  [83%]
                    │  PagedAttention               │
                    │  Batching Continu             │
                    │  CUDA Kernels                 │
                    │                               │
PHASE 3 : PROXY INTELLIGENT
                                        ├───────────────────────────────┤
                                        │  ████████████░░░░░░░░░░░░░░░  │  [50%]
                                        │  HTTP Server                  │
                                        │  OpenAI API                   │
                                        │  Tool Calling                 │
                                        │  Caching                      │
                                        │  Métriques (en cours)         │
                                        │                               │
PHASE 4 : ÉCOSYSTÈME
                                                            ├─────────────────────┤
                                                            │  ░░░░░░░░░░░░░░░░░  │
                                                            │  GUI Dashboard      │
                                                            │  Intégrations IDE   │
                                                            │                     │
```

**Phase 1** (Complète) : MVP avec outils CLI
- Détection matérielle, benchmarking, configuration Ollama

**Phase 2** (83% Complète) : Optimisations avancées
- PagedAttention, scheduler de batching continu, kernels CUDA

**Phase 3** (En cours) : Proxy intelligent
- Compatibilité OpenAI, amélioration tool calling, cache, métriques

**Phase 4** (Planifiée) : Écosystème
- Dashboard GUI, intégrations IDE, marketplace de configurations

Détails complets : [ROADMAP.md](../../ROADMAP.md) | Suivi des tâches : [TODO.md](../../TODO.md)

---

## Contribution

Nous accueillons les contributions ! Domaines d'intérêt :

- **Performance** : Optimiser le proxy, réduire la latence
- **Tool Calling** : Améliorer la correction JSON, ajouter plus de cas limites
- **Cache** : Améliorer le cache sémantique avec de meilleurs embeddings
- **Tests** : Ajouter des tests d'intégration, benchmarker sur divers matériels
- **Documentation** : Améliorer les guides et la doc API

Voir [CONTRIBUTING.md](../../CONTRIBUTING.md) pour les directives.

---

## Licence

Licence MIT - voir [LICENSE](../../LICENSE) pour les détails.

---

## Liens

- **Dépôt** : [github.com/ArthurDEV44/velollm](https://github.com/ArthurDEV44/velollm)
- **Issues** : [GitHub Issues](https://github.com/ArthurDEV44/velollm/issues)
- **Discussions** : [GitHub Discussions](https://github.com/ArthurDEV44/velollm/discussions)

---

**Statut** : Phase 3 - Développement du proxy en cours (50% complète)

Construit avec Rust par la communauté VeloLLM.
