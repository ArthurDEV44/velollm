# VeloLLM

**Pilote automatique pour l'inférence LLM locale** - Optimisation des performances sans configuration pour Ollama, llama.cpp et plus encore.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml)

## Le problème

L'inférence LLM locale est **35 à 50 fois plus lente** que les solutions cloud (vLLM, Morph) malgré un matériel comparable. VeloLLM comble cet écart en apportant des optimisations de niveau production aux déploiements locaux.

**État actuel** :
- Cloud (vLLM) : 10 000+ jetons/s avec décodage spéculatif
- Local (Ollama) : 200-300 jetons/s (utilisateur moyen)

**Objectif de VeloLLM** : Réduire cet écart de performance grâce à des optimisations intelligentes et automatiques.

---

## Démarrage rapide

### Installation

```bash
# Depuis crates.io (bientôt disponible)
cargo install velollm

# Depuis les sources
git clone https://github.com/ArthurDEV44/velollm.git
cd velollm
cargo install --path velollm-cli
```

### Utilisation

```bash
# 1. Détecter votre matériel
velollm detect

# 2. Optimiser la configuration Ollama
velollm optimize --dry-run  # Aperçu des modifications
velollm optimize -o velollm.sh
source velollm.sh

# 3. Benchmarker les performances
velollm benchmark

# 4. Comparer avant/après
velollm benchmark --compare baseline.json optimized.json
```

---

## Fonctionnalités

### Phase 1 (MVP - Actuelle)

- **Détection matérielle** : Détection automatique du GPU (NVIDIA/AMD/Apple), CPU, RAM
- **Auto-configuration Ollama** : Optimiser l'utilisation de la VRAM, la taille des lots, la fenêtre de contexte
- **Suite de benchmarks** : Mesurer les jetons/s, le temps jusqu'au premier jeton, l'utilisation de la mémoire
- **Décodage spéculatif** : Accélération de 1,5 à 2,5x via l'intégration d'un modèle brouillon

### Phase 2 (Mois 4-6)

- **PagedAttention** : Réduction de 70% de la fragmentation du cache KV
- **Batching continu** : Gérer efficacement 4 à 8 utilisateurs simultanés
- **Hybride CPU-GPU** : Placement intelligent des couches et déchargement
- **Multi-backend** : Support pour llama.cpp, LocalAI, vLLM

### Phase 3 (Mois 7-12)

- **Interface graphique** : Surveillance des performances en temps réel
- **Intégrations IDE** : VSCode, Continue.dev, Cursor
- **Support Mamba/MoE** : Architectures de modèles de nouvelle génération
- **Place de marché de configurations** : Base de données d'optimisation pilotée par la communauté

Voir [ROADMAP.md](../../ROADMAP.md) pour tous les détails.

---

## Résultats des benchmarks

### Performance attendue (Objectifs Phase 1)

| Matériel | Modèle | Base | VeloLLM | Accélération |
|----------|--------|------|---------|--------------|
| RTX 4090 24GB | Llama 3.1 8B | ~28 tok/s | 60-70 tok/s | 2,1-2,5x |
| RTX 3060 12GB | Llama 3.2 3B | ~35 tok/s | 70-85 tok/s | 2,0-2,4x |
| M2 Max 32GB | Llama 3.1 8B | ~22 tok/s | 45-55 tok/s | 2,0-2,5x |

Voir [BENCHMARKS.md](../../BENCHMARKS.md) pour la méthodologie et les résultats détaillés.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│      Couche d'orchestration VeloLLM             │
│  • Détection matérielle                         │
│  • Auto-configuration                           │
│  • Profilage des performances                   │
└────┬──────────┬──────────┬──────────┬───────────┘
     │          │          │          │
┌────▼─────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐
│  Ollama  │ │llama.cpp│ │LocalAI │ │  vLLM  │
│ Adapteur │ │Adapteur │ │Adapteur│ │Adapteur│
└──────────┘ └─────────┘ └────────┘ └────────┘
```

**Technologies principales** :
- **Backend** : Rust (optimisations critiques pour les performances)
- **CLI/Outils** : TypeScript/Node.js (expérience développeur)
- **Bindings** : Python (compatibilité écosystème ML)

---

## Statut du développement

**Phase actuelle** : Phase 1 - Développement MVP

| Tâche | Statut |
|-------|--------|
| Configuration du dépôt | ✅ Terminé |
| Système de build | ✅ Terminé |
| Détection matérielle | ⏳ Planifié |
| Suite de benchmarks | ⏳ Planifié |
| PoC décodage spéculatif | ⏳ Planifié |
| Optimisation Ollama | ⏳ Planifié |

Suivre la progression : [TODO.md](../../TODO.md)

---

## Contribution

Nous accueillons les contributions ! VeloLLM est en développement précoce et a besoin d'aide pour :

- **Optimisations principales** : PagedAttention, décodage spéculatif
- **Adaptateurs backend** : Support pour plus de moteurs d'inférence
- **Benchmarking** : Tests sur diverses configurations matérielles
- **Documentation** : Guides, tutoriels, documentation API

Voir [CONTRIBUTING.md](../../CONTRIBUTING.md) pour les directives.

---

## Feuille de route

**Phase 1 (Mois 1-3)** : MVP avec accélération 2-3x
- Intégration du décodage spéculatif
- Auto-configuration Ollama
- Benchmarking de base

**Phase 2 (Mois 4-6)** : Optimisations avancées (3-5x)
- Implémentation PagedAttention
- Batching continu pour le local
- Support multi-backend

**Phase 3 (Mois 7-12)** : Écosystème (5-10x)
- Interface graphique et surveillance
- Intégrations IDE
- Alternatives d'architecture (Mamba, MoE)

Détails complets : [ROADMAP.md](../../ROADMAP.md)

---

## Pourquoi VeloLLM ?

### Différenciation

| Fonctionnalité | Ollama | vLLM | LM Studio | VeloLLM |
|----------------|--------|------|-----------|---------|
| Cible | Simplicité | Prod cloud | Utilisateurs desktop | Performance locale |
| Décodage spéculatif | ❌ | ❌ | ✅ | ✅ Auto-configuré |
| PagedAttention | ❌ | ✅ | ❌ | ✅ Adapté local |
| Batching continu | ❌ | ✅ | ❌ | ✅ Multi-utilisateur |
| Auto-optimisation | ❌ | ❌ | Partiel | ✅ Adapté au matériel |
| Open Source | ✅ | ✅ | ❌ | ✅ |

### Proposition de valeur

**VeloLLM = "Pilote automatique pour l'inférence IA locale"**

1. **Sans configuration** : Détecte le matériel, applique automatiquement les paramètres optimaux
2. **Adapté au matériel** : S'adapte dynamiquement (ordinateur portable vs station de travail vs serveur)
3. **Multi-backend** : Fonctionne avec Ollama, llama.cpp, LocalAI de manière transparente
4. **Transparent** : Surveillance détaillée, métriques, explications des optimisations
5. **Piloté par la communauté** : Open source, extensible, bien documenté

---

## Recherche & Références

Ce projet s'appuie sur :

- [llama.cpp](https://github.com/ggml-org/llama.cpp) : Fondation pour le décodage spéculatif
- [vLLM](https://github.com/vllm-project/vllm) : PagedAttention et batching continu
- [Ollama](https://github.com/ollama/ollama) : Expérience utilisateur et conception API
- [Mamba](https://github.com/state-spaces/mamba) : Exploration d'architectures alternatives

Articles clés :
- [PagedAttention](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) : Optimisation de la mémoire
- [Décodage spéculatif](https://arxiv.org/abs/2211.17192) : Accélération de l'inférence

---

## Licence

Licence MIT - voir [LICENSE](../../LICENSE) pour les détails.

---

## Contact

- **Issues** : [GitHub Issues](https://github.com/ArthurDEV44/velollm/issues)
- **Discussions** : [GitHub Discussions](https://github.com/ArthurDEV44/velollm/discussions)
- **Dépôt** : [github.com/ArthurDEV44/velollm](https://github.com/ArthurDEV44/velollm)

---

**Statut** : 🚧 Développement précoce - MVP Phase 1 en cours

Construit avec ❤️ par la communauté VeloLLM.
