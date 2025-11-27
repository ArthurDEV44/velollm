# Guide de benchmarking VeloLLM

## Aperçu

VeloLLM inclut une suite de benchmarking complète pour mesurer les performances d'inférence LLM sur différents matériels et configurations.

## Démarrage rapide

### Prérequis

1. **Ollama installé et en cours d'exécution** :
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Modèle téléchargé** :
   ```bash
   ollama pull llama3.2:3b
   ```

### Exécuter les benchmarks

```bash
# Exécuter avec le modèle par défaut (llama3.2:3b)
velollm benchmark

# Spécifier un modèle différent
velollm benchmark -m llama3.1:8b

# Sauvegarder les résultats en JSON
velollm benchmark -o results.json

# Utiliser un backend différent (futur)
velollm benchmark -b llamacpp
```

## Benchmarks standard

La suite inclut trois benchmarks standard :

### 1. Complétion courte (50 jetons)
**Prompt** : "Write a hello world program in Python"
**Itérations** : 5
**Objectif** : Mesurer la vitesse de génération de base

### 2. Complétion moyenne (150 jetons)
**Prompt** : "Explain how neural networks learn through backpropagation in detail"
**Itérations** : 3
**Objectif** : Mesurer le débit soutenu

### 3. Génération de code (200 jetons)
**Prompt** : "Write a Rust function to compute the Fibonacci sequence using dynamic programming"
**Itérations** : 3
**Objectif** : Mesurer les performances de génération de code

## Métriques collectées

### Jetons par seconde (tok/s)
- **Définition** : Nombre de jetons générés par seconde
- **Plus haut est mieux**
- **Plages typiques** :
  - CPU uniquement : 5-20 tok/s
  - GPU milieu de gamme (RTX 3060) : 30-60 tok/s
  - GPU haut de gamme (RTX 4090) : 80-150 tok/s

### Temps jusqu'au premier jeton (TTFT)
- **Définition** : Temps entre la requête et le premier jeton généré
- **Plus bas est mieux**
- **Composants** :
  - Temps d'évaluation du prompt
  - Temps de génération du premier jeton
- **Plages typiques** :
  - Petits modèles (3B) : 50-200ms
  - Grands modèles (70B) : 500-2000ms

### Temps total
- **Définition** : Durée complète de la requête
- **Inclut** :
  - Traitement du prompt
  - Toute la génération de jetons
  - Formatage de la réponse

## Exemple de sortie

```
🚀 VeloLLM Benchmark Suite

Backend: ollama
Model: llama3.2:3b

Checking Ollama availability... ✓

Running 3 benchmarks...

═══════════════════════════════════════════════════════

Running benchmark: short_completion (5 iterations)
  Iteration 1/5... 82.3 tok/s (612ms)
  Iteration 2/5... 85.1 tok/s (593ms)
  Iteration 3/5... 84.7 tok/s (596ms)
  Iteration 4/5... 83.9 tok/s (602ms)
  Iteration 5/5... 84.2 tok/s (599ms)
  Average: 84.0 tok/s, TTFT: 127.3ms

Running benchmark: medium_completion (3 iterations)
  Iteration 1/3... 78.5 tok/s (1913ms)
  Iteration 2/3... 79.2 tok/s (1896ms)
  Iteration 3/3... 78.9 tok/s (1903ms)
  Average: 78.9 tok/s, TTFT: 145.6ms

Running benchmark: code_generation (3 iterations)
  Iteration 1/3... 76.3 tok/s (2621ms)
  Iteration 2/3... 77.1 tok/s (2596ms)
  Iteration 3/3... 76.8 tok/s (2605ms)
  Average: 76.7 tok/s, TTFT: 152.1ms

═══════════════════════════════════════════════════════

📊 Benchmark Summary

short_completion:
  Tokens/s: 84.0
  TTFT: 127.3ms
  Total tokens: 252
  Total time: 3.0s

medium_completion:
  Tokens/s: 78.9
  TTFT: 145.6ms
  Total tokens: 447
  Total time: 5.7s

code_generation:
  Tokens/s: 76.7
  TTFT: 152.1ms
  Total tokens: 593
  Total time: 7.7s

Overall Average:
  Tokens/s: 79.9
  TTFT: 141.7ms

💡 Tip: Use -o <file> to save results to JSON
```

## Format de sortie JSON

```json
[
  {
    "config": {
      "name": "short_completion",
      "model": "llama3.2:3b",
      "prompt": "Write a hello world program in Python",
      "max_tokens": 50,
      "iterations": 5
    },
    "tokens_per_second": 84.0,
    "time_to_first_token_ms": 127.3,
    "total_time_ms": 3002.5,
    "total_tokens": 252,
    "prompt_eval_count": 12,
    "eval_count": 50,
    "timestamp": "2025-01-15T10:30:00Z"
  },
  ...
]
```

## Comparer les résultats

### Avant/Après optimisation

```bash
# Exécuter la ligne de base
velollm benchmark -o baseline.json

# Appliquer les optimisations
velollm optimize -o velollm.sh
source velollm.sh

# Exécuter optimisé
velollm benchmark -o optimized.json

# Comparer (manuel)
jq '.[0].tokens_per_second' baseline.json
jq '.[0].tokens_per_second' optimized.json
```

### Entre différents matériels

Créer une base de données de benchmarks :

```bash
# Sur chaque système
velollm detect > hardware.json
velollm benchmark -o benchmark.json

# Organiser
mkdir benchmarks/rtx-4090
mv hardware.json benchmarks/rtx-4090/
mv benchmark.json benchmarks/rtx-4090/
```

## Utilisation avancée

### Benchmarks personnalisés

Créer votre propre configuration de benchmark :

```rust
use velollm_benchmarks::{BenchmarkConfig, BenchmarkRunner};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        name: "custom_test".to_string(),
        model: "llama3.2:3b".to_string(),
        prompt: "Votre prompt personnalisé ici".to_string(),
        max_tokens: 100,
        iterations: 5,
    };

    let runner = BenchmarkRunner::new("ollama");
    let result = runner.run(&config).await?;

    println!("Tokens/s: {}", result.tokens_per_second);
    Ok(())
}
```

### Différents modèles

Tester plusieurs modèles :

```bash
for model in llama3.2:1b llama3.2:3b llama3.1:8b; do
    echo "=== Testing $model ==="
    velollm benchmark -m $model -o results-$model.json
done
```

## Interpréter les résultats

### Indicateurs de bonnes performances

✅ **Jetons/s élevés** : Utilisation efficace du GPU
✅ **TTFT faible** : Traitement rapide du prompt
✅ **Cohérence entre itérations** : Performance stable

### Problèmes de performance

❌ **Jetons/s faibles** : Vérifier l'utilisation GPU, pression mémoire
❌ **TTFT élevé** : Prompt trop long ou encodeur de prompt lent
❌ **Variance entre itérations** : Limitation thermique ou processus en arrière-plan

## Dépannage

### "Ollama is not running"

```bash
# Vérifier le statut d'Ollama
systemctl status ollama  # Linux
ollama serve            # Démarrage manuel

# Vérifier que le modèle est disponible
ollama list
ollama pull llama3.2:3b
```

### "Model not found"

```bash
# Lister les modèles disponibles
ollama list

# Télécharger le modèle
ollama pull llama3.2:3b
```

### Performance lente

**Vérifier l'utilisation GPU** :
```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi
```

**Vérifier la VRAM** :
- S'assurer que le modèle tient dans la VRAM
- Fermer les autres applications GPU
- Essayer un modèle plus petit ou la quantification

**Vérifier l'utilisation CPU** :
```bash
top
htop
```

### Erreurs réseau

```bash
# Vérifier l'API Ollama
curl http://localhost:11434/api/tags

# Essayer un port différent
velollm benchmark --ollama-url http://localhost:11434
```

## Conseils d'optimisation des performances

### 1. Utiliser une taille de modèle appropriée

- **<8Go VRAM** : llama3.2:1b ou 3b
- **8-16Go VRAM** : llama3.1:8b (quantification Q4)
- **>16Go VRAM** : llama3.1:13b ou plus grand

### 2. Optimiser les paramètres Ollama

```bash
# Augmenter la fenêtre de contexte
export OLLAMA_NUM_CTX=4096

# Taille de lot
export OLLAMA_NUM_BATCH=512

# Keep alive
export OLLAMA_KEEP_ALIVE=5m
```

### 3. Déchargement GPU

```bash
# Décharger toutes les couches vers le GPU
export OLLAMA_NUM_GPU=99
```

### 4. Réduire la charge en arrière-plan

- Fermer les navigateurs
- Arrêter les autres applications GPU
- Désactiver les effets de bureau accélérés par GPU

## Prochaines étapes

- Comparer vos résultats avec les benchmarks communautaires
- Expérimenter avec différents modèles
- Essayer le décodage spéculatif (à venir en Phase 2)
- Contribuer vos résultats à la base de données de benchmarks VeloLLM

## Références

- [Documentation API Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [ROADMAP VeloLLM](../../ROADMAP.md)
- [Guide de détection matérielle](hardware_detection.md)
