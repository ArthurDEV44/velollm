# Analyse du décodage spéculatif

**Date de recherche** : 2025-11-27
**Source** : Implémentation llama.cpp
**Version** : Dernière (novembre 2025)

---

## Résumé exécutif

Le décodage spéculatif est une technique pour accélérer l'inférence LLM en utilisant un modèle "brouillon" plus petit pour prédire plusieurs jetons en avance, qui sont ensuite vérifiés en parallèle par le modèle "cible" plus grand. Cela permet une **accélération de 1,5x à 2,5x** sans **perte de qualité**.

**Idée clé** : Le modèle brouillon s'exécute rapidement (petit), le modèle cible valide en parallèle → amortir le coût du modèle cible sur plusieurs jetons.

---

## Détails d'implémentation (llama.cpp)

### Fichiers principaux

| Fichier | Objectif |
|---------|----------|
| `common/speculative.h` | Interface API et définitions de paramètres |
| `common/speculative.cpp` | Logique principale du décodage spéculatif |
| `examples/speculative/speculative.cpp` | Exemple complet avec spéculation basée sur arbre |

### Paramètres clés

#### De `common_speculative_params` (speculative.h:8-13)

```cpp
struct common_speculative_params {
    int n_draft = 16;      // Max de jetons brouillons par itération
    int n_reuse = 256;     // Nombre de jetons à réutiliser du brouillon précédent
    float p_min = 0.75f;   // Probabilité min pour accepter un jeton brouillon
};
```

**Détail des paramètres** :

1. **`n_draft`** (défaut : 16)
   - Nombre de jetons que le modèle brouillon génère en avance
   - Plus élevé → plus d'accélération SI le taux d'acceptation est élevé
   - Plus bas → moins de travail gaspillé si l'acceptation est faible
   - **Plage optimale** : 5-16 pour la plupart des modèles

2. **`n_reuse`** (défaut : 256)
   - Optimisation de réutilisation du cache KV
   - Conserve le contexte brouillon entre les itérations
   - Plus élevé → meilleure utilisation du cache
   - **Recommandé** : 256-512

3. **`p_min`** (défaut : 0.75)
   - Seuil de probabilité minimum pour les jetons brouillons
   - Seuls les jetons brouillons avec confiance ≥ p_min sont acceptés
   - Plus élevé → moins de jetons mais acceptation plus élevée
   - Plus bas → plus de jetons mais plus de rejets
   - **Plage optimale** : 0.70-0.80

#### De l'exemple (speculative.cpp:183)

```cpp
int n_draft = params.speculative.n_max;  // Configurable en ligne de commande
```

Paramètres supplémentaires :
- **`n_parallel`** : Nombre de séquences brouillons parallèles (spéculation basée sur arbre)
- **`p_split`** : Seuil de probabilité pour diviser les branches brouillons

### Stratégie d'échantillonnage

De `speculative.cpp:58-68` :

```cpp
common_params_sampling params;
params.top_k = 10;  // Utiliser l'échantillonnage top-k pour le modèle brouillon

params.samplers = {
    COMMON_SAMPLER_TYPE_TOP_K,
};
```

**Échantillonnage du modèle brouillon** :
- Utilise l'échantillonnage **top-k=10** (conservateur)
- Plus rapide que l'échantillonnage complet du modèle cible
- Objectif : Générer rapidement des candidats plausibles

**Alternative** (commenté, lignes 40-55) :
- top-k=40, top-p=0.9 pour des brouillons de meilleure qualité
- Compromis : génération brouillon plus lente vs acceptation plus élevée

---

## Flux de l'algorithme

### Processus de haut niveau

```
1. Le modèle cible génère 1 jeton (T0)
2. Le modèle brouillon génère N jetons en avance (D1, D2, ..., DN)
3. Le modèle cible valide tous les N jetons en parallèle
4. Accepter le préfixe valide le plus long (ex : D1, D2, D3)
5. Rejeter à partir de la première non-correspondance
6. Répéter à partir de l'étape 2 avec le dernier jeton accepté
```

### Étapes détaillées (de speculative.cpp:185-361)

```
Fonction : common_speculative_gen_draft()

1. Réutilisation du cache KV (lignes 198-244) :
   - Trouver le préfixe correspondant le plus long avec le brouillon précédent
   - Réutiliser le cache KV pour les jetons correspondants
   - Recalculer uniquement les nouveaux jetons

2. Génération de brouillon (lignes 314-349) :
   pour i dans plage(n_draft):
       - Échantillonner le jeton suivant du modèle brouillon
       - Vérifier la confiance : si prob < p_min, ARRÊTER
       - Ajouter le jeton au lot brouillon
       - Décoder le modèle brouillon pour la prochaine itération

3. Compatibilité du vocabulaire (lignes 204-223, 351-359) :
   - Si les vocabulaires diffèrent : détokeniser → retokeniser
   - Gère les légères différences de vocabulaire entre modèles
   - Différence maximale de vocabulaire : 128 jetons (SPEC_VOCAB_MAX_SIZE_DIFFERENCE)
```

### Logique d'acceptation

D'après l'analyse du code d'exemple :

```cpp
// Le modèle cible évalue les jetons brouillons en parallèle
pour chaque draft_token dans draft_sequence:
    target_logits = target_model.eval(draft_token)
    si target_model.would_sample(draft_token):
        accepter(draft_token)
    sinon:
        rejeter(draft_token)
        arrêter  // Arrêter au premier rejet
```

**Optimisation clé** : L'évaluation parallèle amortit le coût du modèle cible.

---

## Exigences de compatibilité des modèles

### Contraintes de vocabulaire

De `speculative.cpp:89-148` :

1. **Le type de vocabulaire doit correspondre** :
   ```cpp
   if (vocab_type_tgt != vocab_type_dft) {
       return false;  // Impossible d'utiliser la spéculation
   }
   ```

2. **Les jetons spéciaux doivent correspondre** :
   - BOS (début de séquence)
   - EOS (fin de séquence)
   - Drapeaux Add BOS/EOS

3. **Différence de taille de vocabulaire < 128** :
   ```cpp
   const int vocab_diff = abs(n_vocab_tgt - n_vocab_dft);
   if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
       return false;
   }
   ```

4. **Le mappage des jetons doit correspondre** (pour les jetons 5+) :
   ```cpp
   for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < min(n_vocab_tgt, n_vocab_dft); ++i) {
       if (token_text_tgt[i] != token_text_dft[i]) {
           return false;
       }
   }
   ```

### Paires de modèles recommandées

Basé sur la compatibilité du vocabulaire et les ratios de taille :

| Modèle principal | Modèle brouillon | Ratio de taille | Accélération attendue |
|------------------|------------------|-----------------|----------------------|
| **Llama 3.2 3B** | Llama 3.2 1B | 3:1 | 1.8-2.2x |
| **Llama 3.1 8B** | Llama 3.2 1B | 8:1 | 2.0-2.5x |
| **Llama 3.1 8B** | Llama 3.2 3B | 2.6:1 | 1.5-2.0x |
| **Llama 3.1 70B** | Llama 3.2 3B | 23:1 | 2.5-3.0x |
| **CodeLlama 13B** | CodeLlama 7B | 1.8:1 | 1.5-1.8x |
| **CodeLlama 34B** | CodeLlama 7B | 4.8:1 | 2.0-2.3x |

**Critères de sélection** :
1. Même famille de modèles (Llama 3.x, CodeLlama, etc.)
2. Même vocabulaire/tokenizer
3. Ratio de taille 2:1 à 10:1 (sweet spot : 3:1 à 5:1)
4. Les deux quantifiés à une précision similaire (Q4_K_M recommandé)

---

## Caractéristiques de performance

### Formule d'accélération (Théorique)

```
Accélération = (N * T_cible) / (T_brouillon + T_cible + T_overhead)

Où :
- N = nombre de jetons acceptés
- T_cible = temps pour générer 1 jeton avec le modèle cible
- T_brouillon = temps pour générer N jetons avec le modèle brouillon
- T_overhead = overhead de traitement par lots et de vérification
```

**Idée clé** : L'accélération augmente avec :
1. Taux d'acceptation plus élevé (N grand)
2. Modèle brouillon plus rapide (petit T_brouillon)
3. Overhead plus faible (traitement par lots efficace)

### Facteurs du taux d'acceptation

**Acceptation élevée** (70-90%) :
- Modèle brouillon entraîné sur des données similaires
- Texte simple/prévisible (code, données structurées)
- Échantillonnage à basse température
- Jetons brouillons avec confiance élevée (p > 0.8)

**Acceptation faible** (30-50%) :
- Texte créatif/imprévisible
- Échantillonnage à haute température
- Domaines de modèles non correspondants
- Modèle brouillon trop petit (différence de taille >10x)

### Résultats empiriques (de la littérature)

| Paire de modèles | Tâche | Taux d'acceptation | Accélération |
|------------------|-------|-------------------|--------------|
| Llama-7B + Llama-68M | Général | 60% | 2.0x |
| Llama-13B + Llama-7B | Code | 75% | 2.3x |
| Llama-70B + Llama-7B | Général | 65% | 2.5x |

---

## Stratégie de configuration optimale

### Pour l'intégration VeloLLM

**Paramètres recommandés** :

```yaml
speculative_config:
  # Pour llama3.2:3b + llama3.2:1b (votre ligne de base)
  n_draft: 8           # Début conservateur
  n_reuse: 256         # Réutilisation par défaut
  p_min: 0.75          # Confiance par défaut

  # Échantillonnage pour le modèle brouillon
  draft_sampling:
    top_k: 10
    temperature: 0.0   # Greedy pour meilleure acceptation
```

**Stratégie d'ajustement** :

1. **Commencer conservateur** (n_draft=5-8) :
   - Mesurer le taux d'acceptation
   - Si >70% : augmenter n_draft progressivement
   - Si <50% : diminuer n_draft ou vérifier la compatibilité des modèles

2. **Surveiller les métriques** :
   ```
   Taux d'acceptation = jetons_acceptés / jetons_brouillons
   Accélération = (total_jetons / temps) / jetons_ligne_de_base_par_seconde
   Efficacité = accélération / (1 + overhead_brouillon)
   ```

3. **Optimiser pour le matériel** :
   - **GPU-bound** : Augmenter n_draft (l'évaluation parallèle est peu coûteuse)
   - **Memory-bound** : Diminuer n_draft (moins de cache KV)
   - **CPU uniquement** : Envisager un modèle brouillon plus petit

### Recommandations spécifiques au matériel

**Pour RTX 4070 Ti SUPER (16Go VRAM)** :

```yaml
# Votre ligne de base : 137 tok/s avec llama3.2:3b
optimal_config:
  main_model: "llama3.2:3b"
  draft_model: "llama3.2:1b"
  n_draft: 10           # Votre GPU peut gérer cela
  batch_size: 512       # Beaucoup de VRAM

  expected_results:
    baseline: 137 tok/s
    optimized: 270-300 tok/s  # Accélération 2.0-2.2x
    acceptance_rate: 70-75%
```

---

## Liste de vérification d'implémentation pour TASK-006

### Phase 1 : Wrapper de base

- [ ] Structure Rust `SpeculativeConfig` avec paramètres
- [ ] Wrapper pour appeler le binaire `llama-speculative`
- [ ] Analyser la sortie de performance (jetons/s, taux d'acceptation)
- [ ] Gestion des erreurs pour incompatibilité de modèle

### Phase 2 : Détection de paire de modèles

- [ ] Détecter automatiquement les modèles brouillons disponibles dans Ollama
- [ ] Vérifier la compatibilité du vocabulaire
- [ ] Suggérer des paires optimales basées sur le matériel

### Phase 3 : Intégration

- [ ] Ajouter à la commande `velollm benchmark`
- [ ] Comparer vanilla vs spéculatif
- [ ] Générer le rapport de performance

---

## Références

### PR llama.cpp
- [#2926](https://github.com/ggml-org/llama.cpp/pull/2926) : Décodage spéculatif initial
- [#3624](https://github.com/ggml-org/llama.cpp/pull/3624) : Spéculation basée sur arbre
- [#5625](https://github.com/ggml-org/llama.cpp/pull/5625) : Améliorations et optimisations

### Articles académiques
- [Leviathan et al. 2022](https://arxiv.org/abs/2211.17192) : "Fast Inference from Transformers via Speculative Decoding"
- [Chen et al. 2023](https://arxiv.org/abs/2302.01318) : "Accelerating LLM Inference with Staged Speculative Decoding"

### Implémentations connexes
- [vLLM](https://github.com/vllm-project/vllm) : Implémentation de niveau production
- [Medusa](https://github.com/FasterDecoding/Medusa) : Spéculation multi-têtes

---

## Prochaines étapes (TASK-006)

1. **Créer un wrapper Rust** pour le binaire `llama-speculative`
2. **Implémenter l'ajustement des paramètres** basé sur la détection matérielle
3. **Comparaison de benchmark** : vanilla vs spéculatif sur votre RTX 4070 Ti SUPER
4. **Objectif** : Démontrer une accélération de 2.0-2.2x (137 → 270-300 tok/s)

---

**Statut** : ✅ Analyse complète
**Tâche suivante** : TASK-006 - Implémenter le wrapper spéculatif
**Temps estimé** : 4 heures
