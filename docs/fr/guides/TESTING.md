# Tester VeloLLM

## Exécuter les tests

### Tous les tests

```bash
# Exécuter tous les tests
cargo test --all

# Avec sortie détaillée
cargo test --all -- --nocapture

# Ou utiliser Make
make test
make test-verbose
```

### Tests de crate spécifique

```bash
# Tester uniquement la bibliothèque principale
cargo test -p velollm-core

# Tester uniquement le CLI
cargo test -p velollm-cli

# Tester uniquement les benchmarks
cargo test -p velollm-benchmarks
```

### Fonction de test spécifique

```bash
# Exécuter un seul test
cargo test test_hardware_detection

# Exécuter tous les tests correspondant au motif
cargo test hardware
```

## Tests de détection matérielle

### Exécuter les tests matériels

```bash
# Tester la détection matérielle
cargo test -p velollm-core -- --nocapture

# Cela affichera le matériel détecté dans la console
```

**Sortie attendue :**
```
running 8 tests
test hardware_tests::tests::test_cpu_detection ... ok
test hardware_tests::tests::test_gpu_detection ... ok
test hardware_tests::tests::test_hardware_detection ... ok
test hardware_tests::tests::test_json_serialization ... ok
test hardware_tests::tests::test_memory_detection ... ok
test hardware_tests::tests::test_nvidia_detection_on_linux ... ok
test hardware_tests::tests::test_platform_string ... ok
test hardware_tests::tests::test_gpu_vendor_serialization ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Résultats des tests de détection GPU

Les tests afficheront des résultats différents selon le matériel disponible :

**GPU NVIDIA présent :**
```
GPU detected:
  Name: NVIDIA GeForce RTX 4090
  Vendor: Nvidia
  VRAM Total: 24564 MB (24 GB)
  VRAM Free: 23012 MB (22 GB)
  Driver: 535.129.03
  Compute Capability: 8.9
```

**Pas de GPU (CPU uniquement) :**
```
No GPU detected (running on CPU-only system)
```

**Apple Silicon :**
```
GPU detected:
  Name: Apple M2
  Vendor: Apple
  VRAM Total: 32768 MB (32 GB)
  VRAM Free: 32768 MB (32 GB)
```

## Tester le CLI

### Construire et tester localement

```bash
# Construire en mode debug
cargo build

# Exécuter la commande detect
./target/debug/velollm detect

# Tester avec Make
make run-detect
```

### Sortie attendue

```
🔍 Detecting hardware configuration...

=== System Information ===
OS: linux
Platform: linux-x86_64

=== CPU ===
Model: AMD Ryzen 9 7950X 16-Core Processor
Cores: 16
Threads: 32
Frequency: 4500 MHz

=== Memory ===
Total: 65536 MB (64.0 GB)
Available: 42384 MB (41.4 GB)
Used: 23152 MB (22.6 GB)

=== GPU ===
Name: NVIDIA GeForce RTX 4090
Vendor: Nvidia
VRAM Total: 24564 MB (24.0 GB)
VRAM Free: 23012 MB (22.5 GB)
Driver: 535.129.03
Compute Capability: 8.9

=== JSON Output ===
{
  "gpu": { ... },
  "cpu": { ... },
  ...
}
```

## Tests spécifiques à la plateforme

### Linux

**Prérequis :**
- NVIDIA : `nvidia-smi` installé
- AMD : `rocm-smi` installé
- Intel : `lspci` disponible (généralement pré-installé)

**Commandes de test :**
```bash
# Vérifier si nvidia-smi est disponible
which nvidia-smi

# Vérifier si rocm-smi est disponible
which rocm-smi

# Exécuter les tests
cargo test -p velollm-core
```

### macOS

**Prérequis :**
- `system_profiler` (intégré)
- `sysctl` (intégré)

**Commandes de test :**
```bash
# Tester la détection Apple Silicon
cargo test -p velollm-core -- --nocapture

# Devrait détecter M1/M2/M3 si exécuté sur Apple Silicon
```

### Windows

**Prérequis :**
- NVIDIA : `nvidia-smi.exe` dans PATH

**Commandes de test :**
```powershell
# Vérifier nvidia-smi
where nvidia-smi

# Exécuter les tests
cargo test -p velollm-core
```

## Intégration continue

Les tests s'exécutent automatiquement :
- À chaque push vers la branche `main`
- À chaque pull request

Voir `.github/workflows/ci.yml` pour la configuration.

**Le CI exécute les tests sur :**
- Ubuntu (Linux)
- macOS
- Windows

## Liste de vérification des tests manuels

Avant de créer une PR, vérifier :

- [ ] `cargo test --all` passe
- [ ] `cargo clippy --all` n'a pas d'avertissements
- [ ] `cargo fmt --all -- --check` passe
- [ ] `velollm detect` fonctionne sur votre système
- [ ] La sortie JSON est valide (tester avec `jq`)

```bash
# Valider la sortie JSON
./target/debug/velollm detect | tail -n +17 | jq .
```

## Dépannage des tests

### Test échoue : "nvidia-smi not found"

**Cause :** Pilotes NVIDIA non installés ou nvidia-smi pas dans PATH

**Solution :** C'est attendu sur les systèmes sans GPU NVIDIA. Le test devrait passer avec `gpu: None`.

### Test échoue : "Memory detection returns 0"

**Cause :** Problèmes de permissions de la crate sysinfo

**Solution :** Exécuter avec les permissions appropriées ou vérifier les exigences spécifiques à l'OS.

### Test bloqué sur macOS

**Cause :** `system_profiler` peut être lent à la première exécution

**Solution :** Attendre 5-10 secondes ou exécuter `system_profiler SPDisplaysDataType` manuellement d'abord.

## Couverture de test

Pour générer la couverture de code :

```bash
# Installer tarpaulin
cargo install cargo-tarpaulin

# Générer la couverture
cargo tarpaulin --all --out Html

# Ouvrir le rapport
open tarpaulin-report.html
```

**Objectif de couverture :** >80% pour les modules principaux

## Écrire de nouveaux tests

### Modèle de test

```rust
#[test]
fn test_new_feature() {
    // Arranger
    let expected = ...;

    // Agir
    let result = function_to_test();

    // Affirmer
    assert_eq!(result, expected);
}
```

### Modèle de test matériel

```rust
#[test]
fn test_new_hardware_detection() {
    let hw = HardwareSpec::detect().unwrap();

    // Valider les résultats
    assert!(hw.some_field.is_some(), "Le champ devrait être détecté");

    // Afficher pour vérification manuelle
    println!("Detected: {:?}", hw.some_field);
}
```

## Tests de performance

Pour le code critique en performance :

```rust
#[test]
fn test_performance() {
    use std::time::Instant;

    let start = Instant::now();
    expensive_function();
    let elapsed = start.elapsed();

    // Devrait se terminer en <100ms
    assert!(elapsed.as_millis() < 100, "Trop lent: {:?}", elapsed);
}
```

## Prochaines étapes

- Voir [DEVELOPMENT.md](DEVELOPMENT.md) pour les instructions de construction
- Voir [CONTRIBUTING.md](../../CONTRIBUTING.md) pour les directives de contribution
