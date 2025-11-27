# Guide de développement VeloLLM

## Prérequis

### Requis

- **Rust** 1.70+ (installer via [rustup](https://rustup.rs/))
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

- **Git** pour le contrôle de version

### Optionnel mais recommandé

- **cargo-watch** : Reconstruction automatique lors des modifications de fichiers
  ```bash
  cargo install cargo-watch
  ```

- **cargo-tarpaulin** : Couverture de code
  ```bash
  cargo install cargo-tarpaulin
  ```

## Construire le projet

### Démarrage rapide

```bash
# Cloner le dépôt
git clone https://github.com/yourusername/velollm
cd velollm

# Construction en mode debug
cargo build

# Construction en mode release (optimisé)
cargo build --release

# Ou utiliser les raccourcis Make
make build        # Construction release
make build-dev    # Construction debug
```

### Structure du workspace

VeloLLM utilise un workspace Cargo avec trois crates :

```
velollm/
├── velollm-core/          # Bibliothèque principale (détection matérielle, optimisation)
├── velollm-cli/           # Application CLI
├── velollm-benchmarks/    # Bibliothèque de benchmarking
└── Cargo.toml             # Configuration du workspace
```

## Flux de travail de développement

### 1. Effectuer des modifications

```bash
# Éditer les fichiers dans src/
vim velollm-core/src/hardware.rs
```

### 2. Formater le code

```bash
cargo fmt --all
# Ou
make fmt
```

### 3. Vérifier avec Clippy

```bash
cargo clippy --all -- -D warnings
# Ou
make clippy
```

### 4. Exécuter les tests

```bash
cargo test --all
# Ou
make test

# Avec sortie
cargo test --all -- --nocapture
# Ou
make test-verbose
```

### 5. Construire et exécuter

```bash
# Construire le CLI
cargo build

# Exécuter les commandes
./target/debug/velollm detect
./target/debug/velollm benchmark
./target/debug/velollm optimize --dry-run

# Ou utiliser les raccourcis Make
make run-detect
make run-benchmark
make run-optimize
```

## Commandes de développement

### Utiliser Make

```bash
make help           # Afficher toutes les commandes disponibles
make build          # Construction release
make test           # Exécuter les tests
make fmt            # Formater le code
make clippy         # Linter le code
make doc            # Générer et ouvrir la documentation
make ci             # Exécuter toutes les vérifications CI (fmt + clippy + test)
```

### Utiliser Cargo directement

```bash
# Vérifier sans construire
cargo check --all

# Construire une crate spécifique
cargo build -p velollm-core

# Exécuter les tests pour une crate spécifique
cargo test -p velollm-benchmarks

# Générer la documentation
cargo doc --all --no-deps --open

# Installer le CLI localement
cargo install --path velollm-cli
```

## Mode surveillance (Reconstruction automatique)

Installer cargo-watch :
```bash
cargo install cargo-watch
```

Puis utiliser :
```bash
# Surveiller et reconstruire lors des modifications
cargo watch -x build

# Surveiller et exécuter les tests
cargo watch -x test

# Ou utiliser Make
make watch
make watch-test
```

## Tests

### Tests unitaires

Écrire les tests dans le même fichier :

```rust
// Dans velollm-core/src/hardware.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let hw = HardwareSpec::detect().unwrap();
        assert!(hw.cpu.cores > 0);
    }
}
```

### Tests d'intégration

Créer des fichiers dans `tests/` :

```rust
// tests/integration_test.rs
use velollm_core::hardware::HardwareSpec;

#[test]
fn test_full_detection() {
    let hw = HardwareSpec::detect().unwrap();
    assert!(!hw.os.is_empty());
}
```

### Exécuter des tests spécifiques

```bash
# Exécuter tous les tests
cargo test

# Exécuter les tests correspondant à un motif
cargo test hardware

# Exécuter les tests pour une crate spécifique
cargo test -p velollm-core

# Afficher la sortie des tests
cargo test -- --nocapture
```

## Couverture de code

```bash
# Installer tarpaulin
cargo install cargo-tarpaulin

# Générer le rapport de couverture
cargo tarpaulin --all --out Html

# Ouvrir le rapport de couverture
open tarpaulin-report.html
```

## Débogage

### Utiliser rust-gdb/rust-lldb

```bash
# Construire avec symboles de débogage
cargo build

# Déboguer avec gdb (Linux)
rust-gdb target/debug/velollm

# Déboguer avec lldb (macOS)
rust-lldb target/debug/velollm
```

### Utiliser VSCode

Créer `.vscode/launch.json` :

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug velollm",
      "cargo": {
        "args": ["build", "--bin=velollm"]
      },
      "args": ["detect"],
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

## Profilage de performance

### Utiliser perf (Linux)

```bash
# Construire avec release + symboles de débogage
cargo build --release

# Profiler avec perf
perf record --call-graph dwarf ./target/release/velollm benchmark
perf report
```

### Utiliser Instruments (macOS)

```bash
# Construction release
cargo build --release

# Ouvrir avec Instruments
instruments -t "Time Profiler" ./target/release/velollm benchmark
```

## Documentation

### Générer la documentation

```bash
# Générer et ouvrir la documentation
cargo doc --all --no-deps --open

# Ou utiliser Make
make doc
```

### Écrire la documentation

Utiliser les commentaires de documentation :

```rust
/// Détecte la configuration matérielle actuelle.
///
/// # Retourne
///
/// Une structure `HardwareSpec` contenant les informations GPU, CPU et mémoire.
///
/// # Erreurs
///
/// Retourne une erreur si les informations système ne peuvent pas être accédées.
///
/// # Exemples
///
/// ```
/// use velollm_core::hardware::HardwareSpec;
///
/// let hw = HardwareSpec::detect()?;
/// println!("Cœurs CPU: {}", hw.cpu.cores);
/// ```
pub fn detect() -> anyhow::Result<HardwareSpec> {
    // Implémentation
}
```

## Intégration continue

Le CI s'exécute à chaque push et PR :

- **Formatage** : `cargo fmt --check`
- **Linting** : `cargo clippy`
- **Tests** : `cargo test --all`
- **Constructions** : Constructions debug et release

Voir `.github/workflows/ci.yml` pour les détails.

## Dépannage

### Erreurs de construction Cargo

```bash
# Nettoyer et reconstruire
cargo clean
cargo build

# Ou
make clean
make build
```

### Avertissements Clippy

Corriger tous les avertissements clippy avant de commiter :

```bash
# Voir les avertissements
cargo clippy --all

# Correction automatique (quand possible)
cargo clippy --all --fix
```

### Échecs de tests

```bash
# Exécuter un test échoué spécifique
cargo test test_name -- --nocapture --test-threads=1

# Activer la journalisation
RUST_LOG=debug cargo test
```

## Prochaines étapes

- Lire [CONTRIBUTING.md](../../CONTRIBUTING.md) pour les directives de contribution
- Consulter [TODO.md](../../TODO.md) pour les tâches à traiter
- Voir [ROADMAP.md](../../ROADMAP.md) pour la direction du projet

## Questions ?

Ouvrir une Discussion GitHub ou contacter les mainteneurs.
