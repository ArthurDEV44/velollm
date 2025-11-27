# API de détection matérielle

## Aperçu

Le module `velollm-core::hardware` fournit une détection matérielle complète pour les informations GPU, CPU, RAM et système d'exploitation.

## Utilisation

```rust
use velollm_core::hardware::HardwareSpec;

fn main() -> anyhow::Result<()> {
    let hw = HardwareSpec::detect()?;

    println!("Plateforme: {}", hw.platform);
    println!("CPU: {}", hw.cpu.model);
    println!("RAM: {} Go", hw.memory.total_mb / 1024);

    if let Some(gpu) = hw.gpu {
        println!("GPU: {}", gpu.name);
        println!("VRAM: {} Go", gpu.vram_total_mb / 1024);
    }

    Ok(())
}
```

## Structures de données

### `HardwareSpec`

Structure principale contenant toutes les informations matérielles.

```rust
pub struct HardwareSpec {
    pub gpu: Option<GpuInfo>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub os: String,
    pub platform: String,
}
```

**Champs :**
- `gpu` : Informations GPU (si disponible)
- `cpu` : Informations CPU
- `memory` : Informations mémoire système
- `os` : Nom du système d'exploitation ("linux", "macos", "windows")
- `platform` : Chaîne de plateforme (ex: "linux-x86_64", "macos-aarch64")

### `GpuInfo`

Informations spécifiques au GPU.

```rust
pub struct GpuInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
}
```

**Fabricants supportés :**
- `GpuVendor::Nvidia` : GPU NVIDIA (via nvidia-smi)
- `GpuVendor::Amd` : GPU AMD (via rocm-smi)
- `GpuVendor::Apple` : Apple Silicon (M1/M2/M3)
- `GpuVendor::Intel` : GPU intégrés Intel

### `CpuInfo`

Informations spécifiques au CPU.

```rust
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency_mhz: Option<u64>,
}
```

### `MemoryInfo`

Informations mémoire système.

```rust
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
    pub used_mb: u64,
}
```

## Support des plateformes

### Linux

**Détection GPU :**
- NVIDIA : commande `nvidia-smi`
- AMD : commande `rocm-smi`
- Intel : analyse de `lspci`

**CPU/Mémoire :**
- Crate sysinfo (lit /proc/cpuinfo, /proc/meminfo)

### macOS

**Détection GPU :**
- Apple Silicon : `system_profiler SPDisplaysDataType` + `sysctl hw.memsize`

**CPU/Mémoire :**
- Crate sysinfo

### Windows

**Détection GPU :**
- NVIDIA : `nvidia-smi.exe` (si dans PATH)

**CPU/Mémoire :**
- Crate sysinfo (requêtes WMI)

## Sortie JSON

La structure `HardwareSpec` est sérialisable en JSON :

```bash
velollm detect
```

**Exemple de sortie :**

```json
{
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "vendor": "Nvidia",
    "vram_total_mb": 24564,
    "vram_free_mb": 23012,
    "driver_version": "535.129.03",
    "compute_capability": "8.9"
  },
  "cpu": {
    "model": "AMD Ryzen 9 7950X 16-Core Processor",
    "cores": 16,
    "threads": 32,
    "frequency_mhz": 4500
  },
  "memory": {
    "total_mb": 65536,
    "available_mb": 42384,
    "used_mb": 23152
  },
  "os": "linux",
  "platform": "linux-x86_64"
}
```

## Gestion des erreurs

La méthode `detect()` retourne `anyhow::Result<HardwareSpec>` :

- **Détection CPU/Mémoire** : Réussit toujours (repli sur valeurs sûres par défaut)
- **Détection GPU** : Retourne `None` si aucun GPU trouvé ou outils de détection indisponibles

```rust
match HardwareSpec::detect() {
    Ok(hw) => {
        // Matériel détecté avec succès
    }
    Err(e) => {
        eprintln!("Échec de détection du matériel: {}", e);
    }
}
```

## Prérequis

### Commandes externes

Pour la détection GPU, les commandes suivantes doivent être dans PATH :

- **NVIDIA** : `nvidia-smi`
- **AMD** : `rocm-smi`
- **Apple** : `system_profiler`, `sysctl` (intégrés sur macOS)
- **Intel (Linux)** : `lspci`

Si ces commandes ne sont pas disponibles, la détection GPU retournera `None`.

## Exemples

### Détecter et afficher le matériel

```rust
use velollm_core::hardware::HardwareSpec;

fn main() {
    let hw = HardwareSpec::detect().unwrap();
    println!("{:#?}", hw);
}
```

### Exporter vers un fichier JSON

```rust
use velollm_core::hardware::HardwareSpec;
use std::fs;

fn main() -> anyhow::Result<()> {
    let hw = HardwareSpec::detect()?;
    let json = serde_json::to_string_pretty(&hw)?;
    fs::write("hardware.json", json)?;
    Ok(())
}
```

### Vérifier la disponibilité du GPU

```rust
use velollm_core::hardware::{HardwareSpec, GpuVendor};

fn main() -> anyhow::Result<()> {
    let hw = HardwareSpec::detect()?;

    match hw.gpu {
        Some(gpu) => {
            println!("GPU disponible: {}", gpu.name);

            match gpu.vendor {
                GpuVendor::Nvidia => {
                    println!("GPU NVIDIA avec {} Go de VRAM", gpu.vram_total_mb / 1024);
                }
                GpuVendor::Apple => {
                    println!("Apple Silicon avec mémoire unifiée");
                }
                _ => {
                    println!("Autre fabricant GPU: {:?}", gpu.vendor);
                }
            }
        }
        None => {
            println!("Aucun GPU détecté - mode CPU uniquement");
        }
    }

    Ok(())
}
```

## Tests

Exécuter les tests avec :

```bash
cargo test -p velollm-core

# Avec sortie
cargo test -p velollm-core -- --nocapture
```

Les tests valident :
- Détection CPU (cœurs, threads, modèle)
- Détection mémoire (total, disponible, utilisé)
- Détection GPU (si disponible)
- Sérialisation/désérialisation JSON
- Format de chaîne de plateforme

## Améliorations futures

Améliorations prévues :

- [ ] Détection GPU Windows (AMD via DirectX)
- [ ] Détection de capacité de calcul GPU Intel
- [ ] Détection de bande passante mémoire GPU
- [ ] Détection taille cache CPU (L1/L2/L3)
- [ ] Version/voies PCIe pour GPU
- [ ] Support multi-GPU
- [ ] Métriques de consommation électrique
