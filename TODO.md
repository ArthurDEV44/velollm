# VeloLLM - TODO Détaillé pour Implémentation par Agent IA

Ce document décompose la roadmap en tâches atomiques, exécutables par un agent IA (Claude Code, Gemini CLI, etc.) avec des instructions précises et des critères de validation.

---

## 🎯 Phase 1: MVP - COMPLETE ✅

**Status**: 12/12 tasks (100%)

Phase 1 MVP est terminée et disponible en release [v0.1.0](https://github.com/ArthurDEV44/velollm/releases/tag/v0.1.0).

Pour les détails complets des tâches Phase 1 (TASK-001 à TASK-012), voir [TODO_MVP.md](TODO_MVP.md).

### Summary

| Task | Description | Status |
|------|-------------|--------|
| TASK-001 | Repository setup | ✅ |
| TASK-002 | Build system (Cargo workspace) | ✅ |
| TASK-003 | Hardware detection (GPU/CPU/Memory) | ✅ |
| TASK-004 | Benchmark suite (Ollama API) | ✅ |
| TASK-005 | Speculative decoding analysis | ✅ |
| TASK-006 | llama.cpp speculative wrapper | ✅ |
| TASK-007 | Benchmark comparison | ✅ |
| TASK-008 | Ollama config parser | ✅ |
| TASK-009 | Hardware-based optimizer | ✅ |
| TASK-010 | CLI `velollm optimize` command | ✅ |
| TASK-011 | End-to-end integration tests | ✅ |
| TASK-012 | Documentation | ✅ |

---

## 🚀 Phase 2: Optimisations Avancées (Mois 4-6)

### Sprint 5: PagedAttention Implementation (Semaine 9-12)

#### TASK-013: Étude de PagedAttention dans vLLM ✅
**Priority**: P1
**Estimated effort**: 6h
**Dependencies**: None

**Completed**: Documentation complète dans `docs/research/paged_attention.md`

Couvre:
- Core concept (KV cache as virtual memory)
- vLLM implementation details
- llama.cpp integration challenges
- Implementation strategy for VeloLLM (3 phases)

---

#### TASK-014: Block Manager Implementation ✅
**Priority**: P1
**Estimated effort**: 8h
**Dependencies**: TASK-013

**Completed**: Module `velollm-core/src/paged_attention/`

Files:
- `mod.rs`: BlockManager, BlockManagerConfig, PagedAttentionError
- `block_allocator.rs`: BlockAllocator with reference counting, CoW
- `block_table.rs`: SequenceBlockTable for per-sequence mapping

Features:
- Block size: 16 tokens (configurable)
- FIFO allocation with reference counting
- Copy-on-write for memory sharing (beam search)
- Model-specific configs (Llama 8B, 3B presets)
- 35 unit tests

---

#### TASK-015: llama.cpp Paged KV Cache Integration
**Priority**: P1
**Estimated effort**: 12h
**Dependencies**: TASK-014

**Instructions**:

1. **Étudier l'architecture KV cache de llama.cpp**:
   ```bash
   cd /home/sauron/code/llama.cpp

   # Fichiers clés
   cat src/llama-kv-cache.h
   cat src/llama-kv-cache.cpp
   grep -r "kv_cache" src/
   ```

2. **Identifier les points d'intégration**:
   - `llama_kv_cache_init`: Allocation initiale
   - `llama_kv_cache_find_slot`: Placement des tokens
   - `llama_kv_cache_seq_add/rm`: Gestion des séquences

3. **Créer wrapper Rust pour FFI**:

   **File**: `adapters/llamacpp/src/kv_cache.rs`

   ```rust
   use crate::ffi;
   use velollm_core::paged_attention::{BlockManager, BlockId};

   pub struct PagedKvCache {
       block_manager: BlockManager,
       // Mapping from llama.cpp sequence to our block table
       sequence_blocks: HashMap<u32, Vec<BlockId>>,
   }

   impl PagedKvCache {
       pub fn new(config: BlockManagerConfig) -> Self {
           Self {
               block_manager: BlockManager::new(config),
               sequence_blocks: HashMap::new(),
           }
       }

       /// Allocate blocks for a new sequence
       pub fn init_sequence(&mut self, seq_id: u32, initial_tokens: usize) -> Result<(), Error> {
           let blocks_needed = (initial_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
           let mut blocks = Vec::with_capacity(blocks_needed);

           for _ in 0..blocks_needed {
               let block = self.block_manager.allocate()
                   .ok_or(Error::OutOfMemory)?;
               blocks.push(block);
           }

           self.sequence_blocks.insert(seq_id, blocks);
           Ok(())
       }

       /// Get block table for attention computation
       pub fn get_block_table(&self, seq_id: u32) -> Option<&[BlockId]> {
           self.sequence_blocks.get(&seq_id).map(|v| v.as_slice())
       }
   }
   ```

4. **Modifier le build llama.cpp pour intégration**:
   - Option: Build llama.cpp as library (`BUILD_SHARED_LIBS=ON`)
   - Link avec Rust via cc crate ou bindgen

**Validation criteria**:
- [ ] Wrapper compile avec llama.cpp
- [ ] Allocation de blocs fonctionne
- [ ] Séquences peuvent grandir dynamiquement
- [ ] Tests d'intégration avec modèle réel

---

#### TASK-016: CUDA Paged Attention Kernel
**Priority**: P1
**Estimated effort**: 16h
**Dependencies**: TASK-015

**Instructions**:

1. **Étudier les kernels existants**:
   ```bash
   # vLLM kernels (référence)
   git clone https://github.com/vllm-project/vllm
   cat vllm/csrc/attention/attention_kernels.cu

   # llama.cpp CUDA
   cat /home/sauron/code/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu
   ```

2. **Implémenter kernel paged attention**:

   **File**: `adapters/llamacpp/cuda/paged_attention.cu`

   ```cuda
   // Paged attention kernel
   // Q: [batch, heads, head_dim]
   // K_cache, V_cache: [num_blocks, block_size, heads, head_dim]
   // block_tables: [batch, max_blocks]

   __global__ void paged_attention_kernel(
       float* __restrict__ output,
       const float* __restrict__ query,
       const float* __restrict__ key_cache,
       const float* __restrict__ value_cache,
       const int* __restrict__ block_tables,
       const int* __restrict__ seq_lens,
       const int num_heads,
       const int head_dim,
       const int block_size,
       const int max_seq_len
   ) {
       // Thread indices
       const int batch_idx = blockIdx.x;
       const int head_idx = blockIdx.y;
       const int thread_idx = threadIdx.x;

       const int seq_len = seq_lens[batch_idx];
       const int* block_table = block_tables + batch_idx * (max_seq_len / block_size);

       // Load query for this head
       float q[HEAD_DIM];
       for (int i = 0; i < head_dim; i += WARP_SIZE) {
           q[i + thread_idx] = query[batch_idx * num_heads * head_dim
                                      + head_idx * head_dim
                                      + i + thread_idx];
       }

       // Compute attention scores over paged KV cache
       float max_score = -INFINITY;
       for (int token_idx = 0; token_idx < seq_len; token_idx++) {
           int block_idx = token_idx / block_size;
           int block_offset = token_idx % block_size;
           int physical_block = block_table[block_idx];

           // Load key from paged cache
           const float* k = key_cache + physical_block * block_size * num_heads * head_dim
                                       + block_offset * num_heads * head_dim
                                       + head_idx * head_dim;

           // Compute dot product
           float score = 0.0f;
           for (int i = 0; i < head_dim; i += WARP_SIZE) {
               score += q[i + thread_idx] * k[i + thread_idx];
           }
           score = warpReduceSum(score);

           max_score = max(max_score, score);
       }

       // Softmax and weighted sum (simplified)
       // ... (full implementation needed)
   }
   ```

3. **Wrapper Rust pour kernel**:
   ```rust
   // adapters/llamacpp/src/cuda_paged.rs

   #[link(name = "paged_attention", kind = "static")]
   extern "C" {
       fn paged_attention_forward(
           output: *mut f32,
           query: *const f32,
           key_cache: *const f32,
           value_cache: *const f32,
           block_tables: *const i32,
           seq_lens: *const i32,
           batch_size: i32,
           num_heads: i32,
           head_dim: i32,
           block_size: i32,
           max_seq_len: i32,
       );
   }

   pub fn run_paged_attention(...) -> Result<Vec<f32>> {
       // ... CUDA memory management and kernel launch
   }
   ```

**Validation criteria**:
- [ ] Kernel compile avec CUDA 12+
- [ ] Résultats numériquement corrects vs référence
- [ ] Performance: >80% de la version vLLM
- [ ] Memory usage réduit de >50%

---

#### TASK-017: Continuous Batching Scheduler
**Priority**: P1
**Estimated effort**: 10h
**Dependencies**: TASK-016

**Instructions**:

**File**: `velollm-core/src/scheduler.rs`

```rust
use std::collections::{VecDeque, HashMap};
use crate::paged_attention::BlockManager;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub generated_tokens: Vec<u32>,
    pub state: RequestState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    Waiting,
    Running,
    Preempted,
    Finished,
}

pub struct Scheduler {
    waiting_queue: VecDeque<Request>,
    running: HashMap<u64, Request>,
    block_manager: BlockManager,
    max_batch_size: usize,
    max_tokens_per_step: usize,
}

impl Scheduler {
    pub fn new(block_manager: BlockManager, config: SchedulerConfig) -> Self {
        Self {
            waiting_queue: VecDeque::new(),
            running: HashMap::new(),
            block_manager,
            max_batch_size: config.max_batch_size,
            max_tokens_per_step: config.max_tokens_per_step,
        }
    }

    /// Add a new request to the waiting queue
    pub fn add_request(&mut self, request: Request) {
        self.waiting_queue.push_back(request);
    }

    /// Schedule next batch for execution
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();

        // 1. Try to add waiting requests
        while !self.waiting_queue.is_empty()
              && self.running.len() < self.max_batch_size {
            let request = self.waiting_queue.front().unwrap();

            // Check if we have enough blocks
            let blocks_needed = self.blocks_for_request(request);
            if self.block_manager.can_allocate(blocks_needed) {
                let mut request = self.waiting_queue.pop_front().unwrap();
                let seq_id = self.block_manager.add_sequence().unwrap();
                request.state = RequestState::Running;

                output.new_sequences.push(seq_id);
                self.running.insert(request.id, request);
            } else {
                break; // Can't fit more requests
            }
        }

        // 2. Collect running requests for this step
        for (id, request) in &self.running {
            if request.state == RequestState::Running {
                output.running_requests.push(*id);
            }
        }

        output
    }

    /// Called after inference step to update state
    pub fn update(&mut self, completed: Vec<u64>, preempted: Vec<u64>) {
        for id in completed {
            if let Some(mut request) = self.running.remove(&id) {
                request.state = RequestState::Finished;
                // Free blocks
                // ...
            }
        }

        for id in preempted {
            if let Some(mut request) = self.running.remove(&id) {
                request.state = RequestState::Preempted;
                self.waiting_queue.push_front(request);
            }
        }
    }

    fn blocks_for_request(&self, request: &Request) -> usize {
        let total_tokens = request.prompt_tokens.len() + request.max_new_tokens as usize;
        (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE
    }
}

#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub new_sequences: Vec<u64>,
    pub running_requests: Vec<u64>,
    pub preempted_requests: Vec<u64>,
}
```

**Validation criteria**:
- [ ] Scheduler gère dynamiquement les requêtes
- [ ] Preemption fonctionne sous pression mémoire
- [ ] Throughput: >1.5x vs séquentiel
- [ ] Tests avec charges variées (burst, steady)

---

### Sprint 6: Multi-Backend & Performance (Semaine 13-16)

#### TASK-018: LocalAI Adapter
**Priority**: P2
**Estimated effort**: 6h
**Dependencies**: TASK-010

*(Détails similaires au format des tâches précédentes)*

---

#### TASK-019: vLLM Adapter
**Priority**: P2
**Estimated effort**: 8h
**Dependencies**: TASK-010

*(Détails similaires au format des tâches précédentes)*

---

#### TASK-020: Performance Profiler
**Priority**: P2
**Estimated effort**: 6h
**Dependencies**: TASK-016, TASK-017

*(Détails similaires au format des tâches précédentes)*

---

## 📊 Progress Tracking

### Phase 1 MVP (Mois 1-3) ✅
**Progress**: 12/12 tasks (100%)

Voir [TODO_MVP.md](TODO_MVP.md) pour les détails.

### Phase 2 Advanced (Mois 4-6)
- [x] TASK-013: PagedAttention research ✅
- [x] TASK-014: Block manager ✅
- [ ] TASK-015: llama.cpp paged KV cache integration
- [ ] TASK-016: CUDA paged attention kernel
- [ ] TASK-017: Continuous batching scheduler
- [ ] TASK-018: LocalAI adapter
- [ ] TASK-019: vLLM adapter
- [ ] TASK-020: Performance profiler

**Progress**: 2/8 tasks (25%)

**Tests Status**:
- velollm-core: 48/48 tests passing ✅ (+35 paged_attention tests)
- velollm-benchmarks: 3/3 tests passing ✅
- velollm-adapters-llamacpp: 6/6 tests passing ✅
- velollm-adapters-ollama: 6/6 tests passing ✅
- velollm-cli: 8/8 integration tests passing ✅
- Doc tests: 4/4 passing ✅

**Total: 75 tests passing**

---

## 🤖 Instructions pour Agent IA

**Comment utiliser ce TODO**:

1. **Démarrer par les P1**: Exécuter les tâches dans l'ordre de priorité
2. **Validation stricte**: Ne pas marquer "done" sans passer TOUS les critères
3. **Tests obligatoires**: Chaque tâche avec code doit avoir des tests
4. **Documentation inline**: Commenter le code complexe
5. **Git commits granulaires**: 1 commit par tâche complétée
6. **Reporting**: Après chaque tâche, résumer: ce qui marche, ce qui bloque, métriques

**Format de reporting**:
```
TASK-XXX: [DONE/BLOCKED/IN_PROGRESS]
- Completed: [description]
- Tests: [X/Y passing]
- Blockers: [liste ou "None"]
- Metrics: [benchmarks si applicable]
- Next: TASK-YYY
```

**Gestion des blockers**:
- Documenter le problème dans `docs/issues/task-XXX-blocker.md`
- Proposer 2-3 solutions alternatives
- Escalate si besoin d'input humain

---

## 📚 Ressources Rapides

### Commandes utiles
```bash
# Build & test
cargo build --release
cargo test --all
cargo clippy --all -- -D warnings

# Run CLI
velollm detect
velollm optimize --dry-run
velollm benchmark

# Docs
cargo doc --open
```

### Repos de référence
- llama.cpp: `/home/sauron/code/llama.cpp`
- vLLM: `git clone https://github.com/vllm-project/vllm`
- Ollama: `https://github.com/ollama/ollama`

### Papers critiques
- PagedAttention: [vLLM blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- Speculative Decoding: [llama.cpp PR#2926](https://github.com/ggml-org/llama.cpp/pull/2926)

---

**Next task: TASK-015 (llama.cpp Paged KV Cache Integration) 🚀**
