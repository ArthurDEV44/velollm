# VeloLLM

**Autopilot for Local LLM Inference** - High-performance proxy and optimization toolkit for Ollama and llama.cpp.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurDEV44/velollm/actions/workflows/ci.yml)

---

## The Problem

Local LLM inference is **19x slower** than production solutions like vLLM. VeloLLM bridges this gap by providing a high-performance Rust proxy that optimizes requests, improves tool calling reliability, and brings production-grade features to local deployments.

| Metric | Production (vLLM) | Local (Ollama) | Gap |
|--------|-------------------|----------------|-----|
| Throughput | 793 tokens/s | 41 tokens/s | 19x |
| P99 Latency | 80ms | 673ms | 8x |

**VeloLLM Goal**: Bring vLLM-level performance to Ollama users while keeping the simplicity.

---

## What is VeloLLM?

VeloLLM is a **transparent proxy** that sits between your applications and Ollama. It intercepts API calls, applies intelligent optimizations, and forwards them to Ollama. Your existing tools work without modification - just change the API endpoint.

### Key Benefits

- **Drop-in replacement**: Full OpenAI API compatibility
- **Tool calling improvements**: JSON fixing, deduplication, schema validation
- **Performance optimization**: Request batching, intelligent caching, continuous scheduling
- **Metrics & observability**: Track tokens/s, latency, cache hit rates
- **Advanced memory management**: PagedAttention for efficient KV cache

### Supported Models for Tool Calling

- Mistral (mistral:7b, mistral-small:24b)
- Llama (llama3.2:3b, llama3.1:8b, llama3.1:70b)

---

## Architecture

```mermaid
flowchart TB
    subgraph Applications["Your Applications"]
        CC[Claude Code]
        CW[Continue]
        OW[Open WebUI]
        CA[Custom Apps]
    end

    subgraph Proxy["VeloLLM Proxy :8000"]
        direction TB

        subgraph Layer1["API Layer"]
            OAI[OpenAI Compatibility]
            NAT[Ollama Native API]
            SSE[SSE Streaming]
        end

        subgraph Layer2["Optimization Layer"]
            TO[Tool Optimizer]
            RB[Request Batcher]
            SC[Semantic Cache]
        end

        subgraph Layer3["Scheduling Layer"]
            CBS[Continuous Batching Scheduler]
            BM[Block Manager]
            PA[PagedAttention]
        end

        subgraph Layer4["Observability"]
            MET[Metrics Collector]
        end
    end

    subgraph Backend["Inference Backend"]
        OLL[Ollama :11434]
        LCPP[llama.cpp]
    end

    Applications --> Layer1
    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Backend
    Layer4 -.-> Layer1
    Layer4 -.-> Layer2
    Layer4 -.-> Layer3
```

### Request Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Proxy as VeloLLM Proxy
    participant Cache as Semantic Cache
    participant Batch as Request Batcher
    participant Sched as Scheduler
    participant Ollama as Ollama

    App->>Proxy: POST /v1/chat/completions
    Proxy->>Cache: Check cache (exact + semantic)

    alt Cache Hit
        Cache-->>Proxy: Cached response
        Proxy-->>App: Return cached response
    else Cache Miss
        Proxy->>Batch: Add to batch queue
        Batch->>Sched: Submit batch
        Sched->>Ollama: Forward optimized request
        Ollama-->>Sched: Stream response
        Sched-->>Batch: Distribute responses
        Batch-->>Proxy: Response
        Proxy->>Cache: Store in cache
        Proxy-->>App: Return response
    end
```

---

## Features

### Implemented

#### Phase 1: MVP (Complete)

| Feature | Description |
|---------|-------------|
| **Hardware Detection** | Auto-detect GPU (NVIDIA, AMD, Apple Silicon, Intel), CPU, and memory |
| **Benchmarking Suite** | Measure tokens/s, TTFT, total latency with multiple profiles |
| **Ollama Auto-Configuration** | Generate optimized environment variables based on hardware |
| **Speculative Decoding Analysis** | Research and parameter recommendations for draft models |

#### Phase 2: Advanced Optimizations (83% Complete)

| Feature | Description |
|---------|-------------|
| **PagedAttention Block Manager** | Memory-efficient KV cache with 16-token blocks, reference counting, and CoW |
| **llama.cpp KV Cache Integration** | Paged cache wrapper compatible with llama_memory_* API |
| **CUDA Paged Attention Kernel** | GPU-accelerated attention with FP16/FP32 and GQA support |
| **Continuous Batching Scheduler** | Dynamic request scheduling with priority and preemption |

#### Phase 3: Intelligent Proxy (50% Complete)

| Feature | Description |
|---------|-------------|
| **HTTP Server** | Axum-based server with Tower middleware |
| **OpenAI API Compatibility** | Full support for `/v1/chat/completions`, `/v1/models` |
| **Tool Calling Enhancement** | Automatic JSON fixing, deduplication, schema validation |
| **Request Batching** | Group concurrent requests, priority-based scheduling |
| **Semantic Cache** | Embedding-based similarity matching, exact + semantic caching |

### Coming Soon

| Feature | Status |
|---------|--------|
| Metrics & Observability | Planned |
| CLI Integration (`velollm serve`) | Planned |
| Prompt Compression | Planned |
| Speculative Prefetch | Planned |
| Multi-Model Load Balancing | Planned |

---

## Memory Management: PagedAttention

VeloLLM implements PagedAttention for efficient KV cache management, inspired by vLLM.

```mermaid
flowchart LR
    subgraph VirtualMemory["Virtual Memory (Logical)"]
        S1[Sequence 1]
        S2[Sequence 2]
        S3[Sequence 3]
    end

    subgraph BlockTable["Block Tables"]
        BT1["S1: [0, 3, 5]"]
        BT2["S2: [1, 4]"]
        BT3["S3: [2, 6, 7]"]
    end

    subgraph PhysicalBlocks["Physical GPU Memory"]
        B0[Block 0]
        B1[Block 1]
        B2[Block 2]
        B3[Block 3]
        B4[Block 4]
        B5[Block 5]
        B6[Block 6]
        B7[Block 7]
    end

    S1 --> BT1
    S2 --> BT2
    S3 --> BT3

    BT1 --> B0
    BT1 --> B3
    BT1 --> B5
    BT2 --> B1
    BT2 --> B4
    BT3 --> B2
    BT3 --> B6
    BT3 --> B7
```

**Benefits**:
- **70% reduction** in memory fragmentation
- **Dynamic allocation**: Sequences grow without pre-allocation
- **Memory sharing**: Copy-on-Write for beam search and parallel sampling
- **Efficient preemption**: Swap sequences without losing context

---

## Tool Calling Optimization

VeloLLM fixes common tool calling issues that occur with local models.

```mermaid
flowchart LR
    subgraph Input["Model Response"]
        RAW["```json
{name: 'get_weather',
 args: {city: 'Paris',}}
```"]
    end

    subgraph Processing["Tool Optimizer"]
        FIX[JSON Fixer]
        VAL[Schema Validator]
        DED[Deduplicator]
    end

    subgraph Output["Clean Tool Call"]
        CLEAN["{
  \"name\": \"get_weather\",
  \"arguments\": {
    \"city\": \"Paris\"
  }
}"]
    end

    RAW --> FIX
    FIX --> VAL
    VAL --> DED
    DED --> CLEAN
```

**Fixes applied**:
- Remove markdown code blocks
- Fix trailing commas
- Quote unquoted keys
- Extract JSON from mixed content
- Validate against function schemas
- Deduplicate repeated tool calls

---

## Caching System

VeloLLM implements a two-tier caching system for optimal performance.

```mermaid
flowchart TB
    REQ[Incoming Request]

    subgraph ExactCache["Tier 1: Exact Cache"]
        HASH[XXH3 Hash]
        LRU[LRU Cache]
    end

    subgraph SemanticCache["Tier 2: Semantic Cache"]
        EMB[Embedding Model]
        VEC[Vector Store]
        SIM[Similarity Search]
    end

    REQ --> HASH
    HASH --> LRU

    LRU -->|Miss| EMB
    EMB --> VEC
    VEC --> SIM

    LRU -->|Hit| HIT1[Cache Hit < 1ms]
    SIM -->|Match > 0.95| HIT2[Semantic Hit < 5ms]
    SIM -->|No Match| MISS[Forward to Ollama]
```

**Performance targets**:
- Exact cache hit: < 1ms latency
- Semantic cache hit: < 5ms latency
- Cache hit rate: > 30% on repetitive workloads

---

## Project Structure

```mermaid
graph TB
    subgraph Workspace["VeloLLM Workspace"]
        CLI[velollm-cli]
        PROXY[velollm-proxy]
        CORE[velollm-core]
        BENCH[velollm-benchmarks]

        subgraph Adapters["adapters/"]
            OLLAMA[ollama]
            LCPP[llamacpp]
        end
    end

    CLI --> CORE
    CLI --> BENCH
    PROXY --> CORE
    CORE --> OLLAMA
    CORE --> LCPP
    BENCH --> OLLAMA
```

| Crate | Description |
|-------|-------------|
| `velollm-core` | Core library: hardware detection, PagedAttention, scheduler |
| `velollm-cli` | CLI binary: detect, benchmark, optimize commands |
| `velollm-proxy` | Proxy server: OpenAI compatibility, optimizations |
| `velollm-benchmarks` | Benchmarking library for Ollama |
| `adapters/ollama` | Ollama configuration parser and optimizer |
| `adapters/llamacpp` | llama.cpp integration, CUDA kernels |

---

## Development Status

### Progress by Phase

```mermaid
pie title Project Progress
    "Phase 1 (Complete)" : 12
    "Phase 2 (Active)" : 5
    "Phase 2 (Standby)" : 2
    "Phase 3 (Active)" : 5
    "Phase 3 (Remaining)" : 5
```

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: MVP | Complete | 12/12 (100%) |
| Phase 2: Advanced Optimizations | In Progress | 5/6 active (83%) |
| Phase 3: Intelligent Proxy | In Progress | 5/10 (50%) |

### Test Coverage

| Crate | Tests |
|-------|-------|
| velollm-core | 63 |
| velollm-benchmarks | 3 |
| velollm-adapters-llamacpp | 29 |
| velollm-adapters-ollama | 6 |
| velollm-cli (integration) | 8 |
| Doc tests | 8 |
| **Total** | **117** |

---

## Comparison

| Feature | Ollama | vLLM | LM Studio | VeloLLM |
|---------|--------|------|-----------|---------|
| Target Use Case | Simplicity | Cloud production | Desktop GUI | Local performance |
| OpenAI API Compat | Partial | Full | Partial | Full |
| Tool Calling Fix | No | N/A | No | Yes |
| PagedAttention | No | Yes | No | Yes (local) |
| Request Batching | No | Yes | No | Yes |
| Semantic Cache | No | No | No | Yes |
| Auto-optimization | No | No | Partial | Yes |
| Language | Go | Python | Electron | Rust |
| Open Source | Yes | Yes | No | Yes |

---

## Roadmap

```mermaid
gantt
    title VeloLLM Development Roadmap
    dateFormat  YYYY-MM
    section Phase 1
    MVP (CLI tools)           :done, p1, 2024-01, 2024-03
    section Phase 2
    PagedAttention            :done, p2a, 2024-04, 2024-05
    Continuous Batching       :done, p2b, 2024-05, 2024-06
    Performance Profiler      :active, p2c, 2024-06, 2024-07
    section Phase 3
    HTTP Server & OpenAI API  :done, p3a, 2024-07, 2024-08
    Tool Calling & Caching    :done, p3b, 2024-08, 2024-09
    Metrics & CLI Integration :active, p3c, 2024-09, 2024-10
    section Phase 4
    GUI Dashboard             :p4a, 2024-10, 2024-12
    IDE Integrations          :p4b, 2024-11, 2025-01
```

**Phase 1** (Complete): MVP with CLI tools
- Hardware detection, benchmarking, Ollama configuration

**Phase 2** (83% Complete): Advanced optimizations
- PagedAttention, continuous batching scheduler, CUDA kernels

**Phase 3** (In Progress): Intelligent proxy
- OpenAI compatibility, tool calling enhancement, caching, metrics

**Phase 4** (Planned): Ecosystem
- GUI dashboard, IDE integrations, configuration marketplace

Full details: [ROADMAP.md](ROADMAP.md) | Task tracking: [TODO.md](TODO.md)

---

## Contributing

We welcome contributions! Areas of interest:

- **Performance**: Optimize the proxy, reduce latency
- **Tool Calling**: Improve JSON fixing, add more edge cases
- **Caching**: Enhance semantic cache with better embeddings
- **Testing**: Add integration tests, benchmark on diverse hardware
- **Documentation**: Improve guides and API docs

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Links

- **Repository**: [github.com/ArthurDEV44/velollm](https://github.com/ArthurDEV44/velollm)
- **Issues**: [GitHub Issues](https://github.com/ArthurDEV44/velollm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ArthurDEV44/velollm/discussions)

---

**Status**: Phase 3 - Proxy development in progress (50% complete)

Built with Rust by the VeloLLM community.
