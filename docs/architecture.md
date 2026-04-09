# frameforge Architecture

## System Overview

frameforge provides a unified video decoding/encoding interface designed for robotics foundation model data pipelines. The library is structured as a layered system where each layer builds on the one below it.

```mermaid
graph TD
    A[User Code / Training Loop] --> B[DataLoader Layer]
    B --> C[Sampling Layer]
    C --> D[VideoReader]
    D --> E[Backend Layer]
    E --> F1[PyAV]
    E --> F2[torchcodec]
    E --> F3[decord]
    E --> F4[OpenCV]
    F1 --> G[FFmpeg]
    F2 --> H[NVDEC / CPU]
    F3 --> H
    F4 --> G
```

## Core Abstractions

### Backend Layer

All backends implement `AbstractBackend`, which enforces a consistent contract for frame access and metadata. The backend is the only layer that touches library-specific APIs.

```mermaid
classDiagram
    class AbstractBackend {
        <<abstract>>
        +open(path) None
        +close() None
        +read_frame(idx) ndarray
        +read_clip(start, end) Tensor
        +seek(timestamp_sec) int
        +fps float
        +num_frames int
        +duration_sec float
        +width int
        +height int
        +codec str
    }

    AbstractBackend <|-- PyAVBackend
    AbstractBackend <|-- TorchCodecBackend
    AbstractBackend <|-- DecordBackend
    AbstractBackend <|-- OpenCVBackend

    class PyAVBackend {
        -_container
        -_stream
    }
    class TorchCodecBackend {
        -_decoder
        -_device
    }
    class DecordBackend {
        -_reader
        -_ctx
    }
    class OpenCVBackend {
        -_cap
    }
```

### VideoReader

The central user-facing class. Wraps any backend behind Pythonic indexing and lazy initialization.

```mermaid
flowchart LR
    subgraph VideoReader
        IDX["__getitem__"]
        LAZY[Lazy Init]
        PICK[Pickle Support]
    end

    IDX -- "reader[0]" --> FRAME["(H, W, C) Tensor"]
    IDX -- "reader[10:20]" --> CLIP["(T, H, W, C) Tensor"]
    IDX -- "reader[[0,5,10]]" --> SEL["(N, H, W, C) Tensor"]

    LAZY -- first access --> AUTO{Auto Select}
    AUTO -- "try order" --> TC[torchcodec]
    AUTO --> DE[decord]
    AUTO --> PA[pyav]
    AUTO --> CV[opencv]
```

**Key property: worker safety.** The backend is instantiated lazily on first frame access, not at construction time. This means `VideoReader` can be pickled and sent to DataLoader workers without leaking file descriptors across fork boundaries.

## Data Flow

### Single Clip Read

```mermaid
sequenceDiagram
    participant User
    participant VideoReader
    participant Backend
    participant FFmpeg/Decoder

    User->>VideoReader: reader[100:116]
    VideoReader->>VideoReader: _ensure_backend()
    VideoReader->>Backend: read_clip(100, 116)
    Backend->>FFmpeg/Decoder: seek(keyframe ≤ 100)
    FFmpeg/Decoder-->>Backend: decoded frames
    Backend->>Backend: collect frames [100, 116)
    Backend-->>VideoReader: Tensor (16, H, W, C)
    VideoReader-->>User: Tensor (16, H, W, C)
```

### DataLoader Pipeline

```mermaid
flowchart TD
    VP["video_paths[]"] --> DS[VideoClipDataset]
    DS --> |"per worker"| GI["__getitem__(idx)"]
    GI --> VR["VideoReader (lazy init)"]
    VR --> S["Sampler.sample()"]
    S --> T["transform()"]
    T --> COL["video_collate"]
    COL --> BATCH["(B, T, H, W, C)"]
    BATCH --> MODEL[Model Forward Pass]

    style GI fill:#1a1a2e,stroke:#58a6ff
    style VR fill:#1a1a2e,stroke:#58a6ff
    style S fill:#1a1a2e,stroke:#58a6ff
```

Each worker independently constructs its own `VideoReader` instance inside `__getitem__`. No shared state crosses process boundaries.

## Sampling Strategies

Samplers decouple *what frames to read* from *how to read them*. Every sampler produces frame indices, then delegates the actual I/O to `VideoReader`.

```mermaid
flowchart LR
    subgraph Samplers
        U[UniformSampler]
        R[RandomSampler]
        D[DenseSampler]
        E[EpisodeSampler]
    end

    U -- "linspace indices" --> IDX[Frame Indices]
    R -- "random window + sample" --> IDX
    D -- "range(0, N, stride)" --> IDX
    E -- "pick episode → uniform within" --> IDX

    IDX --> VR["VideoReader[indices]"]
    VR --> OUT["(T, H, W, C) Tensor"]
```

| Sampler | Use Case | Index Strategy |
|---------|----------|---------------|
| Uniform | General video understanding | Evenly spaced across full video |
| Random | Data augmentation, pretraining | Random window, random frames within |
| Dense | Action detection, dense prediction | Every Nth frame |
| Episode | Robotics manipulation data | Uniform within episode boundaries |

## Benchmark System

```mermaid
flowchart TD
    CFG[BenchmarkConfig] --> RUN[run_benchmark]
    RUN --> WU[Warmup Trials]
    WU --> TM[Timed Trials]
    TM --> MEM[tracemalloc]
    TM --> LAT[Latency Recording]
    MEM --> RES[BenchmarkResult]
    LAT --> RES
    RES --> JSON[results/*.json]
    JSON --> RPT[generate_report]
    RPT --> HTML[docs/index.html]
    HTML --> GHP[GitHub Pages]
```

The benchmark runner measures decode throughput, latency percentiles (p50/p95/p99), and peak memory. Results serialize to JSON and feed into a self-contained HTML report using Chart.js.

## Threading and Process Model

| Backend | Thread-safe | Fork-safe | GPU Decode |
|---------|------------|-----------|------------|
| PyAV | Yes | No (lazy init) | No |
| torchcodec | Yes | No (lazy init) | Yes (NVDEC) |
| decord | No | No (lazy init) | Yes (NVDEC) |
| OpenCV | Yes | No (lazy init) | No |

All backends use **lazy initialization** to sidestep fork-safety issues. The `VideoReader` drops its backend reference during pickling (`__getstate__`) and reconstructs it on first access in the new process (`_ensure_backend`).
