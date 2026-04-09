# Contributing to frameforge

Thanks for your interest in contributing to frameforge.

## Setup

```bash
# Clone
git clone https://github.com/arpitg1304/frameforge.git
cd frameforge

# Create virtual environment (Python 3.10+)
python3.12 -m venv .venv
source .venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Verify
pytest
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_reader.py

# Verbose
pytest -v

# Stop on first failure
pytest -x
```

All 71 tests should pass. Tests generate synthetic videos using PyAV — no real video files needed.

## Project Structure

```
frameforge/
  backends/       Pluggable decode backends (PyAV, torchcodec, decord, OpenCV)
  sampling/       Temporal samplers + multi-camera sync
  dataloader/     PyTorch Dataset/IterableDataset wrappers
  benchmark/      Runner, configs, HTML report generator
  packing/        Shard packing for training throughput
  cache.py        Disk-cached frame index
  prefetch.py     Background decode-ahead thread
  reader.py       Unified VideoReader
  writer.py       VideoWriter
```

## Adding a New Backend

1. Create `frameforge/backends/your_backend.py`
2. Subclass `AbstractBackend` from `frameforge/backends/base.py`
3. Implement all abstract methods: `open`, `close`, `read_frame`, `read_clip`, `seek`, and properties
4. Raise `BackendNotAvailable` at module level if the library isn't installed
5. Add the backend to `_BACKEND_PRIORITY` in `frameforge/reader.py`
6. Add to optional dependencies in `pyproject.toml`
7. Add tests in `tests/test_backends.py`

## Code Style

- Type annotations on all public APIs
- Use `loguru` for logging (not print)
- No unnecessary abstractions — three similar lines > one premature helper
- Tests use pytest with synthetic video fixtures from `conftest.py`

## Pull Requests

- One feature or fix per PR
- Add tests for new functionality
- Run `pytest` before submitting — all tests must pass
- Keep PRs focused — don't bundle unrelated changes

## Benchmarks

To run benchmarks, generate a test video first:

```bash
python -c "
import av, numpy as np
c = av.open('sample.mp4', mode='w')
s = c.add_stream('h264', rate=30)
s.width, s.height, s.pix_fmt = 640, 480, 'yuv420p'
for i in range(300):
    f = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for p in s.encode(av.VideoFrame.from_ndarray(f, format='rgb24')): c.mux(p)
for p in s.encode(): c.mux(p)
c.close()
"

python -m frameforge.benchmark run --backend pyav --video sample.mp4
```
