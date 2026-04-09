"""Script to run benchmarks across all available backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from frameforge.benchmark.configs import BenchmarkConfig
from frameforge.benchmark.report import generate_report
from frameforge.benchmark.runner import BenchmarkResult, run_benchmark

BACKENDS = ["pyav", "torchcodec", "decord", "opencv"]
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video_path> [--trials N]")
        sys.exit(1)

    video_path = sys.argv[1]
    num_trials = 50
    if "--trials" in sys.argv:
        num_trials = int(sys.argv[sys.argv.index("--trials") + 1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkResult] = []

    for backend in BACKENDS:
        config = BenchmarkConfig(
            backend=backend,
            video_path=video_path,
            num_trials=num_trials,
            clip_length_frames=16,
            seek_mode="sequential",
        )
        try:
            result = run_benchmark(config)
            results.append(result)

            out_path = RESULTS_DIR / f"{backend}.json"
            out_path.write_text(result.to_json())
            logger.info("Saved {} results to {}", backend, out_path)
        except Exception as e:
            logger.warning("Skipping {} — {}", backend, e)

    if results:
        report_path = generate_report(results, Path(__file__).parent.parent / "docs" / "index.html")
        logger.info("Report generated at {}", report_path)
    else:
        logger.error("No backends succeeded")


if __name__ == "__main__":
    main()
