"""CLI entry point: python -m frameforge.benchmark run --backend pyav --video sample.mp4"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from frameforge.benchmark.configs import BenchmarkConfig
from frameforge.benchmark.report import generate_report
from frameforge.benchmark.runner import BenchmarkResult, run_benchmark


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="frameforge-bench",
        description="frameforge benchmark runner",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = sub.add_parser("run", help="Run a benchmark")
    run_parser.add_argument("--backend", required=True, help="Backend name (pyav, decord, etc.)")
    run_parser.add_argument("--video", required=True, help="Path to video file")
    run_parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    run_parser.add_argument("--clip-length", type=int, default=16, help="Frames per clip")
    run_parser.add_argument("--seek-mode", default="sequential", choices=["sequential", "random"])
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--output", default=None, help="Output JSON path")

    # --- report ---
    report_parser = sub.add_parser("report", help="Generate HTML report from JSON results")
    report_parser.add_argument("--results-dir", default="benchmarks/results", help="Dir with JSON files")
    report_parser.add_argument("--output", default="docs/benchmarks.html", help="Output HTML path")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = BenchmarkConfig(
            backend=args.backend,
            video_path=args.video,
            num_trials=args.trials,
            clip_length_frames=args.clip_length,
            seek_mode=args.seek_mode,
            device=args.device,
        )

        result = run_benchmark(config)
        print(result.to_json())

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.to_json())
            logger.info("Results written to {}", out)

    elif args.command == "report":
        results_dir = Path(args.results_dir)
        results: list[BenchmarkResult] = []
        for f in sorted(results_dir.glob("*.json")):
            data = json.loads(f.read_text())
            results.append(BenchmarkResult.from_json(data))

        if not results:
            logger.error("No JSON result files found in {}", results_dir)
            sys.exit(1)

        out = generate_report(results, args.output)
        logger.info("Report generated at {}", out)


if __name__ == "__main__":
    main()
