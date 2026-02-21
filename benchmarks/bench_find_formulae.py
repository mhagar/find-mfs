"""
Profiling script for find_formulae pipeline.

Measures per-stage timing for representative queries at different mass ranges.
Run with: python benchmarks/bench_find_formulae.py
"""
import time
import statistics
import numpy as np

from find_mfs import FormulaFinder


def bench_find_formulae(
    finder: FormulaFinder,
    mass: float,
    error_ppm: float = 5.0,
    filter_rdbe: tuple[float, float] | None = None,
    check_octet: bool = False,
    label: str = "",
    n_runs: int = 10,
    warmup: int = 2,
):
    """Run a single benchmark configuration and report median timing."""
    times = []
    result_count = 0

    for i in range(warmup + n_runs):
        t0 = time.perf_counter()
        results = finder.find_formulae(
            mass=mass,
            error_ppm=error_ppm,
            filter_rdbe=filter_rdbe,
            check_octet=check_octet,
        )
        t1 = time.perf_counter()

        if i >= warmup:
            times.append(t1 - t0)
            result_count = len(results)

    median_ms = statistics.median(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000

    print(f"  {label}")
    print(f"    candidates: {result_count}")
    print(f"    median: {median_ms:.2f} ms  (min: {min_ms:.2f}, max: {max_ms:.2f})")
    print()


def main():
    print("Initializing FormulaFinder (CHNOPS)...")
    t0 = time.perf_counter()
    finder = FormulaFinder('CHNOPS')
    print(f"  init: {(time.perf_counter() - t0)*1000:.1f} ms\n")

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    finder.find_formulae(mass=100.0, error_ppm=5.0)
    print()

    configs = [
        {
            "mass": 180.063,
            "error_ppm": 5.0,
            "label": "180 Da / 5 ppm (small, glucose-range)",
        },
        {
            "mass": 500.0,
            "error_ppm": 5.0,
            "label": "500 Da / 5 ppm (medium)",
        },
        {
            "mass": 500.0,
            "error_ppm": 5.0,
            "filter_rdbe": (0, 30),
            "check_octet": True,
            "label": "500 Da / 5 ppm + RDBE(0,30) + octet",
        },
        {
            "mass": 1000.0,
            "error_ppm": 10.0,
            "label": "1000 Da / 10 ppm (large)",
        },
        {
            "mass": 1000.0,
            "error_ppm": 10.0,
            "filter_rdbe": (0, 40),
            "check_octet": True,
            "label": "1000 Da / 10 ppm + RDBE(0,40) + octet",
        },
    ]

    print("=" * 60)
    print("Benchmarks (median of 10 runs, 2 warmup)")
    print("=" * 60)

    for cfg in configs:
        bench_find_formulae(finder, **cfg)


if __name__ == "__main__":
    main()
