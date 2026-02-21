#!/usr/bin/env python3
"""
Exhaustive benchmark: local find-mfs vs upstream master (github.com/mhagar/find-mfs).

Compares find_formulae() across diverse inputs covering:
  - Mass ranges: small (100-200), medium (300-600), large (800-1500) Da
  - Tolerances: tight (1 ppm), moderate (5 ppm), wide (20 ppm), Da-based
  - Filters: none, RDBE, octet, RDBE+octet
  - Adducts: H, Na, -H
  - Element constraints: min_counts, max_counts, combined
  - Charged ions: +1, -1, +2
  - Edge cases: very small mass, very large mass, zero results expected

Measures:
  - Wall-clock time (median of N runs after warmup)
  - Number of candidates returned
  - Correctness: formula set agreement between versions

Usage:
    python benchmarks/benchmark_comparison.py
"""
from __future__ import annotations

import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# helpers to import both versions
# ---------------------------------------------------------------------------

def _import_local():
    """Import the local (optimized) find_mfs package."""
    import find_mfs as local_pkg
    return local_pkg.FormulaFinder


def _import_upstream():
    """Import the upstream (master) find_mfs package from /tmp clone."""
    upstream_path = "/tmp/find-mfs-upstream"
    if not os.path.isdir(upstream_path):
        raise RuntimeError(
            f"Upstream clone not found at {upstream_path}. "
            "Run: git clone https://github.com/mhagar/find-mfs.git /tmp/find-mfs-upstream"
        )
    # We need to temporarily replace find_mfs with the upstream version.
    # Use importlib to load from the upstream path directly.
    # Save and restore sys.modules to avoid contamination.
    saved_modules = {}
    to_remove = [k for k in sys.modules if k == "find_mfs" or k.startswith("find_mfs.")]
    for k in to_remove:
        saved_modules[k] = sys.modules.pop(k)

    sys.path.insert(0, upstream_path)
    try:
        import find_mfs as upstream_pkg
        FormulaFinder = upstream_pkg.FormulaFinder
    finally:
        sys.path.remove(upstream_path)
        # Remove upstream modules
        upstream_mods = [k for k in sys.modules if k == "find_mfs" or k.startswith("find_mfs.")]
        for k in upstream_mods:
            del sys.modules[k]
        # Restore local modules
        sys.modules.update(saved_modules)

    return FormulaFinder


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    label: str
    category: str
    n_candidates_local: int
    n_candidates_upstream: int
    median_ms_local: float
    median_ms_upstream: float
    min_ms_local: float
    min_ms_upstream: float
    max_ms_local: float
    max_ms_upstream: float
    speedup: float  # upstream_median / local_median
    formulas_match: bool  # whether both versions return same formula set
    only_in_local: list[str] = field(default_factory=list)
    only_in_upstream: list[str] = field(default_factory=list)


@dataclass
class BenchmarkCase:
    label: str
    category: str
    kwargs: dict[str, Any]
    n_runs: int = 5
    warmup: int = 1


def _run_single(finder, kwargs: dict, n_runs: int, warmup: int):
    """Run a single benchmark and return (median_ms, min_ms, max_ms, n_candidates, formula_set)."""
    times = []
    result = None
    for i in range(warmup + n_runs):
        t0 = time.perf_counter()
        result = finder.find_formulae(**kwargs)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)

    n_cand = len(result) if result else 0
    formula_set = set()
    if result:
        for c in result:
            formula_set.add(c.formula.formula)

    return (
        statistics.median(times) * 1000,
        min(times) * 1000,
        max(times) * 1000,
        n_cand,
        formula_set,
    )


def run_benchmark(
    case: BenchmarkCase,
    local_finder,
    upstream_finder,
) -> BenchmarkResult:
    """Run a benchmark case on both versions and return comparison."""
    med_l, min_l, max_l, n_l, formulas_l = _run_single(
        local_finder, case.kwargs, case.n_runs, case.warmup,
    )
    med_u, min_u, max_u, n_u, formulas_u = _run_single(
        upstream_finder, case.kwargs, case.n_runs, case.warmup,
    )

    speedup = med_u / med_l if med_l > 0 else float("inf")

    only_in_local = sorted(formulas_l - formulas_u)
    only_in_upstream = sorted(formulas_u - formulas_l)
    formulas_match = len(only_in_local) == 0 and len(only_in_upstream) == 0

    return BenchmarkResult(
        label=case.label,
        category=case.category,
        n_candidates_local=n_l,
        n_candidates_upstream=n_u,
        median_ms_local=med_l,
        median_ms_upstream=med_u,
        min_ms_local=min_l,
        min_ms_upstream=min_u,
        max_ms_local=max_l,
        max_ms_upstream=max_u,
        speedup=speedup,
        formulas_match=formulas_match,
        only_in_local=only_in_local[:10],  # cap for readability
        only_in_upstream=only_in_upstream[:10],
    )


# ---------------------------------------------------------------------------
# Benchmark cases — designed for exhaustive coverage
# ---------------------------------------------------------------------------

# Known compound masses (monoisotopic, neutral)
GLUCOSE_MASS = 180.06339  # C6H12O6
CAFFEINE_MH = 195.08765   # C8H10N4O2 + H (M+H+)
NOVOBIOCIN_MASS = 612.2526  # C31H36N2O11
SUCROSE_MASS = 342.11621  # C12H22O11
TAXOL_MASS = 853.33089    # C47H51NO14
ATP_MASS = 507.00000      # Approximate

CASES = [
    # -----------------------------------------------------------------------
    # Category 1: Mass range scaling (no filters)
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="100 Da / 5 ppm (tiny)",
        category="Mass range scaling",
        kwargs=dict(mass=100.0, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="180 Da / 5 ppm (glucose range)",
        category="Mass range scaling",
        kwargs=dict(mass=GLUCOSE_MASS, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="342 Da / 5 ppm (sucrose range)",
        category="Mass range scaling",
        kwargs=dict(mass=SUCROSE_MASS, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm (medium)",
        category="Mass range scaling",
        kwargs=dict(mass=500.0, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="612 Da / 5 ppm (novobiocin)",
        category="Mass range scaling",
        kwargs=dict(mass=NOVOBIOCIN_MASS, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="853 Da / 5 ppm (taxol range)",
        category="Mass range scaling",
        kwargs=dict(mass=TAXOL_MASS, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="1000 Da / 5 ppm (large)",
        category="Mass range scaling",
        kwargs=dict(mass=1000.0, error_ppm=5.0),
        n_runs=5, warmup=1,
    ),

    # -----------------------------------------------------------------------
    # Category 2: Tolerance scaling
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="500 Da / 1 ppm (tight)",
        category="Tolerance scaling",
        kwargs=dict(mass=500.0, error_ppm=1.0),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm (moderate)",
        category="Tolerance scaling",
        kwargs=dict(mass=500.0, error_ppm=5.0),
    ),
    BenchmarkCase(
        label="500 Da / 10 ppm (wide)",
        category="Tolerance scaling",
        kwargs=dict(mass=500.0, error_ppm=10.0),
    ),
    BenchmarkCase(
        label="500 Da / 20 ppm (very wide)",
        category="Tolerance scaling",
        kwargs=dict(mass=500.0, error_ppm=20.0),
    ),
    BenchmarkCase(
        label="500 Da / 0.005 Da (Da-based tight)",
        category="Tolerance scaling",
        kwargs=dict(mass=500.0, error_da=0.005),
    ),
    BenchmarkCase(
        label="500 Da / 0.05 Da (Da-based wide)",
        category="Tolerance scaling",
        kwargs=dict(mass=500.0, error_da=0.05),
    ),

    # -----------------------------------------------------------------------
    # Category 3: Chemical filters
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="500 Da / 5 ppm + RDBE(0,20)",
        category="Chemical filters",
        kwargs=dict(mass=500.0, error_ppm=5.0, filter_rdbe=(0, 20)),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm + RDBE(0,30) + octet",
        category="Chemical filters",
        kwargs=dict(mass=500.0, error_ppm=5.0, filter_rdbe=(0, 30), check_octet=True),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm + octet only",
        category="Chemical filters",
        kwargs=dict(mass=500.0, error_ppm=5.0, check_octet=True),
    ),
    BenchmarkCase(
        label="1000 Da / 10 ppm + RDBE(0,40) + octet",
        category="Chemical filters",
        kwargs=dict(mass=1000.0, error_ppm=10.0, filter_rdbe=(0, 40), check_octet=True),
        n_runs=5, warmup=1,
    ),
    BenchmarkCase(
        label="180 Da / 5 ppm + RDBE(0,10) + octet",
        category="Chemical filters",
        kwargs=dict(mass=GLUCOSE_MASS, error_ppm=5.0, filter_rdbe=(0, 10), check_octet=True),
    ),

    # -----------------------------------------------------------------------
    # Category 4: Element constraints
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="500 Da / 5 ppm + max_counts C30H60",
        category="Element constraints",
        kwargs=dict(mass=500.0, error_ppm=5.0, max_counts={"C": 30, "H": 60}),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm + min_counts C5H10",
        category="Element constraints",
        kwargs=dict(mass=500.0, error_ppm=5.0, min_counts={"C": 5, "H": 10}),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm + no P or S",
        category="Element constraints",
        kwargs=dict(mass=500.0, error_ppm=5.0, max_counts={"P": 0, "S": 0}),
    ),
    BenchmarkCase(
        label="342 Da / 5 ppm + constrain to C12H22O11",
        category="Element constraints",
        kwargs=dict(
            mass=SUCROSE_MASS, error_ppm=5.0,
            max_counts={"C": 12, "H": 22, "O": 11, "N": 0, "P": 0, "S": 0},
        ),
    ),
    BenchmarkCase(
        label="500 Da / 5 ppm + min C10 + max C30H60",
        category="Element constraints",
        kwargs=dict(
            mass=500.0, error_ppm=5.0,
            min_counts={"C": 10}, max_counts={"C": 30, "H": 60},
        ),
    ),

    # -----------------------------------------------------------------------
    # Category 5: Adducts and charge states
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="[M+H]+ glucose (charge=1, adduct=H)",
        category="Adducts & charge",
        kwargs=dict(mass=181.07066, charge=1, adduct="H", error_ppm=5.0),
    ),
    BenchmarkCase(
        label="[M+Na]+ glucose (charge=1, adduct=Na)",
        category="Adducts & charge",
        kwargs=dict(mass=203.05261, charge=1, adduct="Na", error_ppm=5.0),
    ),
    BenchmarkCase(
        label="[M-H]- glucose (charge=-1, adduct=-H)",
        category="Adducts & charge",
        kwargs=dict(mass=179.05612, charge=-1, adduct="-H", error_ppm=5.0),
    ),
    BenchmarkCase(
        label="[M+2H]2+ 500 Da (charge=2, adduct=H)",
        category="Adducts & charge",
        kwargs=dict(mass=251.0, charge=2, adduct="H", error_ppm=5.0),
    ),
    BenchmarkCase(
        label="Neutral 500 Da (charge=0)",
        category="Adducts & charge",
        kwargs=dict(mass=500.0, charge=0, error_ppm=5.0),
    ),

    # -----------------------------------------------------------------------
    # Category 6: Combined filters (realistic workflows)
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="Novobiocin: all filters",
        category="Realistic workflows",
        kwargs=dict(
            mass=NOVOBIOCIN_MASS, error_ppm=3.0,
            filter_rdbe=(0, 20), check_octet=True,
            max_counts={"P": 0, "S": 2},
        ),
    ),
    BenchmarkCase(
        label="Glucose [M+H]+: all filters",
        category="Realistic workflows",
        kwargs=dict(
            mass=181.07066, charge=1, adduct="H", error_ppm=3.0,
            filter_rdbe=(0, 10), check_octet=True,
        ),
    ),
    BenchmarkCase(
        label="Taxol range: constrained + filtered",
        category="Realistic workflows",
        kwargs=dict(
            mass=TAXOL_MASS, error_ppm=5.0,
            filter_rdbe=(0, 30), check_octet=True,
            min_counts={"C": 20},
            max_counts={"P": 0, "S": 0},
        ),
        n_runs=5, warmup=1,
    ),
    BenchmarkCase(
        label="1000 Da wide search + all filters",
        category="Realistic workflows",
        kwargs=dict(
            mass=1000.0, error_ppm=10.0,
            filter_rdbe=(0, 40), check_octet=True,
            max_counts={"P": 2, "S": 2},
        ),
        n_runs=5, warmup=1,
    ),

    # -----------------------------------------------------------------------
    # Category 7: Edge cases
    # -----------------------------------------------------------------------
    BenchmarkCase(
        label="18 Da / 10 ppm (water mass, very small)",
        category="Edge cases",
        kwargs=dict(mass=18.01056, error_ppm=10.0),
    ),
    # 2000 Da omitted — too slow for quick benchmarks
    BenchmarkCase(
        label="500 Da / 0.1 ppm (extremely tight)",
        category="Edge cases",
        kwargs=dict(mass=500.0, error_ppm=0.1),
    ),
    BenchmarkCase(
        label="Impossible mass (expect zero results)",
        category="Edge cases",
        kwargs=dict(mass=0.5, error_ppm=1.0),
    ),
]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: list[BenchmarkResult]):
    """Print a formatted benchmark report."""
    line = "=" * 120
    print()
    print(line)
    print("BENCHMARK REPORT: Local (optimized) vs Upstream (master)")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cases: {len(results)}")
    print(line)
    print()

    # Group by category
    categories: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, cat_results in categories.items():
        print(f"\n{'─' * 120}")
        print(f"  {cat}")
        print(f"{'─' * 120}")
        print(
            f"  {'Case':<47} {'Upstream':>10} {'Local':>10} {'Speedup':>8} "
            f"{'#Up':>7} {'#Loc':>7} {'Match':>6}"
        )
        print(f"  {'':─<47} {'(ms)':─>10} {'(ms)':─>10} {'':─>8} {'':─>7} {'':─>7} {'':─>6}")

        for r in cat_results:
            match_str = "YES" if r.formulas_match else "DIFF"
            print(
                f"  {r.label:<47} {r.median_ms_upstream:>10.2f} {r.median_ms_local:>10.2f} "
                f"{r.speedup:>7.1f}x {r.n_candidates_upstream:>7} {r.n_candidates_local:>7} "
                f"{match_str:>6}"
            )
            if not r.formulas_match:
                n_only_l = len(r.only_in_local)
                n_only_u = len(r.only_in_upstream)
                # Count total diffs (not just truncated list)
                total_diff_l = r.n_candidates_local - (r.n_candidates_local - n_only_l) if n_only_l else 0
                total_diff_u = r.n_candidates_upstream - (r.n_candidates_upstream - n_only_u) if n_only_u else 0
                if n_only_l > 0:
                    print(f"    Only in local ({n_only_l}): {', '.join(r.only_in_local[:5])}")
                if n_only_u > 0:
                    print(f"    Only in upstream ({n_only_u}): {', '.join(r.only_in_upstream[:5])}")

    # Summary statistics
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")

    speedups = [r.speedup for r in results if r.speedup != float("inf")]
    if speedups:
        print(f"  Median speedup:  {statistics.median(speedups):.1f}x")
        print(f"  Mean speedup:    {statistics.mean(speedups):.1f}x")
        print(f"  Min speedup:     {min(speedups):.1f}x (case: {results[speedups.index(min(speedups))].label})")
        print(f"  Max speedup:     {max(speedups):.1f}x (case: {results[speedups.index(max(speedups))].label})")

    n_match = sum(1 for r in results if r.formulas_match)
    n_diff = sum(1 for r in results if not r.formulas_match)
    print(f"\n  Correctness:     {n_match}/{len(results)} cases produce identical formula sets")
    if n_diff > 0:
        print(f"  Differences:     {n_diff} cases have differing formula sets (see details above)")

    # Timing breakdown per category
    print(f"\n  {'Category':<30} {'Upstream median':>16} {'Local median':>14} {'Avg speedup':>12}")
    print(f"  {'':─<30} {'':─>16} {'':─>14} {'':─>12}")
    for cat, cat_results in categories.items():
        avg_u = statistics.mean([r.median_ms_upstream for r in cat_results])
        avg_l = statistics.mean([r.median_ms_local for r in cat_results])
        avg_sp = statistics.mean([r.speedup for r in cat_results if r.speedup != float("inf")])
        print(f"  {cat:<30} {avg_u:>14.2f}ms {avg_l:>12.2f}ms {avg_sp:>11.1f}x")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("find-mfs Benchmark: Local (optimized) vs Upstream (master)")
    print("=" * 80)

    # Import both versions
    print("\n[1/4] Importing local version...")
    t0 = time.perf_counter()
    LocalFormulaFinder = _import_local()
    print(f"  done ({(time.perf_counter() - t0)*1000:.0f} ms)")

    print("\n[2/4] Importing upstream version...")
    t0 = time.perf_counter()
    UpstreamFormulaFinder = _import_upstream()
    print(f"  done ({(time.perf_counter() - t0)*1000:.0f} ms)")

    # Initialize finders
    print("\n[3/4] Initializing finders (CHNOPS)...")
    t0 = time.perf_counter()
    local_finder = LocalFormulaFinder("CHNOPS")
    t_local_init = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    upstream_finder = UpstreamFormulaFinder("CHNOPS")
    t_upstream_init = (time.perf_counter() - t0) * 1000

    print(f"  Local init:    {t_local_init:.1f} ms")
    print(f"  Upstream init: {t_upstream_init:.1f} ms")

    # Warm up Numba JIT for both
    print("\n  Warming up JIT (both versions)...")
    local_finder.find_formulae(mass=100.0, error_ppm=5.0)
    upstream_finder.find_formulae(mass=100.0, error_ppm=5.0)
    print("  done")

    # Run benchmarks
    print(f"\n[4/4] Running {len(CASES)} benchmark cases...")
    print()

    results: list[BenchmarkResult] = []
    for i, case in enumerate(CASES, 1):
        sys.stdout.write(f"  [{i:>2}/{len(CASES)}] {case.label:<50}")
        sys.stdout.flush()
        try:
            result = run_benchmark(case, local_finder, upstream_finder)
            results.append(result)
            sys.stdout.write(
                f" {result.speedup:>5.1f}x  "
                f"(up={result.median_ms_upstream:.1f}ms, "
                f"loc={result.median_ms_local:.1f}ms)\n"
            )
        except Exception as e:
            sys.stdout.write(f" ERROR: {e}\n")

    # Print report
    print_report(results)

    # Save raw data as JSON
    output_path = Path(__file__).parent / "benchmark_results.json"
    raw_data = []
    for r in results:
        raw_data.append({
            "label": r.label,
            "category": r.category,
            "n_candidates_local": r.n_candidates_local,
            "n_candidates_upstream": r.n_candidates_upstream,
            "median_ms_local": round(r.median_ms_local, 4),
            "median_ms_upstream": round(r.median_ms_upstream, 4),
            "min_ms_local": round(r.min_ms_local, 4),
            "min_ms_upstream": round(r.min_ms_upstream, 4),
            "max_ms_local": round(r.max_ms_local, 4),
            "max_ms_upstream": round(r.max_ms_upstream, 4),
            "speedup": round(r.speedup, 2),
            "formulas_match": r.formulas_match,
        })
    with open(output_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw results saved to: {output_path}")


if __name__ == "__main__":
    main()
