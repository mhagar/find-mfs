"""
Head-to-head benchmark: Cython vs Numba for _decompose_mass_range.

Compares steady-state throughput, cold-start time, correctness, and
scaling behavior across multiple problem sizes.

Usage:
    cd benchmarks
    python setup_cython.py build_ext --inplace
    python bench_cython_vs_numba.py
"""
import sys
import os
import time
import math
import statistics
import numpy as np

# Ensure find_mfs is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def bench(name, fn, args, n_runs=50, warmup=5):
    """Benchmark a function, returning (median_ms, min_ms, max_ms, result)."""
    for _ in range(warmup):
        result = fn(*args)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = statistics.median(times) * 1e3
    mn = min(times) * 1e3
    mx = max(times) * 1e3
    return med, mn, mx, result


def cold_start(name, import_and_call):
    """Measure cold-start time (first call including import/JIT)."""
    t0 = time.perf_counter()
    result = import_and_call()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1e3
    n = len(result)
    print(f"    {name:30s}: {elapsed:8.1f} ms  (n={n})")
    return elapsed


def main():
    from find_mfs.core.decomposer import MassDecomposer
    from find_mfs.utils.filtering import BOND_ELECTRONS
    from molmass.elements import ELECTRON

    # Try importing Cython version
    try:
        from cython_decompose import decompose_mass_range as cython_decompose
    except ImportError:
        print("ERROR: Cython extension not built.")
        print("Run: cd benchmarks && python setup_cython.py build_ext --inplace")
        sys.exit(1)

    from find_mfs.core.algorithms import _decompose_mass_range as numba_decompose

    # Set up decomposer with real ERT data
    decomposer = MassDecomposer("CHNOPS")
    symbols = decomposer.element_symbols
    rdbe_coeffs = np.array(
        [0.5 * (BOND_ELECTRONS[s] - 2) for s in symbols], dtype=np.float64
    )

    ERT = decomposer.ERT
    integer_masses = decomposer.integer_masses
    real_masses = decomposer.real_masses

    # Benchmark configurations
    configs = [
        {
            "label": "500 Da / 5ppm / NO filter (pure decomposition)",
            "mass": 500.0, "ppm": 5.0,
            "rdbe_min": -np.inf, "rdbe_max": np.inf,
            "check_octet": False, "do_rdbe_filter": False,
        },
        {
            "label": "500 Da / 5ppm / RDBE(0,20) + octet",
            "mass": 500.0, "ppm": 5.0,
            "rdbe_min": 0.0, "rdbe_max": 20.0,
            "check_octet": True, "do_rdbe_filter": True,
        },
        {
            "label": "750 Da / 5ppm / RDBE(0,25) + octet",
            "mass": 750.0, "ppm": 5.0,
            "rdbe_min": 0.0, "rdbe_max": 25.0,
            "check_octet": True, "do_rdbe_filter": True,
        },
        {
            "label": "1000 Da / 5ppm / RDBE(0,30) + octet (stress test)",
            "mass": 1000.0, "ppm": 5.0,
            "rdbe_min": 0.0, "rdbe_max": 30.0,
            "check_octet": True, "do_rdbe_filter": True,
        },
    ]

    # ---- Cold start measurement ----
    print("=" * 76)
    print("  COLD START (first call, including JIT / import)")
    print("=" * 76)

    # Build args for a small test case (500 Da, no filter)
    def build_args(mass, ppm, rdbe_min, rdbe_max, check_octet, do_rdbe_filter):
        charge = 0
        adjusted_mass = mass + ELECTRON.mass * charge
        error = mass * ppm / 1e6
        orig_min = adjusted_mass - error
        orig_max = adjusted_mass + error

        bounds = np.array(
            [math.floor(orig_max / e.mass) for e in decomposer.elements],
            dtype=np.float64,
        )
        min_values = np.zeros(len(symbols), dtype=np.int64)

        from_int = math.ceil(
            (1 + decomposer.min_error) * orig_min / decomposer.precision
        )
        to_int = math.floor(
            (1 + decomposer.max_error) * orig_max / decomposer.precision
        )
        min_int = max(0, int(from_int))
        max_int = max(min_int, int(to_int))
        charge_mass_offset = ELECTRON.mass * charge
        max_results = 50000

        return (
            ERT, integer_masses, real_masses, bounds, min_values,
            min_int, max_int, orig_min, orig_max, charge_mass_offset,
            max_results, rdbe_coeffs, rdbe_min, rdbe_max,
            check_octet, True, do_rdbe_filter,
        )

    cold_args = build_args(500.0, 5.0, -np.inf, np.inf, False, False)

    # Cold start: Numba (includes JIT compilation)
    cold_start("Numba (includes JIT)", lambda: numba_decompose(*cold_args))

    # Cold start: Cython (pre-compiled, just import + call)
    cold_start("Cython (pre-compiled)", lambda: cython_decompose(*cold_args))

    # ---- Steady-state benchmarks ----
    for cfg in configs:
        args = build_args(
            cfg["mass"], cfg["ppm"],
            cfg["rdbe_min"], cfg["rdbe_max"],
            cfg["check_octet"], cfg["do_rdbe_filter"],
        )

        print(f"\n{'=' * 76}")
        print(f"  {cfg['label']}")
        print(f"{'=' * 76}")

        # Numba
        n_med, n_min, n_max, n_result = bench("Numba", numba_decompose, args)
        n_count = len(n_result)

        # Cython
        c_med, c_min, c_max, c_result = bench("Cython", cython_decompose, args)
        c_count = len(c_result)

        print(f"    {'Numba':30s}: median {n_med:7.3f} ms  min {n_min:7.3f}  max {n_max:7.3f}  n={n_count}")
        print(f"    {'Cython':30s}: median {c_med:7.3f} ms  min {c_min:7.3f}  max {c_max:7.3f}  n={c_count}")

        # Speed ratio
        if c_med > 0:
            ratio = n_med / c_med
            if ratio > 1:
                print(f"    --> Cython is {ratio:.2f}x FASTER")
            else:
                print(f"    --> Numba is {1/ratio:.2f}x FASTER")
        else:
            print(f"    --> Cython time too small to compare")

        # Correctness check
        if n_count != c_count:
            print(f"    CORRECTNESS FAIL: count mismatch {n_count} vs {c_count}")
        else:
            # Sort both result arrays for comparison (order may differ)
            n_sorted = n_result[np.lexsort(n_result.T[::-1])]
            c_sorted = c_result[np.lexsort(c_result.T[::-1])]
            if np.array_equal(n_sorted, c_sorted):
                print(f"    Correctness: PASS ({n_count} results match)")
            else:
                # Find first difference
                diff_mask = ~np.all(n_sorted == c_sorted, axis=1)
                first_diff = np.argmax(diff_mask)
                print(f"    CORRECTNESS FAIL: arrays differ at row {first_diff}")
                print(f"      Numba:  {n_sorted[first_diff]}")
                print(f"      Cython: {c_sorted[first_diff]}")

    print(f"\n{'=' * 76}")
    print("  DONE")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()
