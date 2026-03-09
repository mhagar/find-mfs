#!/usr/bin/env python
"""
Cross-branch benchmark for FormulaFinder.find_formulae().
Works on both master (Numba) and isospec (Cython) branches.
Outputs JSON for easy comparison.
"""
import json
import sys
import time
import subprocess
import numpy as np


def get_branch():
    return subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        text=True,
    ).strip()


def get_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        text=True,
    ).strip()


def timeit(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # us
    arr = np.array(times)
    return {
        'median_us': float(np.median(arr)),
        'mean_us': float(np.mean(arr)),
        'p5_us': float(np.percentile(arr, 5)),
        'p95_us': float(np.percentile(arr, 95)),
        'min_us': float(np.min(arr)),
        'n': n,
    }


def run_benchmarks():
    from find_mfs import FormulaFinder

    # Check if isotope matching is available
    has_isotope = True
    try:
        from find_mfs.isotopes import SingleEnvelopeMatch
    except (ImportError, Exception):
        has_isotope = False

    finder = FormulaFinder('CHNOPS')

    results = {}

    # 1. Simple search (mass=180, 5 ppm)
    def simple():
        return finder.find_formulae(mass=180.063388, error_ppm=5.0)
    r = simple()
    n_simple = len(r)
    results['simple_180'] = timeit(simple, 2000)
    results['simple_180']['n_results'] = n_simple

    # 2. RDBE + octet (mass=180)
    def rdbe():
        return finder.find_formulae(
            mass=180.063388, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )
    r = rdbe()
    n_rdbe = len(r)
    results['rdbe_octet_180'] = timeit(rdbe, 2000)
    results['rdbe_octet_180']['n_results'] = n_rdbe

    # 3. With adduct
    def adduct():
        return finder.find_formulae(
            mass=203.05261, charge=1, adduct="Na", error_ppm=5.0,
        )
    r = adduct()
    n_adduct = len(r)
    results['adduct_Na_203'] = timeit(adduct, 1000)
    results['adduct_Na_203']['n_results'] = n_adduct

    # 4. Large mass (800 Da) — lazy only
    def large_lazy():
        return finder.find_formulae(mass=800.0, error_ppm=5.0)
    r = large_lazy()
    n_large = len(r)
    results['large_800_lazy'] = timeit(large_lazy, 200)
    results['large_800_lazy']['n_results'] = n_large

    # 5. Large mass — full materialization (iterate all)
    def large_eager():
        r = finder.find_formulae(mass=800.0, error_ppm=5.0)
        for x in r:
            _ = x.formula
        return r
    results['large_800_eager'] = timeit(large_eager, 50)
    results['large_800_eager']['n_results'] = n_large

    # 6. Medium mass (500 Da) with RDBE
    def medium_rdbe():
        return finder.find_formulae(
            mass=500.0, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )
    r = medium_rdbe()
    n_med = len(r)
    results['medium_500_rdbe'] = timeit(medium_rdbe, 500)
    results['medium_500_rdbe']['n_results'] = n_med

    # 7. Medium mass — full materialization
    def medium_eager():
        r = finder.find_formulae(
            mass=500.0, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )
        for x in r:
            _ = x.formula
        return r
    results['medium_500_eager'] = timeit(medium_eager, 200)
    results['medium_500_eager']['n_results'] = n_med

    # 8. Isotope matching (if available)
    if has_isotope:
        envelope = np.array([
            [180.063388, 1.00],
            [181.066743, 0.065],
            [182.068, 0.012],
        ])
        iso_config = SingleEnvelopeMatch(envelope, mz_tolerance_da=0.01)

        def isotope():
            return finder.find_formulae(
                mass=180.063388, error_ppm=5.0,
                isotope_match=iso_config,
            )
        r = isotope()
        n_iso = len(r)
        results['isotope_180'] = timeit(isotope, 500)
        results['isotope_180']['n_results'] = n_iso

    # 9. Tiny mass (exact match scenario, very few results)
    def tiny():
        return finder.find_formulae(mass=60.021, error_ppm=2.0)
    r = tiny()
    results['tiny_60'] = timeit(tiny, 5000)
    results['tiny_60']['n_results'] = len(r)

    # 10. Very large mass (1200 Da)
    def vlarge():
        return finder.find_formulae(mass=1200.0, error_ppm=3.0)
    r = vlarge()
    results['vlarge_1200_lazy'] = timeit(vlarge, 50)
    results['vlarge_1200_lazy']['n_results'] = len(r)

    # 11. FormulaFinder construction (cold start)
    def construct():
        return FormulaFinder('CHNOPS')
    results['constructor'] = timeit(construct, 100, warmup=1)

    # 12. Halogen set
    finder_hal = FormulaFinder('CHNOPSClBrFI')

    def halogen():
        return finder_hal.find_formulae(mass=400.0, error_ppm=5.0)
    r = halogen()
    results['halogen_400_lazy'] = timeit(halogen, 100)
    results['halogen_400_lazy']['n_results'] = len(r)

    return results


def main():
    branch = get_branch()
    commit = get_commit()
    print(f"Running benchmarks on branch: {branch} ({commit})", file=sys.stderr)

    results = run_benchmarks()

    output = {
        'branch': branch,
        'commit': commit,
        'benchmarks': results,
    }

    json_str = json.dumps(output, indent=2)
    print(json_str)

    # Also print human-readable table
    print("\n" + "=" * 75, file=sys.stderr)
    print(f"  Branch: {branch} ({commit})", file=sys.stderr)
    print("=" * 75, file=sys.stderr)
    print(f"{'Benchmark':<28} {'Median (us)':>12} {'P95 (us)':>12} {'Results':>8}", file=sys.stderr)
    print("-" * 75, file=sys.stderr)
    for name, data in results.items():
        n_res = data.get('n_results', '')
        print(f"{name:<28} {data['median_us']:>12.1f} {data['p95_us']:>12.1f} {str(n_res):>8}", file=sys.stderr)
    print("=" * 75, file=sys.stderr)


if __name__ == '__main__':
    main()
