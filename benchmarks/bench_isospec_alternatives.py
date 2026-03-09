"""
Benchmark alternative IsoSpec calling strategies.

1. Current: setupIso + setupThreshold + read + manual sort/combine/normalize
2. Binned: setupIso + setupBinnedFixedEnvelope (pre-sorted, pre-combined)
3. Threshold + IsoSpec sort/normalize (let IsoSpec do the work)
"""
import time
import ctypes
import numpy as np
from find_mfs.isotopes._isospec_bridge import get_isotope_arrays


def load_isospec_lib():
    """Load IsoSpec and return function pointers."""
    from IsoSpecPy.isoFFI import isoFFI
    lib_path = str(isoFFI.libpath)
    lib = ctypes.CDLL(lib_path)

    # setupIso
    lib.setupIso.restype = ctypes.c_void_p
    lib.setupIso.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    # setupThresholdFixedEnvelope
    lib.setupThresholdFixedEnvelope.restype = ctypes.c_void_p
    lib.setupThresholdFixedEnvelope.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_bool, ctypes.c_bool,
    ]

    # setupBinnedFixedEnvelope
    lib.setupBinnedFixedEnvelope.restype = ctypes.c_void_p
    lib.setupBinnedFixedEnvelope.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]

    # confs_no
    lib.confs_noFixedEnvelope.restype = ctypes.c_size_t
    lib.confs_noFixedEnvelope.argtypes = [ctypes.c_void_p]

    # masses/probs
    lib.massesFixedEnvelope.restype = ctypes.POINTER(ctypes.c_double)
    lib.massesFixedEnvelope.argtypes = [ctypes.c_void_p]
    lib.probsFixedEnvelope.restype = ctypes.POINTER(ctypes.c_double)
    lib.probsFixedEnvelope.argtypes = [ctypes.c_void_p]

    # sort/normalize
    lib.sortEnvelopeByMass.restype = None
    lib.sortEnvelopeByMass.argtypes = [ctypes.c_void_p]
    lib.normalizeEnvelope.restype = None
    lib.normalizeEnvelope.argtypes = [ctypes.c_void_p]

    # cleanup
    lib.deleteFixedEnvelope.restype = None
    lib.deleteFixedEnvelope.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.deleteIso.restype = None
    lib.deleteIso.argtypes = [ctypes.c_void_p]
    lib.freeReleasedArray.restype = None
    lib.freeReleasedArray.argtypes = [ctypes.c_void_p]

    return lib


def bench_strategies():
    lib = load_isospec_lib()

    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    atom_counts_list = [6, 12, 0, 6, 0, 0]  # glucose

    iso_numbers_np, flat_masses_np, flat_probs_np = get_isotope_arrays(symbols)

    # Convert to ctypes
    n_elem = len(symbols)
    iso_numbers = (ctypes.c_int * n_elem)(*iso_numbers_np.tolist())
    atom_counts = (ctypes.c_int * n_elem)(*atom_counts_list)
    flat_masses = (ctypes.c_double * len(flat_masses_np))(*flat_masses_np.tolist())
    flat_probs = (ctypes.c_double * len(flat_probs_np))(*flat_probs_np.tolist())

    threshold = 0.001
    combine_tol = 0.05

    n = 5000

    # Strategy 1: Current approach (setupThreshold, manual sort/combine)
    times_current = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
        env_ptr = lib.setupThresholdFixedEnvelope(iso_ptr, threshold, False, False)
        n_peaks = lib.confs_noFixedEnvelope(env_ptr)
        if n_peaks > 0:
            masses_p = lib.massesFixedEnvelope(env_ptr)
            probs_p = lib.probsFixedEnvelope(env_ptr)
            # Copy out
            masses = np.array([masses_p[i] for i in range(n_peaks)])
            probs = np.array([probs_p[i] for i in range(n_peaks)])
            lib.freeReleasedArray(masses_p)
            lib.freeReleasedArray(probs_p)
        lib.deleteFixedEnvelope(env_ptr, False)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times_current.append((t1 - t0) / 1e3)

    arr = np.array(times_current)
    print(f"Strategy 1 (threshold, ctypes): median={np.median(arr):.1f} us, n_peaks={n_peaks}")

    # Strategy 1b: Just setupIso + setupThreshold (no read)
    times_setup_only = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
        env_ptr = lib.setupThresholdFixedEnvelope(iso_ptr, threshold, False, False)
        lib.deleteFixedEnvelope(env_ptr, False)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times_setup_only.append((t1 - t0) / 1e3)

    arr = np.array(times_setup_only)
    print(f"Strategy 1b (setup+thresh only): median={np.median(arr):.1f} us")

    # Strategy 1c: Just setupIso (no envelope)
    times_iso_only = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times_iso_only.append((t1 - t0) / 1e3)

    arr = np.array(times_iso_only)
    print(f"Strategy 1c (setupIso only):     median={np.median(arr):.1f} us")

    # Strategy 2: Binned envelope (pre-sorted, pre-combined)
    times_binned = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
        env_ptr = lib.setupBinnedFixedEnvelope(iso_ptr, 0.999, combine_tol, 0.0)
        n_peaks_b = lib.confs_noFixedEnvelope(env_ptr)
        if n_peaks_b > 0:
            masses_p = lib.massesFixedEnvelope(env_ptr)
            probs_p = lib.probsFixedEnvelope(env_ptr)
            masses = np.array([masses_p[i] for i in range(n_peaks_b)])
            probs = np.array([probs_p[i] for i in range(n_peaks_b)])
            lib.freeReleasedArray(masses_p)
            lib.freeReleasedArray(probs_p)
        lib.deleteFixedEnvelope(env_ptr, False)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times_binned.append((t1 - t0) / 1e3)

    arr = np.array(times_binned)
    print(f"Strategy 2 (binned):             median={np.median(arr):.1f} us, n_peaks={n_peaks_b}")

    # Strategy 3: Threshold + IsoSpec sort/normalize
    times_isosort = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
        env_ptr = lib.setupThresholdFixedEnvelope(iso_ptr, threshold, False, False)
        lib.sortEnvelopeByMass(env_ptr)
        lib.normalizeEnvelope(env_ptr)
        n_peaks_s = lib.confs_noFixedEnvelope(env_ptr)
        if n_peaks_s > 0:
            masses_p = lib.massesFixedEnvelope(env_ptr)
            probs_p = lib.probsFixedEnvelope(env_ptr)
            masses = np.array([masses_p[i] for i in range(n_peaks_s)])
            probs = np.array([probs_p[i] for i in range(n_peaks_s)])
            lib.freeReleasedArray(masses_p)
            lib.freeReleasedArray(probs_p)
        lib.deleteFixedEnvelope(env_ptr, False)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times_isosort.append((t1 - t0) / 1e3)

    arr = np.array(times_isosort)
    print(f"Strategy 3 (thresh+sort+norm):   median={np.median(arr):.1f} us, n_peaks={n_peaks_s}")

    # Now test with a larger molecule
    print("\n--- Larger molecule (C30H50N5O10P1S2) ---")
    atom_counts_large = (ctypes.c_int * n_elem)(30, 50, 5, 10, 1, 2)

    # setupIso only (large)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts_large, flat_masses, flat_probs)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"  {'setupIso only':<25} median={np.median(arr):.1f} us")

    # threshold (large)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts_large, flat_masses, flat_probs)
        env_ptr = lib.setupThresholdFixedEnvelope(iso_ptr, threshold, False, False)
        np_count = lib.confs_noFixedEnvelope(env_ptr)
        lib.deleteFixedEnvelope(env_ptr, False)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"  {'threshold':<25} median={np.median(arr):.1f} us, n_peaks={np_count}")

    # binned (large)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts_large, flat_masses, flat_probs)
        env_ptr = lib.setupBinnedFixedEnvelope(iso_ptr, 0.999, combine_tol, 0.0)
        np_count = lib.confs_noFixedEnvelope(env_ptr)
        lib.deleteFixedEnvelope(env_ptr, False)
        lib.deleteIso(iso_ptr)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"  {'binned':<25} median={np.median(arr):.1f} us, n_peaks={np_count}")


def bench_binned_output_quality():
    """Check that binned envelope gives same/similar results as threshold+combine."""
    lib = load_isospec_lib()

    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    atom_counts_list = [6, 12, 0, 6, 0, 0]

    iso_numbers_np, flat_masses_np, flat_probs_np = get_isotope_arrays(symbols)
    n_elem = len(symbols)
    iso_numbers = (ctypes.c_int * n_elem)(*iso_numbers_np.tolist())
    atom_counts = (ctypes.c_int * n_elem)(*atom_counts_list)
    flat_masses = (ctypes.c_double * len(flat_masses_np))(*flat_masses_np.tolist())
    flat_probs = (ctypes.c_double * len(flat_probs_np))(*flat_probs_np.tolist())

    # Threshold
    iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
    env_ptr = lib.setupThresholdFixedEnvelope(iso_ptr, 0.001, False, False)
    n_peaks = lib.confs_noFixedEnvelope(env_ptr)
    masses_p = lib.massesFixedEnvelope(env_ptr)
    probs_p = lib.probsFixedEnvelope(env_ptr)
    t_masses = np.array([masses_p[i] for i in range(n_peaks)])
    t_probs = np.array([probs_p[i] for i in range(n_peaks)])
    lib.freeReleasedArray(masses_p)
    lib.freeReleasedArray(probs_p)
    lib.deleteFixedEnvelope(env_ptr, False)
    lib.deleteIso(iso_ptr)

    # Sort and combine manually
    idx = np.argsort(t_masses)
    t_masses = t_masses[idx]
    t_probs = t_probs[idx]

    print(f"\n--- Threshold output (raw) ---")
    print(f"  n_peaks={n_peaks}")
    # Normalize
    t_probs_n = t_probs / t_probs.max()
    for i in range(min(10, n_peaks)):
        print(f"  {t_masses[i]:.6f}  {t_probs_n[i]:.6f}")

    # Binned
    iso_ptr = lib.setupIso(n_elem, iso_numbers, atom_counts, flat_masses, flat_probs)
    env_ptr = lib.setupBinnedFixedEnvelope(iso_ptr, 0.999, 0.05, 0.0)
    n_peaks_b = lib.confs_noFixedEnvelope(env_ptr)
    masses_p = lib.massesFixedEnvelope(env_ptr)
    probs_p = lib.probsFixedEnvelope(env_ptr)
    b_masses = np.array([masses_p[i] for i in range(n_peaks_b)])
    b_probs = np.array([probs_p[i] for i in range(n_peaks_b)])
    lib.freeReleasedArray(masses_p)
    lib.freeReleasedArray(probs_p)
    lib.deleteFixedEnvelope(env_ptr, False)
    lib.deleteIso(iso_ptr)

    print(f"\n--- Binned output ---")
    print(f"  n_peaks={n_peaks_b}")
    b_probs_n = b_probs / b_probs.max()
    for i in range(min(10, n_peaks_b)):
        print(f"  {b_masses[i]:.6f}  {b_probs_n[i]:.6f}")


if __name__ == '__main__':
    print("="*60)
    print("ISOSPEC ALTERNATIVE STRATEGIES")
    print("="*60)
    bench_strategies()
    bench_binned_output_quality()
