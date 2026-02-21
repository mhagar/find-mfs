"""
Quick test: can we use OpenMP prange for the candidate loop?
IsoSpec's C++ is thread-safe for independent Iso objects.
"""
import numpy as np
import os
import time

# Check if OpenMP is available
from find_mfs.isotopes._isospec import score_isotope_batch
from find_mfs.isotopes._isospec_bridge import get_isotope_arrays

symbols = ['C', 'H', 'N', 'O', 'P', 'S']
envelope = np.array([
    [180.063388, 1.00],
    [181.066743, 0.065],
    [182.068, 0.012],
])

# Generate many candidates
np.random.seed(42)
n_cands = 500
counts = np.zeros((n_cands, 6), dtype=np.int32)
counts[:, 0] = np.random.randint(3, 15, n_cands)  # C
counts[:, 1] = np.random.randint(5, 30, n_cands)  # H
counts[:, 2] = np.random.randint(0, 5, n_cands)   # N
counts[:, 3] = np.random.randint(0, 10, n_cands)  # O
counts[:, 4] = np.random.randint(0, 2, n_cands)   # P
counts[:, 5] = np.random.randint(0, 2, n_cands)   # S

n = 50
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    rmse, mf, nm = score_isotope_batch(symbols, counts, 0, envelope, 0.01)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)

arr = np.array(times)
print(f"500 candidates (current, serial): median={np.median(arr):.1f} us, per-cand={np.median(arr)/n_cands:.1f} us")

# Check CPU count
print(f"CPU count: {os.cpu_count()}")
