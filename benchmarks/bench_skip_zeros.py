"""
Bench: does skipping zero-count elements in setupIso help?
e.g., C6H12O6 has P=0, S=0, N=0 — skip those 3 elements.
"""
import time
import ctypes
import numpy as np
from find_mfs.isotopes._isospec_bridge import get_isotope_arrays

from IsoSpecPy.isoFFI import isoFFI
lib = ctypes.CDLL(str(isoFFI.libpath))

lib.setupIso.restype = ctypes.c_void_p
lib.setupIso.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)]
lib.setupThresholdFixedEnvelope.restype = ctypes.c_void_p
lib.setupThresholdFixedEnvelope.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool, ctypes.c_bool]
lib.confs_noFixedEnvelope.restype = ctypes.c_size_t
lib.confs_noFixedEnvelope.argtypes = [ctypes.c_void_p]
lib.deleteFixedEnvelope.restype = None
lib.deleteFixedEnvelope.argtypes = [ctypes.c_void_p, ctypes.c_bool]
lib.deleteIso.restype = None
lib.deleteIso.argtypes = [ctypes.c_void_p]
lib.massesFixedEnvelope.restype = ctypes.POINTER(ctypes.c_double)
lib.massesFixedEnvelope.argtypes = [ctypes.c_void_p]
lib.probsFixedEnvelope.restype = ctypes.POINTER(ctypes.c_double)
lib.probsFixedEnvelope.argtypes = [ctypes.c_void_p]
lib.freeReleasedArray.restype = None
lib.freeReleasedArray.argtypes = [ctypes.c_void_p]

n = 5000

# Full 6 elements (CHNOPS) with glucose C6H12O6
symbols_full = ['C', 'H', 'N', 'O', 'P', 'S']
counts_full = [6, 12, 0, 6, 0, 0]

iso_n_full, fm_full, fp_full = get_isotope_arrays(symbols_full)
n_full = len(symbols_full)
iso_n_c = (ctypes.c_int * n_full)(*iso_n_full.tolist())
ac_c = (ctypes.c_int * n_full)(*counts_full)
fm_c = (ctypes.c_double * len(fm_full))(*fm_full.tolist())
fp_c = (ctypes.c_double * len(fp_full))(*fp_full.tolist())

times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    iso = lib.setupIso(n_full, iso_n_c, ac_c, fm_c, fp_c)
    env = lib.setupThresholdFixedEnvelope(iso, 0.001, False, False)
    np_count = lib.confs_noFixedEnvelope(env)
    lib.deleteFixedEnvelope(env, False)
    lib.deleteIso(iso)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"6 elements (CHNOPS, 3 zero): median={np.median(arr):.2f} us, n_peaks={np_count}")

# Only non-zero elements (CHO)
symbols_nz = ['C', 'H', 'O']
counts_nz = [6, 12, 6]
iso_n_nz, fm_nz, fp_nz = get_isotope_arrays(symbols_nz)
n_nz = len(symbols_nz)
iso_n_c2 = (ctypes.c_int * n_nz)(*iso_n_nz.tolist())
ac_c2 = (ctypes.c_int * n_nz)(*counts_nz)
fm_c2 = (ctypes.c_double * len(fm_nz))(*fm_nz.tolist())
fp_c2 = (ctypes.c_double * len(fp_nz))(*fp_nz.tolist())

times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    iso = lib.setupIso(n_nz, iso_n_c2, ac_c2, fm_c2, fp_c2)
    env = lib.setupThresholdFixedEnvelope(iso, 0.001, False, False)
    np_count = lib.confs_noFixedEnvelope(env)
    lib.deleteFixedEnvelope(env, False)
    lib.deleteIso(iso)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"3 elements (CHO, no zeros):  median={np.median(arr):.2f} us, n_peaks={np_count}")

# 1 element (just C)
symbols_1 = ['C']
counts_1 = [6]
iso_n_1, fm_1, fp_1 = get_isotope_arrays(symbols_1)
n_1 = 1
iso_n_c3 = (ctypes.c_int * n_1)(*iso_n_1.tolist())
ac_c3 = (ctypes.c_int * n_1)(*counts_1)
fm_c3 = (ctypes.c_double * len(fm_1))(*fm_1.tolist())
fp_c3 = (ctypes.c_double * len(fp_1))(*fp_1.tolist())

times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    iso = lib.setupIso(n_1, iso_n_c3, ac_c3, fm_c3, fp_c3)
    env = lib.setupThresholdFixedEnvelope(iso, 0.001, False, False)
    np_count = lib.confs_noFixedEnvelope(env)
    lib.deleteFixedEnvelope(env, False)
    lib.deleteIso(iso)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"1 element (C only):          median={np.median(arr):.2f} us, n_peaks={np_count}")
