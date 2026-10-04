"""csrc/rng.h must reproduce NumPy's legacy RandomState draws bit for bit."""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _lib():
    so = os.path.join(tempfile.gettempdir(), "fastsim_rng_shim.so")
    subprocess.run(["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", so,
                    os.path.join(HERE, "..", "csrc", "rng_shim.cpp")], check=True)
    lib = ctypes.CDLL(so)
    lib.draw.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                         ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    return lib


def _c_draws(lib, rs, kind, p0, p1, n):
    _, key, pos, has_gauss, gauss = rs.get_state()
    key = np.ascontiguousarray(key, dtype=np.uint32)
    out = np.empty(n, dtype=np.float64)
    key_out = np.empty(624, dtype=np.uint32)
    pos_out = ctypes.c_int()
    lib.draw(key.ctypes.data, pos, has_gauss, gauss, kind, p0, p1, n, out.ctypes.data,
             key_out.ctypes.data, ctypes.byref(pos_out))
    return out, key_out, pos_out.value


CASES = [
    # (kind, numpy call, p0, p1)
    (0, lambda r, a, b: r.random_sample(), 0, 0),
    (1, lambda r, a, b: r.normal(a, b), 10.0, 2.0),
    (1, lambda r, a, b: r.normal(a, b), 100003.0, 31.622776601683793),
    (2, lambda r, a, b: r.lognormal(mean=a, sigma=b), float(np.log(500.0)), 0.3),
    (2, lambda r, a, b: r.lognormal(mean=a, sigma=b), float(np.log(25000.0)), 1.1),
    (3, lambda r, a, b: r.uniform(a, b), 100.0, 2000000.0),
    (4, lambda r, a, b: r.pareto(a), 1.5, 0),
    (4, lambda r, a, b: r.pareto(a), 2.7, 0),
    (5, lambda r, a, b: r.exponential(scale=a), 1.0 / 2.77778e-18, 0),
    (5, lambda r, a, b: r.exponential(scale=a), 1.0 / 2e-9, 0),
    (6, lambda r, a, b: r.randint(a, b), 0, 2),
    (6, lambda r, a, b: r.randint(a, b), 0, 6),
    (6, lambda r, a, b: r.randint(a, b), 0, 1),
    (6, lambda r, a, b: r.randint(a, b), 0, 11),
    (6, lambda r, a, b: r.randint(a, b), 0, 1000),
    (7, lambda r, a, b: r.randint(low=0, high=2**32, dtype="uint64"), 0, 0),
]


def test_rng_exact(n=200_000):
    lib = _lib()
    for seed in (1, 1040981657, 2**32 - 1):
        for kind, call, p0, p1 in CASES:
            rs = np.random.RandomState(seed)
            rs.normal()  # leave a cached gaussian in the state, as agents' streams often do
            got, key_out, pos_out = _c_draws(lib, rs, kind, p0, p1, n)
            p0n = int(p0) if kind == 6 else p0
            p1n = int(p1) if kind == 6 else p1
            want = np.array([call(rs, p0n, p1n) for _ in range(n)], dtype=np.float64)
            assert np.array_equal(got, want), (kind, p0, p1, seed, int(np.argmax(got != want)))
            _, key, pos, _, _ = rs.get_state()
            assert pos == pos_out and np.array_equal(key, key_out), ("state diverged", kind, p0, seed)


if __name__ == "__main__":
    test_rng_exact(int(sys.argv[1]) if len(sys.argv) > 1 else 200_000)
    print("rng exact: OK")
