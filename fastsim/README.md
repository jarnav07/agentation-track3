# fastsim — a fast, byte-identical re-implementation of the Track 3 ABIDES baseline

## Executive summary

`fastsim` is a Track 3 submission image. It answers the same `simulate` and `simulate-batch`
commands as the ABIDES baseline (`baselines/`) and writes the **same files, byte for byte**:
`trace.parquet`, `message_trace.parquet`, `events.json` and, for batches, `batch_events.json`.
It does this between roughly 20× and 200× faster, depending on whether you count the whole
container or only the simulation. It keeps the exact rules of the exchange (price-time priority,
self-trade prevention, latency, the order in which messages arrive) and every random draw, so it
passes the correctness gates by reproducing the baseline rather than by approximating it.

```bash
docker build --platform=linux/amd64 -t track3-fastsim:latest fastsim/
docker run --rm --network none -v <unit>:/input:ro -v <out>:/output track3-fastsim:latest \
    simulate --config /input/scenario.json --out /output/trace.parquet
docker run --rm --network none -v <batch-unit>:/input:ro -v <out>:/output track3-fastsim:latest \
    simulate-batch --batch-dir /input/scenarios --out-dir /output
```

## How it is built

| layer | file | role |
|---|---|---|
| native CLI | `csrc/native_main.cpp` | `simulate` / `simulate-batch` with no interpreter. Batch subs run on a thread pool, one thread per available CPU |
| setup | `csrc/native_setup.cpp` | scenario JSON → engine spec, mirroring `baselines/abides_fork/config.py`, including NumPy's legacy seeding and draw order |
| engine | `csrc/engine.cpp`, `csrc/rng.h` | kernel event queue, exchange, order book, the four agent types, oracle; MT19937 + NumPy legacy distributions, bit-exact |
| writer | `csrc/writer.cpp` | parquet through the pyarrow 15.0.2 wheel's own libarrow/libparquet, with `pyarrow.parquet.write_table`'s default options |
| `np.log` | `csrc/svml/` | NumPy computes the log-normal `mu` with Intel SVML on AVX512_SKX CPUs (libm otherwise); this is the same SVML routine, vendored (BSD-3) |
| Python path | `fastsim/*.py` | the readable specification (`pyengine.py`) plus the CPython extension wrapper (`cengine.py`). The native binary hands any run to it when it is unsure |

**When the Python path is used instead.** The native binary falls back to
`python3 -m fastsim.simulate` (or `fastsim.simulate_batch`) in these cases: the JSON uses a shape
`native_setup.cpp` does not handle (a string where a number is expected, an unknown agent type, a
seed outside `[0, 2**32)`, a non-ASCII `scenario_id`, …); there is no `latency_config`, which
selects the baseline's line-distance latency model; or the engine reports an error. The Python
path reproduces the baseline's behaviour in those cases too, including the errors it raises for
malformed scenarios. Set `FASTSIM_VERBOSE=1` to log fallbacks and `FASTSIM_NO_FALLBACK=1` to
forbid them (useful in tests).

## How it is checked

* `fastsim/tests/test_rng.py` compares the C++ RNG with NumPy: 16 call shapes × 3 seeds × 100k
  draws, including the generator state afterwards.
* `scripts/check_identical.py` compares outputs with the baseline image by SHA-256 and then row
  by row. It covers the 65 public scenarios at their own seed and at fresh seeds, the 6 batch units
  per sub, and the edge cases from `scripts/make_edge_scenarios.py`.
* `regression_suite/run_regression.py` and `scripts/run_batch_units.py` are the official public
  gates.

`NOTES.md` at the repository root logs every step with its measured result.

## Environment variables

| variable | effect |
|---|---|
| `FASTSIM_BATCH_WORKERS` | number of batch worker threads (default: the CPUs available, cgroup quota included) |
| `FASTSIM_VERBOSE=1` | log when and why the Python path is used |
| `FASTSIM_NO_FALLBACK=1` | exit 3 instead of falling back (tests) |
| `FASTSIM_TIMING=1` | print the engine and write timings to stderr |
| `FASTSIM_ENGINE=py` | (Python path only) use the pure-Python engine instead of the extension |
