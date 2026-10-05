# NOTES — fast, semantics-preserving ABIDES (Track 3)

A running log of what was read, built, measured, kept and reverted. Newest phase last.

## Executive summary

The goal is a Docker image whose `simulate` and `simulate-batch` verbs produce **byte-identical**
output to the ABIDES baseline, only faster. Correctness gates admission and speed only ranks
admissible entries, so every change is checked against a byte-equality oracle
(`scripts/check_identical.py`) before the official regression runner.

---

## Phase 0 — Orientation

Read, in order: `README.md`, `AGENTS.md`, `docs/CONCEPTS.md`, `docs/CATEGORIES.md`,
`SUBMISSION_CLI.md`, `baselines/README.md`, the exemplar (now at
`examples/t3-EXAMPLE-vectorized-matching/`, not `units/`), `docs/PROFILING.md`,
`docs/NVIDIA-STACK.md`. Skimmed `baselines/abides_fork/*`, `qfbench2_track_simulation/{semantics,
batch,telemetry}.py`, and the pinned ABIDES source (`abides-core` kernel/agent/message,
`abides-markets` exchange agent, order book, trading agent, sparse mean-reverting oracle) with the
four patches applied.

### Brief vs. repo conflicts (AGENTS.md wins)

| Brief says | Repo says | What I did |
|---|---|---|
| Install `qfbench2-common` at `v2.3.1` | README: "Do not pin `v2.3.1`"; AGENTS.md/CI pin **`v2.5.1`** (the scorer imports `digest_members`, absent before v2.4.4) | Installed **v2.5.1** (`pip … @v2.5.1`). |
| Use Python 3.13 | 3.13 for the **toolkit / scoring tools**; the ABIDES baseline image is deliberately **Python 3.11** (pandas 1.5.3 / numpy 1.26.4 publish no cp313 wheels; that stack generated the references) | Scoring venv = 3.13; baseline/profiling venv = 3.11. |
| "All 65 public scenarios" | 71 public units: 65 single-scenario + 6 batch (`t3-gbatch-*`) | Run both. |

### Output contract

**`simulate --config /input/scenario.json --out /output/trace.parquet`** (unit dir is mounted at
`/input`, read-only; `--network none`; exit 0). Writes to `/output/`:

`trace.parquet` — Snappy parquet, one row per exchange event, 7 columns, exactly these dtypes
(as written by the baseline through pandas 1.5.3 → pyarrow 15.0.2):

| column | dtype | notes |
|---|---|---|
| `t_ns` | int64 | event time, ns since epoch (absolute kernel clock, date 2021-02-05) |
| `agent_id` | int32 | exchange is 0 |
| `msg_type` | string | `ORDER_SUBMITTED`, `ORDER_ACCEPTED`, `ORDER_FILLED`, `PARTIAL_FILL`, `ORDER_CANCELLED`, `ORDER_REPLACED`, `QUOTE_UPDATE` |
| `side` | string | `BID` / `ASK` / null |
| `price` | int64 | ticks; 0 where n/a |
| `size` | int64 | 0 where n/a |
| `order_id` | int64 | -1 for `QUOTE_UPDATE` |

Row order: order-lifecycle rows (stable-sorted by `EventTime` over the per-agent log concatenation)
followed by quote rows (last `BEST_BID`/`BEST_ASK` per `(t_ns, side)`, ranked by first appearance),
then the whole frame stable-sorted by `(t_ns, order_id)`. The final `ORDER_EXECUTED` per order id
becomes `ORDER_FILLED`, earlier ones `PARTIAL_FILL`.

`message_trace.parquet` — one row per **delivered** (sub)message from the kernel ledger, sorted by
delivery `seq`:

| column | dtype |
|---|---|
| `seq` | int64 (contiguous 0..N-1) |
| `t_recv_ns` | int64 |
| `t_send_ns` | Int64 (nullable; null for `AGENT_WAKEUP`) |
| `latency_ns` | int64 (`t_recv - t_send`; 0 for wakeups) |
| `src_id`, `dst_id` | int32 |
| `message_id` | int64 |
| `msg_type` | string (ABIDES class name, or `AGENT_WAKEUP`) |
| `order_id` | Int64 (nullable) |
| `causal_parent` | Int64 (nullable; message id of the message/wakeup being processed when this was sent) |

`events.json` — required six: `scenario_id` (str), `seed` (int), `n_events` (int, must equal trace
row count), `wall_clock_sec`, `events_per_sec` (within ±5% of `n_events / wall_clock_sec`),
`trace_sha256`; plus telemetry `peak_memory_bytes` (int) and `gpu_seconds` (number). The baseline
also writes `n_messages` and `message_trace_sha256`.

**`simulate-batch --batch-dir /input/scenarios --out-dir /output`** — for each `<sub>.json`, writes
`/output/<sub>/{trace,message_trace}.parquet` + `events.json`, and `/output/batch_events.json` with
`n_scenarios`, `total_events`, `wall_clock_sec` (required — a missing key fails the whole batch),
`events_per_sec` (= total / wall), `per_scenario[{sub, n_events, trace_sha256}]`. Each sub must
reproduce its isolated reference (isolation gate).

The image must carry `LABEL qfbench2.interface_version="2.0"` and resolve the verb from `PATH`.

**Timing.** Development ranks the self-reported `events_per_sec` (capped at 1e7/unit). Final ranks
the organizer's measurement: parquet row count ÷ the Docker window `State.StartedAt → FinishedAt`
— so **interpreter start-up, imports and parquet writing count in Final** even though the baseline's
self-reported `wall_clock_sec` covers only `abides.run`. Ranking = arithmetic mean of per-unit rates
over the whole roster; a failed unit contributes 0. 300 s per-run limit in Final.

### Families and tiers

| # | family (`scenario_family`) | tier | public units |
|---|---|---|---|
| 1 | matching-engine-semantics (incl. `t3-mr-*`, `t3-s00*`, `t3-stp-*`) | A | 14 |
| 2 | agent-mix | B | 11 |
| 3 | latency-profile | A | 9 |
| 4 | oracle-noise | B | 0 public |
| 5 | calibration-stylized-facts | B (+ 4-metric stylized-fact gate) | 12 |
| 6 | throughput-scale (`t3-gb-*` single + 6 `t3-gbatch-*` batch) | A | 6 + 6 |
| 7 | exchange-protocol (`t3-mp*`) | A (+ g3.5 protocol-fidelity on the ledger) | 7 |
| 8 | reactive-agent (`t3-ra*`) | A | 6 |

Tier A = exact row count, exact fill `order_id/price/size` sequence, ±1 µs fill timestamps, exact
bidirectional event coverage, Kendall-τ ≥ 0.999. Tier B = return KS ≤ 0.08 and time-averaged spread
within ±10 bps. No majority rule anywhere. Latency models in public scenarios: `log_normal` 58,
`uniform` 3, `pareto` 3, `deterministic` 1.

### Message-ledger requirement (`[scoring.params].requires_message_ledger`, read from every card)

**59 / 71 true.** The 12 `false` are all throughput-scale: `t3-gb-base-30agent-30s`,
`t3-gb-highfreq-40hz-60s`, `t3-gb-horizon-240s`, `t3-gb-mega-throughput`, `t3-gb-pop-128-agents`,
`t3-gb-pop-horizon-scale`, and the six `t3-gbatch-*`. The batch cards' `false` does **not** exempt
them: `batch.score_subs` requires a ledger for every sub unconditionally. Decision: **always write
`message_trace.parquet`** (as the baseline does) — it costs little and removes a failure mode.

### Determinism and RNG constraints (from `baselines/abides_fork/config.py` + ABIDES source)

1. `build_config` calls `np.random.seed(scenario.seed)` and then draws, from the **global** NumPy
   RNG, one `uint64` in `[0, 2**32)` per `RandomState`, in this fixed order: oracle → (oracle
   constructor's `np.random.exponential` megashock draw, also global) → exchange → each scenario
   agent in `agent_configs` order × `count` → latency model → kernel. Agent ids follow the same
   order (exchange = 0).
2. Per-agent draws: `NoiseTrader.act` = `normal(size_mean, size_std)` → `randint(0,2)` →
   `randint(0, offset+1)`; `ValueTrader.act` = oracle `observe_price` (`normal`, after the oracle
   advances its own OU state with the oracle `RandomState` and possibly global `exponential`
   megashock draws); `MomentumTrader` and `MarketMaker` draw nothing.
3. Latency: one draw per `send_message` from the latency `RandomState` (log-normal / uniform /
   pareto, clipped, `int(round(...))` = banker's rounding), in **kernel send order**. Self-messages
   draw nothing (`sender == recipient → 0`). Wakeups draw nothing.
4. Event ordering: kernel `queue.PriorityQueue` of `(deliver_time, (sender_id, recipient_id,
   message))`. **Ties break on sender id, then recipient id, then `message_id`** (creation order) —
   *not* "ascending order_id" as the docs say. A requeued message (agent still "in the future")
   keeps its `message_id` and is re-inserted at the agent's free time.
5. `Message._Message__message_id_counter` and `Order._order_id_counter` are class globals reset per
   run by the adapter; ids are therefore deterministic and order-dependent. Only `Order.__init__`
   with `order_id=None` advances the order counter (`deepcopy` keeps the id); every `Message`
   constructed advances the message counter, including messages that are never delivered.
6. Timing details that must be reproduced: per-agent computation delay 50 ns (exchange: its
   `computation_delay`, 0 unless `protocol_enforcement`; reset to 0 for `MarketHoursRequestMsg`),
   exchange pipeline delay on accepted/executed/cancelled reports, wakeup re-arm at
   `t + interval_ns`, market open/close from `horizon_ns`, kernel stop = close + 1 s.
7. No wall-clock time, no dict-order hazards beyond insertion order, no threads in the baseline.
   Fresh-seed verification ("organizer verification reruns on fresh seeds") means the candidate
   must match the baseline **at any seed**, not just the scenario seeds — the oracle therefore
   tests fresh seeds too.

### Open questions

1. **Toolkit tag.** Resolved in favour of AGENTS.md (v2.5.1, not v2.3.1). Flagging in case the
   brief's pin was deliberate.
2. **Tie-break documentation mismatch** (item 4 above). I reproduce the baseline's actual rule,
   which is what the references were generated with. Not a blocker.
3. **`QFBENCH_SEED`.** SUBMISSION_CLI says the harness sets it; the baseline ignores it and reads
   the seed from `scenario.json`, and `regression_suite/README.md` says the seed is read from the
   file. I follow the baseline (ignore the env var).
4. **Self-reported `wall_clock_sec` boundary.** The baseline times only `abides.run`. I keep the
   same boundary (simulation, excluding parquet writing) so Dev numbers are comparable, while
   optimising whole-container time for Final.
5. **3 stylized-fact reports read `hill_abs = NaN`, not 0.0** (`s001`, `mp03`, `mp04` — small
   Tier-A units, too few returns for a top-100 Hill estimate). They are not Family-5 units and pass.

---

## Phase 1 — Baseline reproduced

Environment notes (this sandbox, not the eval host): no Docker daemon was running (started
`dockerd`); the build container could not reach `deb.debian.org` over plain HTTP (403) and needed
the proxy CA for PyPI. Workaround, used only for the local build and producing the same installed
packages: clone ABIDES at `f9cbe513…` on the host, apply the four patches in Dockerfile order, and
`COPY` the patched tree into a scratch copy of `baselines/` (pip via a BuildKit secret CA mount).
The committed `baselines/Dockerfile` is untouched. A local run of the built stack reproduces
`units/t3-as06-throughput-fast/{trace,message_trace}.parquet` **byte for byte**.

| check | result |
|---|---|
| `run_regression.py`, 65 single-scenario units | **65 / 65 PASS** (4 m 33 s wall, `--workers 4`) |
| `stylized_fact_report.json` | 62 / 65 all four = 0.0; 3 have `hill_abs = NaN` (see open question 5) |
| batch units via `simulate-batch` (`scripts/run_batch_units.py`, isolation + aggregate gates) | **6 / 6 PASS** |
| `timer.py`, `as06_throughput_fast`, 5 runs, warm-up discarded | **median 9,086 events/s** (host-measured container wall, ≈8.2 s/run; 74,502 events) |
| self-reported `events_per_sec` over the 65 units (loop-only clock) | geomean 12,661 (7,848 – 17,159) |

Baseline image: `track3-abides-baseline:latest`. Scoring venv: Python 3.13 + `qfbench2-common`
2.5.1. Profiling venv: Python 3.11 + the pinned stack + patched ABIDES.

---

## Phase 2 — Byte-equality oracle (`scripts/check_identical.py`)

For each (scenario, seed) it runs the candidate (`--candidate-image`, `--network none`, or
`--candidate-local "<cmd>"` on the host for fast iteration), fetches the baseline image's output for
the same pair from a cache keyed by baseline image id (`run_outputs/baseline_cache/<id>/`, filled by
running the baseline once), compares `trace.parquet` and `message_trace.parquet` by SHA-256 and, on
a mismatch, reports schema / row-count differences and the first diverging row with ±2 rows of
context from both sides. It also checks `events.json` (all 8 fields, `n_events` = row count,
`events_per_sec` within 5% of `n_events / wall_clock_sec`, `trace_sha256` correct).
`--extra-seeds N` adds N fresh seeds per scenario (derived like `timer.py`'s seed family), because
organizer verification re-runs on fresh seeds and Tier-A exactness has to hold at any seed.
`--batch` runs every batch unit through `simulate-batch` and compares each sub.

Validation:
* baseline vs. itself, 65 scenarios × (own seed + 2 fresh seeds) + 6 batch units: all identical
  (determinism of the baseline at fresh seeds confirmed);
* mutation test: a candidate that bumps one `ORDER_FILLED` price by one tick → `DIFF`, first
  diverging row 27, column `price`, shown with context. The oracle fails loudly.

Run order after every change: `check_identical.py` (stricter), then `run_regression.py`, then
`timer.py`.

---

## Phase 3 — Profile of the baseline (Python 3.11, outside Docker)

Tools: `scripts/profile_baseline.py` (cProfile, self-time attributed to components) over
`as06_throughput_fast` + one scenario from each family (`as01`, `ca_fat_tail_jumps`,
`fastlob_core`, `eq001_pareto_latency_tail`, `gb_base_30agent_30s`, `mp01`, `ra01`), and
`py-spy record --native` on `as06` (sampling, no per-call overhead). Raw profiles in
`run_outputs/profile/` (not committed).

**as06, py-spy (sampling), share of the whole process:**

| rank | component | share |
|---|---|---|
| 1 | trace & ledger extraction after the run (`parse_logs_df` + pandas in `abides_fork/trace.py`) | 17.2% |
| 2 | start-up & imports (interpreter, pandas, numpy, ABIDES) | 15.1% |
| 3 | agent `wakeup` / `receive_message` logic (TradingAgent/ExchangeAgent dispatch, `isinstance` chains, dataclass messages) | 13.8% |
| 4 | debug string formatting that runs even with logging off (eager f-strings → `Order.__str__` → `fmt_ts`, plus `warnings.warn`) | 13.1% |
| 5 | order book insert / cancel / match | 6.7% |
| 6 | exchange end-of-run metrics (`get_time_dropout`: `DataFrame.iterrows` over every book snapshot) | 6.4% |
| 7 | latency model (`np.clip` on a scalar dominates the draw itself) | 6.3% |
| 8 | `deepcopy` (orders, holdings dicts in `logEvent`, spread lists) | 5.8% |
| 9 | kernel event queue (`queue.PriorityQueue`: a lock + condition per put/get) | 5.3% |
| 10 | kernel dispatch loop | 3.1% |
| 11 | RNG (the NumPy draws themselves) | 2.4% |
| 12 | parquet writing | 2.2% |
| 13 | message passing (`Kernel.send_message` + ledger dict per message) | 2.0% |
| 14 | in-loop event logging (`logEvent` excl. its deepcopies) | 0.3% |

**All 8 scenarios, cProfile self-time (inflates small-call-heavy parts):** pandas 22.0% (mostly
`get_time_dropout` iterrows and `parse_logs_df`), agent logic 12.5%, kernel queue/dispatch 11.0%,
builtins misc 10.6%, debug formatting 9.8%, latency/message passing 9.1%, deepcopy 8.0%, order book
7.8%, post-run trace extraction 3.8%, numpy misc 1.9%, RNG 1.6%, parquet 0.6%. Fresh-interpreter
`import abides_fork.simulate`: 0.59 s. The ranking is the same in every family; no family has a
different bottleneck.

**Reading.** The simulation proper (order book + RNG + queue) is under a fifth of the time. Most of
the cost is bookkeeping nobody reads (string formatting, deep copies, book snapshots, end-of-run
metrics, a log-then-parse round trip through pandas) and interpreter/import overhead. Under Final's
container-window timing, start-up and output writing count, so they are first-class targets.

---

## Phase 4 — Optimisation log

Every step is checked in this order: `check_identical.py` (own seed + 2 fresh seeds per scenario,
plus batch) → `run_regression.py` → batch units → `timer.py` on `as06`.

Where the code lives: `fastsim/` (a new top-level directory — the submission image; nothing under
`baselines/` is modified). `fastsim/fastsim/scenario.py` reproduces `abides_fork/config.py`,
including every pre-loop NumPy draw in the baseline's order.

### Step 4.1 — Tier 1: a specialised pure-Python engine (`fastsim/fastsim/pyengine.py`) — KEPT

Rather than patching ABIDES piecemeal, I wrote a line-by-line re-statement of what the kernel,
exchange, order book, `TradingAgent` and the four scheduled agents *do* for the configurations a
scenario can express, keeping every observable and dropping every unobservable (Tier-1 items from
the brief, all at once):

* events recorded straight into columnar lists; the trace is assembled at the end with the
  baseline's exact row-order rules (`assemble.py`), with no log → pandas → parse round trip;
* no logging / eager f-strings / `warnings`, no deep copies of objects that are never mutated
  afterwards, no holdings or cash bookkeeping (no agent reads it), no order history, no `book_log2`
  snapshots, no end-of-run metrics, no summary-log pickle;
* `heapq` on `(t, sender, recipient, message_id, …)` tuples instead of `queue.PriorityQueue`
  (same order, no locks);
* parquet written with pyarrow directly, with the same `pandas` schema-metadata string — byte
  identical without importing pandas (`output.py`).

RNG: unchanged — the same NumPy `RandomState` objects, the same calls, in the same order.

| check | result |
|---|---|
| `check_identical.py --extra-seeds 2 --batch` (image `track3-fastsim:py`) | **201 / 201 byte-identical** (195 scenario×seed + 6 batch units) |
| `run_regression.py` | **65 / 65 PASS** |
| batch units | **6 / 6 PASS** |
| `timer.py` as06 (median, container wall) | **53,785 ev/s** — **5.9×** baseline (9,086); ≈1.3 s per run, mostly start-up |
| self-reported `events_per_sec`, geomean over 65 | **125,745** — **9.9×** baseline (12,661) |
| as06 loop time (local) | 0.61 s vs 6.43 s |

### Step 4.2 — Tier 2: the event loop, exchange, book and agents in C++ (`fastsim/csrc/`) — KEPT

A direct port of `pyengine.py` into a CPython extension (`fastsim._engine`), same algorithm,
compiled `-O3 -ffp-contract=off -fno-fast-math` (no FMA contraction, no reassociation: every
floating-point operation happens in the same order and precision as in NumPy / CPython).

* **RNG.** `csrc/rng.h` re-implements MT19937 and the NumPy *legacy* distributions the baseline
  calls (`normal` via the cached polar Box–Muller, `lognormal`, `uniform`, `pareto` = `exp(E/a)-1`,
  `exponential`, `randint` with masked rejection — and `randint(0, 1)` consuming nothing). State is
  imported verbatim from each `RandomState.get_state()` after Python has done all pre-loop seeding,
  so seeding stays NumPy's own. `fastsim/tests/test_rng.py` checks 16 call shapes × 3 seeds ×
  100k draws against NumPy, *including the generator state afterwards*: exact.
* **Kernel.** Binary heap of 32-byte events ordered `(t, sender, recipient, message_id)` with a
  payload pool; same re-queue rule for agents "in the future"; same "one more message past
  stop_time" loop condition.
* **Order book.** Price levels in a sorted vector with the best level at the back (insertions
  land near the back), FIFO intrusive lists per level with a running visible total, `oid → slot`
  index for O(1) cancel lookup (only valid when the request's side and price match, exactly like
  ABIDES's level search).
* **Oracle.** Timestamps are Python ints until a megashock turns them into `np.float64`; the port
  tracks that type and uses NumPy's int→float64 comparison semantics where the baseline compares a
  `np.float64` with an int (they differ from exact Python int/float comparison above 2**53).
* **Trace assembly** in C++ (stable sort by `(t, oid)`, quote de-duplication, merge); Python only
  wraps the column buffers for pyarrow.

| check | result |
|---|---|
| `check_identical.py --extra-seeds 2 --batch` (image `track3-fastsim:c1`) | **201 / 201 byte-identical** |
| `run_regression.py` | **65 / 65 PASS** |
| batch units | **6 / 6 PASS** |
| `timer.py` as06 (median, container wall) | **116,148 ev/s — 12.8×** baseline; 0.57 s per run |
| self-reported `events_per_sec`, geomean over 65 | **2,268,476 — 179×** baseline (min 488k, max 3.94M) |
| as06 engine time (local) | 33 ms vs 6.43 s baseline loop |

### Step 4.3 — Start-up and output: a native `simulate` binary — KEPT

**Measured problem.** Under Final's container-window timing (Docker `StartedAt → FinishedAt`) the
C++-engine image spent ~440 ms on `as06`: ~95 ms is the container runtime itself (a no-op
container), ~170 ms Python + NumPy + pyarrow imports, ~90 ms lazy imports and first-call costs, and
only ~25 ms in the engine. Profiling the in-process remainder showed parquet writing (~50 ms) as
the largest real cost.

**Change.** `fastsim-native` (`csrc/native_main.cpp`) does everything without an interpreter:

* scenario JSON parsed natively (`csrc/json.h`; Python `json` semantics: ints stay ints, last
  duplicate key wins, `strtod` = correctly rounded `float()`);
* `csrc/native_setup.cpp` mirrors `config.py` line by line with Python's `int()`/`float()`/
  truthiness rules, and reproduces NumPy's **legacy seeding** — verified that `np.random.seed(s)`
  and `RandomState(np.uint64(s))` are exactly `init_genrand(s)`;
* **`np.log` hazard found and handled.** The baseline computes the log-normal `mu` as
  `np.log(mean_ns)`. On this AVX-512 CPU, NumPy 1.26's float64 log is Intel **SVML**
  (`__svml_log8`), which differs from libm `log` in ~27% of inputs (measured over 2.7 M values).
  Using libm would silently change every latency draw. The native path vendors the exact SVML
  routine NumPy 1.26.4 links (`csrc/svml/`, numpy/SVML @ `1b21e45`, BSD-3) and uses it when the
  CPU has AVX512_SKX — NumPy's own dispatch condition — else libm, as NumPy does. Verified 0
  mismatches against `np.log` over 1.6 M inputs;
* parquet written by **the pyarrow wheel's own libarrow/libparquet** (linked from site-packages)
  with `pyarrow.parquet.write_table`'s default writer properties and the same pandas metadata
  string → identical bytes (checked also on a 300k-row table, where dictionary encoding falls back
  to plain pages); the two files are written and hashed concurrently;
* `simulate-batch` runs subs on a thread pool (one thread per available CPU, cgroup quota
  respected) — the engine has no global state, so each sub is byte-identical to its isolated run;
* **fallback:** anything the native setup is not sure about (non-numeric types, unknown agent
  types, no `latency_config` → line-distance model, seed out of range, engine error) re-execs the
  Python implementation, which is byte-identical by the same oracle. `FASTSIM_NO_FALLBACK=1` makes
  the oracle runs prove the native path itself was used.

Tried and **reverted**: Arrow's internal writer threading (`set_use_threads`) — byte-identical but
slower at these sizes (thread-pool start-up > gain).

**Deferred (not done):** a value-identical but not byte-identical writer (no dictionary, no
compression) would cut writing by ~60% (as06: 45 → 18 ms in pyarrow). It would keep every scored
value identical but lose SHA equality with the baseline; kept byte identity as the safer property.

| check | result |
|---|---|
| `check_identical` local native, 65 × 3 seeds, fallback disabled | **195 / 195 byte-identical** |
| as06 official window (StartedAt→FinishedAt, 5 runs) | **≈ 225 ms** (C++-engine Python image ≈ 440 ms; baseline ≈ 7,700 ms; no-op container ≈ 95 ms) |
| as06 process, local | 94–124 ms total: engine 22–30 ms, write+hash ≈ 55 ms, library load ≈ 11 ms |

### Step 4.4 — Engine micro-optimisations — KEPT

callgrind on `as06` (instructions inside `run_engine`): heap operations, `memcpy` from payload
copies and vector growth, the Gaussian draws, and out-of-line `push_back`s dominated. Changes: a
4-ary heap, a chunked message pool read by reference (no payload copy per delivery), reserved
output buffers. 120 M → 112 M instructions (−7%); `memcpy` halved. Byte-identical (195/195 local).
Further engine work has low return now: the engine is ~25 ms of a ~200 ms container window.

### Step 4.5 — Edge-case safety net for the sealed scenarios

`scripts/make_edge_scenarios.py` mutates public scenarios to reach paths the public set never or
barely exercises (exchange compute/pipeline delays → agent re-queueing, STP under delays, no
`latency_config` → Python fallback with the line-distance model, deterministic latency ties,
degenerate and float-typed parameters, tiny prices with negative limit prices, very heavy latency
tails, frequent megashocks + scheduled jump, seeds 0 and 2**32-1, all-default params, single agent).
Compared with the baseline image at two seeds each:

* **44 / 46 byte-identical.** The other 2 are `e22_no_liquidity_providers`, where **the baseline
  itself crashes** (`KeyError: 'order_id'` in `abides_fork/trace.py` when no order is ever placed);
  fastsim writes an empty trace. No reference can exist for such a scenario, so it cannot be sealed.
* `e17` (market makers re-quoting every 300 ns) was dropped: the baseline did not finish within
  15 minutes; same reasoning. An earlier variant with a 50 µs exchange compute delay was also
  pathological for the baseline (re-queue storm) and was softened to realistic delays.

---

## Phase 5 — Package and final checks (image `track3-fastsim:latest` = build `n3`)

Image: `fastsim/Dockerfile`, `LABEL qfbench2.interface_version="2.0"`, `simulate` and
`simulate-batch` on `PATH` (no `ENTRYPOINT`), 117 MB, fully offline at run time.

| check | result |
|---|---|
| `check_identical.py --extra-seeds 2 --batch` | **201 / 201 byte-identical** (65 × 3 seeds + 6 batch units) |
| determinism: two more full own-seed runs (+ batch) | **71 / 71** each; 190 output parquet files identical across the 3 runs |
| edge cases (Step 4.5) | 44 / 46 identical; the 2 others crash the baseline |
| `run_regression.py` | **65 / 65 PASS**; stylized-fact reports all 0.0 (same 3 NaN `hill_abs` as the baseline) |
| batch units (`simulate-batch`, isolation + aggregate gates) | **6 / 6 PASS**; `batch_events.json` carries `n_scenarios`, `total_events`, `wall_clock_sec`, `events_per_sec`, `per_scenario` |
| `events.json` | all 8 fields; `n_events` = row count; `events_per_sec` = `n_events / wall_clock_sec` (checked by the oracle on every run) |
| message ledger | written for every unit and every batch sub |
| firewall self-check (`qfbench2 manifest assert-public-safe` on all units) | pass (units untouched) |
| repo test suite (`pytest tests`) | 287 passed, 1 failed — the same `test_malicious_output` case fails on the untouched upstream commit (pre-existing, environment-related) |

**Throughput vs the baseline (same machine):**

| metric | baseline | fastsim | speedup |
|---|---|---|---|
| `timer.py` as06, median (host wall incl. docker client) | 9,086 ev/s | **190,949 ev/s** | **21.0×** |
| Final proxy: mean over 65 units of rows / Docker window (`scripts/window_rates.py`) | 9,181 ev/s | **349,261 ev/s** | **38.0×** |
| self-reported `events_per_sec` (Development's input), geomean over 65 | 12,661 | **2,459,234** | **194×** |
| batch units, aggregate `events_per_sec` | 11.9k–13.8k | 1.09M–1.76M | **91–135×** |

At these sizes the container runtime itself (~95 ms for a no-op container, identical for every
submission) is now about half of each window; the rest is ~11 ms library loading, ~25 ms engine
and ~55 ms parquet writing for a 75k-event unit. Larger scenarios (SS-BENCH) shift weight to the
engine and writer, which scale linearly.

Not done (optional): `profile.json` SimProfile for the Best Systems Diagnosis award.

### Open items for the organizer / user

1. **Byte identity vs. writer speed.** A value-identical, non-byte-identical parquet writer would
   save ~35 ms per 75k-event unit (~15–20% of the Final window at public sizes). Kept byte identity;
   say if you want the faster writer (oracle would then compare rows, not SHAs).
2. **Toolkit pin** v2.5.1 (repo rule) rather than v2.3.1 (brief).

---

## Phase 6 — Second speed pass (image build `n4`)

**Executive summary.** The simulator's own speed (what Development scores, self-reported
`events_per_sec`) went from a mean of ~2.5M to **~6.1M events/s** over the 65 single-scenario
units, with every output still byte-identical to the baseline. The 10M ev/s per-unit cap is not
reached: the remaining time is genuine per-event work. The Final-style number (rows ÷ the whole
container lifetime) is unchanged at ~360k ev/s, because Docker start-up and parquet writing dominate
it, not the simulation.

### What changed (all in `fastsim/csrc/`)

| change | why it is exact |
|---|---|
| **Timed window = the event loop only** (trace assembly moved after the timer) | The baseline times `abides.run` only; its `extract_trace` runs after the timer. Our assembly is the same step. Batch units still time the whole loop, writes included, as the baseline does |
| **Busy-recipient wait lists** instead of ABIDES's requeue loop | ABIDES re-pushes a message whose recipient is busy and pops it again, possibly many times (a burst of k messages to one agent = O(k²) pops; 711k of 981k pops on `mr-deep-book-state-size`). A requeue pop changes nothing but `now`, to a time already reached. Waiting events now sit in a per-agent heap with one proxy in the queue carrying exactly the key the first of them would have. Comment in `Engine::run` gives the argument |
| **Event queue = calendar + heap** | Near-future events (messages, proxies) in a ring of 64 ns buckets with a bitmap; far ones (next wakeups) in a 4-ary heap; same total order `(t, sender, recipient, mid)` over the union. Keys packed as `(t, sender<<32 \| recipient)` |
| **Memory: no mid-run growth, no zero-fill, pre-faulting** | Record buffers are reserved from a wakeup/order estimate (calibrated so none of the 65 public + 22 edge scenarios grows), output columns are default-initialised (fresh mappings are zero), and a helper thread populates pages a few MiB ahead of each fill point (`MADV_POPULATE_WRITE`, which never changes contents). Page faults had been ~half of the loop's wall time here |
| AoS record structs, inline append buffer, one per-order struct | fewer stores and no out-of-line `emplace_back` |

Tried and dropped: a thread pre-drawing the latency RNG stream (fixed-order stream, so exact, but
cross-core traffic cost what it saved); transparent huge pages (fewer faults, but compaction
stalls gave a long tail of 3–4× slower runs); a sorted-vector message queue (bad when ~100
messages are in flight); branch-free 128-bit key compares (more instructions).

### Checks on build `n4`

| check | result |
|---|---|
| local oracle, native binary, 65 × 3 seeds | **195 / 195 byte-identical** |
| image oracle, 65 × 2 seeds + 6 batch units | **136 / 136 byte-identical** |
| edge cases through the image (22 scenarios × 2 seeds, Python hand-off path included) | **44 / 44** |
| `run_regression.py` | **65 / 65 PASS** |
| batch units (`run_batch_units.py`) | **6 / 6 PASS** |

### Throughput (this sandbox; it is a noisy VM, ±15% run to run)

| metric | `n3` (Phase 5) | `n4` |
|---|---|---|
| self-reported `events_per_sec`, mean over 65 units, in-container (regression run) | ~2.52M | **6.10M** (median 6.16M) |
| batch units, mean of 6, in-container, back-to-back pairs | 1.09M / 1.31M | 1.17M / 1.50M (write-dominated: the batch timer includes parquet writing) |
| Final proxy (`window_rates.py`, mean over 65) | 358k | 362k |

Where the remaining loop time goes (as06, instructions): RNG (bit-exact MT19937 + glibc
`log`/`exp`) ~25%, order handling/book ~25%, queue ~15%, record writing ~10%. Reaching the 10M cap
on every unit would need roughly another 1.6× on the loop.

---

## Phase 7 — Third speed pass (image build `n5`)

**Executive summary.** Development scored build `n4` at 4.6M events/s (mean over 71 units). This
pass makes the simulation loop ~15–20% faster again and hashes output files ~4× faster, still
byte-identical. Locally, the single-scenario mean (one run per unit, like Development) went from
7.1M to 7.3M for the PGO step alone, and from 6.4M to 7.4M for the whole pass.

| change | why it is exact |
|---|---|
| Latency draws generated in blocks of 256 | The latency stream (`rngs[2]`) is consumed in a fixed order whatever the simulation does, so drawing ahead changes nothing but when the work happens; a tight loop of independent draws overlaps the libm calls (~10–14% loop time) |
| Noise-trader actions and value-trader observations pre-drawn per agent (blocks of 16) | Each agent's stream is consumed in a fixed per-action pattern (normal, randint, randint / one gaussian); `normal(loc, s) == loc + s * std_gauss()` exactly |
| MT19937 tempers a whole 624-word block at once | Same outputs; a draw becomes a load |
| `__exp_finite` / `__log_finite` instead of `exp` / `log` | glibc's same ifunc-selected cores without the errno wrapper; 0 mismatches over 2×10⁸ inputs, `tests/test_rng.py` passes |
| SHA-256 with the SHA-NI instructions when present | Same digest (checked against `sha256sum`, edge lengths 0–128 and large files); 40 ms → 9 ms for a 9.7 MB file — counts in batch units' timer and in Final's container window |
| PGO in the Docker build (`fastsim/pgo/`, 10 public scenarios) | Code layout only; floating-point flags unchanged. Falls back to a plain build if the training run fails |

Tried and dropped: an x86-64-v3 (AVX2) build behind a CPU-dispatching launcher (+0.3%, not worth
a second binary).

### Checks on build `n5`

| check | result |
|---|---|
| image oracle, 65 × 3 seeds + 6 batch units | **201 / 201 byte-identical** |
| edge cases through the image (22 × 2 seeds) | **44 / 44** |
| `run_regression.py` | **65 / 65 PASS** |
| batch units | **6 / 6 PASS** |
| `tests/test_rng.py` | pass |

| metric (this sandbox) | `n4` | `n5` |
|---|---|---|
| self-reported eps, mean over 65 units, in-container (regression run) | 6.10M | **7.40M** |
| batch units, in-container | 0.97M–1.50M | **1.57M–2.24M** |
