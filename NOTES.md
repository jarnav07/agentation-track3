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
