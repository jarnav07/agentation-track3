"""Profile the ABIDES baseline outside Docker and attribute runtime to simulator components.

Runs ``abides_fork.simulate`` in-process under cProfile for each named scenario, then assigns every
function's *self* time (``tottime``) to one component by the file/function it lives in, so the
shares add up to the profiled wall time. Run it in the Python 3.11 baseline environment (pinned
stack + the four-patch ABIDES), with ``baselines/`` on ``PYTHONPATH``::

    PYTHONPATH=baselines python scripts/profile_baseline.py --out run_outputs/profile \
        regression_suite/scenarios/as06_throughput_fast.json ...

Writes ``<out>/<stem>.prof`` (raw cProfile) and ``<out>/profile_summary.json``; prints a table.
Startup/import time is measured separately in a fresh interpreter (``-X importtime`` is too noisy;
we time ``import abides_fork.simulate`` with ``perf_counter``).
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Ordered: first matching rule wins. (substring of filename, substring of function name or "").
RULES: list[tuple[str, str, str]] = [
    ("abides_fork/trace.py", "", "trace & ledger extraction (post-run)"),
    ("abides_core/utils.py", "parse_logs_df", "trace & ledger extraction (post-run)"),
    ("pyarrow", "", "parquet writing"),
    ("to_parquet", "", "parquet writing"),
    ("parquet", "", "parquet writing"),
    ("abides_core/kernel.py", "send_message", "message passing & latency model"),
    ("abides_fork/config.py", "get_latency", "message passing & latency model"),
    ("numpy/core/fromnumeric.py", "", "message passing & latency model"),  # np.clip in get_latency
    ("numpy/core/_methods.py", "_clip", "message passing & latency model"),
    ("'clip'", "", "message passing & latency model"),
    ("queue.py", "", "kernel event queue & dispatch"),
    ("heapq", "", "kernel event queue & dispatch"),
    ("threading.py", "", "kernel event queue & dispatch"),  # PriorityQueue's lock/condition
    ("abides_core/kernel.py", "", "kernel event queue & dispatch"),
    ("abides_markets/order_book.py", "", "order book insert/cancel/match"),
    ("abides_markets/price_level.py", "", "order book insert/cancel/match"),
    ("abides_core/agent.py", "logEvent", "trace & ledger recording (in-loop logging)"),
    ("copy.py", "", "deepcopy (orders/holdings/log events)"),
    ("abides_markets/orders.py", "__deepcopy__", "deepcopy (orders/holdings/log events)"),
    ("abides_markets/orders.py", "__str__", "debug string formatting (eager f-strings)"),
    ("abides_core/utils.py", "fmt_ts", "debug string formatting (eager f-strings)"),
    ("abides_markets/utils", "dollarize", "debug string formatting (eager f-strings)"),
    ("'format' of 'str'", "", "debug string formatting (eager f-strings)"),
    ("logging/__init__", "", "debug string formatting (eager f-strings)"),
    ("warnings", "", "debug string formatting (eager f-strings)"),
    ("mtrand", "", "RNG"),
    ("'lognormal'", "", "RNG"),
    ("'normal'", "", "RNG"),
    ("'randint'", "", "RNG"),
    ("'uniform'", "", "RNG"),
    ("'pareto'", "", "RNG"),
    ("'exponential'", "", "RNG"),
    ("sparse_mean_reverting_oracle", "", "oracle (fundamental value)"),
    ("abides_markets/agents/exchange_agent.py", "", "agent wakeup/receive_message logic"),
    ("abides_markets/agents/trading_agent.py", "", "agent wakeup/receive_message logic"),
    ("abides_fork/agents.py", "", "agent wakeup/receive_message logic"),
    ("abides_markets/messages", "", "agent wakeup/receive_message logic"),
    ("abides_markets/orders.py", "", "agent wakeup/receive_message logic"),
    ("abides_core/message.py", "", "agent wakeup/receive_message logic"),
    ("<string>", "__init__", "agent wakeup/receive_message logic"),  # dataclass __init__s
    ("pandas", "", "pandas (exchange metrics / summary log / book_log2)"),
    ("numpy", "", "numpy misc (book_log2 arrays etc.)"),
    ("abides_fork/config.py", "", "setup (config build)"),
    ("<frozen", "", "startup & import (in-process)"),
    ("importlib", "", "startup & import (in-process)"),
    ("isinstance", "", "agent wakeup/receive_message logic"),
]


def classify(func: tuple[str, int, str]) -> str:
    filename, _, name = func
    key = f"{filename}:{name}"
    for fsub, nsub, comp in RULES:
        if fsub in key and (not nsub or nsub in name):
            return comp
    return "other (builtins, misc)"


def profile_one(scenario: Path, out_dir: Path) -> dict:
    from abides_fork.simulate import simulate

    prof_path = out_dir / f"{scenario.stem}.prof"
    trace_out = out_dir / scenario.stem / "trace.parquet"
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr.enable()
        events = simulate(scenario, trace_out)
        pr.disable()
    wall = time.perf_counter() - t0
    pr.dump_stats(prof_path)
    st = pstats.Stats(str(prof_path))
    comps: dict[str, float] = {}
    total = 0.0
    for func, (_cc, _nc, tt, _ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
        comp = classify(func)
        comps[comp] = comps.get(comp, 0.0) + tt
        total += tt
    return {
        "scenario": scenario.stem,
        "n_events": events["n_events"],
        "profiled_wall_sec": wall,
        "profiled_total_tottime": total,
        "loop_wall_sec_selfreport": events["wall_clock_sec"],
        "components_sec": dict(sorted(comps.items(), key=lambda kv: -kv[1])),
    }


def startup_cost() -> float:
    code = "import time; t=time.perf_counter(); import abides_fork.simulate; print(time.perf_counter()-t)"
    vals = []
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        vals.append(float(out.stdout.strip()))
    return sorted(vals)[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--out", default="run_outputs/profile")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [profile_one(Path(s), out_dir) for s in args.scenarios]
    imp = startup_cost()
    agg: dict[str, float] = {}
    for r in results:
        for comp, sec in r["components_sec"].items():
            agg[comp] = agg.get(comp, 0.0) + sec
    total = sum(agg.values())
    summary = {"import_sec_median_of_3": imp, "per_scenario": results, "aggregate_sec": agg}
    (out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"import abides_fork.simulate (fresh interpreter, median of 3): {imp:.3f}s")
    print(f"{'component':55s} {'sec':>8s} {'share':>7s}")
    for comp, sec in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"{comp:55s} {sec:8.2f} {100 * sec / total:6.1f}%")
    for r in results:
        top = list(r["components_sec"].items())[:3]
        print(f"  {r['scenario']:32s} events={r['n_events']:7d} wall={r['profiled_wall_sec']:.2f}s top: "
              + ", ".join(f"{c.split(' (')[0]} {100 * s / r['profiled_total_tottime']:.0f}%" for c, s in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
